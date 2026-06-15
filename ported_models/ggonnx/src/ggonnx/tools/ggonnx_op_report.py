#!/usr/bin/env python3
"""Report which ONNX ops in a model run on GGML vs fall back to other EPs.

Builds a session with the GGONNX EP registered, runs one inference with
dummy inputs, and prints a breakdown from the ORT profile grouping node
events by (provider, op_name). Useful to see at a glance which ops the
GGONNX partition still loses to CPU.

Dynamic input dims must be pinned with --dim NAME AXIS VALUE (repeatable).
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

_EP_NAME = "GGMLExecutionProvider"


def register_ep(ep_library: Path) -> None:
    ort.register_execution_provider_library("GGONNX", str(ep_library))


def ggml_devices():
    devices = [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    if not devices:
        raise RuntimeError(f"{_EP_NAME} not discovered after library registration")
    return devices


_ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def _concretize_input_dims(
    model_path: Path, dim_overrides: dict[tuple[str, int], int]
) -> Path:
    if not dim_overrides:
        return model_path
    static_path = Path(tempfile.gettempdir()) / (
        model_path.stem + "_ggonnx_report_static.onnx"
    )
    model = onnx.load(str(model_path))
    for graph_input in model.graph.input:
        shape = graph_input.type.tensor_type.shape
        for axis, slot in enumerate(shape.dim):
            override = dim_overrides.get((graph_input.name, axis))
            if override is None:
                continue
            slot.ClearField("dim_param")
            slot.dim_value = int(override)
    onnx.save(shape_inference.infer_shapes(model), str(static_path))
    return static_path


def _make_dummy_input(info, rng: np.random.Generator) -> np.ndarray:
    dtype = _ORT_DTYPE_TO_NUMPY.get(info.type)
    if dtype is None:
        raise RuntimeError(
            f"unsupported input dtype {info.type!r} for input {info.name!r}"
        )
    shape = []
    for axis, dim in enumerate(info.shape):
        if not isinstance(dim, int) or dim <= 0:
            raise RuntimeError(
                f"input {info.name!r} has non-static dim at axis {axis} ({dim!r}); "
                f"pin it with --dim {info.name} {axis} <value>"
            )
        shape.append(dim)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    return rng.integers(0, 2, size=shape).astype(dtype)


def _build_session(
    model_path: Path, provider: str, ep_library: Path | None
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(Path(tempfile.gettempdir()) / "ggonnx_op_report")
    if provider == "cpu":
        so.add_provider("CPUExecutionProvider", {})
    elif provider == "ggml":
        register_ep(ep_library)
        so.add_provider_for_devices(ggml_devices(), {})
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return ort.InferenceSession(str(model_path), sess_options=so)


def _static_op_counts(model_path: Path) -> tuple[Counter, Counter, int, int]:
    # Walks the ONNX model and returns (top_level_counts, subgraph_counts,
    # top_level_total, subgraph_total). Subgraph counts come from Loop/If/Scan
    # bodies — these ops re-execute per iteration at runtime, so event counts
    # above inflate them relative to the static graph size.
    model = onnx.load(str(model_path))
    top: Counter = Counter()
    sub: Counter = Counter()

    def walk(graph, into_counter):
        for node in graph.node:
            into_counter[node.op_type] += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g, sub)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        walk(g, sub)

    walk(model.graph, top)
    return top, sub, sum(top.values()), sum(sub.values())


def _collect_node_events(session: ort.InferenceSession) -> list[dict]:
    profile_path = Path(session.end_profiling())
    with profile_path.open() as f:
        profile = json.load(f)
    return [e for e in profile if e.get("cat") == "Node"]


def _print_counter(title: str, counter: Counter) -> None:
    if not counter:
        return
    width = max(len(key) for key in counter)
    total = sum(counter.values())
    print(f"\n{title} (total {total}):")
    for key, count in counter.most_common():
        print(f"  {key.ljust(width)}  {count}")


def parse_dim(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--dim must be NAME:AXIS:VALUE, got {value!r}"
        )
    name, axis, size = parts
    return name, int(axis), int(size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path, help="Path to .onnx model")
    p.add_argument(
        "--ep-library",
        type=Path,
        default=Path("build/libggonnx_ep.so"),
        help="Path to libggonnx_ep.so",
    )
    p.add_argument(
        "--dim",
        action="append",
        default=[],
        type=parse_dim,
        metavar="NAME:AXIS:VALUE",
        help="Pin a dynamic input dim (repeatable), e.g. --dim input_1:2:416",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for dummy inputs",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    dim_overrides = {(name, axis): value for name, axis, value in args.dim}
    model_path = _concretize_input_dims(args.model, dim_overrides)

    ep_library = args.ep_library.resolve()
    session = _build_session(model_path, "ggml", ep_library)

    rng = np.random.default_rng(args.seed)
    feeds = {inp.name: _make_dummy_input(inp, rng) for inp in session.get_inputs()}
    output_names = [o.name for o in session.get_outputs()]
    session.run(output_names, feeds)

    node_events = _collect_node_events(session)
    if not node_events:
        print("no node events in profile", file=sys.stderr)
        return 1

    by_provider: Counter = Counter()
    cpu_ops: Counter = Counter()
    other_by_provider: dict[str, Counter] = {}
    # GGML fuses a partition into a single node, so each GGML event == one
    # partition. For CPU we approximate "islands" — contiguous runs of CPU ops
    # between GGML partitions — by ordering events by start timestamp. This is
    # not exact (ORT can interleave independent chains) but it's a useful
    # fragmentation signal.
    ts_ordered: list[tuple[int, str]] = []
    for event in node_events:
        args_ = event.get("args", {})
        provider = args_.get("provider", "?")
        op_name = args_.get("op_name", "?")
        by_provider[provider] += 1
        if provider == _EP_NAME:
            pass
        elif provider == "CPUExecutionProvider":
            cpu_ops[op_name] += 1
        else:
            other_by_provider.setdefault(provider, Counter())[op_name] += 1
        ts_ordered.append((int(event.get("ts", 0)), provider))

    ts_ordered.sort(key=lambda x: x[0])
    islands: Counter = Counter()
    prev = None
    for _, provider in ts_ordered:
        if provider != prev:
            islands[provider] += 1
        prev = provider

    ggml_partitions = by_provider.get(_EP_NAME, 0)

    top_counts, sub_counts, top_total, sub_total = _static_op_counts(model_path)

    print(f"model:  {args.model}")
    print(
        f"static graph: {top_total} top-level nodes, "
        f"{sub_total} nodes in Loop/If/Scan bodies "
        f"(body ops re-execute per iteration, event counts below may be inflated)"
    )
    print(f"runtime events: {len(node_events)}")
    print(f"ggml partitions (fused nodes, exact): {ggml_partitions}")
    _print_counter("static top-level op counts", top_counts)
    if sub_counts:
        _print_counter("static subgraph-body op counts per iteration", sub_counts)
    _print_counter("events by provider", by_provider)
    _print_counter(
        "contiguous islands by provider",
        islands,
    )
    _print_counter("CPU-side ops (unsupported or bounced)", cpu_ops)
    for provider, counter in other_by_provider.items():
        _print_counter(f"{provider} ops", counter)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
