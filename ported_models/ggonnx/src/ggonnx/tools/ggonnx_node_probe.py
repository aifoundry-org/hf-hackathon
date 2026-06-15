#!/usr/bin/env python3
"""Probe individual nodes in isolation to distinguish support-check failures
from fragmentation (nodes the EP supports but can't reach because neighbours
are unsupported).

Each probe runs in a forked child so a native GGML assertion crash only kills
that child and doesn't abort the whole sweep.

For each node whose op_type is registered in the EP, the script constructs a
minimal single-node ONNX model (preserving exact shapes, dtypes, and attributes)
and creates an EP session for it.  If the EP accepts the node in isolation we
know the support check passes and any CPU assignment in the full model is purely
fragmentation.  If the EP rejects it in isolation the support check itself is
failing and that is the thing to fix.

Dynamic input dims must be pinned the same way as in ggonnx_op_report.py:
  --dim NAME:AXIS:VALUE   (repeatable)

Run with GGONNX_TRACE_SUPPORT=1 to get per-condition rejection reasons from the
EP (requires the support_trace.hpp macros to be active in ops.cpp).

Usage examples
--------------
# Full sweep of all registered op types (slow — probes every supported node):
  python3 tools/ggonnx_node_probe.py model.onnx \\
    --ep-library build/libggonnx_ep.so \\
    --dim input:0:1 --dim input:1:20

# Probe specific ops only (fast — use this while iterating on a fix):
  python3 tools/ggonnx_node_probe.py model.onnx \\
    --ep-library build/libggonnx_ep.so \\
    --dim input:0:1 --dim input:1:20 \\
    --op Slice --op Expand

# Show only ops with failures (skip the "ok" rows):
  python3 tools/ggonnx_node_probe.py model.onnx \\
    --ep-library build/libggonnx_ep.so \\
    --dim input:0:1 --dim input:1:20 \\
    --only-failing

# Get rejection reasons for a specific op (requires SUPPORT_REJECT macros in
# ops.cpp and a rebuild; src/debug/support_trace.hpp has the instructions):
  GGONNX_TRACE_SUPPORT=1 python3 tools/ggonnx_node_probe.py model.onnx \\
    --ep-library build/libggonnx_ep.so \\
    --dim input:0:1 --dim input:1:20 \\
    --op ReduceMean

Interpreting the output
-----------------------
  ggml   — EP accepted the node in isolation.  If it still goes to CPU in the
            full model (see ggonnx_op_report.py) the cause is fragmentation, not
            a support-check bug.
  cpu    — EP rejected the node in isolation.  The support check is the problem.
  bad    — crash (support check too permissive: GGML asserted) or session build
            failure.  Either the support check lets through a shape it shouldn't,
            or the node config is fundamentally unsupported.
"""

import argparse
import json
import os
import signal
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference

_EP_NAME = "GGMLExecutionProvider"

# Op types registered in the EP.  Keep in sync with FindOpDefinition in ops.cpp.
_SUPPORTED_OPS = {
    "Add", "Sub", "Mul", "Div", "Max", "Min",
    "Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log",
    "Erf", "Softplus", "Elu", "LeakyRelu", "PRelu", "Clip",
    "Pow", "CumSum",
    "GRU", "LSTM",
    "MatMul", "Gemm",
    "Conv", "ConvTranspose",
    "Softmax",
    "InstanceNormalization", "BatchNormalization",
    "Reshape", "Flatten", "Squeeze", "Unsqueeze",
    "Expand", "Transpose", "Slice", "Split", "Concat", "DepthToSpace", "Pad",
    "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    "Upsample", "Resize",
    "ReduceMean", "ReduceSum",
    "Identity",
}

_ONNX_DTYPE_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}


def _register_ep(ep_library: Path) -> list:
    ort.register_execution_provider_library("GGONNX", str(ep_library))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    if not devices:
        raise RuntimeError(f"{_EP_NAME} not found after library registration")
    return devices


def _concretize(model_path: Path, dim_overrides: dict) -> Path:
    if not dim_overrides:
        return model_path
    out = Path(tempfile.gettempdir()) / (model_path.stem + "_probe_static.onnx")
    m = onnx.load(str(model_path))
    for inp in m.graph.input:
        for (name, axis), val in dim_overrides.items():
            if inp.name == name:
                inp.type.tensor_type.shape.dim[axis].ClearField("dim_param")
                inp.type.tensor_type.shape.dim[axis].dim_value = val
    onnx.save(shape_inference.infer_shapes(m), str(out))
    return out


def _build_value_info_map(model: onnx.ModelProto) -> dict:
    vi = {}
    for v in model.graph.input:
        vi[v.name] = v
    for v in model.graph.output:
        vi[v.name] = v
    for v in model.graph.value_info:
        vi[v.name] = v
    return vi


def _build_initializer_map(model: onnx.ModelProto) -> dict:
    return {init.name: init for init in model.graph.initializer}


def _vi_shape(vi) -> list:
    t = vi.type.tensor_type
    return [
        d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
        for d in t.shape.dim
    ]


def _vi_dtype(vi) -> int:
    return vi.type.tensor_type.elem_type


def _probe_node(
    node: onnx.NodeProto,
    vi_map: dict,
    init_map: dict,
    opset: int,
    devices: list,
) -> str:
    """Return 'ggml', 'cpu', or 'error:<msg>'."""
    inputs_vi = []
    graph_inputs = []
    initializers = []
    feeds = {}

    for inp_name in node.input:
        if not inp_name:
            # optional absent input — skip
            continue
        if inp_name in init_map:
            # Constant initializer: include as-is.
            initializers.append(init_map[inp_name])
        elif inp_name in vi_map:
            vi = vi_map[inp_name]
            dtype_int = _vi_dtype(vi)
            shape = _vi_shape(vi)
            np_dtype = _ONNX_DTYPE_TO_NUMPY.get(dtype_int, np.float32)
            if np.issubdtype(np_dtype, np.floating):
                arr = np.ones(shape, dtype=np_dtype)
            else:
                arr = np.ones(shape, dtype=np_dtype)
            # Non-float small tensors (axes, shapes, indices) must look like
            # compile-time constants to the EP's support check.  Inject them as
            # ONNX initializers so IsConstantInitializer() returns true.
            total_elems = int(np.prod(shape)) if shape else 1
            is_const_like = (not np.issubdtype(np_dtype, np.floating) and total_elems <= 16)
            if is_const_like:
                initializers.append(numpy_helper.from_array(arr, name=inp_name))
            else:
                graph_inputs.append(vi)
                feeds[inp_name] = arr
        # else: unknown input — omit (node may reject, that's fine)

    if not graph_inputs and not initializers:
        return "error:no_inputs"

    # Build output value_info for the graph.
    graph_outputs = []
    for out_name in node.output:
        if not out_name:
            continue
        if out_name in vi_map:
            graph_outputs.append(vi_map[out_name])
        else:
            # Shape unknown — create a scalar float placeholder.
            graph_outputs.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            )

    if not graph_outputs:
        return "error:no_outputs"

    graph = helper.make_graph(
        [node],
        "probe",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model_proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model_proto.ir_version = 8

    try:
        model_proto = shape_inference.infer_shapes(model_proto)
    except Exception:
        pass  # best-effort

    tmp = Path(tempfile.gettempdir()) / "ggonnx_probe_node.onnx"
    onnx.save(model_proto, str(tmp))

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(Path(tempfile.gettempdir()) / "ggonnx_probe")
    so.add_provider_for_devices(devices, {})

    try:
        sess = ort.InferenceSession(str(tmp), sess_options=so)
    except Exception as e:
        return f"error:session:{e}"

    out_names = [o.name for o in sess.get_outputs()]
    actual_feeds = {i.name: feeds[i.name] for i in sess.get_inputs() if i.name in feeds}

    inference_error = None
    try:
        sess.run(out_names, actual_feeds)
    except Exception as e:
        inference_error = e

    # Always read the profile — even a failed run may have dispatched to GGML
    # before hitting an error (e.g. shape mismatch with dummy inputs).
    try:
        profile_path = Path(sess.end_profiling())
        with profile_path.open() as f:
            events = json.load(f)
    except Exception:
        return f"error:profile:{inference_error}"

    for ev in events:
        if ev.get("cat") != "Node":
            continue
        provider = ev.get("args", {}).get("provider", "")
        if _EP_NAME in provider:
            return "ggml"

    if inference_error:
        return f"error:inference:{inference_error}"
    return "cpu"


def _probe_with_fork(
    node: onnx.NodeProto,
    vi_map: dict,
    init_map: dict,
    opset: int,
    devices: list,
) -> str:
    """Run _probe_node in a forked child; return 'crash:<sig>' if it dies."""
    result_r_fd, result_w_fd = os.pipe()
    stderr_r_fd, stderr_w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(result_r_fd)
        os.close(stderr_r_fd)
        try:
            os.dup2(stderr_w_fd, 2)
            result = _probe_node(node, vi_map, init_map, opset, devices)
            encoded = result.encode()
            os.write(result_w_fd, len(encoded).to_bytes(4, "little") + encoded)
        finally:
            os.close(result_w_fd)
            os.close(stderr_w_fd)
            os._exit(0)

    os.close(result_w_fd)
    os.close(stderr_w_fd)
    result_data = b""
    stderr_data = b""
    while True:
        chunk = os.read(result_r_fd, 4096)
        if not chunk:
            break
        result_data += chunk
    while True:
        chunk = os.read(stderr_r_fd, 4096)
        if not chunk:
            break
        stderr_data += chunk
    os.close(result_r_fd)
    os.close(stderr_r_fd)
    _, status = os.waitpid(pid, 0)

    stderr_text = stderr_data.decode(errors="replace")
    support_lines = [
        line.strip()
        for line in stderr_text.splitlines()
        if "[ggonnx][support] reject" in line
    ]

    if os.WIFSIGNALED(status):
        detail = f" [{support_lines[-1]}]" if support_lines else ""
        return f"crash:signal_{signal.Signals(os.WTERMSIG(status)).name}{detail}"
    if len(result_data) >= 4:
        length = int.from_bytes(result_data[:4], "little")
        result = result_data[4 : 4 + length].decode(errors="replace")
        if result == "cpu" and support_lines:
            return f"cpu:{support_lines[-1]}"
        if result.startswith("error:session") and support_lines:
            return f"{result} [{support_lines[-1]}]"
        return result
    return "crash:no_result"


def parse_dim(value: str):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--dim must be NAME:AXIS:VALUE, got {value!r}")
    return parts[0], int(parts[1]), int(parts[2])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--ep-library", type=Path, default=Path("build/libggonnx_ep.so"))
    p.add_argument("--dim", action="append", default=[], type=parse_dim, metavar="NAME:AXIS:VALUE")
    p.add_argument("--op", action="append", default=[], dest="ops",
                   help="Only probe nodes of this op type (repeatable)")
    p.add_argument("--only-failing", action="store_true",
                   help="Only print nodes that fail in isolation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    ep_library = args.ep_library.resolve()
    devices = _register_ep(ep_library)

    dim_overrides = {(name, axis): val for name, axis, val in args.dim}
    model_path = _concretize(args.model, dim_overrides)
    model = onnx.load(str(model_path))

    opset = next(
        (o.version for o in model.opset_import if o.domain == ""),
        15,
    )

    vi_map = _build_value_info_map(model)
    init_map = _build_initializer_map(model)

    filter_ops = set(args.ops) if args.ops else _SUPPORTED_OPS

    # Collect results grouped by op type.
    results: dict[str, list[tuple[str, str]]] = defaultdict(list)

    nodes_to_probe = [n for n in model.graph.node if n.op_type in filter_ops]
    total = len(nodes_to_probe)
    print(f"Probing {total} nodes of supported op types...", file=sys.stderr)

    for i, node in enumerate(nodes_to_probe, 1):
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}", file=sys.stderr)
        result = _probe_with_fork(node, vi_map, init_map, opset, devices)
        node_id = node.name or f"{node.op_type}_{i}"
        results[node.op_type].append((node_id, result))

    # Summary
    print()
    print(f"{'Op':<22} {'total':>6} {'ggml':>6} {'cpu':>6} {'bad':>6}  verdict")
    print("-" * 72)

    any_failures = False
    for op_type in sorted(results):
        items = results[op_type]
        n_ggml  = sum(1 for _, r in items if r == "ggml")
        n_cpu   = sum(1 for _, r in items if r.startswith("cpu"))
        # crash = support check too permissive (accepted then died in GGML)
        # error:session = node build failed (likely genuinely unsupported shape/config)
        # error:inference = accepted but dummy inputs caused a CPU-side failure
        n_bad   = sum(1 for _, r in items if r.startswith("crash") or r.startswith("error:session"))
        total_n = len(items)

        if n_cpu == 0 and n_bad == 0:
            verdict = "ok - all accepted"
        elif n_ggml == 0 and n_bad == 0:
            verdict = "FAIL - all rejected in isolation"
            any_failures = True
        elif n_bad > 0 and n_ggml == 0 and n_cpu == 0:
            verdict = "BUG - accepted then crashed in GGML"
            any_failures = True
        else:
            parts = []
            if n_cpu:   parts.append(f"{n_cpu} rejected")
            if n_ggml:  parts.append(f"{n_ggml} accepted")
            if n_bad:   parts.append(f"{n_bad} crash/session-err")
            verdict = "MIXED - " + ", ".join(parts)
            any_failures = True

        if args.only_failing and n_cpu == 0 and n_bad == 0:
            continue

        print(f"  {op_type:<20} {total_n:>6} {n_ggml:>6} {n_cpu:>6} {n_bad:>6}  {verdict}")

    if any_failures:
        print()
        print("Failure details:")
        for op_type in sorted(results):
            for node_id, result in results[op_type]:
                if result == "ggml":
                    continue
                print(f"  {op_type:<20} {node_id}: {result}")
        if os.environ.get("GGONNX_TRACE_SUPPORT") in (None, "", "0"):
            print()
            print("Set GGONNX_TRACE_SUPPORT=1 to capture per-condition rejection reasons.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
