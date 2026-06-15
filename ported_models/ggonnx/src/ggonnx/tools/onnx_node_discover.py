#!/usr/bin/env python3
"""Discover ONNX node test outcomes per case, with fork-based isolation.

Each case runs in a forked child so a native crash (SIGSEGV/SIGABRT) only
kills that child — the parent records the outcome and moves on. Prints a
summary plus the list of crashers in a form you can paste into
``_KNOWN_CRASHERS`` in ``tests/test_onnx_node_conformance.py``.

Usage:
    tools/onnx_node_discover.py --ep-library build/libggonnx_ep.so
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import tempfile
from pathlib import Path

# Make `tests/` importable so we reuse the same session helpers.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

NODE_TEST_ROOT = Path(onnx.__file__).parent / "backend" / "test" / "data" / "node"

_RTOL = 1e-3
_ATOL = 1e-4


def _load_tensor(path: Path):
    return numpy_helper.to_array(onnx.load_tensor(str(path)))


def _graph_input_names(model: onnx.ModelProto) -> list[str]:
    init_names = {i.name for i in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in init_names]


def _run_case_in_process(case: Path, ep_library: Path) -> tuple[str, str]:
    """Returns (outcome, detail). Called inside a forked child."""
    from test_support import end_profiling_profile, ggml_session

    model_path = case / "model.onnx"
    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        return "skip", f"onnx.load: {e}"

    input_names = _graph_input_names(model)
    output_names = [o.name for o in model.graph.output]
    datasets = sorted(case.glob("test_data_set_*"))
    if not datasets:
        return "skip", "no test_data_set_*"

    try:
        session = ggml_session(model_path, ep_library)
    except Exception as e:
        return "skip", f"session create: {e}"

    results = []
    try:
        for ds in datasets:
            ins = sorted(ds.glob("input_*.pb"))
            outs = sorted(ds.glob("output_*.pb"))
            if len(ins) != len(input_names):
                return "skip", f"{ds.name}: input count mismatch"
            feeds = {n: _load_tensor(p) for n, p in zip(input_names, ins)}
            expected = [_load_tensor(p) for p in outs]
            try:
                actual = session.run(None, feeds)
            except Exception as e:
                return "skip", f"{ds.name}: run: {e}"
            results.append((ds.name, expected, actual))
    finally:
        profile = end_profiling_profile(session)

    node_events = [ev for ev in profile if ev.get("cat") == "Node"]
    if not node_events:
        return "skip", "no node events"
    non_ggml = [
        ev
        for ev in node_events
        if ev.get("args", {}).get("provider") != "GGMLExecutionProvider"
    ]
    if non_ggml:
        ops = sorted({ev.get("args", {}).get("op_name", "?") for ev in non_ggml})
        return "skip", f"fallback: {ops}"

    for ds_name, expected, actual in results:
        if len(actual) != len(expected):
            return "fail", f"{ds_name}: output count mismatch"
        for name, exp, got in zip(output_names, expected, actual):
            if exp.dtype.kind in ("U", "S", "O"):
                return "skip", f"{ds_name}: non-numeric dtype {exp.dtype}"
            try:
                np.testing.assert_allclose(got, exp, rtol=_RTOL, atol=_ATOL)
            except AssertionError as e:
                return "fail", f"{ds_name}: {name}: {str(e).splitlines()[0]}"
    return "pass", ""


def _worker_loop(cases: list[Path], ep_library: Path, write_fd: int) -> None:
    """Child process: run each case sequentially, streaming one result per case.

    Writing goes length-prefixed (4-byte big-endian) so the parent can split
    records even if the child dies mid-case (the parent will see no record
    for the crasher). GGML gets initialized once for the whole batch, which
    is the point of the long-lived worker."""
    import struct
    with os.fdopen(write_fd, "wb", buffering=0) as fp:
        for case in cases:
            try:
                result = _run_case_in_process(case, ep_library)
            except BaseException as e:
                result = ("skip", f"child exception: {type(e).__name__}: {e}")
            blob = pickle.dumps((case.name, result))
            fp.write(struct.pack(">I", len(blob)))
            fp.write(blob)
            fp.flush()


def _read_records(read_fd: int):
    """Generator yielding (case_name, (outcome, detail)) for each record
    the worker wrote before dying (or finishing)."""
    import struct
    buf = b""
    with os.fdopen(read_fd, "rb") as fp:
        while True:
            header = fp.read(4)
            if len(header) < 4:
                return
            (n,) = struct.unpack(">I", header)
            data = b""
            while len(data) < n:
                chunk = fp.read(n - len(data))
                if not chunk:
                    return
                data += chunk
            yield pickle.loads(data)


def _run_batch_with_resume(cases: list[Path], ep_library: Path):
    """Yield (case, outcome, detail) for every case, respawning the worker
    whenever it dies so one SIGSEGV doesn't abort the whole sweep."""
    remaining = list(cases)
    while remaining:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                _worker_loop(remaining, ep_library, w)
            finally:
                os._exit(0)
        os.close(w)

        done_names: set[str] = set()
        try:
            for name, (outcome, detail) in _read_records(r):
                done_names.add(name)
                case = next(c for c in remaining if c.name == name)
                yield case, outcome, detail
        finally:
            _, status = os.waitpid(pid, 0)

        # Trim off everything the worker reported and handle the next case as
        # a crash if the worker died before reporting it.
        unreported = [c for c in remaining if c.name not in done_names]
        if not unreported:
            return
        next_case = unreported[0]
        if os.WIFSIGNALED(status):
            detail = f"signal {signal.Signals(os.WTERMSIG(status)).name}"
        elif os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
            detail = f"exit status {os.WEXITSTATUS(status)}"
        else:
            # Clean exit with no record for this case shouldn't normally
            # happen — treat as a crash so it's visible instead of silently
            # skipped.
            detail = "worker exited without reporting case"
        yield next_case, "crash", detail
        remaining = unreported[1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-library", required=True, type=Path)
    ap.add_argument("--filter", default=None, help="substring filter on case names")
    ap.add_argument(
        "--limit", type=int, default=None, help="only process first N matching cases"
    )
    args = ap.parse_args()

    ep_library = args.ep_library.resolve()
    cases = sorted(
        p for p in NODE_TEST_ROOT.iterdir() if p.is_dir() and (p / "model.onnx").exists()
    )
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]
    if args.limit:
        cases = cases[: args.limit]

    tally = {"pass": 0, "skip": 0, "fail": 0, "crash": 0}
    crashers: list[str] = []
    failures: list[tuple[str, str]] = []

    # Dedicated cache dir per run so parallel invocations don't race (and so
    # the parent's temp clutter lands somewhere obvious).
    tempfile.tempdir = tempfile.mkdtemp(prefix="onnx_node_discover_")

    total = len(cases)
    for i, (case, outcome, detail) in enumerate(
        _run_batch_with_resume(cases, ep_library), 1
    ):
        tally[outcome] += 1
        if outcome == "crash":
            crashers.append(case.name)
            print(f"[{i}/{total}] CRASH  {case.name}  ({detail})", flush=True)
        elif outcome == "fail":
            failures.append((case.name, detail))
            print(f"[{i}/{total}] FAIL   {case.name}  ({detail})", flush=True)
        elif outcome == "pass":
            print(f"[{i}/{total}] pass   {case.name}", flush=True)
        # skips are quiet unless we want them

    print()
    print(f"pass:  {tally['pass']}")
    print(f"fail:  {tally['fail']}")
    print(f"crash: {tally['crash']}")
    print(f"skip:  {tally['skip']}")
    print(f"total: {sum(tally.values())}")

    if crashers:
        print("\nKnown crashers (paste into tests/test_onnx_node_conformance.py):")
        for name in crashers:
            print(f'    "{name}",')
    if failures:
        print("\nFailures:")
        for name, detail in failures:
            print(f"  {name}: {detail}")

    return 0 if tally["fail"] == 0 and tally["crash"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
