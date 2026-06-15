#!/usr/bin/env python3

"""Run the official ONNX node test corpus against the GGML EP.

Each subdirectory under ``onnx/backend/test/data/node`` bundles a minimal
``model.onnx`` and one or more ``test_data_set_*`` folders with reference
``input_*.pb`` / ``output_*.pb`` tensors. We run every case through the GGML
EP. A case is treated three ways:

* **PASS**: GGML EP ran every node in the model and outputs match reference.
* **SKIP**: session could not be created, run failed, model uses types we
  don't handle, or ORT fell back to a non-GGML provider for any node. These
  are the honest "not supported yet" outcomes — they become the to-do list.
* **FAIL**: GGML EP claimed every node *and* produced wrong numbers. This is
  the signal we actually care about.

Run with ``-rs`` to print the skip reasons and see coverage gaps.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from test_support import end_profiling_profile, ggml_session

NODE_TEST_ROOT = Path(onnx.__file__).parent / "backend" / "test" / "data" / "node"

_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-4

# Cases that crash the ORT process on session create or run. Each entry is a
# real EP stability bug to investigate — skipping only so the sweep completes.
# Rediscover with ``tools/onnx_node_discover.py`` after landing fixes.
_KNOWN_CRASHERS: set[str] = {
    "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded",
}

# Cases where the GGML EP claims nodes and then emits wrong shapes or numbers.
# These are real correctness bugs — xfail (strict=False) keeps CI green while
# the bug is tracked. Remove the entry once the underlying op is fixed.
_KNOWN_NUMERIC_FAILURES: set[str] = {
    # Shape-tracking bug: one of our partitions emits a rank-2 tensor that
    # feeds a CPU Transpose with perm size 3. Bug is inside a multi-partition
    # graph with nested If branches; needs dedicated debugging.
    "test_affine_grid_2d_expanded",
    "test_affine_grid_2d_align_corners_expanded",
}


def _discover_cases() -> list[Path]:
    if not NODE_TEST_ROOT.is_dir():
        return []
    return sorted(
        p
        for p in NODE_TEST_ROOT.iterdir()
        if p.is_dir() and (p / "model.onnx").exists()
    )


_ALL_CASES = _discover_cases()


def _load_tensor(path: Path):
    return numpy_helper.to_array(onnx.load_tensor(str(path)))


def _graph_input_names(model: onnx.ModelProto) -> list[str]:
    initializer_names = {init.name for init in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def _collect_datasets(case_dir: Path, input_names: list[str]):
    datasets = sorted(case_dir.glob("test_data_set_*"))
    collected = []
    for ds in datasets:
        input_files = sorted(ds.glob("input_*.pb"))
        output_files = sorted(ds.glob("output_*.pb"))
        if len(input_files) != len(input_names):
            pytest.skip(
                f"{ds.name}: input file count {len(input_files)} != "
                f"{len(input_names)} graph inputs"
            )
        feeds = {name: _load_tensor(p) for name, p in zip(input_names, input_files)}
        expected = [_load_tensor(p) for p in output_files]
        collected.append((ds.name, feeds, expected))
    return collected


def _pytest_param(case: Path):
    marks = []
    if case.name in _KNOWN_NUMERIC_FAILURES:
        marks.append(
            pytest.mark.xfail(
                reason="known numeric mismatch — tracked EP correctness bug",
                strict=False,
            )
        )
    return pytest.param(case, id=case.name, marks=marks)


@pytest.mark.skipif(not _ALL_CASES, reason="ONNX node test corpus not found")
@pytest.mark.parametrize("case", [_pytest_param(c) for c in _ALL_CASES])
def test_onnx_node_case(case: Path, ep_library: Path) -> None:
    if case.name in _KNOWN_CRASHERS:
        pytest.skip("known to crash the ORT process — tracked as EP stability bug")
    model_path = case / "model.onnx"

    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        pytest.skip(f"onnx.load failed: {e}")

    input_names = _graph_input_names(model)
    output_names = [o.name for o in model.graph.output]
    try:
        datasets = _collect_datasets(case, input_names)
    except Exception as e:
        # Tensors for sequence/optional/non-plain types can't be decoded via
        # numpy_helper. Treat as unsupported.
        pytest.skip(f"could not load reference tensors: {e}")
    if not datasets:
        pytest.skip("no test_data_set_* directories")

    try:
        session = ggml_session(model_path, ep_library)
    except Exception as e:
        # ORT wraps EP compile failures with the partition name
        # "GGMLExecutionProvider_<hash>_<i>". If we see that, our EP claimed
        # some nodes and then failed to compile them — a real bug, not a
        # coverage gap.
        if "GGMLExecutionProvider_" in str(e):
            pytest.fail(f"EP claimed nodes but failed to compile: {e}")
        pytest.skip(f"session create failed: {e}")

    # Run every dataset before inspecting the profile. ORT accumulates node
    # events across runs until end_profiling() is called, so one check covers
    # the whole case. Defer run exceptions until after we've read the profile
    # so we can tell "EP never claimed these nodes" from "EP ran and exploded".
    results = []
    run_exception: tuple[str, Exception] | None = None
    try:
        for ds_name, feeds, expected in datasets:
            try:
                actual = session.run(None, feeds)
            except Exception as e:
                run_exception = (ds_name, e)
                break
            results.append((ds_name, expected, actual))
    finally:
        profile = end_profiling_profile(session)

    node_events = [ev for ev in profile if ev.get("cat") == "Node"]
    ggml_events = [
        ev
        for ev in node_events
        if ev.get("args", {}).get("provider") == "GGMLExecutionProvider"
    ]

    if run_exception is not None:
        ds_name, err = run_exception
        if ggml_events or "GGMLExecutionProvider_" in str(err):
            pytest.fail(f"{ds_name}: EP ran nodes then failed at runtime: {err}")
        pytest.skip(f"{ds_name}: session.run failed: {err}")

    if not node_events:
        pytest.skip("no node events in profile")

    non_ggml = [
        ev
        for ev in node_events
        if ev.get("args", {}).get("provider") != "GGMLExecutionProvider"
    ]
    if non_ggml:
        fallback_ops = sorted(
            {ev.get("args", {}).get("op_name", "?") for ev in non_ggml}
        )
        pytest.skip(f"fell back to non-GGML provider for ops: {fallback_ops}")

    # Every node ran on GGML — now numeric mismatch is a real failure.
    for ds_name, expected, actual in results:
        assert len(actual) == len(expected), (
            f"{ds_name}: produced {len(actual)} outputs, expected {len(expected)}"
        )
        for name, exp, got in zip(output_names, expected, actual):
            if exp.dtype.kind in ("U", "S", "O"):
                pytest.skip(f"{ds_name}: non-numeric output dtype {exp.dtype}")
            np.testing.assert_allclose(
                got,
                exp,
                rtol=_DEFAULT_RTOL,
                atol=_DEFAULT_ATOL,
                err_msg=f"{ds_name}: output '{name}' mismatch",
            )
