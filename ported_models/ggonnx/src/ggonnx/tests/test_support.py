#!/usr/bin/env python3

import atexit
import ctypes
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import helper

_EP_REGISTERED = False
_GGML_DEVICES = None

def model_cache_dir() -> Path:
    override = os.environ.get("GGONNX_TEST_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ggonnx-test" / "models"


def register_ep(ep_library: Path) -> None:
    global _EP_REGISTERED
    if _EP_REGISTERED:
        return
    ort.register_execution_provider_library("GGONNX", str(ep_library))
    _EP_REGISTERED = True


def ggml_devices(ep_library: Path):
    global _GGML_DEVICES
    register_ep(ep_library)
    if _GGML_DEVICES is None:
        _GGML_DEVICES = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "GGMLExecutionProvider"
        ]
    if not _GGML_DEVICES:
        raise RuntimeError(
            "GGMLExecutionProvider was not discovered after library registration"
        )
    return _GGML_DEVICES


class DebugApi:
    def __init__(self, ep_library: Path) -> None:
        self.lib = ctypes.CDLL(str(ep_library))
        self.lib.GGONNX_DebugGetGraphBuildCount.restype = ctypes.c_uint64
        self.lib.GGONNX_DebugGetLastGraphNodeCount.restype = ctypes.c_uint64
        self.lib.GGONNX_DebugGetLastGraphOutputCount.restype = ctypes.c_uint64
        self.lib.GGONNX_DebugResetGraphBuildCount.argtypes = []

    def reset(self) -> None:
        self.lib.GGONNX_DebugResetGraphBuildCount()

    def graph_build_count(self) -> int:
        return int(self.lib.GGONNX_DebugGetGraphBuildCount())

    def last_graph_node_count(self) -> int:
        return int(self.lib.GGONNX_DebugGetLastGraphNodeCount())

    def last_graph_output_count(self) -> int:
        return int(self.lib.GGONNX_DebugGetLastGraphOutputCount())


def save_model(path: Path, graph) -> None:
    model = helper.make_model(
        graph,
        producer_name="ggonnx-test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


def ensure_model(tmpdir: Path, graph) -> Path:
    return ensure_model_with_opset(tmpdir, graph, 13)


def ensure_model_with_opset(tmpdir: Path, graph, opset_version: int) -> Path:
    model = helper.make_model(
        graph,
        producer_name="ggonnx-test",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    content = model.SerializeToString()
    content_hash = hashlib.sha256(content).hexdigest()
    model_path = tmpdir / f"{graph.name}_{content_hash[:16]}.onnx"
    if not model_path.exists():
        onnx.save(model, model_path)
    return model_path


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with urllib.request.urlopen(url) as response, temp_path.open("wb") as out:
        total_bytes = response.headers.get("Content-Length")
        total = int(total_bytes) if total_bytes is not None else None
        downloaded = 0
        bar_width = 32

        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total:
                filled = min(bar_width, int(bar_width * downloaded / total))
                bar = "#" * filled + "-" * (bar_width - filled)
                percent = 100.0 * downloaded / total
                message = (
                    f"\rDownloading {destination.name} [{bar}] "
                    f"{percent:5.1f}% ({downloaded}/{total} bytes)"
                )
            else:
                message = f"\rDownloading {destination.name}: {downloaded} bytes"

            print(message, end="", file=sys.stderr, flush=True)

    os.replace(temp_path, destination)
    print(file=sys.stderr, flush=True)


def cached_model_path(filename: str, url: str) -> Path:
    destination = model_cache_dir() / filename
    if not destination.exists():
        _download_with_progress(url, destination)
    return destination


def cpu_session(model_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.add_provider("CPUExecutionProvider", {})
    return ort.InferenceSession(str(model_path), sess_options=so)


_PROFILE_DIR = Path(tempfile.mkdtemp(prefix="ggonnx_profile_"))
_PROFILE_COUNTER = 0
atexit.register(shutil.rmtree, _PROFILE_DIR, ignore_errors=True)


def ggml_session(model_path: Path, ep_library: Path) -> ort.InferenceSession:
    # ORT names profile files `{prefix}_{timestamp}.json` with second-
    # granularity timestamps, so sessions created in the same second share a
    # file. A per-session counter in the prefix keeps the paths distinct,
    # which matters when many tests run back-to-back and each inspects its
    # own profile to check fallback behavior.
    global _PROFILE_COUNTER
    _PROFILE_COUNTER += 1
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(_PROFILE_DIR / f"session_{_PROFILE_COUNTER:06d}")
    so.add_provider_for_devices(ggml_devices(ep_library), {})
    return ort.InferenceSession(str(model_path), sess_options=so)


def assert_all_nodes_run_on_ggml(session: ort.InferenceSession) -> None:
    profile = end_profiling_profile(session)

    node_events = [event for event in profile if event.get("cat") == "Node"]
    assert node_events, "expected at least one profiled node event"

    non_ggml_events = [
        event
        for event in node_events
        if event.get("args", {}).get("provider") != "GGMLExecutionProvider"
    ]
    assert not non_ggml_events, (
        f"found non-GGML node events in profile: {non_ggml_events}"
    )


def assert_provider_does_not_run_ops(
    session: ort.InferenceSession,
    provider: str,
    op_names: set[str],
) -> None:
    profile = end_profiling_profile(session)

    forbidden_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == provider
        and event.get("args", {}).get("op_name") in op_names
    ]
    assert not forbidden_events, (
        f"found forbidden {provider} node events in profile: {forbidden_events}"
    )


def assert_provider_runs_ops(
    session: ort.InferenceSession,
    provider: str,
    op_names: set[str],
) -> None:
    profile = end_profiling_profile(session)

    matching_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == provider
        and event.get("args", {}).get("op_name") in op_names
    ]
    assert matching_events, (
        f"expected {provider} to run one of {sorted(op_names)}, "
        f"but found none in profile"
    )


def assert_provider_runs_any_node(
    session: ort.InferenceSession,
    provider: str,
) -> None:
    profile = end_profiling_profile(session)

    matching_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == provider
    ]
    assert matching_events, f"expected at least one node event on {provider}"


def end_profiling_profile(session: ort.InferenceSession) -> list[dict]:
    profile_path = Path(session.end_profiling())
    try:
        with profile_path.open() as f:
            return json.load(f)
    finally:
        profile_path.unlink(missing_ok=True)
