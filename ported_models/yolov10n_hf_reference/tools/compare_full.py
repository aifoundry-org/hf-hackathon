#!/usr/bin/env python3
"""Strictly compare a full-graph C dump with pinned-artifact ORT checkpoints.

The schema-v2 full package uses a liveness arena, so most intermediate
locations are deliberately reused.  Only tensors declared in ``checkpoints``
are pinned until the end of execution and are therefore meaningful in the
final dump.  This comparator never interprets other workspace bytes as
goldens; it does, however, verify the YRF1 FNV-1a over the entire arena.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
)

EXPECTED_REPO = "onnx-community/yolov10n"
EXPECTED_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"
EXPECTED_FILENAME = "onnx/model.onnx"
EXPECTED_SOURCE_SHA256 = (
    "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
)
EXPECTED_LICENSE = "AGPL-3.0"
EXPECTED_HEADER_BYTES = 92881
EXPECTED_HEADER_SHA256 = (
    "79be5b751842df025a3612ebb690e283813ea9ac8e373fd1bc44b706ca7a2a7e"
)
EXPECTED_SELECTOR = "N000:N307"
EXPECTED_NODE_COUNT = 308

RESULT_MAGIC = 0x31465259
RESULT_VERSION = 1
MATH_VERSION = 1
RESULT_STRUCT_BYTES = 128
HEADER_U32 = struct.Struct("<16I")
HEADER_U64 = struct.Struct("<8Q")
STATUS_NAMES = {
    0: "ok",
    1: "bad_manifest",
    2: "unsupported_op",
    3: "unsupported_shape",
}
DTYPE_INFO = {
    "FLOAT": ("<f4", 4),
    "INT64": ("<i8", 8),
}


class CompareError(RuntimeError):
    """The package or dump does not satisfy the full-graph contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompareError(message)


def integer(value: Any, field: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        "{} must be an integer".format(field),
    )
    return int(value)


def shape_elements(shape: Sequence[Any], field: str) -> int:
    require(isinstance(shape, list), "{} must be a list".format(field))
    elements = 1
    for index, dimension_value in enumerate(shape):
        dimension = integer(
            dimension_value, "{}[{}]".format(field, index)
        )
        require(dimension >= 0, "{} has a negative dimension".format(field))
        elements *= dimension
    return elements


def safe_package_path(full_dir: Path, relative: Any, field: str) -> Path:
    require(
        isinstance(relative, str) and relative != "",
        "{} must be a non-empty relative path".format(field),
    )
    path = (full_dir / relative).resolve()
    try:
        path.relative_to(full_dir)
    except ValueError:
        raise CompareError("{} escapes the full package".format(field))
    return path


def parse_node_id(value: Any, field: str) -> int:
    require(
        isinstance(value, str)
        and len(value) == 4
        and value[0] == "N"
        and value[1:].isdigit(),
        "{} must have the form N000".format(field),
    )
    return int(value[1:])


def result_header(data: bytes) -> Dict[str, Any]:
    require(
        len(data) >= RESULT_STRUCT_BYTES,
        "dump is shorter than the 128-byte YRF1 result structure",
    )
    words = HEADER_U32.unpack_from(data, 0)
    longs = HEADER_U64.unpack_from(data, 64)
    return {
        "magic": words[0],
        "version": words[1],
        "status": words[2],
        "status_name": STATUS_NAMES.get(words[2], "unknown"),
        "failed_node": words[3],
        "failed_op": words[4],
        "first_node": words[5],
        "last_node": words[6],
        "node_count": words[7],
        "tensor_count": words[8],
        "workspace_bytes": words[9],
        "input_blob_bytes": words[10],
        "weight_blob_bytes": words[11],
        "math_version": words[12],
        "reserved32": list(words[13:16]),
        "workspace_fnv1a": longs[0],
        "reserved64": list(longs[1:8]),
    }


def compare_float(
    actual: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    finite = np.isfinite(actual64) & np.isfinite(reference64)
    absolute = np.zeros(actual64.shape, dtype=np.float64)
    absolute[finite] = np.abs(actual64[finite] - reference64[finite])
    allowed = atol + rtol * np.abs(reference64)
    relative = np.zeros(actual64.shape, dtype=np.float64)
    relative[finite] = absolute[finite] / np.maximum(
        np.abs(reference64[finite]), np.finfo(np.float32).tiny
    )
    mismatch = ~finite
    mismatch[finite] |= absolute[finite] > allowed[finite]
    nonfinite_indices = np.flatnonzero(~finite)
    if nonfinite_indices.size:
        worst = int(nonfinite_indices[0])
    else:
        worst = int(np.argmax(absolute)) if absolute.size else 0
    finite_absolute = absolute[finite]
    finite_relative = relative[finite]
    return {
        "comparison": "tolerance",
        "pass": bool(not np.any(mismatch)),
        "elements": int(actual.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "nonfinite_actual_count": int(
            np.count_nonzero(~np.isfinite(actual64))
        ),
        "nonfinite_reference_count": int(
            np.count_nonzero(~np.isfinite(reference64))
        ),
        "max_abs": (
            float(np.max(finite_absolute)) if finite_absolute.size else 0.0
        ),
        "mean_abs": (
            float(np.mean(finite_absolute)) if finite_absolute.size else 0.0
        ),
        "max_rel": (
            float(np.max(finite_relative)) if finite_relative.size else 0.0
        ),
        "worst_flat_index": worst,
        "worst_actual": (
            float(actual.flat[worst])
            if actual.size and np.isfinite(actual.flat[worst])
            else None
        ),
        "worst_reference": (
            float(reference.flat[worst])
            if reference.size and np.isfinite(reference.flat[worst])
            else None
        ),
        "atol": atol,
        "rtol": rtol,
    }


def compare_int64(
    actual: np.ndarray, reference: np.ndarray
) -> Dict[str, Any]:
    mismatch = actual != reference
    # longdouble avoids signed-int64 subtraction overflow.  Equality itself is
    # evaluated in INT64 and remains the exact pass/fail criterion.
    absolute = np.abs(
        actual.astype(np.longdouble) - reference.astype(np.longdouble)
    )
    denominator = np.maximum(
        np.abs(reference.astype(np.longdouble)), np.longdouble(1.0)
    )
    relative = absolute / denominator
    worst = int(np.argmax(absolute)) if absolute.size else 0
    return {
        "comparison": "exact",
        "pass": bool(not np.any(mismatch)),
        "elements": int(actual.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "nonfinite_actual_count": 0,
        "nonfinite_reference_count": 0,
        "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
        "mean_abs": float(np.mean(absolute)) if absolute.size else 0.0,
        "max_rel": float(np.max(relative)) if relative.size else 0.0,
        "worst_flat_index": worst,
        "worst_actual": int(actual.flat[worst]) if actual.size else None,
        "worst_reference": (
            int(reference.flat[worst]) if reference.size else None
        ),
        "atol": 0,
        "rtol": 0,
    }


def json_float_records(array: np.ndarray) -> List[List[Optional[float]]]:
    records: List[List[Optional[float]]] = []
    for row in array:
        records.append(
            [
                float(value) if np.isfinite(value) else None
                for value in row
            ]
        )
    return records


def stable_topk(values: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Replay ONNX Runtime's observed value-desc/index-asc TopK contract."""
    require(values.ndim == 1, "TopK replay input must be rank one")
    require(
        bool(np.all(np.isfinite(values))),
        "TopK replay input contains a non-finite value",
    )
    require(0 < k <= values.size, "TopK replay K is out of range")
    source_indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((source_indices, -values.astype(np.float64)))
    selected = order[:k].astype(np.int64, copy=False)
    return values[selected], selected


def topk_boundary(values: np.ndarray, k: int) -> Dict[str, Any]:
    sorted_values, indices = stable_topk(values, values.size)
    cutoff = sorted_values[k - 1]
    next_value = sorted_values[k] if k < values.size else cutoff
    return {
        "k": k,
        "cutoff": float(cutoff),
        "next": float(next_value),
        "margin": float(cutoff - next_value),
        "tie_count_at_cutoff": int(np.count_nonzero(values == cutoff)),
        "strictly_above_cutoff": int(np.count_nonzero(values > cutoff)),
        "selected_indices": indices[:k],
    }


def replay_selection_tail(candidate: np.ndarray) -> Dict[str, Any]:
    """Independently replay the exact pinned N289:N307 tensor program."""
    require(
        candidate.shape == (1, 8400, 84),
        "selection replay requires N288 shape [1,8400,84]",
    )
    boxes = candidate[0, :, :4]
    class_scores = candidate[0, :, 4:]
    anchor_scores = np.max(class_scores, axis=1)
    first_values, first_indices = stable_topk(anchor_scores, 300)
    del first_values
    gathered_boxes = boxes[first_indices, :]
    gathered_scores = class_scores[first_indices, :]
    flattened = gathered_scores.reshape(-1)
    final_scores, flattened_indices = stable_topk(flattened, 300)
    selected_slots = flattened_indices // 80
    selected_classes = flattened_indices % 80
    output = np.concatenate(
        (
            gathered_boxes[selected_slots, :],
            final_scores[:, np.newaxis],
            selected_classes.astype(np.float32)[:, np.newaxis],
        ),
        axis=1,
    )[np.newaxis, ...].astype(np.float32, copy=False)
    anchor_class = np.stack(
        (
            first_indices[selected_slots],
            selected_classes,
        ),
        axis=1,
    )
    return {
        "output": output,
        "first_indices": first_indices,
        "anchor_class": anchor_class,
        "first_boundary": topk_boundary(anchor_scores, 300),
        "second_boundary": topk_boundary(flattened, 300),
    }


def validate_selection_tail(
    candidate_actual: np.ndarray,
    candidate_reference: np.ndarray,
    output_actual: np.ndarray,
    output_reference: np.ndarray,
    candidate_pass: bool,
    direct_output_pass: bool,
) -> Dict[str, Any]:
    actual_replay = replay_selection_tail(candidate_actual)
    reference_replay = replay_selection_tail(candidate_reference)
    actual_exact = (
        actual_replay["output"].tobytes(order="C")
        == output_actual.tobytes(order="C")
    )
    reference_exact = (
        reference_replay["output"].tobytes(order="C")
        == output_reference.tobytes(order="C")
    )
    first_reference = reference_replay["first_boundary"]
    second_reference = reference_replay["second_boundary"]
    discontinuous = (
        first_reference["margin"] == 0.0
        or second_reference["margin"] == 0.0
    )
    actual_pairs = {
        (int(item[0]), int(item[1]))
        for item in actual_replay["anchor_class"]
    }
    reference_pairs = {
        (int(item[0]), int(item[1]))
        for item in reference_replay["anchor_class"]
    }
    actual_anchors = {
        int(item) for item in actual_replay["first_indices"]
    }
    reference_anchors = {
        int(item) for item in reference_replay["first_indices"]
    }
    tie_aware_pass = bool(
        direct_output_pass
        or (
            candidate_pass
            and discontinuous
            and actual_exact
            and reference_exact
        )
    )
    return {
        "pass": tie_aware_pass,
        "mode": (
            "direct_ort_tolerance"
            if direct_output_pass
            else "proven_topk_tie_discontinuity"
        ),
        "direct_ort_pass": direct_output_pass,
        "candidate_checkpoint_pass": candidate_pass,
        "reference_has_exact_cutoff_tie": discontinuous,
        "actual_replay_matches_c_output_bitwise": actual_exact,
        "reference_replay_matches_ort_output_bitwise": reference_exact,
        "first_topk": {
            key: value
            for key, value in first_reference.items()
            if key != "selected_indices"
        },
        "second_topk": {
            key: value
            for key, value in second_reference.items()
            if key != "selected_indices"
        },
        "selected_anchor_overlap": len(actual_anchors & reference_anchors),
        "selected_anchor_count": 300,
        "final_anchor_class_overlap": len(actual_pairs & reference_pairs),
        "final_anchor_class_count": 300,
    }


def verify_source(
    manifest: Dict[str, Any], model_path: Path
) -> Dict[str, Any]:
    source = manifest.get("source")
    require(isinstance(source, dict), "manifest source must be an object")
    expected_fields = {
        "repo": EXPECTED_REPO,
        "revision": EXPECTED_REVISION,
        "filename": EXPECTED_FILENAME,
        "sha256": EXPECTED_SOURCE_SHA256,
        "license": EXPECTED_LICENSE,
    }
    fields = {
        key: source.get(key) == expected
        for key, expected in expected_fields.items()
    }
    exists = model_path.is_file()
    actual_bytes = model_path.stat().st_size if exists else None
    actual_sha256 = file_sha256(model_path) if exists else None
    passed = (
        all(fields.values())
        and exists
        and actual_sha256 == EXPECTED_SOURCE_SHA256
    )
    return {
        "pass": passed,
        "path": str(model_path),
        "exists": exists,
        "bytes": actual_bytes,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "actual_sha256": actual_sha256,
        "manifest_fields": fields,
    }


def verify_blobs(
    manifest: Dict[str, Any], full_dir: Path
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    blobs = manifest.get("blobs")
    require(isinstance(blobs, dict), "manifest blobs must be an object")
    require(
        {"inputs", "weights", "goldens"}.issubset(blobs),
        "manifest must declare inputs, weights, and goldens blobs",
    )
    checks: Dict[str, Dict[str, Any]] = {}
    for name in sorted(blobs):
        record = blobs[name]
        require(
            isinstance(record, dict),
            "blobs.{} must be an object".format(name),
        )
        path = safe_package_path(
            full_dir, record.get("path"), "blobs.{}.path".format(name)
        )
        expected_bytes = integer(
            record.get("nbytes"), "blobs.{}.nbytes".format(name)
        )
        expected_sha256 = record.get("sha256")
        require(
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64,
            "blobs.{}.sha256 is invalid".format(name),
        )
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = file_sha256(path) if exists else None
        passed = (
            exists
            and actual_bytes == expected_bytes
            and actual_sha256 == expected_sha256
        )
        checks[name] = {
            "pass": passed,
            "path": record["path"],
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
        }
        print(
            "BLOB {} name={} bytes={} sha256={}".format(
                "PASS" if passed else "FAIL",
                name,
                actual_bytes,
                actual_sha256,
            )
        )
    return bool(checks) and all(item["pass"] for item in checks.values()), checks


def verify_generated_header(
    manifest: Dict[str, Any], full_dir: Path
) -> Dict[str, Any]:
    generated = manifest.get("generated")
    require(isinstance(generated, dict), "manifest generated must be an object")
    record = generated.get("header")
    require(
        isinstance(record, dict),
        "manifest generated.header must be an object",
    )
    path = safe_package_path(
        full_dir, record.get("path"), "generated.header.path"
    )
    require(
        path == (full_dir / "slice_manifest.h").resolve(),
        "generated.header.path must name slice_manifest.h",
    )
    expected_bytes = integer(
        record.get("nbytes"), "generated.header.nbytes"
    )
    expected_sha256 = record.get("sha256")
    require(
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "generated.header.sha256 is invalid",
    )
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha256 = file_sha256(path) if exists else None
    return {
        "pass": bool(
            exists
            and actual_bytes == expected_bytes
            and actual_sha256 == expected_sha256
            and expected_bytes == EXPECTED_HEADER_BYTES
            and expected_sha256 == EXPECTED_HEADER_SHA256
        ),
        "path": str(path),
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "pinned_bytes": EXPECTED_HEADER_BYTES,
        "pinned_sha256": EXPECTED_HEADER_SHA256,
    }


def validate_manifest(
    manifest: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    require(
        manifest.get("schema_version") == 2,
        "full manifest schema_version must be 2",
    )
    require(
        manifest.get("manifest_kind") == "full_graph_liveness",
        "manifest_kind must be full_graph_liveness",
    )
    selection = manifest.get("selection")
    require(isinstance(selection, dict), "selection must be an object")
    require(
        selection.get("selector") == EXPECTED_SELECTOR
        and selection.get("first_node") == "N000"
        and selection.get("last_node") == "N307"
        and selection.get("inclusive") is True,
        "selection is not the inclusive full N000:N307 graph",
    )

    nodes = manifest.get("nodes")
    tensors = manifest.get("tensors")
    checkpoints = manifest.get("checkpoints")
    require(
        isinstance(nodes, list) and len(nodes) == EXPECTED_NODE_COUNT,
        "full manifest must contain exactly 308 nodes",
    )
    require(
        isinstance(tensors, list) and bool(tensors),
        "full manifest tensors must be a non-empty list",
    )
    require(
        isinstance(checkpoints, list) and bool(checkpoints),
        "full manifest checkpoints must be a non-empty list",
    )
    tolerances = manifest.get("tolerances")
    require(isinstance(tolerances, dict), "tolerances must be an object")
    require(
        tolerances.get("atol") == 0.00005
        and tolerances.get("rtol") == 0.0001,
        "global tolerances differ from the validated contract",
    )
    overrides = tolerances.get("checkpoint_overrides")
    require(
        isinstance(overrides, dict) and set(overrides) == {"N288"},
        "checkpoint tolerance overrides differ from the validated contract",
    )
    n288_tolerance = overrides["N288"]
    require(
        isinstance(n288_tolerance, dict)
        and n288_tolerance.get("atol") == 0.0002
        and n288_tolerance.get("rtol") == 0.0001,
        "N288 tolerance differs from the validated contract",
    )

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    output_to_node: Dict[str, str] = {}
    for expected_index, node in enumerate(nodes):
        require(isinstance(node, dict), "node record must be an object")
        node_id = node.get("node_id")
        require(
            parse_node_id(node_id, "nodes.node_id") == expected_index
            and integer(node.get("index"), "{}.index".format(node_id))
            == expected_index,
            "nodes are not in exact N000:N307 order",
        )
        require(node_id not in nodes_by_id, "duplicate node {}".format(node_id))
        outputs = node.get("outputs")
        require(
            isinstance(outputs, list) and bool(outputs),
            "{} has no outputs".format(node_id),
        )
        for output in outputs:
            require(
                isinstance(output, str) and output not in output_to_node,
                "invalid or duplicate graph output {!r}".format(output),
            )
            output_to_node[output] = node_id
        nodes_by_id[node_id] = node

    tensors_by_name: Dict[str, Dict[str, Any]] = {}
    for tensor in tensors:
        require(isinstance(tensor, dict), "tensor record must be an object")
        name = tensor.get("name")
        require(
            isinstance(name, str) and name != "",
            "tensor name must be a non-empty string",
        )
        require(
            name not in tensors_by_name,
            "duplicate tensor {!r}".format(name),
        )
        dtype = tensor.get("dtype")
        require(
            dtype in DTYPE_INFO,
            "tensor {!r} has unsupported dtype {!r}".format(name, dtype),
        )
        elements = integer(tensor.get("elements"), "{}.elements".format(name))
        require(
            shape_elements(tensor.get("shape"), "{}.shape".format(name))
            == elements,
            "tensor {!r} shape/elements disagree".format(name),
        )
        require(
            integer(tensor.get("nbytes"), "{}.nbytes".format(name))
            == elements * DTYPE_INFO[dtype][1],
            "tensor {!r} nbytes disagree with dtype/shape".format(name),
        )
        tensors_by_name[name] = tensor

    require(
        set(output_to_node).issubset(tensors_by_name),
        "one or more node outputs have no tensor descriptor",
    )
    return nodes_by_id, tensors_by_name


def compare_checkpoints(
    manifest: Dict[str, Any],
    dump: bytes,
    golden: bytes,
    atol: float,
    rtol: float,
    nodes_by_id: Dict[str, Dict[str, Any]],
    tensors_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    result = manifest["result"]
    memory_map = manifest["memory_map"]
    output_base = integer(
        result.get("workspace_offset_within_dump"),
        "result.workspace_offset_within_dump",
    )
    workspace_bytes = integer(
        memory_map.get("workspace_bytes"), "memory_map.workspace_bytes"
    )
    golden_bytes = integer(
        manifest["blobs"]["goldens"].get("nbytes"),
        "blobs.goldens.nbytes",
    )
    output_tensor = result.get("output_tensor")
    require(
        isinstance(output_tensor, str) and output_tensor != "",
        "result.output_tensor must be a tensor name",
    )

    seen_nodes = set()
    seen_tensors = set()
    workspace_ranges: List[Tuple[int, int, str]] = []
    golden_ranges: List[Tuple[int, int, str]] = []
    tolerance_overrides = manifest["tolerances"].get(
        "checkpoint_overrides", {}
    )
    require(
        isinstance(tolerance_overrides, dict),
        "tolerances.checkpoint_overrides must be an object",
    )
    reports: List[Dict[str, Any]] = []
    final_report: Optional[Dict[str, Any]] = None
    final_checkpoint_report: Optional[Dict[str, Any]] = None
    candidate_report: Optional[Dict[str, Any]] = None
    candidate_actual: Optional[np.ndarray] = None
    candidate_reference: Optional[np.ndarray] = None
    output_actual: Optional[np.ndarray] = None
    output_reference: Optional[np.ndarray] = None
    for checkpoint_index, checkpoint in enumerate(manifest["checkpoints"]):
        field = "checkpoints[{}]".format(checkpoint_index)
        require(isinstance(checkpoint, dict), "{} must be an object".format(field))
        node_id = checkpoint.get("node_id")
        tensor_name = checkpoint.get("tensor")
        require(node_id in nodes_by_id, "{} names an unknown node".format(field))
        require(
            isinstance(tensor_name, str) and tensor_name in tensors_by_name,
            "{} names an unknown tensor".format(field),
        )
        require(
            node_id not in seen_nodes,
            "duplicate checkpoint node {}".format(node_id),
        )
        require(
            tensor_name not in seen_tensors,
            "duplicate checkpoint tensor {!r}".format(tensor_name),
        )
        seen_nodes.add(node_id)
        seen_tensors.add(tensor_name)

        tensor = tensors_by_name[tensor_name]
        node = nodes_by_id[node_id]
        require(
            tensor_name in node["outputs"],
            "{} is not produced by {}".format(tensor_name, node_id),
        )
        require(
            tensor.get("storage") == "workspace"
            and tensor.get("checkpoint") is True
            and tensor.get("producer") == node_id,
            "{} is not a pinned workspace output".format(tensor_name),
        )
        for name in ("dtype", "shape", "elements", "nbytes"):
            require(
                checkpoint.get(name) == tensor.get(name),
                "{} {} disagrees with tensor descriptor".format(field, name),
            )

        dtype = checkpoint["dtype"]
        elements = integer(checkpoint["elements"], "{}.elements".format(field))
        nbytes = integer(checkpoint["nbytes"], "{}.nbytes".format(field))
        workspace_offset = integer(
            checkpoint.get("workspace_offset"),
            "{}.workspace_offset".format(field),
        )
        golden_offset = integer(
            checkpoint.get("golden_offset"),
            "{}.golden_offset".format(field),
        )
        require(
            workspace_offset == integer(
                tensor.get("offset"), "{}.tensor.offset".format(field)
            ),
            "{} workspace offset disagrees with tensor descriptor".format(field),
        )
        require(
            workspace_offset >= 0
            and workspace_offset + nbytes <= workspace_bytes,
            "{} exceeds the workspace".format(field),
        )
        require(
            golden_offset >= 0 and golden_offset + nbytes <= golden_bytes,
            "{} exceeds the golden blob".format(field),
        )
        for start, end, prior_name in workspace_ranges:
            require(
                workspace_offset + nbytes <= start or workspace_offset >= end,
                "{} overlaps checkpoint {!r} in the workspace".format(
                    field, prior_name
                ),
            )
        for start, end, prior_name in golden_ranges:
            require(
                golden_offset + nbytes <= start or golden_offset >= end,
                "{} overlaps checkpoint {!r} in the golden blob".format(
                    field, prior_name
                ),
            )
        workspace_ranges.append(
            (workspace_offset, workspace_offset + nbytes, tensor_name)
        )
        golden_ranges.append(
            (golden_offset, golden_offset + nbytes, tensor_name)
        )

        actual_raw = dump[
            output_base + workspace_offset:
            output_base + workspace_offset + nbytes
        ]
        reference_raw = golden[golden_offset:golden_offset + nbytes]
        require(
            len(actual_raw) == nbytes and len(reference_raw) == nbytes,
            "{} data is truncated".format(field),
        )
        expected_golden_sha = checkpoint.get("golden_sha256")
        require(
            isinstance(expected_golden_sha, str)
            and bytes_sha256(reference_raw) == expected_golden_sha,
            "{} golden segment SHA-256 mismatch".format(field),
        )

        numpy_dtype = DTYPE_INFO[dtype][0]
        actual = np.frombuffer(actual_raw, dtype=numpy_dtype, count=elements)
        reference = np.frombuffer(
            reference_raw, dtype=numpy_dtype, count=elements
        )
        checkpoint_atol = atol
        checkpoint_rtol = rtol
        override = tolerance_overrides.get(node_id)
        if override is not None:
            require(
                isinstance(override, dict),
                "tolerance override {} must be an object".format(node_id),
            )
            checkpoint_atol = float(override.get("atol", atol))
            checkpoint_rtol = float(override.get("rtol", rtol))
            require(
                math.isfinite(checkpoint_atol)
                and math.isfinite(checkpoint_rtol)
                and checkpoint_atol >= 0.0
                and checkpoint_rtol >= 0.0,
                "tolerance override {} is invalid".format(node_id),
            )
        if dtype == "FLOAT":
            metrics = compare_float(
                actual, reference, checkpoint_atol, checkpoint_rtol
            )
        else:
            metrics = compare_int64(actual, reference)
        metrics.update(
            {
                "node_id": node_id,
                "tensor": tensor_name,
                "dtype": dtype,
                "shape": checkpoint["shape"],
                "workspace_offset": workspace_offset,
                "golden_offset": golden_offset,
                "actual_sha256": bytes_sha256(actual_raw),
                "reference_sha256": bytes_sha256(reference_raw),
            }
        )
        reports.append(metrics)
        print(
            "CHECKPOINT {} node={} tensor={} dtype={} shape={} "
            "max_abs={:.9g} max_rel={:.9g} mean_abs={:.9g} "
            "mismatches={}/{} atol={} rtol={}".format(
                "PASS" if metrics["pass"] else "FAIL",
                node_id,
                tensor_name,
                dtype,
                checkpoint["shape"],
                metrics["max_abs"],
                metrics["max_rel"],
                metrics["mean_abs"],
                metrics["mismatch_count"],
                metrics["elements"],
                metrics["atol"],
                metrics["rtol"],
            )
        )
        if node_id == "N288" and checkpoint["shape"] == [1, 8400, 84]:
            candidate_report = metrics
            candidate_actual = actual.reshape(1, 8400, 84).copy()
            candidate_reference = reference.reshape(1, 8400, 84).copy()
        if tensor_name == output_tensor:
            require(
                final_report is None,
                "result.output_tensor has multiple checkpoint records",
            )
            final_checkpoint_report = metrics
            final_report = metrics
            if checkpoint["shape"] == [1, 300, 6] and dtype == "FLOAT":
                output_actual = actual.reshape(1, 300, 6).copy()
                output_reference = reference.reshape(1, 300, 6).copy()
                preview_count = min(5, 300)
                final_report["record_preview"] = {
                    "actual": json_float_records(
                        actual.reshape(1, 300, 6)[0, :preview_count, :]
                    ),
                    "reference": json_float_records(
                        reference.reshape(1, 300, 6)[
                            0, :preview_count, :
                        ]
                    ),
                }

    require(
        final_report is not None,
        "result.output_tensor is not one of the declared checkpoints",
    )
    require(
        final_checkpoint_report is not None
        and candidate_report is not None
        and candidate_actual is not None
        and candidate_reference is not None
        and output_actual is not None
        and output_reference is not None,
        "full checkpoints must include N288 candidates and FLOAT output0",
    )
    direct_output_pass = bool(final_report["pass"])
    selection = validate_selection_tail(
        candidate_actual,
        candidate_reference,
        output_actual,
        output_reference,
        bool(candidate_report["pass"]),
        direct_output_pass,
    )
    final_report["selection_validation"] = selection
    final_report["direct_ort_pass"] = direct_output_pass
    final_report["direct_ort_mismatch_count"] = final_report["mismatch_count"]
    final_report["unexplained_mismatch_count"] = (
        0 if selection["pass"] else final_report["mismatch_count"]
    )
    final_report["pass"] = selection["pass"]
    print(
        "SELECTION_TAIL {} mode={} first_margin={:.9g} "
        "second_margin={:.9g} anchor_overlap={}/300 pair_overlap={}/300 "
        "actual_replay_exact={} reference_replay_exact={}".format(
            "PASS" if selection["pass"] else "FAIL",
            selection["mode"],
            selection["first_topk"]["margin"],
            selection["second_topk"]["margin"],
            selection["selected_anchor_overlap"],
            selection["final_anchor_class_overlap"],
            selection["actual_replay_matches_c_output_bitwise"],
            selection["reference_replay_matches_ort_output_bitwise"],
        )
    )
    print(
        "FINAL_OUTPUT {} tensor={} node={} shape={} mode={} "
        "max_abs={:.9g} max_rel={:.9g} direct_mismatches={}/{} "
        "unexplained_mismatches={} actual_sha256={}".format(
            "PASS" if final_report["pass"] else "FAIL",
            final_report["tensor"],
            final_report["node_id"],
            final_report["shape"],
            selection["mode"],
            final_report["max_abs"],
            final_report["max_rel"],
            final_report["mismatch_count"],
            final_report["elements"],
            final_report["unexplained_mismatch_count"],
            final_report["actual_sha256"],
        )
    )
    return reports, final_report


def write_json(path: Optional[Path], report: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    full_dir = args.full_dir.resolve()
    report: Dict[str, Any] = {
        "schema_version": 1,
        "comparison_kind": "full_graph_checkpoints",
        "pass": False,
        "errors": [],
        "full_dir": str(full_dir),
        "dump_path": str(args.dump.resolve()),
    }
    try:
        require(full_dir.is_dir(), "full package directory does not exist")
        manifest_path = full_dir / "slice_manifest.json"
        require(manifest_path.is_file(), "missing slice_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(isinstance(manifest, dict), "manifest root must be an object")
        nodes_by_id, tensors_by_name = validate_manifest(manifest)

        source_check = verify_source(manifest, args.model.resolve())
        report["source"] = source_check
        print(
            "SOURCE {} revision={} bytes={} sha256={}".format(
                "PASS" if source_check["pass"] else "FAIL",
                EXPECTED_REVISION,
                source_check["bytes"],
                source_check["actual_sha256"],
            )
        )

        blob_pass, blob_checks = verify_blobs(manifest, full_dir)
        report["blob_pass"] = blob_pass
        report["blob_checks"] = blob_checks
        header_identity = verify_generated_header(manifest, full_dir)
        report["generated_header"] = header_identity
        print(
            "GENERATED_HEADER {} bytes={} sha256={}".format(
                "PASS" if header_identity["pass"] else "FAIL",
                header_identity["actual_bytes"],
                header_identity["actual_sha256"],
            )
        )
        require(source_check["pass"], "pinned source identity check failed")
        require(blob_pass, "one or more full-package blobs failed identity checks")
        require(
            header_identity["pass"],
            "generated header identity check failed",
        )

        dump = args.dump.read_bytes()
        expected_dump_bytes = integer(
            manifest["memory_map"].get("dump_size"),
            "memory_map.dump_size",
        )
        dump_pass = len(dump) == expected_dump_bytes
        report["dump"] = {
            "pass": dump_pass,
            "expected_bytes": expected_dump_bytes,
            "actual_bytes": len(dump),
            "sha256": bytes_sha256(dump),
        }
        require(
            dump_pass,
            "dump size mismatch: expected {}, got {}".format(
                expected_dump_bytes, len(dump)
            ),
        )

        result = manifest.get("result")
        memory_map = manifest.get("memory_map")
        memory_plan = manifest.get("memory_plan")
        require(isinstance(result, dict), "result must be an object")
        require(isinstance(memory_map, dict), "memory_map must be an object")
        require(isinstance(memory_plan, dict), "memory_plan must be an object")
        require(result.get("magic") == "YRF1", "result magic is not YRF1")
        require(
            integer(result.get("header_bytes"), "result.header_bytes")
            == integer(
                result.get("workspace_offset_within_dump"),
                "result.workspace_offset_within_dump",
            ),
            "workspace does not immediately follow the result header",
        )
        require(
            integer(
                memory_map.get("result_device_offset"),
                "memory_map.result_device_offset",
            )
            == 0,
            "full dump must begin at result_device_offset zero",
        )
        workspace_bytes = integer(
            memory_map.get("workspace_bytes"), "memory_map.workspace_bytes"
        )
        require(
            workspace_bytes
            == integer(memory_plan.get("arena_bytes"), "memory_plan.arena_bytes"),
            "workspace and liveness arena byte counts disagree",
        )
        output_base = integer(
            result.get("workspace_offset_within_dump"),
            "result.workspace_offset_within_dump",
        )
        workspace_end = output_base + workspace_bytes
        require(
            workspace_end <= len(dump),
            "dump is too short for the declared workspace",
        )

        header = result_header(dump)
        computed_fnv = fnv1a(dump[output_base:workspace_end])
        expected_header = {
            "magic": RESULT_MAGIC,
            "version": RESULT_VERSION,
            "status": 0,
            "failed_node": 0xFFFFFFFF,
            "failed_op": 0,
            "first_node": 0,
            "last_node": 307,
            "node_count": len(manifest["nodes"]),
            "tensor_count": len(manifest["tensors"]),
            "workspace_bytes": workspace_bytes,
            "input_blob_bytes": integer(
                manifest["blobs"]["inputs"].get("nbytes"),
                "blobs.inputs.nbytes",
            ),
            "weight_blob_bytes": integer(
                manifest["blobs"]["weights"].get("nbytes"),
                "blobs.weights.nbytes",
            ),
            "math_version": MATH_VERSION,
            "reserved32": [0, 0, 0],
            "workspace_fnv1a": computed_fnv,
            "reserved64": [0, 0, 0, 0, 0, 0, 0],
        }
        header_checks = {
            name: header.get(name) == expected
            for name, expected in expected_header.items()
        }
        header_pass = all(header_checks.values())
        report["header_pass"] = header_pass
        report["header_checks"] = header_checks
        report["expected_header"] = expected_header
        report["header"] = header
        print(
            "YRF1 {} status={} nodes=N{:03d}:N{:03d} "
            "workspace_bytes={} workspace_fnv1a={:016x}".format(
                "PASS" if header_pass else "FAIL",
                header["status_name"],
                header["first_node"],
                header["last_node"],
                header["workspace_bytes"],
                header["workspace_fnv1a"],
            )
        )
        require(header_pass, "YRF1 header or workspace FNV check failed")

        atol = (
            args.atol
            if args.atol is not None
            else manifest["tolerances"]["atol"]
        )
        rtol = (
            args.rtol
            if args.rtol is not None
            else manifest["tolerances"]["rtol"]
        )
        require(
            isinstance(atol, (int, float))
            and isinstance(rtol, (int, float))
            and math.isfinite(float(atol))
            and math.isfinite(float(rtol))
            and float(atol) >= 0.0
            and float(rtol) >= 0.0,
            "atol and rtol must be finite and non-negative",
        )
        atol = float(atol)
        rtol = float(rtol)
        require(
            atol <= float(manifest["tolerances"]["atol"])
            and rtol <= float(manifest["tolerances"]["rtol"]),
            "command-line tolerances may tighten but not loosen the "
            "validated contract",
        )
        golden_path = safe_package_path(
            full_dir,
            manifest["blobs"]["goldens"]["path"],
            "blobs.goldens.path",
        )
        golden = golden_path.read_bytes()
        checkpoints, final_output = compare_checkpoints(
            manifest,
            dump,
            golden,
            atol,
            rtol,
            nodes_by_id,
            tensors_by_name,
        )
        checkpoint_pass = bool(checkpoints) and all(
            item["pass"] for item in checkpoints
        )
        report["tolerances"] = {"atol": atol, "rtol": rtol}
        report["direct_output_required"] = args.require_direct_output
        report["checkpoint_pass"] = checkpoint_pass
        report["checkpoints"] = checkpoints
        report["final_output"] = final_output
        output_pass = bool(
            final_output is not None
            and (
                final_output["direct_ort_pass"]
                if args.require_direct_output
                else final_output["pass"]
            )
        )
        report["pass"] = bool(
            source_check["pass"]
            and blob_pass
            and header_identity["pass"]
            and dump_pass
            and header_pass
            and checkpoint_pass
            and output_pass
        )
    except (
        CompareError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        report["errors"].append(str(error))
        print("error: {}".format(error), file=sys.stderr)

    print(
        "FULL_COMPARE {} selector={} checkpoints={} runtime_status={}".format(
            "PASS" if report["pass"] else "FAIL",
            EXPECTED_SELECTOR,
            len(report.get("checkpoints", [])),
            report.get("header", {}).get("status_name", "unavailable"),
        )
    )
    return (0 if report["pass"] else 1), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "compare declared full-graph checkpoints in a YRF1 C dump "
            "against exact-artifact ONNX Runtime goldens"
        )
    )
    parser.add_argument(
        "full_dir",
        type=Path,
        help="schema-v2 full package directory",
    )
    parser.add_argument("dump", type=Path, help="full YRF1 launcher dump")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="checksum-verified pinned ONNX source",
    )
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--require-direct-output",
        action="store_true",
        help=(
            "reject tie-aware selection validation unless output0 also "
            "passes direct positional ORT tolerance"
        ),
    )
    parser.add_argument(
        "--json", type=Path, help="write the complete machine-readable report"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status, report = run(args)
    write_json(args.json, report)
    return status


if __name__ == "__main__":
    sys.exit(main())
