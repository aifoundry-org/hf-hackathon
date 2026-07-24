#!/usr/bin/env python3
"""Compare every range output with its pinned ONNX Runtime golden."""

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
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
EXPECTED_REPO = "onnx-community/yolov10n"
EXPECTED_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"
EXPECTED_FILENAME = "onnx/model.onnx"
EXPECTED_SOURCE_SHA256 = (
    "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
)
EXPECTED_LICENSE = "AGPL-3.0"
RESULT_MAGIC = 0x31465259
RESULT_VERSION = 1
MATH_VERSION = 1
RESULT_STRUCT_BYTES = 128
HEADER_U32 = struct.Struct("<16I")
HEADER_U64 = struct.Struct("<8Q")
DTYPES = {"FLOAT": ("<f4", 4), "INT64": ("<i8", 8)}
STORAGES = {"input", "weights", "workspace"}


class CompareError(RuntimeError):
    """The package or dump violates the schema-v2 range contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompareError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fnv1a(data: bytes) -> int:
    result = 14695981039346656037
    for byte in data:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result


def integer(value: Any, field: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        "{} must be an integer".format(field),
    )
    return int(value)


def parse_node_id(value: Any, field: str) -> int:
    require(
        isinstance(value, str)
        and len(value) == 4
        and value[0] == "N"
        and value[1:].isdigit(),
        "{} must have form N000".format(field),
    )
    return int(value[1:])


def shape_elements(shape: Any, field: str) -> int:
    require(isinstance(shape, list), "{} must be a list".format(field))
    elements = 1
    for index, value in enumerate(shape):
        dimension = integer(value, "{}[{}]".format(field, index))
        require(dimension >= 0, "{} contains a negative dimension".format(field))
        elements *= dimension
    return elements


def safe_package_path(
    package_dir: Path, relative: Any, field: str
) -> Path:
    require(
        isinstance(relative, str) and relative,
        "{} must be a non-empty relative path".format(field),
    )
    resolved = (package_dir / relative).resolve()
    try:
        resolved.relative_to(package_dir)
    except ValueError:
        raise CompareError("{} escapes the range package".format(field))
    return resolved


def read_result_header(data: bytes) -> Dict[str, Any]:
    require(
        len(data) >= RESULT_STRUCT_BYTES,
        "dump is shorter than the 128-byte result structure",
    )
    words = HEADER_U32.unpack_from(data, 0)
    longs = HEADER_U64.unpack_from(data, 64)
    return {
        "magic": words[0],
        "version": words[1],
        "status": words[2],
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
    mismatch = ~finite
    mismatch[finite] |= absolute[finite] > allowed[finite]
    relative = np.zeros(actual64.shape, dtype=np.float64)
    relative[finite] = absolute[finite] / np.maximum(
        np.abs(reference64[finite]), np.finfo(np.float32).tiny
    )
    nonfinite = np.flatnonzero(~finite)
    if nonfinite.size:
        worst = int(nonfinite[0])
    else:
        worst = int(np.argmax(absolute)) if absolute.size else 0
    finite_abs = absolute[finite]
    finite_rel = relative[finite]
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
        "max_abs": float(np.max(finite_abs)) if finite_abs.size else 0.0,
        "mean_abs": float(np.mean(finite_abs)) if finite_abs.size else 0.0,
        "max_rel": float(np.max(finite_rel)) if finite_rel.size else 0.0,
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


def verify_source(
    manifest: Dict[str, Any], model_path: Path
) -> Dict[str, Any]:
    source = manifest.get("source")
    require(isinstance(source, dict), "source must be an object")
    expected = {
        "repo": EXPECTED_REPO,
        "revision": EXPECTED_REVISION,
        "filename": EXPECTED_FILENAME,
        "sha256": EXPECTED_SOURCE_SHA256,
        "license": EXPECTED_LICENSE,
    }
    fields = {
        name: source.get(name) == value for name, value in expected.items()
    }
    exists = model_path.is_file()
    actual_sha = file_sha256(model_path) if exists else None
    passed = all(fields.values()) and actual_sha == EXPECTED_SOURCE_SHA256
    return {
        "pass": passed,
        "path": str(model_path),
        "exists": exists,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "actual_sha256": actual_sha,
        "manifest_fields": fields,
    }


def verify_blob(
    package_dir: Path, record: Any, field: str
) -> Tuple[bytes, Dict[str, Any]]:
    require(isinstance(record, dict), "{} must be an object".format(field))
    path = safe_package_path(package_dir, record.get("path"), field + ".path")
    require(path.is_file(), "{} is missing".format(path))
    data = path.read_bytes()
    expected_bytes = integer(record.get("nbytes"), field + ".nbytes")
    expected_sha = record.get("sha256")
    require(
        isinstance(expected_sha, str) and len(expected_sha) == 64,
        "{}.sha256 is invalid".format(field),
    )
    actual_sha = bytes_sha256(data)
    require(
        len(data) == expected_bytes,
        "{} size mismatch: manifest={} actual={}".format(
            field, expected_bytes, len(data)
        ),
    )
    require(
        actual_sha == expected_sha,
        "{} SHA-256 mismatch".format(field),
    )
    return data, {
        "pass": True,
        "path": str(path),
        "nbytes": len(data),
        "sha256": actual_sha,
    }


def compare_package(
    package_dir: Path,
    dump_path: Path,
    model_path: Path,
    atol_override: Optional[float],
    rtol_override: Optional[float],
) -> Dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "slice_manifest.json"
    require(manifest_path.is_file(), "missing {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 2, "expected schema_version 2")
    require(
        manifest.get("manifest_kind") == "contiguous_node_range",
        "expected contiguous_node_range manifest",
    )

    source_report = verify_source(manifest, model_path.resolve())
    require(source_report["pass"], "pinned source verification failed")
    source = manifest["source"]
    instrumented_path = safe_package_path(
        package_dir,
        source.get("instrumented_path"),
        "source.instrumented_path",
    )
    require(
        instrumented_path.is_file()
        and file_sha256(instrumented_path)
        == source.get("instrumented_sha256"),
        "instrumented range model identity mismatch",
    )
    source_report["instrumented_path"] = str(instrumented_path)
    source_report["instrumented_sha256"] = file_sha256(instrumented_path)

    generated = manifest.get("generated")
    require(isinstance(generated, dict), "generated must be an object")
    _, header_report = verify_blob(
        package_dir, generated.get("header"), "generated.header"
    )

    selection = manifest.get("selection")
    require(isinstance(selection, dict), "selection must be an object")
    first = parse_node_id(selection.get("first_node"), "selection.first_node")
    last = parse_node_id(selection.get("last_node"), "selection.last_node")
    node_count = integer(selection.get("node_count"), "selection.node_count")
    require(first <= last, "selection is descending")
    require(node_count == last - first + 1, "selection node count mismatch")
    require(
        selection.get("selector")
        == "N{:03d}:N{:03d}".format(first, last),
        "selector is not canonical",
    )

    nodes = manifest.get("nodes")
    require(
        isinstance(nodes, list) and len(nodes) == node_count,
        "node list does not cover the selected range",
    )
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for local_index, node in enumerate(nodes):
        field = "nodes[{}]".format(local_index)
        require(isinstance(node, dict), "{} must be an object".format(field))
        global_index = first + local_index
        node_id = "N{:03d}".format(global_index)
        require(node.get("node_id") == node_id, "{} id mismatch".format(field))
        require(
            node.get("index") == global_index
            and node.get("local_index") == local_index,
            "{} ordinal mismatch".format(field),
        )
        require(
            isinstance(node.get("inputs"), list)
            and isinstance(node.get("outputs"), list)
            and len(node["outputs"]) >= 1,
            "{} inputs/outputs are invalid".format(field),
        )
        nodes_by_id[node_id] = node

    memory_map = manifest.get("memory_map")
    require(isinstance(memory_map, dict), "memory_map must be an object")
    result_offset = integer(
        memory_map.get("result_device_offset"),
        "memory_map.result_device_offset",
    )
    input_offset = integer(
        memory_map.get("input_device_offset"),
        "memory_map.input_device_offset",
    )
    weight_offset = integer(
        memory_map.get("weight_device_offset"),
        "memory_map.weight_device_offset",
    )
    workspace_bytes = integer(
        memory_map.get("workspace_bytes"),
        "memory_map.workspace_bytes",
    )
    input_bytes = integer(
        memory_map.get("input_blob_bytes"),
        "memory_map.input_blob_bytes",
    )
    weight_bytes = integer(
        memory_map.get("weight_blob_bytes"),
        "memory_map.weight_blob_bytes",
    )
    dump_size = integer(memory_map.get("dump_size"), "memory_map.dump_size")
    mem_size = integer(memory_map.get("mem_size"), "memory_map.mem_size")
    require(result_offset == 0, "result offset must be zero")
    require(dump_size <= mem_size, "dump exceeds memory size")

    result_record = manifest.get("result")
    require(isinstance(result_record, dict), "result must be an object")
    header_bytes = integer(
        result_record.get("header_bytes"), "result.header_bytes"
    )
    require(
        header_bytes >= RESULT_STRUCT_BYTES,
        "result header is shorter than the ABI structure",
    )
    workspace_base = result_offset + header_bytes
    require(
        workspace_base + workspace_bytes <= dump_size,
        "workspace exceeds dump size",
    )
    require(
        input_offset + input_bytes <= mem_size
        and weight_offset + weight_bytes <= mem_size,
        "input or weights exceed memory size",
    )

    blobs = manifest.get("blobs")
    require(isinstance(blobs, dict), "blobs must be an object")
    input_blob, input_report = verify_blob(
        package_dir, blobs.get("inputs"), "blobs.inputs"
    )
    weight_blob, weight_report = verify_blob(
        package_dir, blobs.get("weights"), "blobs.weights"
    )
    golden_blob, golden_report = verify_blob(
        package_dir, blobs.get("goldens"), "blobs.goldens"
    )
    require(len(input_blob) == input_bytes, "input blob/memory map mismatch")
    require(
        len(weight_blob) == weight_bytes, "weight blob/memory map mismatch"
    )

    tensors = manifest.get("tensors")
    require(isinstance(tensors, list) and tensors, "tensors must be non-empty")
    tensors_by_name: Dict[str, Dict[str, Any]] = {}
    for tensor_index, tensor in enumerate(tensors):
        field = "tensors[{}]".format(tensor_index)
        require(isinstance(tensor, dict), "{} must be an object".format(field))
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        storage = tensor.get("storage")
        require(
            isinstance(name, str) and name,
            "{}.name is invalid".format(field),
        )
        require(name not in tensors_by_name, "duplicate tensor {!r}".format(name))
        require(dtype in DTYPES, "{} has unsupported dtype".format(field))
        require(storage in STORAGES, "{} has invalid storage".format(field))
        elements = shape_elements(tensor.get("shape"), field + ".shape")
        require(
            tensor.get("elements") == elements,
            "{} element count mismatch".format(field),
        )
        nbytes = elements * DTYPES[dtype][1]
        require(
            tensor.get("nbytes") == nbytes,
            "{} byte count mismatch".format(field),
        )
        offset = integer(tensor.get("offset"), field + ".offset")
        limit = {
            "input": input_bytes,
            "weights": weight_bytes,
            "workspace": workspace_bytes,
        }[storage]
        require(
            offset >= 0 and offset + nbytes <= limit,
            "{} exceeds its {} blob".format(field, storage),
        )
        require(
            offset % DTYPES[dtype][1] == 0,
            "{} offset is not dtype-aligned".format(field),
        )
        tensors_by_name[name] = tensor

    for boundary_index, boundary in enumerate(manifest.get("boundary_inputs", [])):
        field = "boundary_inputs[{}]".format(boundary_index)
        require(isinstance(boundary, dict), "{} must be an object".format(field))
        name = boundary.get("tensor")
        require(name in tensors_by_name, "{} tensor is unknown".format(field))
        tensor = tensors_by_name[name]
        require(
            tensor.get("storage") == "input"
            and tensor.get("role") == "boundary_input",
            "{} is not an input tensor".format(field),
        )
        start = integer(boundary.get("blob_offset"), field + ".blob_offset")
        end = start + integer(boundary.get("nbytes"), field + ".nbytes")
        raw = input_blob[start:end]
        require(
            bytes_sha256(raw) == boundary.get("sha256")
            == tensor.get("segment_sha256"),
            "{} segment SHA-256 mismatch".format(field),
        )

    for initializer_index, initializer in enumerate(
        manifest.get("initializers", [])
    ):
        field = "initializers[{}]".format(initializer_index)
        require(
            isinstance(initializer, dict), "{} must be an object".format(field)
        )
        name = initializer.get("tensor")
        require(name in tensors_by_name, "{} tensor is unknown".format(field))
        tensor = tensors_by_name[name]
        require(
            tensor.get("storage") == "weights"
            and tensor.get("role") == "initializer",
            "{} is not an initializer tensor".format(field),
        )
        start = integer(initializer.get("blob_offset"), field + ".blob_offset")
        end = start + integer(initializer.get("nbytes"), field + ".nbytes")
        raw = weight_blob[start:end]
        require(
            bytes_sha256(raw) == initializer.get("sha256")
            == tensor.get("segment_sha256"),
            "{} segment SHA-256 mismatch".format(field),
        )

    dump_path = dump_path.resolve()
    require(dump_path.is_file(), "missing dump {}".format(dump_path))
    dump = dump_path.read_bytes()
    require(
        len(dump) == dump_size,
        "dump size mismatch: expected={} actual={}".format(
            dump_size, len(dump)
        ),
    )
    header = read_result_header(dump)
    expected_header = {
        "magic": RESULT_MAGIC,
        "version": RESULT_VERSION,
        "status": 0,
        "failed_node": 0xFFFFFFFF,
        "failed_op": 0,
        "first_node": first,
        "last_node": last,
        "node_count": node_count,
        "tensor_count": len(tensors),
        "workspace_bytes": workspace_bytes,
        "input_blob_bytes": input_bytes,
        "weight_blob_bytes": weight_bytes,
        "math_version": MATH_VERSION,
    }
    for name, expected in expected_header.items():
        require(
            header[name] == expected,
            "result header {} mismatch: expected={} actual={}".format(
                name, expected, header[name]
            ),
        )
    require(
        header["reserved32"] == [0, 0, 0]
        and header["reserved64"] == [0] * 7,
        "result header reserved fields are nonzero",
    )
    workspace = dump[workspace_base:workspace_base + workspace_bytes]
    require(
        fnv1a(workspace) == header["workspace_fnv1a"],
        "workspace FNV-1a mismatch",
    )

    tolerances = manifest.get("tolerances")
    require(isinstance(tolerances, dict), "tolerances must be an object")
    atol = (
        float(atol_override)
        if atol_override is not None
        else float(tolerances.get("atol"))
    )
    rtol = (
        float(rtol_override)
        if rtol_override is not None
        else float(tolerances.get("rtol"))
    )
    require(
        math.isfinite(atol) and math.isfinite(rtol)
        and atol >= 0.0 and rtol >= 0.0,
        "tolerances must be finite and nonnegative",
    )

    outputs = manifest.get("outputs")
    require(isinstance(outputs, list) and outputs, "outputs must be non-empty")
    expected_output_count = sum(len(node["outputs"]) for node in nodes)
    require(
        len(outputs) == expected_output_count,
        "output records do not cover every selected node output",
    )
    seen_output_ids = set()
    seen_tensors = set()
    workspace_ranges: List[Tuple[int, int, str]] = []
    golden_ranges: List[Tuple[int, int, str]] = []
    reports = []
    for record_index, record in enumerate(outputs):
        field = "outputs[{}]".format(record_index)
        require(isinstance(record, dict), "{} must be an object".format(field))
        node_id = record.get("node_id")
        output_index = integer(
            record.get("output_index"), field + ".output_index"
        )
        output_id = record.get("output_id")
        name = record.get("tensor")
        require(node_id in nodes_by_id, "{} node is unknown".format(field))
        require(
            output_id == "{}:O{}".format(node_id, output_index)
            and output_id not in seen_output_ids,
            "{} output id is invalid or duplicate".format(field),
        )
        node = nodes_by_id[node_id]
        require(
            output_index < len(node["outputs"])
            and node["outputs"][output_index] == name,
            "{} does not match the ONNX node output".format(field),
        )
        require(
            name in tensors_by_name and name not in seen_tensors,
            "{} tensor is unknown or duplicate".format(field),
        )
        seen_output_ids.add(output_id)
        seen_tensors.add(name)
        tensor = tensors_by_name[name]
        require(
            tensor.get("storage") == "workspace"
            and tensor.get("role") == "node_output_checkpoint"
            and tensor.get("checkpoint") is True
            and tensor.get("producer") == node_id,
            "{} tensor is not a pinned selected output".format(field),
        )
        for key in ("dtype", "shape", "elements", "nbytes"):
            require(
                record.get(key) == tensor.get(key),
                "{} {} disagrees with tensor descriptor".format(field, key),
            )
        dtype = record["dtype"]
        elements = integer(record["elements"], field + ".elements")
        nbytes = integer(record["nbytes"], field + ".nbytes")
        workspace_offset = integer(
            record.get("workspace_offset"), field + ".workspace_offset"
        )
        golden_offset = integer(
            record.get("golden_offset"), field + ".golden_offset"
        )
        require(
            workspace_offset == tensor["offset"]
            and workspace_offset + nbytes <= workspace_bytes,
            "{} workspace mapping is invalid".format(field),
        )
        require(
            golden_offset >= 0
            and golden_offset + nbytes <= len(golden_blob),
            "{} golden mapping is invalid".format(field),
        )
        for start, end, prior in workspace_ranges:
            require(
                workspace_offset + nbytes <= start
                or workspace_offset >= end,
                "{} overlaps workspace output {}".format(field, prior),
            )
        for start, end, prior in golden_ranges:
            require(
                golden_offset + nbytes <= start or golden_offset >= end,
                "{} overlaps golden output {}".format(field, prior),
            )
        workspace_ranges.append(
            (workspace_offset, workspace_offset + nbytes, output_id)
        )
        golden_ranges.append(
            (golden_offset, golden_offset + nbytes, output_id)
        )
        actual_raw = workspace[
            workspace_offset:workspace_offset + nbytes
        ]
        reference_raw = golden_blob[golden_offset:golden_offset + nbytes]
        require(
            bytes_sha256(reference_raw) == record.get("golden_sha256"),
            "{} golden segment SHA-256 mismatch".format(field),
        )
        numpy_dtype = DTYPES[dtype][0]
        actual = np.frombuffer(
            actual_raw, dtype=numpy_dtype, count=elements
        )
        reference = np.frombuffer(
            reference_raw, dtype=numpy_dtype, count=elements
        )
        if dtype == "FLOAT":
            metrics = compare_float(actual, reference, atol, rtol)
        else:
            metrics = compare_int64(actual, reference)
        metrics.update(
            {
                "output_id": output_id,
                "node_id": node_id,
                "op_type": node["op_type"],
                "output_index": output_index,
                "tensor": name,
                "dtype": dtype,
                "shape": record["shape"],
                "workspace_offset": workspace_offset,
                "golden_offset": golden_offset,
                "actual_sha256": bytes_sha256(actual_raw),
                "reference_sha256": bytes_sha256(reference_raw),
            }
        )
        reports.append(metrics)
        print(
            "NODE_OUTPUT {} id={} op={} tensor={} dtype={} shape={} "
            "max_abs={:.9g} max_rel={:.9g} mismatches={}/{}".format(
                "PASS" if metrics["pass"] else "FAIL",
                output_id,
                node["op_type"],
                name,
                dtype,
                record["shape"],
                metrics["max_abs"],
                metrics["max_rel"],
                metrics["mismatch_count"],
                metrics["elements"],
            )
        )

    passed = all(item["pass"] for item in reports)
    return {
        "schema_version": 1,
        "kind": "contiguous_range_comparison",
        "pass": passed,
        "selector": selection["selector"],
        "source": source_report,
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "header": header_report,
        },
        "dump": {
            "path": str(dump_path),
            "nbytes": len(dump),
            "sha256": file_sha256(dump_path),
        },
        "blobs": {
            "inputs": input_report,
            "weights": weight_report,
            "goldens": golden_report,
        },
        "result_header": header,
        "tolerances": {"atol": atol, "rtol": rtol},
        "summary": {
            "node_count": node_count,
            "output_count": len(reports),
            "passed_outputs": sum(bool(item["pass"]) for item in reports),
            "failed_outputs": sum(not bool(item["pass"]) for item in reports),
            "total_elements": sum(item["elements"] for item in reports),
            "total_mismatches": sum(
                item["mismatch_count"] for item in reports
            ),
        },
        "outputs": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("range_dir", type=Path)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare_package(
            args.range_dir,
            args.dump,
            args.model,
            args.atol,
            args.rtol,
        )
    except (CompareError, OSError, ValueError, KeyError) as exc:
        print("RANGE_COMPARE FAIL {}".format(exc), file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    summary = report["summary"]
    print(
        "RANGE_COMPARE {} selector={} outputs={}/{} "
        "mismatches={}/{} report={}".format(
            "PASS" if report["pass"] else "FAIL",
            report["selector"],
            summary["passed_outputs"],
            summary["output_count"],
            summary["total_mismatches"],
            summary["total_elements"],
            str(args.json) if args.json is not None else "none",
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
