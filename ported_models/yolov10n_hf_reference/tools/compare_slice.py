#!/usr/bin/env python3
"""Compare every selected C tensor against its exact-artifact ORT golden."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List

import numpy as np


RESULT_MAGIC = 0x31465259
EXPECTED_SOURCE_SHA256 = (
    "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
)
HEADER_U32 = struct.Struct("<16I")
HEADER_U64 = struct.Struct("<8Q")
STATUS_NAMES = {
    0: "ok",
    1: "bad_manifest",
    2: "unsupported_op",
    3: "unsupported_shape",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_header(data: bytes) -> Dict[str, Any]:
    if len(data) < 128:
        raise ValueError("dump is shorter than the 128-byte result header")
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
        "workspace_fnv1a": longs[0],
    }


def fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def compare(
    actual: np.ndarray, reference: np.ndarray, atol: float, rtol: float
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
        "pass": bool(not np.any(mismatch)),
        "elements": int(actual.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "nonfinite_actual_count": int(np.count_nonzero(~np.isfinite(actual64))),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slice_dir", type=Path)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--json", type=Path, help="also write a machine-readable report")
    args = parser.parse_args()

    slice_dir = args.slice_dir.resolve()
    manifest = json.loads((slice_dir / "slice_manifest.json").read_text())
    if manifest["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
        raise SystemExit("error: slice manifest does not name the pinned ONNX")
    dump = args.dump.read_bytes()
    blob_checks: Dict[str, Dict[str, Any]] = {}
    for name, record in manifest["blobs"].items():
        path = (slice_dir / record["path"]).resolve()
        try:
            path.relative_to(slice_dir)
        except ValueError:
            raise SystemExit(
                f"error: blob path escapes slice directory: {record['path']}"
            )
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = file_sha256(path) if exists else None
        passed = (
            exists
            and actual_bytes == int(record["nbytes"])
            and actual_sha256 == record["sha256"]
        )
        blob_checks[name] = {
            "pass": passed,
            "path": record["path"],
            "expected_bytes": int(record["nbytes"]),
            "actual_bytes": actual_bytes,
            "expected_sha256": record["sha256"],
            "actual_sha256": actual_sha256,
        }
        print(
            f"BLOB {'PASS' if passed else 'FAIL'} name={name} "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )
    blob_pass = bool(blob_checks) and all(
        check["pass"] for check in blob_checks.values()
    )
    if not blob_pass:
        raise SystemExit("error: one or more slice blobs failed identity checks")
    golden = (
        slice_dir / manifest["blobs"]["goldens"]["path"]
    ).read_bytes()
    header = result_header(dump)
    atol = args.atol if args.atol is not None else manifest["tolerances"]["atol"]
    rtol = args.rtol if args.rtol is not None else manifest["tolerances"]["rtol"]
    if (
        not math.isfinite(atol)
        or not math.isfinite(rtol)
        or atol < 0.0
        or rtol < 0.0
    ):
        raise SystemExit("error: atol and rtol must be finite and non-negative")
    output_base = int(manifest["result"]["workspace_offset_within_dump"])
    workspace_bytes = int(manifest["blobs"]["goldens"]["nbytes"])
    workspace_end = output_base + workspace_bytes
    if workspace_end > len(dump):
        raise SystemExit(
            f"error: dump too short for workspace: need {workspace_end}, "
            f"have {len(dump)}"
        )
    expected_header = {
        "magic": RESULT_MAGIC,
        "version": 1,
        "status": 0,
        "failed_node": 0xFFFFFFFF,
        "failed_op": 0,
        "first_node": int(manifest["selection"]["first_node"][1:]),
        "last_node": int(manifest["selection"]["last_node"][1:]),
        "node_count": len(manifest["nodes"]),
        "tensor_count": len(manifest["tensors"]),
        "workspace_bytes": workspace_bytes,
        "input_blob_bytes": int(manifest["blobs"]["inputs"]["nbytes"]),
        "weight_blob_bytes": int(manifest["blobs"]["weights"]["nbytes"]),
        "math_version": 1,
        "workspace_fnv1a": fnv1a(dump[output_base:workspace_end]),
    }
    header_checks = {
        key: header[key] == expected for key, expected in expected_header.items()
    }
    header_pass = all(header_checks.values())
    reports: List[Dict[str, Any]] = []

    output_to_node = {
        name: node["node_id"]
        for node in manifest["nodes"]
        for name in node["outputs"]
    }
    for tensor in manifest["tensors"]:
        if tensor["storage"] != "workspace":
            continue
        offset = int(tensor["offset"])
        nbytes = int(tensor["nbytes"])
        elements = int(tensor["elements"])
        if output_base + offset + nbytes > len(dump):
            raise SystemExit(
                f"error: dump too short for {tensor['name']}: "
                f"need {output_base + offset + nbytes}, have {len(dump)}"
            )
        if offset + nbytes > len(golden):
            raise SystemExit(f"error: golden blob too short for {tensor['name']}")
        actual = np.frombuffer(
            dump, dtype="<f4", count=elements, offset=output_base + offset
        )
        reference = np.frombuffer(
            golden, dtype="<f4", count=elements, offset=offset
        )
        metrics = compare(actual, reference, atol, rtol)
        metrics.update(
            {
                "node_id": output_to_node.get(tensor["name"]),
                "tensor": tensor["name"],
                "shape": tensor["shape"],
            }
        )
        reports.append(metrics)
        print(
            f"TENSOR {'PASS' if metrics['pass'] else 'FAIL'} "
            f"node={metrics['node_id']} shape={tensor['shape']} "
            f"max_abs={metrics['max_abs']:.9g} "
            f"max_rel={metrics['max_rel']:.9g} "
            f"mean_abs={metrics['mean_abs']:.9g} "
            f"mismatches={metrics['mismatch_count']}/{metrics['elements']} "
            f"nonfinite_actual={metrics['nonfinite_actual_count']} "
            f"nonfinite_reference={metrics['nonfinite_reference_count']} "
            f"atol={atol:g} rtol={rtol:g}"
        )

    overall = (
        blob_pass
        and header_pass
        and bool(reports)
        and all(item["pass"] for item in reports)
    )
    report = {
        "schema_version": 1,
        "pass": overall,
        "blob_pass": blob_pass,
        "blob_checks": blob_checks,
        "header_pass": header_pass,
        "header_checks": header_checks,
        "expected_header": expected_header,
        "header": header,
        "source_sha256": manifest["source"]["sha256"],
        "selection": manifest["selection"],
        "tensors": reports,
    }
    print(
        f"SLICE_COMPARE {'PASS' if overall else 'FAIL'} "
        f"nodes={manifest['selection']['first_node']}:{manifest['selection']['last_node']} "
        f"runtime_status={header['status_name']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
