#!/usr/bin/env python3
"""Pack the pinned YOLOv10n ONNX initializers without changing their values.

The package is deliberately just concatenated tensor bytes plus zero alignment
padding.  Names, shapes, types, offsets, sizes, and hashes live in the readable
JSON manifest; there is no generated C parser or private container format.

Only ONNX FLOAT (IEEE-754 binary32) and integer tensors are accepted.  FLOAT16,
DOUBLE, BOOL, strings, complex values, sparse tensors, and external data are
rejected rather than converted or approximated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper, shape_inference


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_ARTIFACTS = PORT_ROOT / "artifacts.json"
DEFAULT_WEIGHTS = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/package/weights.bin"
)
DEFAULT_OUTPUT_MANIFEST = PORT_ROOT / "manifests/weights_manifest.json"
ALIGNMENT = 64

PINNED_SOURCE = {
    "type": "huggingface",
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "url": (
        "https://huggingface.co/onnx-community/yolov10n/resolve/"
        "57657320425ee34056408a57ad9d29c4d4815bd8/"
        "onnx/model.onnx?download=true"
    ),
}
PINNED_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
PINNED_LICENSE = "AGPL-3.0"

# Explicit canonical storage types.  A leading '<' makes all multi-byte
# elements little-endian even if this tool is run on a big-endian host.
SUPPORTED_DTYPES = {
    TensorProto.FLOAT: np.dtype("<f4"),
    TensorProto.UINT8: np.dtype("<u1"),
    TensorProto.INT8: np.dtype("<i1"),
    TensorProto.UINT16: np.dtype("<u2"),
    TensorProto.INT16: np.dtype("<i2"),
    TensorProto.INT32: np.dtype("<i4"),
    TensorProto.INT64: np.dtype("<i8"),
    TensorProto.UINT32: np.dtype("<u4"),
    TensorProto.UINT64: np.dtype("<u8"),
}

TYPED_STORAGE_FIELD = {
    TensorProto.FLOAT: "float_data",
    TensorProto.UINT8: "int32_data",
    TensorProto.INT8: "int32_data",
    TensorProto.UINT16: "int32_data",
    TensorProto.INT16: "int32_data",
    TensorProto.INT32: "int32_data",
    TensorProto.INT64: "int64_data",
    TensorProto.UINT32: "uint64_data",
    TensorProto.UINT64: "uint64_data",
}

ALL_TYPED_STORAGE_FIELDS = (
    "float_data",
    "int32_data",
    "string_data",
    "int64_data",
    "double_data",
    "uint64_data",
)


class PackError(RuntimeError):
    """A source condition that this transparent packer cannot preserve."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer reproducible repository-relative paths in generated metadata."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_pinned_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as src:
        document = json.load(src)
    try:
        artifact = document["artifacts"]["yolov10n_onnx"]
    except (KeyError, TypeError) as exc:
        raise PackError(f"malformed artifact manifest {path}: {exc}") from exc

    source = artifact.get("source")
    if not isinstance(source, dict):
        raise PackError(f"artifact source is missing or malformed in {path}")
    for key, expected in PINNED_SOURCE.items():
        actual = source.get(key)
        if actual != expected:
            raise PackError(
                f"artifact pin mismatch for source.{key}: "
                f"expected={expected!r} actual={actual!r}"
            )

    actual_sha = str(artifact.get("sha256", "")).lower()
    if actual_sha != PINNED_SHA256:
        raise PackError(
            "artifact pin mismatch for sha256: "
            f"expected={PINNED_SHA256} actual={actual_sha}"
        )
    if artifact.get("license") != PINNED_LICENSE:
        raise PackError(
            "artifact pin mismatch for license: "
            f"expected={PINNED_LICENSE!r} actual={artifact.get('license')!r}"
        )
    if not artifact.get("local_cache"):
        raise PackError(f"artifact local_cache is missing in {path}")
    return artifact


def resolve_model_path(
    artifact: Dict[str, Any], requested: Optional[Path]
) -> Path:
    if requested is not None:
        return requested.resolve()
    cache = Path(str(artifact["local_cache"]))
    if not cache.is_absolute():
        cache = REPO_ROOT / cache
    return cache.resolve()


def iter_graphs(
    graph: onnx.GraphProto, scope: str = "main"
) -> Iterable[Tuple[str, onnx.GraphProto]]:
    yield scope, graph
    for node_index, node in enumerate(graph.node):
        node_label = node.name or f"{node.op_type}[{node_index}]"
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                child_scope = f"{scope}/{node_label}:{attribute.name}"
                yield from iter_graphs(attribute.g, child_scope)
            elif attribute.type == AttributeProto.GRAPHS:
                for graph_index, child in enumerate(attribute.graphs):
                    child_scope = (
                        f"{scope}/{node_label}:{attribute.name}[{graph_index}]"
                    )
                    yield from iter_graphs(child, child_scope)


def reject_sparse_and_nested_initializers(model: onnx.ModelProto) -> None:
    """Reject storage forms that would otherwise be silently omitted."""

    for scope, graph in iter_graphs(model.graph):
        if graph.sparse_initializer:
            raise PackError(
                f"sparse initializers are unsupported: "
                f"scope={scope} count={len(graph.sparse_initializer)}"
            )
        if scope != "main" and graph.initializer:
            raise PackError(
                "nested-graph initializers are unsupported: "
                f"scope={scope} count={len(graph.initializer)}"
            )
        for node_index, node in enumerate(graph.node):
            for attribute in node.attribute:
                if attribute.type in (
                    AttributeProto.SPARSE_TENSOR,
                    AttributeProto.SPARSE_TENSORS,
                ):
                    node_label = node.name or f"{node.op_type}[{node_index}]"
                    raise PackError(
                        "sparse tensor attributes are unsupported: "
                        f"scope={scope} node={node_label} attribute={attribute.name}"
                    )


def element_count(dims: Iterable[int]) -> int:
    count = 1
    for raw_dim in dims:
        dim = int(raw_dim)
        if dim < 0:
            raise PackError(f"initializer has negative dimension {dim}")
        count *= dim
    return count


def populated_typed_fields(tensor: TensorProto) -> List[str]:
    return [
        field_name
        for field_name in ALL_TYPED_STORAGE_FIELDS
        if len(getattr(tensor, field_name))
    ]


def initializer_bytes(tensor: TensorProto) -> Tuple[bytes, str]:
    """Return canonical little-endian bytes and their source storage form."""

    if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
        raise PackError(
            f"external initializer data is unsupported: name={tensor.name!r}"
        )

    storage_dtype = SUPPORTED_DTYPES.get(tensor.data_type)
    if storage_dtype is None:
        try:
            dtype_name = TensorProto.DataType.Name(tensor.data_type)
        except ValueError:
            dtype_name = f"UNKNOWN({tensor.data_type})"
        raise PackError(
            f"unsupported initializer dtype: name={tensor.name!r} "
            f"dtype={dtype_name}"
        )

    elements = element_count(tensor.dims)
    expected_nbytes = elements * storage_dtype.itemsize
    typed_fields = populated_typed_fields(tensor)

    if tensor.raw_data:
        if typed_fields:
            raise PackError(
                f"ambiguous initializer storage: name={tensor.name!r} "
                f"has raw_data and {typed_fields}"
            )
        data = bytes(tensor.raw_data)
        if len(data) != expected_nbytes:
            raise PackError(
                f"raw initializer byte count mismatch: name={tensor.name!r} "
                f"expected={expected_nbytes} actual={len(data)}"
            )
        # ONNX raw_data is defined to use little-endian element bytes.  Keep
        # those bytes verbatim, including FLOAT bit patterns such as signed
        # zero or NaN payloads.
        return data, "raw_data"

    expected_field = TYPED_STORAGE_FIELD[tensor.data_type]
    unexpected_fields = [
        field_name for field_name in typed_fields if field_name != expected_field
    ]
    if unexpected_fields:
        raise PackError(
            f"initializer uses storage fields inconsistent with its dtype: "
            f"name={tensor.name!r} fields={unexpected_fields}"
        )
    actual_values = len(getattr(tensor, expected_field))
    if actual_values != elements:
        raise PackError(
            f"typed initializer element count mismatch: name={tensor.name!r} "
            f"field={expected_field} expected={elements} actual={actual_values}"
        )

    try:
        logical = numpy_helper.to_array(tensor)
    except Exception as exc:
        raise PackError(
            f"could not decode initializer {tensor.name!r}: {exc}"
        ) from exc
    logical = np.asarray(logical)
    if logical.size != elements:
        raise PackError(
            f"decoded initializer element count mismatch: name={tensor.name!r} "
            f"expected={elements} actual={logical.size}"
        )
    if (
        logical.dtype.kind != storage_dtype.kind
        or logical.dtype.itemsize != storage_dtype.itemsize
    ):
        raise PackError(
            f"decoded initializer dtype mismatch: name={tensor.name!r} "
            f"expected={storage_dtype} actual={logical.dtype}"
        )

    canonical = np.ascontiguousarray(logical, dtype=storage_dtype)
    data = canonical.tobytes(order="C")
    if len(data) != expected_nbytes:
        raise PackError(
            f"encoded initializer byte count mismatch: name={tensor.name!r} "
            f"expected={expected_nbytes} actual={len(data)}"
        )
    return data, expected_field


def alignment_padding(offset: int) -> int:
    return (-offset) % ALIGNMENT


def build_package(
    initializers: Iterable[Tuple[int, TensorProto]]
) -> Tuple[bytes, List[Dict[str, Any]], Dict[str, Any]]:
    blob = bytearray()
    records: List[Dict[str, Any]] = []
    dtype_counts: Counter = Counter()
    data_bytes = 0
    padding_bytes = 0

    # Name ordering is independent of incidental protobuf field ordering.
    ordered = sorted(initializers, key=lambda item: item[1].name)
    names = [tensor.name for _, tensor in ordered]
    if any(not name for name in names):
        raise PackError("all initializers must have non-empty names")
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count != 1
    )
    if duplicate_names:
        raise PackError(f"duplicate initializer names: {duplicate_names}")

    for pack_index, (graph_index, tensor) in enumerate(ordered):
        padding_before = alignment_padding(len(blob))
        if padding_before:
            blob.extend(b"\x00" * padding_before)
        offset = len(blob)
        if offset % ALIGNMENT:
            raise AssertionError("internal alignment error")

        data, source_storage = initializer_bytes(tensor)
        blob.extend(data)
        dtype_name = TensorProto.DataType.Name(tensor.data_type)
        shape = [int(dim) for dim in tensor.dims]
        elements = element_count(shape)
        dtype_counts[dtype_name] += 1
        data_bytes += len(data)
        padding_bytes += padding_before
        records.append(
            {
                "pack_index": pack_index,
                "graph_index": graph_index,
                "name": tensor.name,
                "dtype": dtype_name,
                "dtype_code": int(tensor.data_type),
                "shape": shape,
                "elements": elements,
                "offset": offset,
                "nbytes": len(data),
                "end_offset": offset + len(data),
                "padding_before": padding_before,
                "sha256": sha256_bytes(data),
                "source_storage": source_storage,
            }
        )

    trailing_padding = alignment_padding(len(blob))
    if trailing_padding:
        blob.extend(b"\x00" * trailing_padding)
    padding_bytes += trailing_padding
    statistics = {
        "initializer_count": len(records),
        "data_bytes": data_bytes,
        "padding_bytes": padding_bytes,
        "trailing_padding": trailing_padding,
        "total_bytes": len(blob),
        "float_initializer_count": dtype_counts.get("FLOAT", 0),
        "integer_initializer_count": len(records)
        - dtype_counts.get("FLOAT", 0),
    }
    statistics["dtype_counts"] = dict(sorted(dtype_counts.items()))
    return bytes(blob), records, statistics


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        # mkstemp defaults to 0600. These are reproducible model artifacts,
        # not secrets; keep generated manifests and cache blobs readable by
        # the same users as ordinary checkout files.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as dst:
            dst.write(data)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(
    artifact_manifest_path: Path,
    model_path: Path,
    model: onnx.ModelProto,
    inferred: onnx.ModelProto,
    weights_path: Path,
    package: bytes,
    records: List[Dict[str, Any]],
    statistics: Dict[str, Any],
) -> Dict[str, Any]:
    package_record = {
        "path": display_path(weights_path),
        "format": "headerless_concatenated_tensor_bytes",
        "byte_order": "little",
        "alignment_bytes": ALIGNMENT,
        "initializer_order": "name_ascending",
        "padding_value": 0,
        "initializer_count": statistics["initializer_count"],
        "dtype_counts": statistics["dtype_counts"],
        "float_initializer_count": statistics["float_initializer_count"],
        "integer_initializer_count": statistics["integer_initializer_count"],
        "data_bytes": statistics["data_bytes"],
        "padding_bytes": statistics["padding_bytes"],
        "trailing_padding": statistics["trailing_padding"],
        "total_bytes": statistics["total_bytes"],
        "sha256": sha256_bytes(package),
    }
    return {
        "schema_version": 1,
        "source": {
            "type": PINNED_SOURCE["type"],
            "repo": PINNED_SOURCE["repo"],
            "revision": PINNED_SOURCE["revision"],
            "filename": PINNED_SOURCE["filename"],
            "url": PINNED_SOURCE["url"],
            "sha256": PINNED_SHA256,
            "license": PINNED_LICENSE,
            "artifact_manifest": display_path(artifact_manifest_path),
            "local_model": display_path(model_path),
            "model_bytes": model_path.stat().st_size,
        },
        "validation": {
            "model_checksum": "PASS",
            "onnx_checker_original": "PASS",
            "onnx_shape_inference": "PASS",
            "onnx_checker_inferred": "PASS",
            "onnx_ir_version": int(model.ir_version),
            "opsets": {
                item.domain or "ai.onnx": int(item.version)
                for item in model.opset_import
            },
            "graph_nodes": len(model.graph.node),
            "graph_initializers": len(model.graph.initializer),
            "graph_sparse_initializers": len(model.graph.sparse_initializer),
            "value_info_before_inference": len(model.graph.value_info),
            "value_info_after_inference": len(inferred.graph.value_info),
        },
        "package": package_record,
        "initializers": records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    artifacts_path = args.artifacts.resolve()
    artifact = load_pinned_artifact(artifacts_path)
    model_path = resolve_model_path(artifact, args.model)
    weights_path = args.weights_out.resolve()
    manifest_path = args.manifest_out.resolve()

    if not model_path.is_file():
        raise PackError(
            f"pinned model is missing: {model_path}; "
            "run tools/download_model.py first"
        )
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != PINNED_SHA256:
        raise PackError(
            f"model checksum mismatch: path={model_path} "
            f"expected={PINNED_SHA256} actual={actual_model_sha}"
        )
    print(
        f"SOURCE_CHECK PASS path={model_path} "
        f"bytes={model_path.stat().st_size} sha256={actual_model_sha}"
    )

    # Do not resolve external-data references implicitly.  This keeps a
    # one-file pin meaningful and lets the explicit rejection below fire.
    model = onnx.load(str(model_path), load_external_data=False)
    reject_sparse_and_nested_initializers(model)
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(
        model, check_type=True, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred)
    print(
        "MODEL_CHECK PASS "
        f"nodes={len(model.graph.node)} "
        f"initializers={len(model.graph.initializer)} "
        f"inferred_value_info={len(inferred.graph.value_info)}"
    )

    package, records, statistics = build_package(
        enumerate(model.graph.initializer)
    )
    if statistics["initializer_count"] != len(model.graph.initializer):
        raise PackError(
            "initializer coverage mismatch: "
            f"graph={len(model.graph.initializer)} "
            f"packed={statistics['initializer_count']}"
        )
    if any(record["offset"] % ALIGNMENT for record in records):
        raise PackError("one or more initializer offsets are not aligned")
    if statistics["total_bytes"] % ALIGNMENT:
        raise PackError("package total size is not aligned")

    manifest = build_manifest(
        artifacts_path,
        model_path,
        model,
        inferred,
        weights_path,
        package,
        records,
        statistics,
    )
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")

    atomic_write(weights_path, package)
    written_sha = sha256_file(weights_path)
    expected_package_sha = manifest["package"]["sha256"]
    if written_sha != expected_package_sha:
        raise PackError(
            f"written package checksum mismatch: "
            f"expected={expected_package_sha} actual={written_sha}"
        )
    if weights_path.stat().st_size != statistics["total_bytes"]:
        raise PackError(
            f"written package size mismatch: "
            f"expected={statistics['total_bytes']} "
            f"actual={weights_path.stat().st_size}"
        )
    atomic_write(manifest_path, manifest_bytes)

    print(
        "INITIALIZERS PASS "
        f"count={statistics['initializer_count']} "
        f"dtypes={statistics['dtype_counts']} "
        f"data_bytes={statistics['data_bytes']} "
        f"padding_bytes={statistics['padding_bytes']}"
    )
    print(
        f"WEIGHTS PASS path={weights_path} "
        f"bytes={statistics['total_bytes']} sha256={written_sha}"
    )
    print(f"MANIFEST PASS path={manifest_path} entries={len(records)}")
    print("PACK PASS")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and pack every initializer from the pinned YOLOv10n ONNX "
            "artifact into a 64-byte-aligned little-endian byte blob."
        )
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"pinned artifact manifest (default: {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "model path override; it must still match the pinned SHA-256 "
            "(default: artifacts.json local_cache)"
        ),
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"raw package output (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
        help=f"JSON manifest output (default: {DEFAULT_OUTPUT_MANIFEST})",
    )
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except (PackError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PACK FAIL reason={exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
