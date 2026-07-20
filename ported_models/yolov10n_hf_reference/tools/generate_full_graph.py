#!/usr/bin/env python3
"""Generate the full pinned-ONNX scalar execution package.

This is a declarative conversion, not a model export.  The checksum-verified
ONNX remains authoritative for node order, tensor names/types/shapes,
attributes, and initializer values.  The generated C header contains only
descriptors and a liveness-derived arena plan; operator implementations remain
in the small hand-written scalar runtime.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
import onnxruntime as ort


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_WEIGHTS = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/package/weights.bin"
)
DEFAULT_WEIGHTS_MANIFEST = PORT_ROOT / "manifests/weights_manifest.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/full_graph"
)
DEFAULT_EXECUTION_MANIFEST = PORT_ROOT / "manifests/full_execution.json"

EXPECTED_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
EXPECTED_WEIGHTS_SHA256 = (
    "b6d5aa13ef3238328c19ff0f646f72841bcf44dc1651383d089f0d57e37cf850"
)
EXPECTED_HEADER_BYTES = 92881
EXPECTED_HEADER_SHA256 = (
    "79be5b751842df025a3612ebb690e283813ea9ac8e373fd1bc44b706ca7a2a7e"
)
ALIGNMENT = 64
PAGE_ALIGNMENT = 0x10000
RESULT_HEADER_BYTES = 4096
RESULT_DEVICE_OFFSET = 0
PMC_STAGE_STRIDE = 0x10000

DTYPE_CODES = {"FLOAT": 1, "INT64": 2}
DTYPE_BYTES = {"FLOAT": 4, "INT64": 8}
STORAGE_CODES = {"input": 1, "weights": 2, "workspace": 3}
OP_CODES = {
    "Conv": 1,
    "Sigmoid": 2,
    "Mul": 3,
    "Concat": 4,
    "Add": 5,
    "Split": 6,
    "MaxPool": 7,
    "Resize": 8,
    "MatMul": 9,
    "Softmax": 10,
    "Reshape": 11,
    "Transpose": 12,
    "Sub": 13,
    "ReduceMax": 14,
    "TopK": 15,
    "Unsqueeze": 16,
    "Tile": 17,
    "GatherElements": 18,
    "Flatten": 19,
    "Mod": 20,
    "Div": 21,
    "Cast": 22,
}
EXPECTED_OP_COUNTS = {
    "Add": 11,
    "Cast": 1,
    "Concat": 21,
    "Conv": 83,
    "Div": 1,
    "Flatten": 1,
    "GatherElements": 3,
    "MatMul": 2,
    "MaxPool": 3,
    "Mod": 1,
    "Mul": 71,
    "ReduceMax": 1,
    "Reshape": 8,
    "Resize": 2,
    "Sigmoid": 70,
    "Softmax": 2,
    "Split": 13,
    "Sub": 1,
    "Tile": 3,
    "TopK": 2,
    "Transpose": 4,
    "Unsqueeze": 4,
}
ALLOWED_ATTRIBUTES = {
    "Add": set(),
    "Cast": {"to"},
    "Concat": {"axis"},
    "Conv": {"dilations", "group", "kernel_shape", "pads", "strides"},
    "Div": set(),
    "Flatten": {"axis"},
    "GatherElements": {"axis"},
    "MatMul": set(),
    "MaxPool": {
        "ceil_mode",
        "dilations",
        "kernel_shape",
        "pads",
        "strides",
    },
    "Mod": {"fmod"},
    "Mul": set(),
    "ReduceMax": {"axes", "keepdims"},
    "Reshape": {"allowzero"},
    "Resize": {
        "coordinate_transformation_mode",
        "cubic_coeff_a",
        "mode",
        "nearest_mode",
    },
    "Sigmoid": set(),
    "Softmax": {"axis"},
    "Split": {"axis"},
    "Sub": set(),
    "Tile": set(),
    "TopK": {"axis", "largest", "sorted"},
    "Transpose": {"perm"},
    "Unsqueeze": set(),
}

# Measured architecture boundaries used for full-graph PMC intervals.
CHECKPOINT_NODES = (
    5,
    20,
    45,
    71,
    90,
    100,
    128,
    144,
    160,
    178,
    207,
    228,
    249,
    270,
    288,
    307,
)

# Seven requested PMC categories.  They partition N000:N307 exactly.
PMC_STAGES = (
    ("stem", 0, 5),
    ("backbone", 6, 90),
    ("sppf_psa", 91, 128),
    ("neck", 129, 207),
    ("three_scale_head", 208, 270),
    ("dfl_decode", 271, 288),
    ("topk_selection", 289, 307),
)


class GenerationError(RuntimeError):
    """A graph or package condition outside the full-reference contract."""


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(array: np.ndarray) -> bytes:
    if array.dtype == np.float32:
        return np.asarray(array, dtype="<f4", order="C").tobytes(order="C")
    if array.dtype == np.int64:
        return np.asarray(array, dtype="<i8", order="C").tobytes(order="C")
    raise GenerationError("unsupported array dtype {}".format(array.dtype))


def deterministic_input(shape: Sequence[int]) -> np.ndarray:
    elements = int(np.prod(shape, dtype=np.int64))
    index = np.arange(elements, dtype=np.uint64)
    bits = (
        index * np.uint64(1664525) + np.uint64(1013904223)
    ) & np.uint64(0x00FFFFFF)
    values = bits.astype(np.float32) * np.float32(1.0 / 16777215.0)
    return values.reshape(tuple(shape))


def value_shape(value: onnx.ValueInfoProto) -> List[int]:
    shape = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise GenerationError(
                "tensor {!r} has a non-static dimension".format(value.name)
            )
        shape.append(int(dimension.dim_value))
    return shape


def value_dtype(value: onnx.ValueInfoProto) -> str:
    return TensorProto.DataType.Name(value.type.tensor_type.elem_type)


def element_count(shape: Iterable[int]) -> int:
    result = 1
    for dimension in shape:
        if int(dimension) < 0:
            raise GenerationError("negative static dimension {}".format(dimension))
        result *= int(dimension)
    return result


def json_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return [json_attribute(item) for item in value]
    if isinstance(value, list):
        return [json_attribute(item) for item in value]
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "values": array.reshape(-1).tolist(),
        }
    return value


def tensor_metadata(
    model: onnx.ModelProto,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, onnx.ValueInfoProto],
    Dict[str, onnx.TensorProto],
]:
    values = {
        item.name: item
        for item in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
    }
    initializers = {item.name: item for item in model.graph.initializer}
    metadata: Dict[str, Dict[str, Any]] = {}
    for name, value in values.items():
        dtype = value_dtype(value)
        shape = value_shape(value)
        if dtype not in DTYPE_CODES:
            raise GenerationError(
                "runtime tensor {!r} has unsupported dtype {}".format(name, dtype)
            )
        metadata[name] = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "elements": element_count(shape),
            "nbytes": element_count(shape) * DTYPE_BYTES[dtype],
        }
    for name, tensor in initializers.items():
        dtype = TensorProto.DataType.Name(tensor.data_type)
        shape = [int(item) for item in tensor.dims]
        if dtype not in DTYPE_CODES:
            raise GenerationError(
                "initializer {!r} has unsupported dtype {}".format(name, dtype)
            )
        metadata[name] = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "elements": element_count(shape),
            "nbytes": element_count(shape) * DTYPE_BYTES[dtype],
        }
    return metadata, values, initializers


def validate_graph(
    model: onnx.ModelProto,
    metadata: Dict[str, Dict[str, Any]],
) -> None:
    imports = {(item.domain, int(item.version)) for item in model.opset_import}
    if imports != {("", 13)}:
        raise GenerationError(
            "expected the sole ONNX opset import ('', 13), got {}".format(
                sorted(imports)
            )
        )
    if len(model.graph.node) != 308:
        raise GenerationError(
            "expected 308 nodes, got {}".format(len(model.graph.node))
        )
    if len(model.graph.initializer) != 187:
        raise GenerationError(
            "expected 187 initializers, got {}".format(
                len(model.graph.initializer)
            )
        )
    actual_ops = {node.op_type for node in model.graph.node}
    unsupported = sorted(actual_ops - set(OP_CODES))
    if unsupported:
        raise GenerationError("unsupported graph operators: {}".format(unsupported))
    actual_counts = {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in actual_ops
    }
    if actual_counts != EXPECTED_OP_COUNTS:
        raise GenerationError(
            "operator histogram differs from the pin: {}".format(actual_counts)
        )
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        raise GenerationError("expected sole graph input named images")
    if metadata["images"] != {
        "name": "images",
        "dtype": "FLOAT",
        "shape": [1, 3, 640, 640],
        "elements": 1228800,
        "nbytes": 4915200,
    }:
        raise GenerationError("input metadata differs from the pinned contract")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "output0":
        raise GenerationError("expected sole graph output named output0")
    if metadata["output0"]["dtype"] != "FLOAT" or metadata["output0"][
        "shape"
    ] != [1, 300, 6]:
        raise GenerationError("output0 metadata differs from the pinned contract")

    initializer_arrays = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }

    for index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx"):
            raise GenerationError(
                "N{:03d} uses unsupported domain {!r}".format(index, node.domain)
            )
        if len(node.input) > 4 or len(node.output) > 3:
            raise GenerationError(
                "N{:03d} exceeds manifest arity: inputs={} outputs={}".format(
                    index, len(node.input), len(node.output)
                )
            )
        for name in node.input:
            if name and name not in metadata:
                raise GenerationError(
                    "N{:03d} input {!r} lacks metadata".format(index, name)
                )
        for name in node.output:
            if not name or name not in metadata:
                raise GenerationError(
                    "N{:03d} output {!r} lacks metadata".format(index, name)
                )
        attrs = {
            item.name: helper.get_attribute_value(item) for item in node.attribute
        }
        unexpected_attrs = set(attrs) - ALLOWED_ATTRIBUTES[node.op_type]
        if unexpected_attrs:
            raise GenerationError(
                "N{:03d} {} has unimplemented attributes {}".format(
                    index, node.op_type, sorted(unexpected_attrs)
                )
            )
        if node.op_type == "Reshape":
            if attrs.get("allowzero", 0) != 0:
                raise GenerationError(
                    "N{:03d} Reshape allowzero=1 is unsupported".format(index)
                )
            if (
                len(node.input) != 2
                or node.input[1] not in initializer_arrays
                or initializer_arrays[node.input[1]].dtype != np.int64
            ):
                raise GenerationError(
                    "N{:03d} Reshape requires a packed INT64 shape".format(index)
                )
        if node.op_type == "Split":
            if (
                len(node.input) != 2
                or node.input[1] not in initializer_arrays
                or initializer_arrays[node.input[1]].dtype != np.int64
                or initializer_arrays[node.input[1]].size != len(node.output)
            ):
                raise GenerationError(
                    "N{:03d} Split requires one INT64 size per output".format(
                        index
                    )
                )
        if node.op_type == "Resize":
            expected = {
                "coordinate_transformation_mode": b"asymmetric",
                "mode": b"nearest",
                "nearest_mode": b"floor",
            }
            for key, value in expected.items():
                if attrs.get(key) != value:
                    raise GenerationError(
                        "N{:03d} Resize {} is {!r}, expected {!r}".format(
                            index, key, attrs.get(key), value
                        )
                    )
            if (
                len(node.input) != 3
                or node.input[1] != ""
                or node.input[2] not in initializer_arrays
                or initializer_arrays[node.input[2]].dtype != np.float32
                or not np.array_equal(
                    initializer_arrays[node.input[2]],
                    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                )
            ):
                raise GenerationError(
                    "N{:03d} Resize is not exact nearest 2x scales".format(index)
                )
        if node.op_type == "TopK":
            if attrs.get("largest", 1) != 1 or attrs.get("sorted", 1) != 1:
                raise GenerationError(
                    "N{:03d} requires largest sorted TopK".format(index)
                )
            if (
                len(node.input) != 2
                or node.input[1] not in initializer_arrays
                or initializer_arrays[node.input[1]].dtype != np.int64
                or initializer_arrays[node.input[1]].size != 1
            ):
                raise GenerationError(
                    "N{:03d} TopK requires a scalar packed INT64 K".format(index)
                )
        if node.op_type == "MaxPool" and attrs.get("ceil_mode", 0) != 0:
            raise GenerationError(
                "N{:03d} MaxPool ceil_mode is unsupported".format(index)
            )
        if node.op_type == "Mod" and attrs.get("fmod", 0) != 0:
            raise GenerationError(
                "N{:03d} floating-point Mod is unsupported".format(index)
            )
        if node.op_type == "Cast" and attrs.get("to") != TensorProto.FLOAT:
            raise GenerationError(
                "N{:03d} Cast target is not FLOAT".format(index)
            )
        if node.op_type in ("Unsqueeze", "Tile"):
            if (
                len(node.input) != 2
                or node.input[1] not in initializer_arrays
                or initializer_arrays[node.input[1]].dtype != np.int64
            ):
                raise GenerationError(
                    "N{:03d} {} requires a packed INT64 control tensor".format(
                        index, node.op_type
                    )
                )
        if node.op_type in ("Mod", "Div"):
            if (
                len(node.input) != 2
                or node.input[1] not in initializer_arrays
                or initializer_arrays[node.input[1]].dtype != np.int64
                or initializer_arrays[node.input[1]].size != 1
                or int(initializer_arrays[node.input[1]].reshape(-1)[0]) == 0
            ):
                raise GenerationError(
                    "N{:03d} {} requires a nonzero scalar INT64 divisor".format(
                        index, node.op_type
                    )
                )


def verify_weight_package(
    model: onnx.ModelProto,
    weights_path: Path,
    manifest_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    package = json.loads(manifest_path.read_text(encoding="utf-8"))
    if package.get("schema_version") != 1:
        raise GenerationError("unknown weights manifest schema")
    if package.get("source", {}).get("sha256") != EXPECTED_SHA256:
        raise GenerationError("weights manifest source SHA does not match the pin")
    package_record = package.get("package")
    if not isinstance(package_record, dict):
        raise GenerationError("weights manifest lacks package metadata")
    if (
        package_record["sha256"] != EXPECTED_WEIGHTS_SHA256
        or file_sha256(weights_path) != EXPECTED_WEIGHTS_SHA256
        or weights_path.stat().st_size != int(package_record["total_bytes"])
    ):
        raise GenerationError("packed initializer blob identity mismatch")
    if (
        package_record.get("format")
        != "headerless_concatenated_tensor_bytes"
        or package_record.get("byte_order") != "little"
        or package_record.get("alignment_bytes") != ALIGNMENT
        or package_record.get("initializer_order") != "name_ascending"
        or package_record.get("padding_value") != 0
    ):
        raise GenerationError("weights package layout contract differs")

    listed_records = package.get("initializers")
    if not isinstance(listed_records, list):
        raise GenerationError("weights manifest initializers are not a list")
    records = {item.get("name"): item for item in listed_records}
    if len(records) != len(listed_records) or None in records:
        raise GenerationError("weights manifest has duplicate/invalid names")
    if set(records) != {item.name for item in model.graph.initializer}:
        raise GenerationError("weights manifest initializer coverage mismatch")
    graph_indices = {
        initializer.name: index
        for index, initializer in enumerate(model.graph.initializer)
    }
    sorted_initializers = sorted(
        model.graph.initializer, key=lambda initializer: initializer.name
    )
    blob = weights_path.read_bytes()
    cursor = 0
    data_bytes = 0
    dtype_counts: Dict[str, int] = {}
    for pack_index, initializer in enumerate(sorted_initializers):
        record = listed_records[pack_index]
        array = numpy_helper.to_array(initializer)
        data = canonical_bytes(array)
        dtype = TensorProto.DataType.Name(initializer.data_type)
        expected_offset = align(cursor)
        expected_end = expected_offset + len(data)
        if (
            record.get("pack_index") != pack_index
            or record.get("graph_index") != graph_indices[initializer.name]
            or record.get("name") != initializer.name
            or record.get("dtype") != dtype
            or record.get("dtype_code") != int(initializer.data_type)
            or record["shape"] != [int(item) for item in initializer.dims]
            or record.get("elements") != int(array.size)
            or record.get("offset") != expected_offset
            or int(record["nbytes"]) != len(data)
            or record.get("end_offset") != expected_end
            or record.get("padding_before") != expected_offset - cursor
            or record["sha256"] != hashlib.sha256(data).hexdigest()
        ):
            raise GenerationError(
                "initializer package mapping mismatch for {!r}".format(
                    initializer.name
                )
            )
        if any(blob[cursor:expected_offset]):
            raise GenerationError(
                "initializer padding is nonzero before {!r}".format(
                    initializer.name
                )
            )
        if blob[expected_offset:expected_end] != data:
            raise GenerationError(
                "packed initializer segment differs for {!r}".format(
                    initializer.name
                )
            )
        cursor = expected_end
        data_bytes += len(data)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    final_size = align(cursor)
    if len(blob) != final_size or any(blob[cursor:final_size]):
        raise GenerationError("weights package trailing padding differs")
    padding_bytes = len(blob) - data_bytes
    if (
        package_record.get("initializer_count") != len(sorted_initializers)
        or package_record.get("dtype_counts") != dtype_counts
        or package_record.get("float_initializer_count")
        != dtype_counts.get("FLOAT", 0)
        or package_record.get("integer_initializer_count")
        != len(sorted_initializers) - dtype_counts.get("FLOAT", 0)
        or package_record.get("data_bytes") != data_bytes
        or package_record.get("padding_bytes") != padding_bytes
        or package_record.get("trailing_padding") != final_size - cursor
    ):
        raise GenerationError("weights package summary metadata differs")
    return records, package


def allocate_liveness(
    model: onnx.ModelProto,
    metadata: Dict[str, Dict[str, Any]],
    pinned: Set[str],
) -> Tuple[int, Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    consumers: Dict[str, int] = {}
    producer: Dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name] = max(consumers.get(name, -1), index)
        for name in node.output:
            producer[name] = index
    for item in model.graph.output:
        consumers[item.name] = len(model.graph.node)

    live: Dict[str, Tuple[int, int]] = {}
    free: List[Tuple[int, int]] = []
    allocations: Dict[str, Dict[str, int]] = {}
    end = 0
    events: List[Dict[str, Any]] = []

    def merge_free(blocks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        for offset, size in sorted(blocks):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                old_offset, old_size = merged[-1]
                merged[-1] = (old_offset, old_size + size)
            else:
                merged.append((offset, size))
        return merged

    for node_index, node in enumerate(model.graph.node):
        released = []
        for name, (offset, size) in list(live.items()):
            last_use = consumers.get(name, -1)
            if name in pinned:
                last_use = len(model.graph.node)
            if last_use < node_index:
                released.append(
                    {"tensor": name, "offset": offset, "nbytes": size}
                )
                free.append((offset, size))
                del live[name]
        free = merge_free(free)

        allocated = []
        for name in node.output:
            size = align(int(metadata[name]["nbytes"]))
            chosen: Optional[int] = None
            for block_index, (block_offset, block_size) in enumerate(free):
                aligned_offset = align(block_offset)
                prefix = aligned_offset - block_offset
                if block_size - prefix < size:
                    continue
                chosen = aligned_offset
                replacement: List[Tuple[int, int]] = []
                if prefix:
                    replacement.append((block_offset, prefix))
                used_end = aligned_offset + size
                block_end = block_offset + block_size
                if used_end < block_end:
                    replacement.append((used_end, block_end - used_end))
                free = (
                    free[:block_index] + replacement + free[block_index + 1 :]
                )
                break
            if chosen is None:
                chosen = align(end)
                end = chosen + size
            allocation = {
                "offset": chosen,
                "allocated_nbytes": size,
                "live_start": node_index,
                "live_end": (
                    len(model.graph.node)
                    if name in pinned
                    else consumers.get(name, node_index)
                ),
            }
            allocations[name] = allocation
            live[name] = (chosen, size)
            allocated.append({"tensor": name, **allocation})
        events.append(
            {
                "node_id": "N{:03d}".format(node_index),
                "released": released,
                "allocated": allocated,
                "arena_end": end,
            }
        )
    return align(end), allocations, events


def write_aligned_blob(
    path: Path, entries: Iterable[Tuple[str, bytes]]
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    cursor = 0
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("wb") as destination:
        for name, data in entries:
            offset = align(cursor)
            if offset > cursor:
                destination.write(b"\x00" * (offset - cursor))
            destination.write(data)
            records[name] = {
                "offset": offset,
                "nbytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            cursor = offset + len(data)
        final_size = align(cursor)
        if final_size > cursor:
            destination.write(b"\x00" * (final_size - cursor))
    return final_size, records


def c_array(values: Sequence[int], width: int, suffix: str = "") -> str:
    padded = list(values[:width]) + [0] * max(0, width - len(values))
    return "{ " + ", ".join("{}{}".format(int(item), suffix) for item in padded) + " }"


def node_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    return {
        item.name: json_attribute(helper.get_attribute_value(item))
        for item in node.attribute
    }


def node_c_record(
    index: int,
    node: onnx.NodeProto,
    tensor_ids: Dict[str, int],
) -> str:
    attrs = {
        item.name: helper.get_attribute_value(item) for item in node.attribute
    }
    kernel = list(attrs.get("kernel_shape", [0, 0]))
    strides = list(attrs.get("strides", [1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    dilations = list(attrs.get("dilations", [1, 1]))
    axes = list(attrs.get("axes", []))
    perm = list(attrs.get("perm", []))
    inputs = [
        0xFFFFFFFF if not name else tensor_ids[name] for name in node.input
    ]
    outputs = [tensor_ids[name] for name in node.output]
    axis = int(attrs.get("axis", 0))
    values = [
        "{}u".format(index),
        "{}u".format(OP_CODES[node.op_type]),
        "{}u".format(len(node.input)),
        "{}u".format(len(node.output)),
        c_array(inputs, 4, "u"),
        c_array(outputs, 3, "u"),
        str(int(attrs.get("group", 1))),
        str(int(kernel[0]) if len(kernel) > 0 else 0),
        str(int(kernel[1]) if len(kernel) > 1 else 0),
        str(int(strides[0]) if len(strides) > 0 else 1),
        str(int(strides[1]) if len(strides) > 1 else 1),
        str(int(pads[0]) if len(pads) > 0 else 0),
        str(int(pads[1]) if len(pads) > 1 else 0),
        str(int(pads[2]) if len(pads) > 2 else 0),
        str(int(pads[3]) if len(pads) > 3 else 0),
        str(int(dilations[0]) if len(dilations) > 0 else 1),
        str(int(dilations[1]) if len(dilations) > 1 else 1),
        str(axis),
        "{}u".format(len(axes)),
        c_array(axes, 6),
        "{}u".format(len(perm)),
        c_array(perm, 6),
        str(int(attrs.get("ceil_mode", 0))),
        str(int(attrs.get("keepdims", 1))),
        str(int(attrs.get("largest", 1))),
        str(int(attrs.get("sorted", 1))),
        str(int(attrs.get("fmod", 0))),
        str(int(attrs.get("to", 0))),
        "1" if node.op_type == "Resize" else "0",
    ]
    return "    { " + ", ".join(values) + " },"


def write_header(
    path: Path,
    tensor_records: List[Dict[str, Any]],
    model: onnx.ModelProto,
    memory_map: Dict[str, int],
) -> None:
    tensor_ids = {
        tensor["name"]: index for index, tensor in enumerate(tensor_records)
    }
    lines = [
        "/* Generated directly from the pinned ONNX by tools/generate_full_graph.py. */",
        "#ifndef YOLOV10N_HF_SLICE_MANIFEST_H",
        "#define YOLOV10N_HF_SLICE_MANIFEST_H",
        "",
        "#include <stdint.h>",
        "",
        "#define YR_MANIFEST_VERSION 2u",
        "#define YR_FIRST_NODE 0u",
        "#define YR_LAST_NODE 307u",
        "#define YR_NODE_COUNT 308u",
        "#define YR_TENSOR_COUNT {}u".format(len(tensor_records)),
        "#define YR_RESULT_HEADER_BYTES {}u".format(RESULT_HEADER_BYTES),
        "#define YR_RESULT_DEVICE_OFFSET 0x{:08x}u".format(
            memory_map["result_device_offset"]
        ),
        "#define YR_INPUT_DEVICE_OFFSET 0x{:08x}u".format(
            memory_map["input_device_offset"]
        ),
        "#define YR_WEIGHT_DEVICE_OFFSET 0x{:08x}u".format(
            memory_map["weight_device_offset"]
        ),
        "#define YR_PMC_DEVICE_OFFSET 0x{:08x}u".format(
            memory_map["pmc_device_offset"]
        ),
        "#define YR_PMC_STAGE_COUNT {}u".format(len(PMC_STAGES)),
        "#define YR_PMC_STAGE_STRIDE 0x{:08x}u".format(PMC_STAGE_STRIDE),
        "#define YR_INPUT_BLOB_BYTES {}u".format(memory_map["input_blob_bytes"]),
        "#define YR_WEIGHT_BLOB_BYTES {}u".format(memory_map["weight_blob_bytes"]),
        "#define YR_WORKSPACE_BYTES {}u".format(memory_map["workspace_bytes"]),
        "#define YR_DUMP_SIZE 0x{:08x}u".format(memory_map["dump_size"]),
        "#define YR_MEM_SIZE 0x{:08x}u".format(memory_map["mem_size"]),
        "",
        "enum yr_storage { YR_STORAGE_INPUT = 1, YR_STORAGE_WEIGHTS = 2, YR_STORAGE_WORKSPACE = 3 };",
        "enum yr_dtype { YR_DTYPE_FLOAT = 1, YR_DTYPE_INT64 = 2 };",
        (
            "enum yr_op { YR_OP_CONV = 1, YR_OP_SIGMOID = 2, "
            "YR_OP_MUL = 3, YR_OP_CONCAT = 4, YR_OP_ADD = 5, "
            "YR_OP_SPLIT = 6, YR_OP_MAXPOOL = 7, YR_OP_RESIZE = 8, "
            "YR_OP_MATMUL = 9, YR_OP_SOFTMAX = 10, YR_OP_RESHAPE = 11, "
            "YR_OP_TRANSPOSE = 12, YR_OP_SUB = 13, "
            "YR_OP_REDUCEMAX = 14, YR_OP_TOPK = 15, "
            "YR_OP_UNSQUEEZE = 16, YR_OP_TILE = 17, "
            "YR_OP_GATHERELEMENTS = 18, YR_OP_FLATTEN = 19, "
            "YR_OP_MOD = 20, YR_OP_DIV = 21, YR_OP_CAST = 22 };"
        ),
        "",
        "struct yr_tensor_desc {",
        "    uint32_t storage, offset, nbytes, elements, rank, dtype;",
        "    uint32_t dims[6];",
        "};",
        "",
        "struct yr_node_desc {",
        "    uint32_t onnx_index, op, input_count, output_count;",
        "    uint32_t inputs[4], outputs[3];",
        "    int32_t group, kernel_h, kernel_w, stride_h, stride_w;",
        "    int32_t pad_top, pad_left, pad_bottom, pad_right;",
        "    int32_t dilation_h, dilation_w, axis;",
        "    uint32_t axes_count;",
        "    int32_t axes[6];",
        "    uint32_t perm_count;",
        "    int32_t perm[6];",
        "    int32_t ceil_mode, keepdims, largest, sorted, fmod, to;",
        "    int32_t resize_nearest_asymmetric_floor;",
        "};",
        "",
        "struct yr_pmc_stage_desc {",
        "    uint32_t first_local_node, last_local_node;",
        "    uint32_t first_onnx_node, last_onnx_node;",
        "};",
        "",
        "static const struct yr_tensor_desc yr_tensors[YR_TENSOR_COUNT] = {",
    ]
    for tensor in tensor_records:
        lines.append(
            "    {{ "
            "{}u, {}u, {}u, {}u, {}u, {}u, {}"
            " }},".format(
                STORAGE_CODES[tensor["storage"]],
                tensor["offset"],
                tensor["nbytes"],
                tensor["elements"],
                len(tensor["shape"]),
                DTYPE_CODES[tensor["dtype"]],
                c_array(tensor["shape"], 6),
            )
        )
    lines.extend(
        [
            "};",
            "",
            "static const struct yr_node_desc yr_nodes[YR_NODE_COUNT] = {",
        ]
    )
    for index, node in enumerate(model.graph.node):
        lines.append(node_c_record(index, node, tensor_ids))
    lines.extend(
        [
            "};",
            "",
            (
                "static const struct yr_pmc_stage_desc "
                "yr_pmc_stages[YR_PMC_STAGE_COUNT] = {"
            ),
        ]
    )
    for _, first, last in PMC_STAGES:
        lines.append(
            "    {{ {}u, {}u, {}u, {}u }},".format(first, last, first, last)
        )
    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def capture_checkpoints(
    model: onnx.ModelProto,
    model_path: Path,
    input_array: np.ndarray,
    checkpoint_names: List[str],
    output_dir: Path,
) -> Tuple[
    Dict[str, np.ndarray],
    ort.InferenceSession,
    Path,
]:
    instrumented = copy.deepcopy(model)
    values = {
        item.name: item
        for item in (
            list(instrumented.graph.input)
            + list(instrumented.graph.value_info)
            + list(instrumented.graph.output)
        )
    }
    existing = {item.name for item in instrumented.graph.output}
    for name in checkpoint_names:
        if name not in values:
            raise GenerationError(
                "checkpoint {!r} has no inferred ValueInfo".format(name)
            )
        if name not in existing:
            instrumented.graph.output.append(copy.deepcopy(values[name]))
            existing.add(name)
    instrumented_path = output_dir / "instrumented_full.onnx"
    onnx.save(instrumented, str(instrumented_path))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(instrumented_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(checkpoint_names, {"images": input_array})
    return dict(zip(checkpoint_names, outputs)), session, instrumented_path


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = args.model.resolve()
    weights_path = args.weights.resolve()
    weights_manifest_path = args.weights_manifest.resolve()
    output_dir = args.output_root.resolve() / args.name
    execution_manifest_path = args.execution_manifest.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_sha = file_sha256(model_path)
    if actual_sha != EXPECTED_SHA256:
        raise GenerationError(
            "model SHA mismatch: expected={} actual={}".format(
                EXPECTED_SHA256, actual_sha
            )
        )
    original = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(original)
    inferred = shape_inference.infer_shapes(
        original, check_type=True, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred)
    metadata, _, initializers = tensor_metadata(inferred)
    validate_graph(inferred, metadata)
    weight_records, weight_package = verify_weight_package(
        inferred, weights_path, weights_manifest_path
    )

    input_shape = metadata["images"]["shape"]
    if args.input_bin is None:
        input_array = deterministic_input(input_shape)
        input_description = {
            "kind": "deterministic_lcg",
            "formula": (
                "(index * 1664525 + 1013904223) & 0x00ffffff, "
                "divided by 16777215"
            ),
        }
    else:
        input_path = args.input_bin.resolve()
        data = input_path.read_bytes()
        expected_bytes = metadata["images"]["nbytes"]
        if len(data) != expected_bytes:
            raise GenerationError(
                "input blob has {} bytes, expected {}".format(
                    len(data), expected_bytes
                )
            )
        input_array = np.frombuffer(data, dtype="<f4").reshape(tuple(input_shape))
        if not np.all(np.isfinite(input_array)):
            raise GenerationError("input blob contains non-finite FP32 values")
        input_description = {
            "kind": "provided_fp32",
            "source_path": str(input_path),
            "source_sha256": file_sha256(input_path),
        }

    input_path_out = output_dir / "inputs.bin"
    input_path_out.write_bytes(canonical_bytes(input_array))
    copied_weights = output_dir / "weights.bin"
    shutil.copyfile(str(weights_path), str(copied_weights))
    if file_sha256(copied_weights) != EXPECTED_WEIGHTS_SHA256:
        raise GenerationError("copied full weight package failed identity check")

    checkpoint_names = [
        inferred.graph.node[index].output[-1] for index in CHECKPOINT_NODES
    ]
    if checkpoint_names[-1] != "output0":
        raise GenerationError("final checkpoint is not output0")
    captures, session, instrumented_path = capture_checkpoints(
        inferred, model_path, input_array, checkpoint_names, output_dir
    )
    golden_entries = [
        (name, canonical_bytes(captures[name])) for name in checkpoint_names
    ]
    golden_path = output_dir / "goldens.bin"
    golden_size, golden_layout = write_aligned_blob(
        golden_path, golden_entries
    )

    pinned = set(checkpoint_names)
    workspace_bytes, allocations, liveness_events = allocate_liveness(
        inferred, metadata, pinned
    )
    pmc_device_offset = align(
        RESULT_HEADER_BYTES + workspace_bytes, PAGE_ALIGNMENT
    )
    dump_size = pmc_device_offset + len(PMC_STAGES) * PMC_STAGE_STRIDE
    input_device_offset = align(dump_size, PAGE_ALIGNMENT)
    weight_device_offset = align(
        input_device_offset + input_path_out.stat().st_size, PAGE_ALIGNMENT
    )
    mem_size = align(
        weight_device_offset + copied_weights.stat().st_size, PAGE_ALIGNMENT
    )
    memory_map = {
        "result_device_offset": RESULT_DEVICE_OFFSET,
        "input_device_offset": input_device_offset,
        "weight_device_offset": weight_device_offset,
        "pmc_device_offset": pmc_device_offset,
        "pmc_stage_stride": PMC_STAGE_STRIDE,
        "pmc_stage_count": len(PMC_STAGES),
        "workspace_bytes": workspace_bytes,
        "input_blob_bytes": input_path_out.stat().st_size,
        "weight_blob_bytes": copied_weights.stat().st_size,
        "dump_size": dump_size,
        "mem_size": mem_size,
    }

    tensor_records: List[Dict[str, Any]] = []
    input_record = dict(metadata["images"])
    input_record.update({"storage": "input", "offset": 0})
    tensor_records.append(input_record)
    for item in weight_package["initializers"]:
        name = item["name"]
        record = dict(metadata[name])
        record.update(
            {
                "storage": "weights",
                "offset": int(weight_records[name]["offset"]),
                "pack_index": int(weight_records[name]["pack_index"]),
            }
        )
        tensor_records.append(record)
    for node_index, node in enumerate(inferred.graph.node):
        for name in node.output:
            record = dict(metadata[name])
            record.update(
                {
                    "storage": "workspace",
                    "offset": allocations[name]["offset"],
                    "allocated_nbytes": allocations[name]["allocated_nbytes"],
                    "producer": "N{:03d}".format(node_index),
                    "live_start": allocations[name]["live_start"],
                    "live_end": allocations[name]["live_end"],
                    "checkpoint": name in pinned,
                }
            )
            tensor_records.append(record)

    header_path = output_dir / "slice_manifest.h"
    write_header(header_path, tensor_records, inferred, memory_map)
    if (
        header_path.stat().st_size != EXPECTED_HEADER_BYTES
        or file_sha256(header_path) != EXPECTED_HEADER_SHA256
    ):
        raise GenerationError(
            "generated header identity differs from the pinned topology"
        )

    node_records = []
    for index, node in enumerate(inferred.graph.node):
        node_records.append(
            {
                "node_id": "N{:03d}".format(index),
                "index": index,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": node_attributes(node),
            }
        )

    checkpoint_records = []
    for node_index, name in zip(CHECKPOINT_NODES, checkpoint_names):
        checkpoint_records.append(
            {
                "node_id": "N{:03d}".format(node_index),
                "tensor": name,
                "dtype": metadata[name]["dtype"],
                "shape": metadata[name]["shape"],
                "elements": metadata[name]["elements"],
                "nbytes": metadata[name]["nbytes"],
                "workspace_offset": allocations[name]["offset"],
                "golden_offset": golden_layout[name]["offset"],
                "golden_sha256": golden_layout[name]["sha256"],
            }
        )

    manifest = {
        "schema_version": 2,
        "manifest_kind": "full_graph_liveness",
        "source": {
            "repo": "onnx-community/yolov10n",
            "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
            "filename": "onnx/model.onnx",
            "sha256": actual_sha,
            "license": "AGPL-3.0",
            "instrumentation": "shape inference plus checkpoint graph outputs only",
            "instrumented_sha256": file_sha256(instrumented_path),
        },
        "reference": {
            "runtime": "onnxruntime",
            "runtime_version": ort.__version__,
            "providers": session.get_providers(),
            "graph_optimization": "ORT_DISABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "input": input_description,
        },
        "selection": {
            "selector": "N000:N307",
            "first_node": "N000",
            "last_node": "N307",
            "inclusive": True,
            "supported_ops": list(OP_CODES),
        },
        "nodes": node_records,
        "tensors": tensor_records,
        "checkpoints": checkpoint_records,
        "pmc_stages": [
            {
                "name": name,
                "first_node": "N{:03d}".format(first),
                "last_node": "N{:03d}".format(last),
                "pmc_device_offset": pmc_device_offset
                + stage_index * PMC_STAGE_STRIDE,
            }
            for stage_index, (name, first, last) in enumerate(PMC_STAGES)
        ],
        "memory_plan": {
            "algorithm": (
                "deterministic first-fit, allocate outputs before releasing "
                "inputs whose last consumer is the current node"
            ),
            "alignment_bytes": ALIGNMENT,
            "pinned_checkpoint_count": len(pinned),
            "arena_bytes": workspace_bytes,
            "events": liveness_events,
        },
        "generated": {
            "header": {
                "path": "slice_manifest.h",
                "nbytes": header_path.stat().st_size,
                "sha256": file_sha256(header_path),
            },
        },
        "blobs": {
            "inputs": {
                "path": "inputs.bin",
                "nbytes": input_path_out.stat().st_size,
                "sha256": file_sha256(input_path_out),
            },
            "weights": {
                "path": "weights.bin",
                "nbytes": copied_weights.stat().st_size,
                "sha256": file_sha256(copied_weights),
            },
            "goldens": {
                "path": "goldens.bin",
                "nbytes": golden_size,
                "sha256": file_sha256(golden_path),
            },
        },
        "memory_map": memory_map,
        "result": {
            "magic": "YRF1",
            "header_bytes": RESULT_HEADER_BYTES,
            "workspace_offset_within_dump": RESULT_HEADER_BYTES,
            "output_tensor": "output0",
            "pmc_scope": "seven disjoint begin/end intervals around only ONNX nodes",
        },
        "tolerances": {
            "atol": 0.00005,
            "rtol": 0.0001,
            "comparison": "abs(actual-reference) <= atol + rtol*abs(reference)",
            "checkpoint_overrides": {
                "N288": {
                    "atol": 0.0002,
                    "rtol": 0.0001,
                    "rationale": (
                        "DFL decode accumulates scalar FP32 head and "
                        "softmax differences; this covers measured "
                        "near-zero coordinate error without loosening "
                        "the score or final-output gate"
                    ),
                }
            },
        },
    }
    local_manifest = output_dir / "slice_manifest.json"
    serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    local_manifest.write_text(serialized, encoding="utf-8")
    execution_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    execution_manifest_path.write_text(serialized, encoding="utf-8")

    print(
        "FULL_GRAPH PASS nodes=308 tensors={} ops={} checkpoints={}".format(
            len(tensor_records), len(OP_CODES), len(checkpoint_records)
        )
    )
    print(
        "MEMORY PASS arena={} dump=0x{:x} input=0x{:x} "
        "weights=0x{:x} total=0x{:x}".format(
            workspace_bytes,
            dump_size,
            input_device_offset,
            weight_device_offset,
            mem_size,
        )
    )
    print(
        "BLOBS PASS inputs={} weights={} goldens={} out={}".format(
            input_path_out.stat().st_size,
            copied_weights.stat().st_size,
            golden_size,
            output_dir,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--weights-manifest", type=Path, default=DEFAULT_WEIGHTS_MANIFEST
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="deterministic")
    parser.add_argument("--input-bin", type=Path)
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=DEFAULT_EXECUTION_MANIFEST,
    )
    return parser.parse_args()


def main() -> int:
    try:
        generate(parse_args())
    except (GenerationError, OSError, ValueError, KeyError) as exc:
        print("FULL_GRAPH FAIL {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
