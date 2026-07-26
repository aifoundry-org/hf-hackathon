#!/usr/bin/env python3
"""Capture any contiguous pinned-ONNX node range for scalar C execution.

This tool accepts every operator in the full scalar runtime. It preserves
every selected node output at a distinct workspace offset, including all
outputs of multi-output nodes.
Inputs whose producers are outside the range are captured from ONNX Runtime;
initializers are copied directly from the checksum-verified ONNX artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

from generate_full_graph import (
    ALIGNMENT,
    DTYPE_CODES,
    EXPECTED_SHA256,
    PAGE_ALIGNMENT,
    PMC_STAGE_STRIDE,
    RESULT_DEVICE_OFFSET,
    RESULT_HEADER_BYTES,
    STORAGE_CODES,
    align,
    canonical_bytes,
    c_array,
    deterministic_input,
    file_sha256,
    node_attributes,
    node_c_record,
    tensor_metadata,
    validate_graph,
    write_aligned_blob,
)


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/ranges"
)
EXPECTED_REPO = "onnx-community/yolov10n"
EXPECTED_REVISION = "57657320425ee34056408a57ad9d29c4d4815bd8"
EXPECTED_FILENAME = "onnx/model.onnx"
EXPECTED_LICENSE = "AGPL-3.0"
MIN_MEMORY_BYTES = 0x00400000
PMC_REGION_BYTES = 0x00010000


class CaptureError(RuntimeError):
    """A pinned-artifact or selected-range contract was not satisfied."""


def parse_node(value: str, count: int) -> int:
    match = re.fullmatch(r"[Nn]?([0-9]+)", value.strip())
    if match is None:
        raise CaptureError(
            "invalid node id {!r}; expected N000 or 0".format(value)
        )
    index = int(match.group(1))
    if index < 0 or index >= count:
        raise CaptureError(
            "node index {} is outside 0..{}".format(index, count - 1)
        )
    return index


def parse_range(value: str, count: int) -> Tuple[int, int]:
    parts = value.split(":")
    if len(parts) == 1:
        first = last = parse_node(parts[0], count)
    elif len(parts) == 2:
        first = parse_node(parts[0], count)
        last = parse_node(parts[1], count)
    else:
        raise CaptureError("invalid node range {!r}".format(value))
    if first > last:
        raise CaptureError(
            "node range must be ascending, got N{:03d}:N{:03d}".format(
                first, last
            )
        )
    return first, last


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_full_input(
    input_path: Optional[Path], shape: Sequence[int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
    if input_path is None:
        array = deterministic_input(shape)
        description = {
            "kind": "deterministic_lcg",
            "formula": (
                "(index * 1664525 + 1013904223) & 0x00ffffff, "
                "divided by 16777215"
            ),
            "sha256": bytes_sha256(canonical_bytes(array)),
        }
        return array, description
    resolved = input_path.resolve()
    data = resolved.read_bytes()
    if len(data) != expected_bytes:
        raise CaptureError(
            "full input {} has {} bytes, expected {}".format(
                resolved, len(data), expected_bytes
            )
        )
    array = np.frombuffer(data, dtype="<f4").reshape(tuple(shape))
    if not np.all(np.isfinite(array)):
        raise CaptureError("full input contains non-finite FP32 values")
    return array, {
        "kind": "provided_fp32",
        "source_basename": resolved.name,
        "nbytes": len(data),
        "sha256": bytes_sha256(data),
    }


def instrument_and_capture(
    model: onnx.ModelProto,
    capture_names: Sequence[str],
    metadata_values: Dict[str, onnx.ValueInfoProto],
    input_array: np.ndarray,
    output_path: Path,
) -> Tuple[Dict[str, np.ndarray], ort.InferenceSession]:
    instrumented = copy.deepcopy(model)
    existing = {item.name for item in instrumented.graph.output}
    for name in capture_names:
        if name not in metadata_values:
            raise CaptureError(
                "captured runtime tensor {!r} lacks inferred ValueInfo".format(
                    name
                )
            )
        if name not in existing:
            instrumented.graph.output.append(
                copy.deepcopy(metadata_values[name])
            )
            existing.add(name)
    onnx.checker.check_model(instrumented)
    onnx.save(instrumented, str(output_path))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    arrays = session.run(list(capture_names), {"images": input_array})
    captures = dict(zip(capture_names, arrays))
    for name in capture_names:
        expected = metadata_values[name]
        expected_dtype = TensorProto.DataType.Name(
            expected.type.tensor_type.elem_type
        )
        expected_shape = [
            int(dimension.dim_value)
            for dimension in expected.type.tensor_type.shape.dim
        ]
        actual_dtype = (
            "FLOAT" if captures[name].dtype == np.float32
            else "INT64" if captures[name].dtype == np.int64
            else str(captures[name].dtype)
        )
        if actual_dtype != expected_dtype:
            raise CaptureError(
                "ORT dtype mismatch for {!r}: inferred={} captured={}".format(
                    name, expected_dtype, actual_dtype
                )
            )
        if list(captures[name].shape) != expected_shape:
            raise CaptureError(
                "ORT shape mismatch for {!r}: inferred={} captured={}".format(
                    name, expected_shape, list(captures[name].shape)
                )
            )
    return captures, session


def tensor_record(
    metadata: Dict[str, Dict[str, Any]],
    name: str,
    storage: str,
    offset: int,
) -> Dict[str, Any]:
    record = dict(metadata[name])
    record.update({"storage": storage, "offset": offset})
    return record


def c_tensor_record(tensor: Dict[str, Any]) -> str:
    return (
        "    {{ {}u, {}u, {}u, {}u, {}u, {}u, {} }},".format(
            STORAGE_CODES[tensor["storage"]],
            tensor["offset"],
            tensor["nbytes"],
            tensor["elements"],
            len(tensor["shape"]),
            DTYPE_CODES[tensor["dtype"]],
            c_array(tensor["shape"], 6),
        )
    )


def write_header(
    path: Path,
    tensor_records: Sequence[Dict[str, Any]],
    selected_nodes: Sequence[Tuple[int, onnx.NodeProto]],
    memory_map: Dict[str, int],
) -> None:
    first = selected_nodes[0][0]
    last = selected_nodes[-1][0]
    tensor_ids = {
        tensor["name"]: index for index, tensor in enumerate(tensor_records)
    }
    lines = [
        (
            "/* Generated directly from the pinned ONNX by "
            "tools/capture_range.py. */"
        ),
        "#ifndef YOLOV10N_HF_SLICE_MANIFEST_H",
        "#define YOLOV10N_HF_SLICE_MANIFEST_H",
        "",
        "#include <stdint.h>",
        "",
        "#define YR_MANIFEST_VERSION 2u",
        "#define YR_FIRST_NODE {}u".format(first),
        "#define YR_LAST_NODE {}u".format(last),
        "#define YR_NODE_COUNT {}u".format(len(selected_nodes)),
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
        "#define YR_PMC_STAGE_COUNT 1u",
        "#define YR_PMC_STAGE_STRIDE 0x{:08x}u".format(PMC_STAGE_STRIDE),
        "#define YR_INPUT_BLOB_BYTES {}u".format(
            memory_map["input_blob_bytes"]
        ),
        "#define YR_WEIGHT_BLOB_BYTES {}u".format(
            memory_map["weight_blob_bytes"]
        ),
        "#define YR_WORKSPACE_BYTES {}u".format(
            memory_map["workspace_bytes"]
        ),
        "#define YR_DUMP_SIZE 0x{:08x}u".format(memory_map["dump_size"]),
        "#define YR_MEM_SIZE 0x{:08x}u".format(memory_map["mem_size"]),
        "",
        (
            "enum yr_storage { YR_STORAGE_INPUT = 1, "
            "YR_STORAGE_WEIGHTS = 2, YR_STORAGE_WORKSPACE = 3 };"
        ),
        "enum yr_dtype { YR_DTYPE_FLOAT = 1, YR_DTYPE_INT64 = 2 };",
        (
            "enum yr_op { YR_OP_CONV = 1, YR_OP_SIGMOID = 2, "
            "YR_OP_MUL = 3, YR_OP_CONCAT = 4, YR_OP_ADD = 5, "
            "YR_OP_SPLIT = 6, YR_OP_MAXPOOL = 7, YR_OP_RESIZE = 8, "
            "YR_OP_MATMUL = 9, YR_OP_SOFTMAX = 10, "
            "YR_OP_RESHAPE = 11, YR_OP_TRANSPOSE = 12, "
            "YR_OP_SUB = 13, YR_OP_REDUCEMAX = 14, "
            "YR_OP_TOPK = 15, YR_OP_UNSQUEEZE = 16, "
            "YR_OP_TILE = 17, YR_OP_GATHERELEMENTS = 18, "
            "YR_OP_FLATTEN = 19, YR_OP_MOD = 20, "
            "YR_OP_DIV = 21, YR_OP_CAST = 22 };"
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
    lines.extend(c_tensor_record(tensor) for tensor in tensor_records)
    lines.extend(
        [
            "};",
            "",
            "static const struct yr_node_desc yr_nodes[YR_NODE_COUNT] = {",
        ]
    )
    lines.extend(
        node_c_record(global_index, node, tensor_ids)
        for global_index, node in selected_nodes
    )
    lines.extend(
        [
            "};",
            "",
            (
                "static const struct yr_pmc_stage_desc "
                "yr_pmc_stages[YR_PMC_STAGE_COUNT] = {"
            ),
            "    {{ 0u, {}u, {}u, {}u }},".format(
                len(selected_nodes) - 1, first, last
            ),
            "};",
            "",
            "#endif",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def blob_record(path: Path) -> Dict[str, Any]:
    return {
        "path": path.name,
        "nbytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = args.model.resolve()
    output_dir = args.output_root.resolve() / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_sha = file_sha256(model_path)
    if actual_sha != EXPECTED_SHA256:
        raise CaptureError(
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
    metadata, values, initializers = tensor_metadata(inferred)
    validate_graph(inferred, metadata)

    first, last = parse_range(args.range, len(inferred.graph.node))
    selected_nodes = [
        (index, inferred.graph.node[index])
        for index in range(first, last + 1)
    ]
    selected_outputs = ordered_unique(
        output
        for _, node in selected_nodes
        for output in node.output
        if output
    )
    selected_output_set = set(selected_outputs)
    boundary_inputs = ordered_unique(
        name
        for _, node in selected_nodes
        for name in node.input
        if name
        and name not in initializers
        and name not in selected_output_set
    )
    required_initializers = ordered_unique(
        name
        for _, node in selected_nodes
        for name in node.input
        if name and name in initializers
    )
    if not boundary_inputs:
        raise CaptureError("selected range has no external runtime input")
    for name in boundary_inputs + required_initializers + selected_outputs:
        if name not in metadata:
            raise CaptureError(
                "selected tensor {!r} has no static metadata".format(name)
            )
        if metadata[name]["dtype"] not in DTYPE_CODES:
            raise CaptureError(
                "selected tensor {!r} has unsupported dtype {}".format(
                    name, metadata[name]["dtype"]
                )
            )
        if len(metadata[name]["shape"]) > 6:
            raise CaptureError(
                "selected tensor {!r} exceeds rank six".format(name)
            )

    full_input, full_input_description = load_full_input(
        args.input_bin, metadata["images"]["shape"]
    )
    capture_names = boundary_inputs + selected_outputs
    instrumented_path = output_dir / "instrumented_range.onnx"
    captures, session = instrument_and_capture(
        inferred, capture_names, values, full_input, instrumented_path
    )

    input_path = output_dir / "inputs.bin"
    input_size, input_layout = write_aligned_blob(
        input_path,
        (
            (name, canonical_bytes(captures[name]))
            for name in boundary_inputs
        ),
    )
    weights_path = output_dir / "weights.bin"
    weight_arrays = {
        name: numpy_helper.to_array(initializers[name])
        for name in required_initializers
    }
    weight_size, weight_layout = write_aligned_blob(
        weights_path,
        (
            (name, canonical_bytes(weight_arrays[name]))
            for name in required_initializers
        ),
    )
    goldens_path = output_dir / "goldens.bin"
    golden_size, golden_layout = write_aligned_blob(
        goldens_path,
        (
            (name, canonical_bytes(captures[name]))
            for name in selected_outputs
        ),
    )

    allocations: Dict[str, Dict[str, int]] = {}
    workspace_cursor = 0
    for name in selected_outputs:
        offset = align(workspace_cursor)
        allocations[name] = {
            "offset": offset,
            "allocated_nbytes": align(metadata[name]["nbytes"]),
        }
        workspace_cursor = offset + align(metadata[name]["nbytes"])
    workspace_bytes = align(workspace_cursor)

    pmc_device_offset = align(
        RESULT_HEADER_BYTES + workspace_bytes, PAGE_ALIGNMENT
    )
    dump_size = pmc_device_offset + PMC_STAGE_STRIDE
    input_device_offset = align(dump_size, PAGE_ALIGNMENT)
    weight_device_offset = align(
        input_device_offset + input_size, PAGE_ALIGNMENT
    )
    mem_size = max(
        MIN_MEMORY_BYTES,
        align(weight_device_offset + weight_size, PAGE_ALIGNMENT),
    )
    memory_map = {
        "result_device_offset": RESULT_DEVICE_OFFSET,
        "input_device_offset": input_device_offset,
        "weight_device_offset": weight_device_offset,
        "pmc_device_offset": pmc_device_offset,
        "pmc_stage_stride": PMC_STAGE_STRIDE,
        "pmc_stage_count": 1,
        "workspace_bytes": workspace_bytes,
        "input_blob_bytes": input_size,
        "weight_blob_bytes": weight_size,
        "dump_size": dump_size,
        "mem_size": mem_size,
    }

    producer = {
        output: index
        for index, node in enumerate(inferred.graph.node)
        for output in node.output
        if output
    }
    tensor_records: List[Dict[str, Any]] = []
    boundary_records = []
    for name in boundary_inputs:
        record = tensor_record(
            metadata, name, "input", input_layout[name]["offset"]
        )
        record.update(
            {
                "role": "boundary_input",
                "source": (
                    "graph_input"
                    if name == "images"
                    else "N{:03d}".format(producer[name])
                ),
                "segment_sha256": input_layout[name]["sha256"],
            }
        )
        tensor_records.append(record)
        boundary_records.append(
            {
                "tensor": name,
                "dtype": record["dtype"],
                "shape": record["shape"],
                "nbytes": record["nbytes"],
                "source": record["source"],
                "blob_offset": record["offset"],
                "sha256": record["segment_sha256"],
            }
        )
    initializer_records = []
    for name in required_initializers:
        record = tensor_record(
            metadata, name, "weights", weight_layout[name]["offset"]
        )
        raw = canonical_bytes(weight_arrays[name])
        record.update(
            {
                "role": "initializer",
                "segment_sha256": weight_layout[name]["sha256"],
            }
        )
        tensor_records.append(record)
        initializer_records.append(
            {
                "tensor": name,
                "dtype": record["dtype"],
                "shape": record["shape"],
                "nbytes": record["nbytes"],
                "blob_offset": record["offset"],
                "sha256": bytes_sha256(raw),
            }
        )
    selected_output_records = []
    output_index_by_name = {
        name: (global_index, output_index)
        for global_index, node in selected_nodes
        for output_index, name in enumerate(node.output)
    }
    for name in selected_outputs:
        global_index, output_index = output_index_by_name[name]
        record = tensor_record(
            metadata, name, "workspace", allocations[name]["offset"]
        )
        record.update(
            {
                "role": "node_output_checkpoint",
                "producer": "N{:03d}".format(global_index),
                "output_index": output_index,
                "allocated_nbytes": allocations[name]["allocated_nbytes"],
                "checkpoint": True,
            }
        )
        tensor_records.append(record)
        selected_output_records.append(
            {
                "output_id": "N{:03d}:O{}".format(
                    global_index, output_index
                ),
                "node_id": "N{:03d}".format(global_index),
                "output_index": output_index,
                "tensor": name,
                "dtype": record["dtype"],
                "shape": record["shape"],
                "elements": record["elements"],
                "nbytes": record["nbytes"],
                "workspace_offset": record["offset"],
                "golden_offset": golden_layout[name]["offset"],
                "golden_sha256": golden_layout[name]["sha256"],
            }
        )

    header_path = output_dir / "slice_manifest.h"
    write_header(header_path, tensor_records, selected_nodes, memory_map)

    node_records = [
        {
            "node_id": "N{:03d}".format(global_index),
            "local_index": local_index,
            "index": global_index,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": node_attributes(node),
        }
        for local_index, (global_index, node) in enumerate(selected_nodes)
    ]
    manifest = {
        "schema_version": 2,
        "manifest_kind": "contiguous_node_range",
        "source": {
            "repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
            "filename": EXPECTED_FILENAME,
            "sha256": actual_sha,
            "license": EXPECTED_LICENSE,
            "instrumentation": (
                "shape inference plus boundary and selected-output graph "
                "outputs only"
            ),
            "instrumented_path": instrumented_path.name,
            "instrumented_sha256": file_sha256(instrumented_path),
        },
        "reference": {
            "runtime": "onnxruntime",
            "runtime_version": ort.__version__,
            "providers": session.get_providers(),
            "graph_optimization": "ORT_DISABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "full_model_input": full_input_description,
        },
        "selection": {
            "selector": "N{:03d}:N{:03d}".format(first, last),
            "first_node": "N{:03d}".format(first),
            "last_node": "N{:03d}".format(last),
            "inclusive": True,
            "node_count": len(selected_nodes),
            "operator_types": ordered_unique(
                node.op_type for _, node in selected_nodes
            ),
        },
        "boundary_inputs": boundary_records,
        "initializers": initializer_records,
        "nodes": node_records,
        "tensors": tensor_records,
        "outputs": selected_output_records,
        "pmc_stages": [
            {
                "name": "selected_range",
                "first_local_node": 0,
                "last_local_node": len(selected_nodes) - 1,
                "first_node": "N{:03d}".format(first),
                "last_node": "N{:03d}".format(last),
                "pmc_device_offset": pmc_device_offset,
                "scope": "only selected ONNX nodes",
            }
        ],
        "memory_plan": {
            "algorithm": (
                "monotonic 64-byte-aligned allocation; every selected "
                "node output remains materialized through final comparison"
            ),
            "alignment_bytes": ALIGNMENT,
            "workspace_bytes": workspace_bytes,
            "checkpoint_count": len(selected_output_records),
            "no_output_aliasing": True,
        },
        "generated": {
            "header": blob_record(header_path),
        },
        "blobs": {
            "inputs": {
                **blob_record(input_path),
                "segments": boundary_records,
            },
            "weights": {
                **blob_record(weights_path),
                "segments": initializer_records,
            },
            "goldens": {
                **blob_record(goldens_path),
                "segments": selected_output_records,
            },
        },
        "memory_map": memory_map,
        "result": {
            "magic": "YRF1",
            "version": 1,
            "math_version": 1,
            "header_bytes": RESULT_HEADER_BYTES,
            "workspace_offset_within_dump": RESULT_HEADER_BYTES,
            "pmc_region_bytes": PMC_REGION_BYTES,
        },
        "tolerances": {
            "atol": args.atol,
            "rtol": args.rtol,
            "float_comparison": (
                "abs(actual-reference) <= atol + rtol*abs(reference)"
            ),
            "int64_comparison": "exact",
        },
    }
    manifest_path = output_dir / "slice_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        "RANGE_CAPTURE PASS selector={} nodes={} outputs={} "
        "boundary={} initializers={}".format(
            manifest["selection"]["selector"],
            len(selected_nodes),
            len(selected_output_records),
            len(boundary_records),
            len(initializer_records),
        )
    )
    for record in boundary_records:
        print(
            "BOUNDARY tensor={} dtype={} shape={} source={} sha256={}".format(
                record["tensor"],
                record["dtype"],
                record["shape"],
                record["source"],
                record["sha256"],
            )
        )
    print(
        "RANGE_MEMORY PASS workspace={} dump=0x{:x} "
        "input=0x{:x} weights=0x{:x} total=0x{:x}".format(
            workspace_bytes,
            dump_size,
            input_device_offset,
            weight_device_offset,
            mem_size,
        )
    )
    print(
        "RANGE_BLOBS PASS inputs={} weights={} goldens={} out={}".format(
            input_size, weight_size, golden_size, output_dir
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--range",
        required=True,
        help="inclusive ONNX node selector, for example N289:N307",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="artifact directory name below --output-root",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--input-bin",
        type=Path,
        help="optional full-model FP32 [1,3,640,640] input",
    )
    parser.add_argument("--atol", type=float, default=0.00005)
    parser.add_argument("--rtol", type=float, default=0.0001)
    return parser.parse_args()


def main() -> int:
    try:
        generate(parse_args())
    except (
        CaptureError,
        OSError,
        ValueError,
        KeyError,
        onnx.checker.ValidationError,
    ) as exc:
        print("RANGE_CAPTURE FAIL {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
