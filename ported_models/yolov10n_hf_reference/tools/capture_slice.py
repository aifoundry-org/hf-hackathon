#!/usr/bin/env python3
"""Capture an ONNX node range and package it for the scalar C slice runner.

The original pinned ONNX remains the source of graph structure, attributes,
weights, tensor names, and reference values.  This tool only adds requested
intermediate tensors as outputs in a cached instrumented copy, runs that copy
with ONNX Runtime, and writes small aligned blobs plus a declarative manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference
import onnxruntime as ort


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_ROOT = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/slices"
EXPECTED_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
SUPPORTED_OPS = ("Conv", "Sigmoid", "Mul", "Concat")
ALIGNMENT = 64
RESULT_HEADER_BYTES = 4096
RESULT_DEVICE_OFFSET = 0x00000000
MIN_MEMORY_BYTES = 0x00400000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_node(value: str, count: int) -> int:
    match = re.fullmatch(r"[Nn]?([0-9]+)", value.strip())
    if not match:
        raise ValueError(f"invalid node id {value!r}; expected N000 or 0")
    result = int(match.group(1))
    if result < 0 or result >= count:
        raise ValueError(f"node index {result} is outside 0..{count - 1}")
    return result


def parse_range(value: str, count: int) -> Tuple[int, int]:
    parts = value.split(":")
    if len(parts) == 1:
        first = last = parse_node(parts[0], count)
    elif len(parts) == 2:
        first, last = (parse_node(item, count) for item in parts)
    else:
        raise ValueError(f"invalid node range {value!r}")
    if first > last:
        raise ValueError(f"node range must be ascending, got N{first:03d}:N{last:03d}")
    return first, last


def value_shape(value: onnx.ValueInfoProto) -> List[int]:
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ValueError(f"tensor {value.name!r} has non-static shape")
        result.append(int(dim.dim_value))
    return result


def value_dtype(value: onnx.ValueInfoProto) -> str:
    return TensorProto.DataType.Name(value.type.tensor_type.elem_type)


def tensor_shape(
    name: str,
    values: Dict[str, onnx.ValueInfoProto],
    initializers: Dict[str, onnx.TensorProto],
) -> List[int]:
    if name in initializers:
        return [int(item) for item in initializers[name].dims]
    if name not in values:
        raise ValueError(f"no inferred shape for tensor {name!r}")
    return value_shape(values[name])


def tensor_dtype(
    name: str,
    values: Dict[str, onnx.ValueInfoProto],
    initializers: Dict[str, onnx.TensorProto],
) -> str:
    if name in initializers:
        return TensorProto.DataType.Name(initializers[name].data_type)
    if name not in values:
        raise ValueError(f"no inferred dtype for tensor {name!r}")
    return value_dtype(values[name])


def capability_error(
    node: onnx.NodeProto,
    values: Dict[str, onnx.ValueInfoProto],
    initializers: Dict[str, onnx.TensorProto],
) -> Optional[str]:
    """Return why this exact node is outside the scalar runtime contract."""
    attrs = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    names = [name for name in list(node.input) + list(node.output) if name]
    try:
        shapes = {
            name: tensor_shape(name, values, initializers) for name in names
        }
        dtypes = {
            name: tensor_dtype(name, values, initializers) for name in names
        }
    except ValueError as exc:
        return str(exc)
    if any(dtype != "FLOAT" for dtype in dtypes.values()):
        return "all runtime tensors must be FLOAT"
    if any(len(shape) > 6 for shape in shapes.values()):
        return "tensor rank exceeds manifest limit 6"
    if any(int(np.prod(shape, dtype=np.int64)) > 0xFFFFFFFF for shape in shapes.values()):
        return "tensor element count exceeds uint32"
    if len(node.output) != 1 or not node.output[0]:
        return "runtime requires exactly one non-empty output"

    if node.op_type == "Conv":
        if len(node.input) not in (2, 3) or any(not name for name in node.input):
            return "Conv requires data, initializer weight, and optional initializer bias"
        if node.input[1] not in initializers or (
            len(node.input) == 3 and node.input[2] not in initializers
        ):
            return "Conv weight and bias must be initializers"
        input_shape = shapes[node.input[0]]
        weight_shape = shapes[node.input[1]]
        output_shape = shapes[node.output[0]]
        if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
            return "Conv supports only rank-4 NCHW tensors"
        kernel = list(attrs.get("kernel_shape", []))
        stride = list(attrs.get("strides", [1, 1]))
        dilation = list(attrs.get("dilations", [1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        auto_pad = attrs.get("auto_pad", b"NOTSET")
        if auto_pad not in (b"NOTSET", "NOTSET"):
            return "Conv auto_pad other than NOTSET is unsupported"
        if (
            len(kernel) != 2
            or len(stride) != 2
            or len(dilation) != 2
            or len(pads) != 4
            or any(int(item) <= 0 for item in kernel + stride + dilation)
            or any(int(item) < 0 for item in pads)
        ):
            return "Conv requires explicit positive 2-D kernel/stride/dilation and four pads"
        group = int(attrs.get("group", 1))
        input_channels = input_shape[1]
        output_channels = output_shape[1]
        if (
            group <= 0
            or input_channels % group
            or output_channels % group
            or weight_shape[0] != output_channels
            or weight_shape[1] != input_channels // group
            or weight_shape[2:] != [int(item) for item in kernel]
        ):
            return "Conv channel/group/kernel shapes are outside the runtime contract"
        if len(node.input) == 3 and shapes[node.input[2]] != [output_channels]:
            return "Conv bias must have shape [output_channels]"
        return None

    if node.op_type == "Sigmoid":
        if len(node.input) != 1 or not node.input[0]:
            return "Sigmoid requires exactly one input"
        if shapes[node.input[0]] != shapes[node.output[0]]:
            return "Sigmoid input/output shapes must match"
        return None

    if node.op_type == "Mul":
        if len(node.input) != 2 or any(not name for name in node.input):
            return "Mul requires exactly two inputs"
        left = shapes[node.input[0]]
        right = shapes[node.input[1]]
        output = shapes[node.output[0]]
        left_scalar = int(np.prod(left, dtype=np.int64)) == 1
        right_scalar = int(np.prod(right, dtype=np.int64)) == 1
        if not (
            (left == output and right == output)
            or (left == output and right_scalar and len(right) <= len(output))
            or (right == output and left_scalar and len(left) <= len(output))
        ):
            return "Mul supports equal-shape tensors or a single scalar only"
        return None

    if node.op_type == "Concat":
        if not (1 <= len(node.input) <= 4) or any(not name for name in node.input):
            return "Concat supports one to four non-empty inputs"
        output = shapes[node.output[0]]
        if not output:
            return "Concat scalar output is unsupported"
        if "axis" not in attrs:
            return "Concat requires an explicit axis"
        axis = int(attrs["axis"])
        if axis < 0:
            axis += len(output)
        if axis < 0 or axis >= len(output):
            return "Concat axis is outside the output rank"
        axis_sum = 0
        for name in node.input:
            shape = shapes[name]
            if len(shape) != len(output):
                return "Concat input/output ranks must match"
            if any(
                shape[index] != output[index]
                for index in range(len(output))
                if index != axis
            ):
                return "Concat non-axis dimensions must match"
            axis_sum += shape[axis]
        if axis_sum != output[axis]:
            return "Concat axis dimensions do not sum to the output"
        return None

    return f"operator {node.op_type} is not implemented"


def canonical_array_bytes(array: np.ndarray) -> bytes:
    if array.dtype == np.float32:
        canonical = np.asarray(array, dtype="<f4", order="C")
    elif array.dtype == np.int64:
        canonical = np.asarray(array, dtype="<i8", order="C")
    elif array.dtype == np.int32:
        canonical = np.asarray(array, dtype="<i4", order="C")
    else:
        raise ValueError(f"unsupported packaged dtype {array.dtype}")
    return canonical.tobytes(order="C")


def deterministic_input(shape: Sequence[int]) -> np.ndarray:
    """Generate version-independent FP32 values in [0, 1]."""
    elements = int(np.prod(shape, dtype=np.int64))
    index = np.arange(elements, dtype=np.uint64)
    bits = (index * np.uint64(1664525) + np.uint64(1013904223)) & np.uint64(
        0x00FFFFFF
    )
    values = bits.astype(np.float32) * np.float32(1.0 / 16777215.0)
    return values.reshape(tuple(shape))


def write_aligned_blob(
    path: Path, entries: Iterable[Tuple[str, bytes]]
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    cursor = 0
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("wb") as dst:
        for name, data in entries:
            offset = align(cursor)
            if offset > cursor:
                dst.write(b"\x00" * (offset - cursor))
            dst.write(data)
            records[name] = {
                "offset": offset,
                "nbytes": len(data),
                "sha256": bytes_sha256(data),
            }
            cursor = offset + len(data)
        final_size = align(cursor)
        if final_size > cursor:
            dst.write(b"\x00" * (final_size - cursor))
    return final_size, records


def c_array(values: Sequence[int], width: int) -> str:
    padded = list(values[:width]) + [0] * max(0, width - len(values))
    return "{ " + ", ".join(str(int(item)) for item in padded) + " }"


def write_header(
    path: Path,
    first: int,
    last: int,
    tensors: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    input_blob_bytes: int,
    weights_blob_bytes: int,
    workspace_bytes: int,
) -> Dict[str, int]:
    pmc_device_offset = align(RESULT_HEADER_BYTES + workspace_bytes, 0x10000)
    # The PMC record is deliberately given a full 64 KiB page. Its actual
    # structure is much smaller, and the decoder checks the recorded version.
    dump_size = pmc_device_offset + 0x10000
    input_device_offset = align(dump_size, 0x10000)
    weight_device_offset = align(
        input_device_offset + input_blob_bytes, 0x10000
    )
    mem_size = max(
        MIN_MEMORY_BYTES,
        align(input_device_offset + input_blob_bytes, 0x10000),
        align(weight_device_offset + weights_blob_bytes, 0x10000),
        align(dump_size, 0x10000),
    )
    storage_values = {"input": 1, "weights": 2, "workspace": 3}
    op_values = {"Conv": 1, "Sigmoid": 2, "Mul": 3, "Concat": 4}

    lines = [
        "/* Generated from the pinned ONNX by tools/capture_slice.py. */",
        "#ifndef YOLOV10N_HF_SLICE_MANIFEST_H",
        "#define YOLOV10N_HF_SLICE_MANIFEST_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define YR_FIRST_NODE {first}u",
        f"#define YR_LAST_NODE {last}u",
        f"#define YR_NODE_COUNT {len(nodes)}u",
        f"#define YR_TENSOR_COUNT {len(tensors)}u",
        f"#define YR_RESULT_HEADER_BYTES {RESULT_HEADER_BYTES}u",
        f"#define YR_RESULT_DEVICE_OFFSET 0x{RESULT_DEVICE_OFFSET:08x}u",
        f"#define YR_INPUT_DEVICE_OFFSET 0x{input_device_offset:08x}u",
        f"#define YR_WEIGHT_DEVICE_OFFSET 0x{weight_device_offset:08x}u",
        f"#define YR_PMC_DEVICE_OFFSET 0x{pmc_device_offset:08x}u",
        f"#define YR_INPUT_BLOB_BYTES {input_blob_bytes}u",
        f"#define YR_WEIGHT_BLOB_BYTES {weights_blob_bytes}u",
        f"#define YR_WORKSPACE_BYTES {workspace_bytes}u",
        f"#define YR_DUMP_SIZE 0x{dump_size:08x}u",
        f"#define YR_MEM_SIZE 0x{mem_size:08x}u",
        "",
        "enum yr_storage { YR_STORAGE_INPUT = 1, YR_STORAGE_WEIGHTS = 2, YR_STORAGE_WORKSPACE = 3 };",
        "enum yr_op { YR_OP_CONV = 1, YR_OP_SIGMOID = 2, YR_OP_MUL = 3, YR_OP_CONCAT = 4 };",
        "",
        "struct yr_tensor_desc {",
        "    uint32_t storage, offset, nbytes, elements, rank;",
        "    uint32_t dims[6];",
        "};",
        "",
        "struct yr_node_desc {",
        "    uint32_t onnx_index, op, input_count, output_count;",
        "    uint32_t inputs[4], outputs[3];",
        "    int32_t group, kernel_h, kernel_w, stride_h, stride_w;",
        "    int32_t pad_top, pad_left, pad_bottom, pad_right;",
        "    int32_t dilation_h, dilation_w, axis;",
        "};",
        "",
        "static const struct yr_tensor_desc yr_tensors[YR_TENSOR_COUNT] = {",
    ]
    for tensor in tensors:
        lines.append(
            "    { "
            f"{storage_values[tensor['storage']]}u, {tensor['offset']}u, "
            f"{tensor['nbytes']}u, {tensor['elements']}u, {len(tensor['shape'])}u, "
            f"{c_array(tensor['shape'], 6)}"
            " },"
        )
    lines.extend(["};", "", "static const struct yr_node_desc yr_nodes[YR_NODE_COUNT] = {"])
    tensor_id = {tensor["name"]: index for index, tensor in enumerate(tensors)}
    for node in nodes:
        attrs = node["attributes"]
        kernel = attrs.get("kernel_shape", [0, 0])
        stride = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        dilation = attrs.get("dilations", [1, 1])
        inputs = [tensor_id[name] for name in node["inputs"]]
        outputs = [tensor_id[name] for name in node["outputs"]]
        lines.append(
            "    { "
            f"{node['index']}u, {op_values[node['op_type']]}u, "
            f"{len(inputs)}u, {len(outputs)}u, "
            f"{c_array(inputs, 4)}, {c_array(outputs, 3)}, "
            f"{int(attrs.get('group', 1))}, {int(kernel[0])}, {int(kernel[1])}, "
            f"{int(stride[0])}, {int(stride[1])}, "
            f"{int(pads[0])}, {int(pads[1])}, {int(pads[2])}, {int(pads[3])}, "
            f"{int(dilation[0])}, {int(dilation[1])}, {int(attrs.get('axis', 0))}"
            " },"
        )
    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "result_device_offset": RESULT_DEVICE_OFFSET,
        "input_device_offset": input_device_offset,
        "weight_device_offset": weight_device_offset,
        "pmc_device_offset": pmc_device_offset,
        "dump_size": dump_size,
        "mem_size": mem_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--nodes",
        default="N263:N265",
        help="one node or an inclusive contiguous range, for example N263:N265",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--name", help="output directory name")
    parser.add_argument(
        "--list-support", action="store_true", help="print implemented C operators"
    )
    args = parser.parse_args()
    if args.list_support:
        print("SUPPORTED_OPS " + " ".join(SUPPORTED_OPS))
        return 0

    model_path = args.model.resolve()
    actual_sha = file_sha256(model_path)
    if actual_sha != EXPECTED_SHA256:
        print(
            f"SOURCE_CHECK FAIL expected={EXPECTED_SHA256} actual={actual_sha}",
            file=sys.stderr,
        )
        return 1
    print(f"SOURCE_CHECK PASS sha256={actual_sha}")

    original = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(original)
    inferred = shape_inference.infer_shapes(original, strict_mode=True, data_prop=True)
    first, last = parse_range(args.nodes, len(inferred.graph.node))
    selected = list(inferred.graph.node[first : last + 1])
    unsupported = [
        (first + offset, node.op_type)
        for offset, node in enumerate(selected)
        if node.domain not in ("", "ai.onnx") or node.op_type not in SUPPORTED_OPS
    ]
    if unsupported:
        for index, op_type in unsupported:
            print(f"UNSUPPORTED node=N{index:03d} op={op_type}", file=sys.stderr)
        return 2

    initializers = {item.name: item for item in inferred.graph.initializer}
    producers = {
        output: index
        for index, node in enumerate(inferred.graph.node)
        for output in node.output
    }
    values = {
        item.name: item
        for item in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    capability_failures = []
    for offset, node in enumerate(selected):
        reason = capability_error(node, values, initializers)
        if reason is not None:
            capability_failures.append((first + offset, node.op_type, reason))
    if capability_failures:
        for index, op_type, reason in capability_failures:
            print(
                f"UNSUPPORTED node=N{index:03d} op={op_type} reason={reason}",
                file=sys.stderr,
            )
        return 2
    graph_inputs = {item.name: item for item in inferred.graph.input}
    internal_names = {
        output for node in selected for output in node.output if output
    }

    external_names: List[str] = []
    weight_names: List[str] = []
    for node in selected:
        for name in node.input:
            if not name:
                continue
            if name in initializers:
                if name not in weight_names:
                    weight_names.append(name)
            elif name not in internal_names and name not in external_names:
                external_names.append(name)

    capture_names = [
        name
        for name in external_names
        if name not in graph_inputs
    ] + [output for node in selected for output in node.output]
    instrumented = copy.deepcopy(inferred)
    existing_outputs = {item.name for item in instrumented.graph.output}
    for name in capture_names:
        if name not in values:
            raise SystemExit(f"error: no inferred ValueInfo for tensor {name!r}")
        if name not in existing_outputs:
            instrumented.graph.output.append(copy.deepcopy(values[name]))
            existing_outputs.add(name)

    slice_name = args.name or f"n{first:03d}_n{last:03d}"
    out_dir = args.output_root.resolve() / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumented_path = out_dir / "instrumented.onnx"
    onnx.save(instrumented, str(instrumented_path))

    feed: Dict[str, np.ndarray] = {}
    for item in inferred.graph.input:
        if value_dtype(item) != "FLOAT":
            raise SystemExit(
                f"error: deterministic input only supports FLOAT, got {value_dtype(item)}"
            )
        feed[item.name] = deterministic_input(value_shape(item))
    full_input = feed["images"]
    (out_dir / "model_input.bin").write_bytes(canonical_array_bytes(full_input))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(instrumented_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    captured_values = session.run(capture_names, feed)
    captures = dict(zip(capture_names, captured_values))
    captures.update({name: feed[name] for name in external_names if name in feed})

    input_entries = [
        (name, canonical_array_bytes(captures[name])) for name in external_names
    ]
    weight_arrays = {
        name: numpy_helper.to_array(initializers[name]) for name in weight_names
    }
    weight_entries = [
        (name, canonical_array_bytes(weight_arrays[name])) for name in weight_names
    ]
    output_names = [output for node in selected for output in node.output]
    golden_entries = [
        (name, canonical_array_bytes(captures[name])) for name in output_names
    ]

    input_size, input_layout = write_aligned_blob(out_dir / "inputs.bin", input_entries)
    weights_size, weights_layout = write_aligned_blob(
        out_dir / "weights.bin", weight_entries
    )
    workspace_size, golden_layout = write_aligned_blob(
        out_dir / "goldens.bin", golden_entries
    )

    tensor_order = external_names + weight_names + output_names
    tensor_records = []
    for name in tensor_order:
        if name in external_names:
            array = captures[name]
            layout = input_layout[name]
            storage = "input"
        elif name in weight_names:
            array = weight_arrays[name]
            layout = weights_layout[name]
            storage = "weights"
        else:
            array = captures[name]
            layout = golden_layout[name]
            storage = "workspace"
        if array.dtype != np.float32:
            raise SystemExit(
                f"error: C slice currently supports FLOAT tensors; {name} is {array.dtype}"
            )
        tensor_records.append(
            {
                "name": name,
                "dtype": "FLOAT",
                "shape": [int(item) for item in array.shape],
                "elements": int(array.size),
                "storage": storage,
                **layout,
            }
        )

    node_records = []
    for index, node in enumerate(selected, start=first):
        attrs = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        attrs = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in attrs.items()
        }
        node_records.append(
            {
                "node_id": f"N{index:03d}",
                "index": index,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": attrs,
            }
        )

    header_path = out_dir / "slice_manifest.h"
    memory_map = write_header(
        header_path,
        first,
        last,
        tensor_records,
        node_records,
        input_size,
        weights_size,
        workspace_size,
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "repo": "onnx-community/yolov10n",
            "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
            "filename": "onnx/model.onnx",
            "sha256": actual_sha,
            "instrumentation": "shape inference plus additional graph outputs only",
            "instrumented_sha256": file_sha256(instrumented_path),
        },
        "reference": {
            "runtime": "onnxruntime",
            "runtime_version": ort.__version__,
            "providers": session.get_providers(),
            "graph_optimization": "ORT_DISABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "input_generator": "(index * 1664525 + 1013904223) & 0x00ffffff, divided by 16777215",
        },
        "selection": {
            "selector": args.nodes,
            "first_node": f"N{first:03d}",
            "last_node": f"N{last:03d}",
            "inclusive": True,
            "supported_ops": list(SUPPORTED_OPS),
        },
        "nodes": node_records,
        "tensors": tensor_records,
        "blobs": {
            "inputs": {
                "path": "inputs.bin",
                "nbytes": input_size,
                "sha256": file_sha256(out_dir / "inputs.bin"),
            },
            "weights": {
                "path": "weights.bin",
                "nbytes": weights_size,
                "sha256": file_sha256(out_dir / "weights.bin"),
            },
            "goldens": {
                "path": "goldens.bin",
                "nbytes": workspace_size,
                "sha256": file_sha256(out_dir / "goldens.bin"),
            },
            "model_input": {
                "path": "model_input.bin",
                "nbytes": (out_dir / "model_input.bin").stat().st_size,
                "sha256": file_sha256(out_dir / "model_input.bin"),
            },
        },
        "memory_map": memory_map,
        "result": {
            "magic": "YRF1",
            "header_bytes": RESULT_HEADER_BYTES,
            "workspace_offset_within_dump": RESULT_HEADER_BYTES,
            "pmc_scope": "immediately around selected inclusive node range only",
        },
        "tolerances": {
            "atol": 0.00005,
            "rtol": 0.0001,
            "comparison": "abs(actual-reference) <= atol + rtol*abs(reference)",
        },
    }
    manifest_path = out_dir / "slice_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"SLICE_CAPTURE PASS nodes=N{first:03d}:N{last:03d} "
        f"ops={','.join(node.op_type for node in selected)}"
    )
    print(
        f"BLOBS inputs={input_size} weights={weights_size} "
        f"goldens={workspace_size} out={out_dir}"
    )
    print(
        f"MEMORY mem_size=0x{memory_map['mem_size']:x} "
        f"dump_size=0x{memory_map['dump_size']:x}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
