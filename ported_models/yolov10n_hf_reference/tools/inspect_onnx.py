#!/usr/bin/env python3
"""Inspect the pinned graph and emit a stable, readable inventory.

No graph transformation or re-export occurs here. Shape inference augments an
in-memory copy solely so tensor metadata can be attached to the inventory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional

import onnx
from onnx import AttributeProto, TensorProto, helper, shape_inference


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_OUT = PORT_ROOT / "manifests"
EXPECTED_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
EXPECTED_OPERATOR_COUNTS = {
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dim_value(dim: onnx.TensorShapeProto.Dimension) -> Any:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return None


def value_info_record(value: onnx.ValueInfoProto) -> Dict[str, Any]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return {"name": value.name, "dtype": None, "shape": None}
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    shape = None
    if tensor_type.HasField("shape"):
        shape = [dim_value(dim) for dim in tensor_type.shape.dim]
    return {"name": value.name, "dtype": dtype, "shape": shape}


def tensor_record(tensor: onnx.TensorProto) -> Dict[str, Any]:
    dims = [int(dim) for dim in tensor.dims]
    elements = 1
    for dim in dims:
        elements *= dim
    return {
        "name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": dims,
        "elements": elements,
        "raw_bytes": len(tensor.raw_data),
        "external_data": bool(tensor.external_data),
    }


def summarize_tensor(tensor: onnx.TensorProto) -> Dict[str, Any]:
    record = tensor_record(tensor)
    record.pop("name", None)
    return record


def attribute_value(attribute: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if attribute.type == AttributeProto.TENSOR:
        return summarize_tensor(value)
    if attribute.type == AttributeProto.TENSORS:
        return [summarize_tensor(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
            for item in value
        ]
    return value


def normalize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return name or "unnamed"


def build_tensor_metadata(model: onnx.ModelProto) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for value in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        records[value.name] = value_info_record(value)
    for initializer in model.graph.initializer:
        records[initializer.name] = tensor_record(initializer)
    return records


def node_records(
    model: onnx.ModelProto, tensor_metadata: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    op_counts: Dict[str, int] = defaultdict(int)
    records = []
    for index, node in enumerate(model.graph.node):
        op_ordinal = op_counts[node.op_type]
        op_counts[node.op_type] += 1
        records.append(
            {
                "node_id": f"N{index:03d}",
                "layer_id": f"L{index:03d}",
                "op_id": f"{normalize_name(node.op_type)}_{op_ordinal:03d}",
                "index": index,
                "name": node.name,
                "domain": node.domain or "ai.onnx",
                "op_type": node.op_type,
                "inputs": [
                    {
                        "name": name,
                        "dtype": tensor_metadata.get(name, {}).get("dtype"),
                        "shape": tensor_metadata.get(name, {}).get("shape"),
                    }
                    for name in node.input
                ],
                "outputs": [
                    {
                        "name": name,
                        "dtype": tensor_metadata.get(name, {}).get("dtype"),
                        "shape": tensor_metadata.get(name, {}).get("shape"),
                    }
                    for name in node.output
                ],
                "attributes": {
                    attribute.name: attribute_value(attribute)
                    for attribute in node.attribute
                },
            }
        )
    return records


def graph_checks(
    sha256: str,
    model: onnx.ModelProto,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}
    terminal = nodes[-40:]
    terminal_types = [node["op_type"] for node in terminal]
    operator_counts = dict(
        sorted(Counter(node["op_type"] for node in nodes).items())
    )
    terminal_selection_nodes = [
        (node["node_id"], node["op_type"])
        for node in nodes
        if node["op_type"] in ("TopK", "GatherElements")
    ]
    return [
        {
            "name": "sha256",
            "expected": EXPECTED_SHA256,
            "actual": sha256,
            "pass": sha256 == EXPECTED_SHA256,
        },
        {
            "name": "opset_ai_onnx",
            "expected": 13,
            "actual": opsets.get("ai.onnx"),
            "pass": opsets.get("ai.onnx") == 13,
        },
        {
            "name": "sole_opset_import",
            "expected": {"ai.onnx": 13},
            "actual": opsets,
            "pass": opsets == {"ai.onnx": 13},
        },
        {
            "name": "input",
            "expected": [{"name": "images", "dtype": "FLOAT", "shape": [1, 3, 640, 640]}],
            "actual": inputs,
            "pass": inputs
            == [{"name": "images", "dtype": "FLOAT", "shape": [1, 3, 640, 640]}],
        },
        {
            "name": "output",
            "expected": [{"name": "output0", "dtype": "FLOAT", "shape": [1, 300, 6]}],
            "actual": outputs,
            "pass": outputs
            == [{"name": "output0", "dtype": "FLOAT", "shape": [1, 300, 6]}],
        },
        {
            "name": "node_count",
            "expected": 308,
            "actual": len(nodes),
            "pass": len(nodes) == 308,
        },
        {
            "name": "initializer_count",
            "expected": 187,
            "actual": len(model.graph.initializer),
            "pass": len(model.graph.initializer) == 187,
        },
        {
            "name": "operator_histogram",
            "expected": EXPECTED_OPERATOR_COUNTS,
            "actual": operator_counts,
            "pass": operator_counts == EXPECTED_OPERATOR_COUNTS,
        },
        {
            "name": "terminal_topk",
            "expected": "TopK in final 40 nodes",
            "actual": terminal_types,
            "pass": "TopK" in terminal_types,
        },
        {
            "name": "terminal_gather_elements",
            "expected": "GatherElements in final 40 nodes",
            "actual": terminal_types,
            "pass": "GatherElements" in terminal_types,
        },
        {
            "name": "terminal_selection_path",
            "expected": [
                ("N291", "TopK"),
                ("N294", "GatherElements"),
                ("N296", "GatherElements"),
                ("N298", "TopK"),
                ("N303", "GatherElements"),
            ],
            "actual": terminal_selection_nodes,
            "pass": terminal_selection_nodes
            == [
                ("N291", "TopK"),
                ("N294", "GatherElements"),
                ("N296", "GatherElements"),
                ("N298", "TopK"),
                ("N303", "GatherElements"),
            ],
        },
    ]


def write_tsv(path: Path, nodes: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "node_id",
                "layer_id",
                "op_id",
                "name",
                "op_type",
                "inputs",
                "outputs",
            ]
        )
        for node in nodes:
            writer.writerow(
                [
                    node["node_id"],
                    node["layer_id"],
                    node["op_id"],
                    node["name"],
                    node["op_type"],
                    " | ".join(
                        f"{item['name']}:{item['dtype']}:{item['shape']}"
                        for item in node["inputs"]
                    ),
                    " | ".join(
                        f"{item['name']}:{item['dtype']}:{item['shape']}"
                        for item in node["outputs"]
                    ),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="emit inventory without enforcing pinned graph facts",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    out_dir = args.out_dir.resolve()
    sha256 = file_sha256(model_path)
    original = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(original)
    inferred = shape_inference.infer_shapes(original, strict_mode=True, data_prop=True)
    metadata = build_tensor_metadata(inferred)
    nodes = node_records(inferred, metadata)
    inputs = [value_info_record(value) for value in inferred.graph.input]
    outputs = [value_info_record(value) for value in inferred.graph.output]
    initializers = sorted(
        (tensor_record(item) for item in inferred.graph.initializer),
        key=lambda item: item["name"],
    )
    checks = graph_checks(sha256, inferred, inputs, outputs, nodes)

    inventory = {
        "schema_version": 1,
        "source": {
            "repo": "onnx-community/yolov10n",
            "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
            "filename": "onnx/model.onnx",
            "sha256": sha256,
            "license": "AGPL-3.0",
        },
        "model": {
            "ir_version": int(inferred.ir_version),
            "producer_name": inferred.producer_name,
            "producer_version": inferred.producer_version,
            "domain": inferred.domain,
            "model_version": int(inferred.model_version),
            "opsets": {
                item.domain or "ai.onnx": int(item.version)
                for item in inferred.opset_import
            },
            "inputs": inputs,
            "outputs": outputs,
            "node_count": len(nodes),
            "initializer_count": len(initializers),
            "operator_counts": dict(
                sorted(Counter(node["op_type"] for node in nodes).items())
            ),
        },
        "checks": checks,
        "nodes": nodes,
        "initializers": initializers,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "graph_inventory.json"
    tsv_path = out_dir / "layers.tsv"
    with json_path.open("w", encoding="utf-8") as dst:
        json.dump(inventory, dst, indent=2, sort_keys=False)
        dst.write("\n")
    write_tsv(tsv_path, nodes)

    print(f"INVENTORY path={json_path}")
    print(f"LAYERS path={tsv_path}")
    print(
        "GRAPH "
        f"nodes={len(nodes)} initializers={len(initializers)} "
        f"opsets={inventory['model']['opsets']} "
        f"operators={inventory['model']['operator_counts']}"
    )
    failed = False
    for check in checks:
        status = "PASS" if check["pass"] else "FAIL"
        print(
            f"CHECK {status} name={check['name']} "
            f"expected={check['expected']} actual={check['actual']}"
        )
        failed |= not check["pass"]
    if failed and not args.no_check:
        print("GRAPH_CHECK FAIL", file=sys.stderr)
        return 1
    print("GRAPH_CHECK PASS" if not failed else "GRAPH_CHECK SKIPPED_FAILURES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
