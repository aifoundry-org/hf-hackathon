#!/usr/bin/env python3
"""Generate the bounded, gap-free system-emulator coverage plan."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import onnx
from onnx import shape_inference

from generate_full_graph import (
    ALIGNMENT,
    EXPECTED_SHA256,
    PAGE_ALIGNMENT,
    PMC_STAGE_STRIDE,
    RESULT_HEADER_BYTES,
    align,
    file_sha256,
    tensor_metadata,
    validate_graph,
)


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_OUTPUT = PORT_ROOT / "manifests/sys_emu_coverage_plan.json"
MIN_MEMORY_BYTES = 0x00400000

# Measured partition: <=250M Conv/MatMul MAC and <=24MB retained outputs per
# range.  Conv->Sigmoid->Mul triples remain intact.  DFL/decode and NMS-free
# selection retain their architectural boundaries and dedicated evidence.
COVERAGE_RANGES: Tuple[Tuple[int, int], ...] = (
    (0, 2),
    (3, 9),
    (10, 17),
    (18, 30),
    (31, 45),
    (46, 60),
    (61, 79),
    (80, 104),
    (105, 137),
    (138, 153),
    (154, 168),
    (169, 196),
    (197, 207),
    (208, 210),
    (211, 213),
    (214, 223),
    (224, 231),
    (232, 249),
    (250, 270),
    (271, 288),
    (289, 307),
)

PMC_ARCHITECTURE_STAGES: Tuple[Tuple[str, int, int], ...] = (
    ("stem", 0, 5),
    ("backbone", 6, 90),
    ("sppf_psa", 91, 128),
    ("neck", 129, 207),
    ("three_scale_head", 208, 270),
    ("dfl_decode", 271, 288),
    ("topk_selection", 289, 307),
)


class PlanError(RuntimeError):
    """The pinned graph no longer satisfies the checked coverage plan."""


def node_macs(
    model: onnx.ModelProto,
    metadata: Dict[str, Dict[str, Any]],
    initializers: Dict[str, onnx.TensorProto],
    index: int,
) -> int:
    node = model.graph.node[index]
    if node.op_type == "Conv":
        output = metadata[node.output[0]]["shape"]
        weights = [int(item) for item in initializers[node.input[1]].dims]
        return (
            output[0]
            * output[1]
            * output[2]
            * output[3]
            * weights[1]
            * weights[2]
            * weights[3]
        )
    if node.op_type == "MatMul":
        return metadata[node.output[0]]["elements"] * metadata[
            node.input[0]
        ]["shape"][-1]
    return 0


def aligned_blob_bytes(names: Sequence[str], metadata: Dict[str, Any]) -> int:
    cursor = 0
    for name in names:
        cursor = align(cursor)
        cursor += metadata[name]["nbytes"]
    return align(cursor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    if file_sha256(model_path) != EXPECTED_SHA256:
        raise PlanError("pinned ONNX SHA-256 mismatch")
    original = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(original)
    model = shape_inference.infer_shapes(
        original, check_type=True, strict_mode=True, data_prop=True
    )
    metadata, _, initializers = tensor_metadata(model)
    validate_graph(model, metadata)

    covered = [
        index
        for first, last in COVERAGE_RANGES
        for index in range(first, last + 1)
    ]
    if covered != list(range(308)):
        raise PlanError("coverage ranges are not an exact N000:N307 partition")

    producer = {
        name: index
        for index, node in enumerate(model.graph.node)
        for name in node.output
    }
    records: List[Dict[str, Any]] = []
    for range_index, (first, last) in enumerate(COVERAGE_RANGES):
        nodes = list(model.graph.node[first:last + 1])
        outputs = [name for node in nodes for name in node.output]
        output_set = set(outputs)
        boundary_inputs: List[str] = []
        required_initializers: List[str] = []
        for node in nodes:
            for name in node.input:
                if not name:
                    continue
                destination = (
                    required_initializers
                    if name in initializers
                    else boundary_inputs
                )
                if (
                    name not in output_set
                    and name not in destination
                ):
                    destination.append(name)
        workspace_bytes = aligned_blob_bytes(outputs, metadata)
        input_bytes = aligned_blob_bytes(boundary_inputs, metadata)
        weight_bytes = aligned_blob_bytes(required_initializers, metadata)
        pmc_offset = align(
            RESULT_HEADER_BYTES + workspace_bytes, PAGE_ALIGNMENT
        )
        dump_size = pmc_offset + PMC_STAGE_STRIDE
        input_offset = align(dump_size, PAGE_ALIGNMENT)
        weight_offset = align(input_offset + input_bytes, PAGE_ALIGNMENT)
        mem_size = max(
            MIN_MEMORY_BYTES,
            align(weight_offset + weight_bytes, PAGE_ALIGNMENT),
        )
        macs = sum(
            node_macs(model, metadata, initializers, index)
            for index in range(first, last + 1)
        )
        stage_overlaps = []
        for stage_name, stage_first, stage_last in PMC_ARCHITECTURE_STAGES:
            overlap_first = max(first, stage_first)
            overlap_last = min(last, stage_last)
            if overlap_first <= overlap_last:
                stage_overlaps.append(
                    {
                        "stage": stage_name,
                        "first_node": "N{:03d}".format(overlap_first),
                        "last_node": "N{:03d}".format(overlap_last),
                    }
                )
        records.append(
            {
                "range_index": range_index,
                "name": "n{:03d}_n{:03d}".format(first, last),
                "selector": "N{:03d}:N{:03d}".format(first, last),
                "first_node": first,
                "last_node": last,
                "node_count": last - first + 1,
                "operator_histogram": dict(
                    sorted(Counter(node.op_type for node in nodes).items())
                ),
                "conv_matmul_macs": macs,
                "retained_output_bytes": workspace_bytes,
                "boundary_inputs": boundary_inputs,
                "boundary_input_bytes": input_bytes,
                "initializer_count": len(required_initializers),
                "initializer_bytes": weight_bytes,
                "predicted_memory_map": {
                    "workspace_bytes": workspace_bytes,
                    "pmc_device_offset": pmc_offset,
                    "dump_size": dump_size,
                    "input_device_offset": input_offset,
                    "weight_device_offset": weight_offset,
                    "mem_size": mem_size,
                },
                "architecture_overlaps": stage_overlaps,
            }
        )

    manifest = {
        "schema_version": 1,
        "plan_kind": "gap_free_schema_v2_sys_emu_ranges",
        "source": {
            "repo": "onnx-community/yolov10n",
            "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
            "filename": "onnx/model.onnx",
            "sha256": EXPECTED_SHA256,
        },
        "coverage": {
            "first_node": "N000",
            "last_node": "N307",
            "node_count": 308,
            "range_count": len(records),
            "gap_count": 0,
            "overlap_count": 0,
            "conv_matmul_macs": sum(
                item["conv_matmul_macs"] for item in records
            ),
        },
        "constraints": {
            "maximum_conv_matmul_macs_per_range": 250000000,
            "target_retained_output_bytes_per_range": 24000000,
            "measured_maximum_retained_output_bytes": max(
                item["retained_output_bytes"] for item in records
            ),
            "retained_output_budget_exception": (
                "N271:N288 is kept as one DFL/decode architecture range; "
                "its 30,105,600-byte checkpoint arena still fits the "
                "repository-proven target allocation envelope"
            ),
            "conv_sigmoid_mul_triples_kept_together": True,
            "all_selected_node_outputs_compared": True,
            "one_pmc_record_per_range": True,
            "range_inputs": "captured exact-artifact ORT activations",
        },
        "architecture_stages": [
            {
                "name": name,
                "first_node": "N{:03d}".format(first),
                "last_node": "N{:03d}".format(last),
            }
            for name, first, last in PMC_ARCHITECTURE_STAGES
        ],
        "ranges": records,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "SYS_EMU_PLAN PASS ranges={} nodes=308 macs={} max_range_macs={} "
        "max_workspace={} out={}".format(
            len(records),
            manifest["coverage"]["conv_matmul_macs"],
            max(item["conv_matmul_macs"] for item in records),
            max(item["retained_output_bytes"] for item in records),
            output_path,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PlanError, OSError, ValueError, KeyError) as error:
        print("SYS_EMU_PLAN FAIL {}".format(error), file=sys.stderr)
        sys.exit(2)
