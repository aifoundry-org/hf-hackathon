#!/usr/bin/env python3
"""Diagnose numerical conditioning of the pinned YOLOv10n TopK tail.

The full scalar run intentionally uses a deterministic, platform-safe sigmoid
implementation.  Very small differences before TopK can permute nearly tied
records even when every upstream float passes its numerical tolerance.  This
tool separates that conditioning effect from a tail indexing error:

* replay N289:N307 from the captured ONNX Runtime N288 boundary;
* require every replayed tail tensor to equal ONNX Runtime bit-for-bit;
* replay the same operations from the C N288 checkpoint;
* require the replayed final output to equal the C output bit-for-bit; and
* report TopK cutoff ties, margins, and selected-index overlap.

It does not reinterpret or reorder output0 when judging the normal numerical
comparison.  The ordinary row-wise mismatch count is reported alongside the
semantic isolation evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_FULL_DIR = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/full_graph/deterministic"
)
DEFAULT_MODEL = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
)
EXPECTED_MODEL_SHA256 = (
    "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
)
TAIL_FIRST_NODE = 289
TAIL_LAST_NODE = 307


class DiagnosticError(RuntimeError):
    """Raised when an input or semantic invariant fails."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    require(isinstance(value, dict), "{} is not a JSON object".format(path))
    return value


def tensor_dtype(name: str) -> np.dtype:
    if name == "FLOAT":
        return np.dtype("<f4")
    if name == "INT64":
        return np.dtype("<i8")
    raise DiagnosticError("unsupported tensor dtype {!r}".format(name))


def checkpoint_arrays(
    full_dir: Path, manifest: Mapping[str, Any], node_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    checkpoints = {
        item["node_id"]: item for item in manifest.get("checkpoints", [])
    }
    require(node_id in checkpoints, "missing checkpoint {}".format(node_id))
    checkpoint = checkpoints[node_id]
    dtype = tensor_dtype(checkpoint["dtype"])
    elements = int(checkpoint["elements"])
    shape = tuple(int(item) for item in checkpoint["shape"])
    header_bytes = int(
        manifest["result"]["workspace_offset_within_dump"]
    )
    actual = np.fromfile(
        str(full_dir / "host_full_dump.bin"),
        dtype=dtype,
        count=elements,
        offset=header_bytes + int(checkpoint["workspace_offset"]),
    )
    reference = np.fromfile(
        str(full_dir / "goldens.bin"),
        dtype=dtype,
        count=elements,
        offset=int(checkpoint["golden_offset"]),
    )
    require(
        actual.size == elements and reference.size == elements,
        "{} checkpoint blob is truncated".format(node_id),
    )
    return actual.reshape(shape), reference.reshape(shape)


def capture_tail(
    model_path: Path,
    manifest: Mapping[str, Any],
    input_array: np.ndarray,
) -> Dict[str, np.ndarray]:
    model = onnx.load(str(model_path))
    require(len(model.graph.node) == 308, "pinned model must contain 308 nodes")
    values: Dict[str, Any] = {}
    for item in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        values[item.name] = item
    tensor_records = {
        item["name"]: item for item in manifest.get("tensors", [])
    }
    existing = {item.name for item in model.graph.output}
    names: List[str] = []
    for node_index in range(TAIL_FIRST_NODE, TAIL_LAST_NODE + 1):
        for name in model.graph.node[node_index].output:
            names.append(name)
            if name in existing:
                continue
            if name in values:
                model.graph.output.append(copy.deepcopy(values[name]))
            else:
                require(
                    name in tensor_records,
                    "missing manifest metadata for {!r}".format(name),
                )
                record = tensor_records[name]
                dtype = {
                    "FLOAT": TensorProto.FLOAT,
                    "INT64": TensorProto.INT64,
                }.get(record["dtype"])
                require(dtype is not None, "unsupported dtype for {!r}".format(name))
                model.graph.output.append(
                    helper.make_tensor_value_info(
                        name, dtype, record["shape"]
                    )
                )
            existing.add(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    with tempfile.NamedTemporaryFile(suffix=".onnx") as instrumented:
        onnx.save(model, instrumented.name)
        session = ort.InferenceSession(
            instrumented.name,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run(names, {"images": input_array})
    return dict(zip(names, outputs))


def stable_topk_last(
    data: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Value-descending TopK with lower source index first on exact ties."""
    require(data.ndim >= 1, "TopK input must have rank >= 1")
    require(0 < k <= data.shape[-1], "invalid TopK k")
    rows = data.reshape((-1, data.shape[-1]))
    values = np.empty((rows.shape[0], k), dtype=np.float32)
    indices = np.empty((rows.shape[0], k), dtype=np.int64)
    source_indices = np.arange(rows.shape[1], dtype=np.int64)
    for row_index, row in enumerate(rows):
        order = np.lexsort(
            (source_indices, -row.astype(np.float64, copy=False))
        )[:k]
        values[row_index] = row[order]
        indices[row_index] = order
    shape = data.shape[:-1] + (k,)
    return values.reshape(shape), indices.reshape(shape)


def replay_tail(boundary: np.ndarray) -> Dict[str, np.ndarray]:
    require(
        boundary.shape == (1, 8400, 84) and boundary.dtype == np.float32,
        "N288 boundary must be FLOAT [1,8400,84]",
    )
    boxes = boundary[..., :4].copy()
    scores = boundary[..., 4:].copy()
    reduced = np.max(scores, axis=-1)
    first_values, first_indices = stable_topk_last(reduced, 300)
    unsqueezed = first_indices[..., None]
    box_indices = np.tile(unsqueezed, (1, 1, 4))
    selected_boxes = np.take_along_axis(boxes, box_indices, axis=1)
    score_indices = np.tile(unsqueezed, (1, 1, 80))
    selected_scores = np.take_along_axis(scores, score_indices, axis=1)
    flattened = selected_scores.reshape((1, 24000))
    second_values, second_indices = stable_topk_last(flattened, 300)
    classes = second_indices % np.int64(80)
    rows = second_indices // np.int64(80)
    row_unsqueezed = rows[..., None]
    row_indices = np.tile(row_unsqueezed, (1, 1, 4))
    final_boxes = np.take_along_axis(selected_boxes, row_indices, axis=1)
    value_unsqueezed = second_values[..., None]
    class_unsqueezed = classes[..., None]
    class_float = class_unsqueezed.astype(np.float32)
    output = np.concatenate(
        (final_boxes, value_unsqueezed, class_float), axis=-1
    )
    return {
        "/model.23/Split_2_output_0": boxes,
        "/model.23/Split_2_output_1": scores,
        "/model.23/ReduceMax_output_0": reduced,
        "/model.23/TopK_output_0": first_values,
        "/model.23/TopK_output_1": first_indices,
        "/model.23/Unsqueeze_output_0": unsqueezed,
        "/model.23/Tile_output_0": box_indices,
        "/model.23/GatherElements_output_0": selected_boxes,
        "/model.23/Tile_1_output_0": score_indices,
        "/model.23/GatherElements_1_output_0": selected_scores,
        "/model.23/Flatten_output_0": flattened,
        "/model.23/TopK_1_output_0": second_values,
        "/model.23/TopK_1_output_1": second_indices,
        "/model.23/Mod_output_0": classes,
        "/model.23/Div_output_0": rows,
        "/model.23/Unsqueeze_1_output_0": row_unsqueezed,
        "/model.23/Tile_2_output_0": row_indices,
        "/model.23/GatherElements_2_output_0": final_boxes,
        "/model.23/Unsqueeze_2_output_0": value_unsqueezed,
        "/model.23/Unsqueeze_3_output_0": class_unsqueezed,
        "/model.23/Cast_2_output_0": class_float,
        "output0": output,
    }


def exact_comparisons(
    replayed: Mapping[str, np.ndarray],
    captured: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for name, actual in replayed.items():
        require(name in captured, "ORT capture lacks {!r}".format(name))
        reference = captured[name]
        require(
            actual.shape == reference.shape and actual.dtype == reference.dtype,
            "replay metadata differs for {!r}".format(name),
        )
        unequal = int(np.count_nonzero(actual != reference))
        result.append(
            {
                "tensor": name,
                "elements": int(actual.size),
                "exact_mismatch_count": unequal,
                "pass": unequal == 0,
            }
        )
    return result


def float_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    difference = np.abs(
        actual.astype(np.float64) - reference.astype(np.float64)
    )
    tolerance = atol + rtol * np.abs(reference.astype(np.float64))
    return {
        "elements": int(actual.size),
        "exact_mismatch_count": int(np.count_nonzero(actual != reference)),
        "tolerance_mismatch_count": int(np.count_nonzero(difference > tolerance)),
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "atol": atol,
        "rtol": rtol,
    }


def cutoff_metrics(values: np.ndarray, rank: int = 299) -> Dict[str, Any]:
    flattened = values.reshape(-1)
    ordered = np.sort(flattened)[::-1]
    cutoff = ordered[rank]
    next_value = ordered[rank + 1]
    return {
        "rank": rank,
        "value": float(cutoff),
        "next_value": float(next_value),
        "margin": float(cutoff - next_value),
        "strictly_greater_count": int(np.count_nonzero(flattened > cutoff)),
        "equal_count": int(np.count_nonzero(flattened == cutoff)),
        "unique_value_count": int(np.unique(flattened).size),
    }


def selection_metrics(
    reference_replay: Mapping[str, np.ndarray],
    actual_replay: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    ref_first = reference_replay["/model.23/TopK_output_1"][0]
    act_first = actual_replay["/model.23/TopK_output_1"][0]
    ref_second = reference_replay["/model.23/TopK_1_output_1"][0]
    act_second = actual_replay["/model.23/TopK_1_output_1"][0]
    ref_pairs = [
        (int(ref_first[int(index // 80)]), int(index % 80))
        for index in ref_second
    ]
    act_pairs = [
        (int(act_first[int(index // 80)]), int(index % 80))
        for index in act_second
    ]
    return {
        "first_topk": {
            "same_rank_count": int(np.count_nonzero(ref_first == act_first)),
            "set_overlap_count": len(
                set(ref_first.tolist()) & set(act_first.tolist())
            ),
            "k": 300,
        },
        "second_topk_anchor_class": {
            "same_rank_count": sum(
                left == right for left, right in zip(ref_pairs, act_pairs)
            ),
            "set_overlap_count": len(set(ref_pairs) & set(act_pairs)),
            "k": 300,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    full_dir = args.full_dir.resolve()
    model_path = args.model.resolve()
    manifest = load_json(full_dir / "slice_manifest.json")
    require(
        sha256_path(model_path) == EXPECTED_MODEL_SHA256,
        "model SHA-256 does not match the pinned artifact",
    )
    input_records = [
        item
        for item in manifest.get("tensors", [])
        if item.get("storage") == "input"
    ]
    require(
        len(input_records) == 1
        and input_records[0].get("name") == "images"
        and input_records[0].get("dtype") == "FLOAT",
        "manifest must contain sole FLOAT input tensor 'images'",
    )
    input_shape = tuple(int(item) for item in input_records[0]["shape"])
    input_array = np.fromfile(
        str(full_dir / manifest["blobs"]["inputs"]["path"]), dtype="<f4"
    )
    require(
        input_array.size == int(np.prod(input_shape)),
        "input blob size differs from manifest shape",
    )
    input_array = input_array.reshape(input_shape)

    c_boundary, reference_boundary = checkpoint_arrays(
        full_dir, manifest, "N288"
    )
    c_output, reference_output = checkpoint_arrays(
        full_dir, manifest, "N307"
    )
    captured = capture_tail(model_path, manifest, input_array)
    require(
        np.array_equal(captured["output0"], reference_output),
        "captured ORT output0 differs from packaged N307 golden",
    )
    require(
        np.array_equal(
            captured["/model.23/Split_2_output_0"],
            reference_boundary[..., :4],
        )
        and np.array_equal(
            captured["/model.23/Split_2_output_1"],
            reference_boundary[..., 4:],
        ),
        "captured ORT tail boundary differs from packaged N288 golden",
    )

    reference_replay = replay_tail(reference_boundary)
    actual_replay = replay_tail(c_boundary)
    reference_checks = exact_comparisons(reference_replay, captured)
    c_replay_exact_mismatches = int(
        np.count_nonzero(actual_replay["output0"] != c_output)
    )
    boundary_box_metrics = float_metrics(
        c_boundary[..., :4],
        reference_boundary[..., :4],
        args.atol,
        args.rtol,
    )
    boundary_score_metrics = float_metrics(
        c_boundary[..., 4:],
        reference_boundary[..., 4:],
        args.atol,
        args.rtol,
    )
    final_metrics = float_metrics(
        c_output, reference_output, args.atol, args.rtol
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_kind": "n289_n307_tail_conditioning",
        "pass": (
            all(item["pass"] for item in reference_checks)
            and c_replay_exact_mismatches == 0
        ),
        "model": {
            "path": str(model_path),
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "full_dir": str(full_dir),
        "input_sha256": sha256_path(
            full_dir / manifest["blobs"]["inputs"]["path"]
        ),
        "reference_boundary_replay": {
            "pass": all(item["pass"] for item in reference_checks),
            "tensors": reference_checks,
        },
        "c_boundary_replay": {
            "pass": c_replay_exact_mismatches == 0,
            "output0_exact_mismatch_count": c_replay_exact_mismatches,
        },
        "n288": {
            "boxes": boundary_box_metrics,
            "class_scores": boundary_score_metrics,
            "reference_score_unique_count": int(
                np.unique(reference_boundary[..., 4:]).size
            ),
            "c_score_unique_count": int(
                np.unique(c_boundary[..., 4:]).size
            ),
        },
        "topk_conditioning": {
            "reference_first_cutoff": cutoff_metrics(
                reference_replay["/model.23/ReduceMax_output_0"]
            ),
            "c_first_cutoff": cutoff_metrics(
                actual_replay["/model.23/ReduceMax_output_0"]
            ),
            "reference_second_cutoff": cutoff_metrics(
                reference_replay["/model.23/Flatten_output_0"]
            ),
            "c_second_cutoff": cutoff_metrics(
                actual_replay["/model.23/Flatten_output_0"]
            ),
            "selection_overlap": selection_metrics(
                reference_replay, actual_replay
            ),
        },
        "ordinary_output0_comparison": final_metrics,
        "interpretation": (
            "Tail semantics pass only means both boundary replays are exact. "
            "The ordinary output0 comparison remains the numerical full-run "
            "gate; cutoff tie metrics explain when small upstream score "
            "differences can change row identity."
        ),
    }

    output_path = (
        args.output_json.resolve()
        if args.output_json is not None
        else full_dir / "tail_diagnostic.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(
        "TAIL_DIAGNOSTIC {} reference_tensors={} c_replay_mismatches={} "
        "ordinary_output_mismatches={} report={}".format(
            "PASS" if report["pass"] else "FAIL",
            len(reference_checks),
            c_replay_exact_mismatches,
            final_metrics["tolerance_mismatch_count"],
            output_path,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print("TAIL_DIAGNOSTIC FAIL: {}".format(error))
        raise SystemExit(1)
