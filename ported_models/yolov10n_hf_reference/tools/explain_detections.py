#!/usr/bin/env python3
"""Inspect output0 records from a validated full-reference dump.

The pinned ONNX graph constructs output0 at N307 by concatenating the four
decoded corner coordinates selected at N303, the selected confidence at N304,
and the class index converted to FP32 at N306.  This tool deliberately does
not apply NMS or mutate the model output.  A score threshold only controls
which of the already selected top-300 records are displayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional

import numpy as np


RESULT_HEADER_BYTES = 128
RESULT_MAGIC = 0x31465259
RESULT_VERSION = 1
HEADER_U32 = struct.Struct("<16I")
PINNED_SOURCE = {
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "sha256": "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b",
}
COCO_ROOM_RAW_SHA256 = (
    "66b6131da00004bd2eab6a5d2fafab937289839d10d8199b3e95bfa3e76d8ca9"
)
TAIL_OPS_N281_N307 = [
    "Split",
    "Sub",
    "Add",
    "Concat",
    "Mul",
    "Sigmoid",
    "Concat",
    "Transpose",
    "Split",
    "ReduceMax",
    "TopK",
    "Unsqueeze",
    "Tile",
    "GatherElements",
    "Tile",
    "GatherElements",
    "Flatten",
    "TopK",
    "Mod",
    "Div",
    "Unsqueeze",
    "Tile",
    "GatherElements",
    "Unsqueeze",
    "Unsqueeze",
    "Cast",
    "Concat",
]


class DetectionError(RuntimeError):
    """The package, dump, or preprocessing metadata is inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DetectionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "dump",
        type=Path,
        help="full host, board, or system-emulator dump containing YRF1",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--preprocess-metadata",
        type=Path,
        help="optional metadata from preprocess_coco_room.py",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def preprocessing_geometry(
    metadata: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    preprocessing = metadata.get("preprocessing")
    source = metadata.get("source")
    output = metadata.get("output")
    require(
        isinstance(preprocessing, dict)
        and isinstance(source, dict)
        and isinstance(output, dict),
        "preprocessing metadata is incomplete",
    )
    require(
        metadata.get("fixture") == "coco_room_000139"
        and source.get("dtype") == "UINT8"
        and source.get("layout") == "HWC_RGB"
        and source.get("shape") == [480, 640, 3]
        and source.get("nbytes") == 480 * 640 * 3
        and source.get("sha256") == COCO_ROOM_RAW_SHA256,
        "preprocessing metadata is not the pinned COCO-room fixture",
    )
    require(
        preprocessing.get("graph_location") == "host_side_outside_onnx"
        and preprocessing.get("placement")
        == {"row_start": 80, "row_end_exclusive": 560}
        and preprocessing.get("padding_rgb") == [114, 114, 114]
        and preprocessing.get("layout_transform") == "HWC_RGB_to_NCHW_RGB",
        "preprocessing metadata does not describe the captured host transform",
    )
    placement = preprocessing.get("placement")
    canvas_shape = preprocessing.get("canvas_shape")
    source_shape = source.get("shape")
    require(
        isinstance(placement, dict)
        and canvas_shape == [640, 640, 3]
        and isinstance(source_shape, list)
        and len(source_shape) == 3,
        "preprocessing placement/canvas/source metadata is invalid",
    )
    source_height, source_width, source_channels = (
        int(value) for value in source_shape
    )
    row_start = int(placement["row_start"])
    row_end = int(placement["row_end_exclusive"])
    require(
        source_channels == 3
        and source_width == int(canvas_shape[1])
        and source_height == row_end - row_start
        and 0 <= row_start < row_end <= int(canvas_shape[0]),
        "captured preprocessing is not an unscaled full-width row placement",
    )
    input_blob = manifest.get("blobs", {}).get("inputs", {})
    require(
        output.get("dtype") == "FLOAT"
        and output.get("layout") == "NCHW_RGB"
        and output.get("shape") == [1, 3, 640, 640]
        and output.get("sha256") == input_blob.get("sha256"),
        "preprocessing metadata is not bound to the package input blob",
    )
    return {
        "source_shape": [source_height, source_width, source_channels],
        "canvas_shape": canvas_shape,
        "row_start": row_start,
        "row_end_exclusive": row_end,
        "inverse_transform": (
            "x_original=x_canvas; y_original=y_canvas-row_start; "
            "then clip to source bounds"
        ),
    }


def original_box(
    box: np.ndarray, geometry: Optional[Dict[str, Any]]
) -> Optional[List[float]]:
    if geometry is None:
        return None
    row_start = int(geometry["row_start"])
    original_height = int(geometry["source_shape"][0])
    original_width = int(geometry["source_shape"][1])
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        max(0.0, min(float(original_width), x1)),
        max(0.0, min(float(original_height), y1 - row_start)),
        max(0.0, min(float(original_width), x2)),
        max(0.0, min(float(original_height), y2 - row_start)),
    ]


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    dump_path = args.dump.resolve()
    require(args.top >= 0, "--top must be non-negative")
    require(np.isfinite(args.threshold), "--threshold must be finite")

    manifest_path = package / "slice_manifest.json"
    require(manifest_path.is_file(), "missing {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("schema_version") == 2
        and manifest.get("manifest_kind") == "full_graph_liveness",
        "expected a schema-v2 full-graph package",
    )
    source = manifest.get("source")
    require(
        isinstance(source, dict)
        and all(source.get(key) == value for key, value in PINNED_SOURCE.items()),
        "package source does not match the pinned ONNX artifact",
    )
    require(
        manifest.get("selection", {}).get("selector") == "N000:N307",
        "package does not select all 308 nodes",
    )
    require(
        manifest.get("result", {}).get("output_tensor") == "output0",
        "manifest output tensor is not output0",
    )
    nodes = manifest.get("nodes")
    require(
        isinstance(nodes, list)
        and len(nodes) == 308
        and [node.get("index") for node in nodes] == list(range(308)),
        "manifest does not contain the ordered N000:N307 graph",
    )
    require(
        [node.get("op_type") for node in nodes[281:308]]
        == TAIL_OPS_N281_N307,
        "manifest N281:N307 operator sequence differs from the pinned graph",
    )
    final_node = nodes[307]
    require(
        final_node.get("name") == "/model.23/Concat_6"
        and final_node.get("inputs")
        == [
            "/model.23/GatherElements_2_output_0",
            "/model.23/Unsqueeze_2_output_0",
            "/model.23/Cast_2_output_0",
        ]
        and final_node.get("outputs") == ["output0"]
        and final_node.get("attributes") == {"axis": -1},
        "N307 does not concatenate [box4, score1, class1] as output0",
    )

    output_records = [
        tensor
        for tensor in manifest.get("tensors", [])
        if tensor.get("name") == "output0"
    ]
    require(len(output_records) == 1, "expected exactly one output0 tensor")
    output = output_records[0]
    require(
        output.get("dtype") == "FLOAT"
        and output.get("shape") == [1, 300, 6]
        and output.get("storage") == "workspace"
        and output.get("producer") == "N307",
        "output0 metadata does not match the pinned graph contract",
    )

    require(dump_path.is_file(), "missing {}".format(dump_path))
    dump = dump_path.read_bytes()
    memory_map = manifest.get("memory_map")
    result_contract = manifest.get("result")
    require(
        isinstance(memory_map, dict) and isinstance(result_contract, dict),
        "manifest result memory map is missing",
    )
    expected_dump_bytes = int(memory_map["dump_size"])
    require(
        len(dump) == expected_dump_bytes,
        "dump size mismatch: expected {}, got {}".format(
            expected_dump_bytes, len(dump)
        ),
    )
    require(len(dump) >= RESULT_HEADER_BYTES, "dump is shorter than YRF1")
    words = HEADER_U32.unpack_from(dump, 0)
    require(words[0] == RESULT_MAGIC, "YRF1 magic mismatch")
    require(words[1] == RESULT_VERSION, "YRF1 version mismatch")
    require(words[2] == 0, "runtime status is not success")
    require(
        words[5] == 0 and words[6] == 307 and words[7] == 308,
        "runtime did not execute N000:N307",
    )
    require(
        words[3] == 0xFFFFFFFF
        and words[4] == 0
        and words[8] == len(manifest["tensors"])
        and words[9] == int(memory_map["workspace_bytes"])
        and words[10] == int(manifest["blobs"]["inputs"]["nbytes"])
        and words[11] == int(manifest["blobs"]["weights"]["nbytes"])
        and words[12] == 1
        and list(words[13:16]) == [0, 0, 0],
        "YRF1 header does not match the full package",
    )
    header_bytes = int(result_contract["header_bytes"])
    require(
        header_bytes
        == int(result_contract["workspace_offset_within_dump"])
        and int(memory_map["result_device_offset"]) == 0,
        "manifest does not place the full workspace after YRF1",
    )
    start = header_bytes + int(output["offset"])
    end = start + int(output["nbytes"])
    require(end <= len(dump), "output0 exceeds dump size")
    records = np.frombuffer(dump[start:end], dtype="<f4").reshape(300, 6)
    require(np.all(np.isfinite(records)), "output0 contains non-finite values")
    require(
        np.all((records[:, 4] >= 0.0) & (records[:, 4] <= 1.0))
        and np.all(records[:-1, 4] >= records[1:, 4]),
        "output0 scores are outside [0,1] or not in TopK order",
    )
    require(
        np.all(records[:, 0] <= records[:, 2])
        and np.all(records[:, 1] <= records[:, 3]),
        "output0 contains an invalid xyxy corner order",
    )

    geometry: Optional[Dict[str, Any]] = None
    metadata_report: Optional[Dict[str, Any]] = None
    if args.preprocess_metadata is not None:
        metadata_path = args.preprocess_metadata.resolve()
        require(metadata_path.is_file(), "missing {}".format(metadata_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(
            metadata.get("schema_version") == 1,
            "unsupported preprocessing metadata schema",
        )
        geometry = preprocessing_geometry(metadata, manifest)
        metadata_report = {
            "path": str(metadata_path),
            "sha256": sha256(metadata_path),
            "fixture": metadata.get("fixture"),
            "geometry": geometry,
        }

    selected: List[Dict[str, Any]] = []
    for rank, record in enumerate(records):
        score = float(record[4])
        class_float = float(record[5])
        class_id = int(class_float)
        require(
            class_float == float(class_id) and 0 <= class_id < 80,
            "record {} has invalid class index {}".format(rank, class_float),
        )
        if score < args.threshold or len(selected) >= args.top:
            continue
        canvas_box = [float(value) for value in record[:4]]
        selected.append(
            {
                "rank": rank,
                "box_xyxy_canvas": canvas_box,
                "box_xyxy_original": original_box(
                    record[:4], geometry
                ),
                "score": score,
                "class_id": class_id,
            }
        )

    report = {
        "schema_version": 1,
        "pass": True,
        "package": str(package),
        "dump": {
            "path": str(dump_path),
            "nbytes": len(dump),
            "sha256": sha256(dump_path),
        },
        "graph_contract": {
            "nodes": "N000:N307",
            "output": "output0",
            "shape": [1, 300, 6],
            "record": ["x1", "y1", "x2", "y2", "score", "class_id"],
            "coordinate_space": "640x640 ONNX input canvas pixels",
            "selection": (
                "NMS-free two-stage TopK selection; overlapping records can "
                "remain, and the display threshold does not change output0"
            ),
            "labels": (
                "numeric class IDs from output0; no external human-readable "
                "label map is applied"
            ),
        },
        "preprocess_metadata": metadata_report,
        "threshold": args.threshold,
        "display_limit": args.top,
        "records_at_or_above_threshold": int(
            np.count_nonzero(records[:, 4] >= args.threshold)
        ),
        "displayed": selected,
    }
    if args.json is not None:
        output_path = args.json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        "DETECTIONS PASS output0=[1,300,6] threshold={} matches={} "
        "displayed={}".format(
            args.threshold,
            report["records_at_or_above_threshold"],
            len(selected),
        )
    )
    for item in selected:
        print(
            "rank={rank} class_id={class_id} score={score:.8f} "
            "xyxy_canvas={box_xyxy_canvas} "
            "xyxy_original={box_xyxy_original}".format(**item)
        )
    if args.json is not None:
        print("REPORT {}".format(args.json.resolve()))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DetectionError, OSError, ValueError, KeyError) as error:
        print("DETECTIONS FAIL {}".format(error), file=sys.stderr)
        sys.exit(2)
