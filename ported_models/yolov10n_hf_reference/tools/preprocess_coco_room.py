#!/usr/bin/env python3
"""Create the pinned model's FP32 input from the repository COCO-room fixture.

Preprocessing is deliberately outside the ONNX node graph.  The source image
is the repository's checked raw 480x640 RGB fixture.  It is centered without
resizing in a 640x640 RGB canvas, with 80 rows of value 114 above and below,
then normalized to [0, 1] and written as little-endian NCHW FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_RAW = (
    REPO_ROOT
    / "ported_models/yolo/assets/yolo/"
    "coco_room_000139_raw_480x640x3_uint8_rgb.bin"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/fixtures/"
    "coco_room_000139/input_fp32.bin"
)
RAW_SHA256 = "66b6131da00004bd2eab6a5d2fafab937289839d10d8199b3e95bfa3e76d8ca9"
FP32_SHA256 = "65afc38f381c09712cd9a6a78e8e7e9800c713d21679abee588b218a6a137c7b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="default: OUTPUT with .json appended",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = args.raw.resolve()
    output_path = args.output.resolve()
    metadata_path = (
        args.metadata.resolve()
        if args.metadata is not None
        else output_path.with_suffix(output_path.suffix + ".json")
    )

    raw = raw_path.read_bytes()
    actual_raw_sha = sha256(raw)
    if len(raw) != 480 * 640 * 3 or actual_raw_sha != RAW_SHA256:
        print(
            "PREPROCESS FAIL raw identity expected_bytes={} actual_bytes={} "
            "expected_sha256={} actual_sha256={}".format(
                480 * 640 * 3,
                len(raw),
                RAW_SHA256,
                actual_raw_sha,
            ),
            file=sys.stderr,
        )
        return 2

    source = np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[80:560, :, :] = source
    tensor = np.ascontiguousarray(
        (canvas.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)[
            np.newaxis, ...
        ],
        dtype="<f4",
    )
    payload = tensor.tobytes(order="C")
    actual_fp32_sha = sha256(payload)
    if len(payload) != 1 * 3 * 640 * 640 * 4 or actual_fp32_sha != FP32_SHA256:
        print(
            "PREPROCESS FAIL FP32 identity expected_sha256={} actual_sha256={}"
            .format(FP32_SHA256, actual_fp32_sha),
            file=sys.stderr,
        )
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    metadata = {
        "schema_version": 1,
        "fixture": "coco_room_000139",
        "source": {
            "path": str(raw_path.relative_to(REPO_ROOT)),
            "dtype": "UINT8",
            "layout": "HWC_RGB",
            "shape": [480, 640, 3],
            "nbytes": len(raw),
            "sha256": actual_raw_sha,
        },
        "preprocessing": {
            "graph_location": "host_side_outside_onnx",
            "canvas_shape": [640, 640, 3],
            "placement": {"row_start": 80, "row_end_exclusive": 560},
            "padding_rgb": [114, 114, 114],
            "normalization": "uint8 converted to FP32, divided by FP32 255.0",
            "layout_transform": "HWC_RGB_to_NCHW_RGB",
        },
        "output": {
            "path": output_path.name,
            "dtype": "FLOAT",
            "endianness": "little",
            "layout": "NCHW_RGB",
            "shape": [1, 3, 640, 640],
            "nbytes": len(payload),
            "sha256": actual_fp32_sha,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PREPROCESS PASS fixture=coco_room_000139 bytes={} sha256={} out={}"
        .format(len(payload), actual_fp32_sha, output_path)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
