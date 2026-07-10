#!/usr/bin/env python3
"""Generate deterministic YOLO detections for the fixed COCO CI suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = REPO_ROOT / ".github" / "ci" / "reference" / "yolo.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_preprocess(raw_path: Path, input_cfg: dict[str, Any]):
    import numpy as np

    src_h, src_w, channels = (int(value) for value in input_cfg["shape"])
    _, _, dst_h, dst_w = (int(value) for value in input_cfg["model_shape"])
    raw = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8)
    expected_size = src_h * src_w * channels
    if raw.size != expected_size:
        raise ValueError(f"{raw_path}: got {raw.size} bytes, expected {expected_size}")
    image = raw.reshape(src_h, src_w, channels).astype(np.float32)

    y = (np.arange(dst_h, dtype=np.float32) + np.float32(0.5)) * np.float32(
        src_h / dst_h
    ) - np.float32(0.5)
    x = (np.arange(dst_w, dtype=np.float32) + np.float32(0.5)) * np.float32(
        src_w / dst_w
    ) - np.float32(0.5)
    # Match the C cast: negative source coordinates truncate toward zero.
    y0 = np.maximum(np.trunc(y).astype(np.int32), 0)
    x0 = np.maximum(np.trunc(x).astype(np.int32), 0)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    dy = y - y0.astype(np.float32)
    dx = x - x0.astype(np.float32)

    p00 = image[y0[:, None], x0[None, :]]
    p01 = image[y0[:, None], x1[None, :]]
    p10 = image[y1[:, None], x0[None, :]]
    p11 = image[y1[:, None], x1[None, :]]
    wx = dx[None, :, None]
    wy = dy[:, None, None]
    resized = (
        (np.float32(1.0) - wy) * ((np.float32(1.0) - wx) * p00 + wx * p01)
        + wy * ((np.float32(1.0) - wx) * p10 + wx * p11)
    ) * np.float32(1.0 / 255.0)
    return resized.transpose(2, 0, 1)[None].astype(np.float32)


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def postprocess(output, config: dict[str, Any], names: Any) -> list[dict[str, Any]]:
    import numpy as np

    expected_shape = tuple(int(value) for value in config["model"]["canonical_export"]["output_shape"])
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"host output shape {tuple(output.shape)} != {expected_shape}")

    post = config["postprocess"]
    boxes = output[0, :4, :].T
    probabilities = output[0, 4:, :].T
    class_ids = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    keep = scores >= float(post["confidence_threshold"])
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    candidates: list[dict[str, Any]] = []
    for box, class_id, score in zip(boxes, class_ids, scores):
        cx, cy, width, height = (float(value) for value in box)
        candidates.append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "box": [
                    cx - width * 0.5,
                    cy - height * 0.5,
                    cx + width * 0.5,
                    cy + height * 0.5,
                ],
            }
        )

    kept: list[dict[str, Any]] = []
    iou_threshold = float(post["nms_iou_threshold"])
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(
            item["class_id"] == candidate["class_id"]
            and box_iou(item["box"], candidate["box"]) > iou_threshold
            for item in kept
        ):
            continue
        class_id = candidate["class_id"]
        if isinstance(names, dict):
            candidate["label"] = str(names.get(class_id, class_id))
        elif class_id < len(names):
            candidate["label"] = str(names[class_id])
        kept.append(candidate)
        if len(kept) == int(post["max_detections"]):
            break
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("error: numpy, torch, and ultralytics are required", file=sys.stderr)
        return 2

    contract = json.loads(args.contract.read_text())
    source = contract["model"]["source"]
    actual_checkpoint_sha = file_sha256(args.checkpoint)
    if actual_checkpoint_sha != source["sha256"]:
        print(
            f"error: checkpoint SHA256 {actual_checkpoint_sha} != {source['sha256']}",
            file=sys.stderr,
        )
        return 2
    required_version = contract["model"]["host_runtime"]["ultralytics"]
    if ultralytics.__version__ != required_version:
        print(
            f"error: ultralytics {ultralytics.__version__} != {required_version}",
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(1)
    torch.manual_seed(0)
    detector = YOLO(str(args.checkpoint))
    head = detector.model.model[-1]
    if not hasattr(head, "end2end"):
        print("error: checkpoint has no YOLOv10 end2end head", file=sys.stderr)
        return 2
    head.end2end = bool(contract["model"]["host_runtime"]["end2end_head"])
    detector.model.eval()

    score_threshold = float(contract["agreement"]["comparison_score_threshold"])
    minimum_detections = int(contract["agreement"]["minimum_reference_detections_per_case"])
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for case in contract["fixtures"]["cases"]:
            asset = REPO_ROOT / case["asset"]
            actual_asset_sha = file_sha256(asset)
            if actual_asset_sha != case["asset_sha256"]:
                raise SystemExit(
                    f"{case['name']}: fixture SHA256 {actual_asset_sha} != {case['asset_sha256']}"
                )
            tensor = torch.from_numpy(board_preprocess(asset, contract["input"]))
            prediction = detector.model(tensor)
            output = prediction[0] if isinstance(prediction, tuple) else prediction
            detections = postprocess(output.detach().cpu().numpy(), contract, detector.names)
            compared = [item for item in detections if item["score"] >= score_threshold]
            if len(compared) < minimum_detections:
                raise SystemExit(
                    f"{case['name']}: host reference produced only {len(compared)} "
                    f"detections at score >= {score_threshold}"
                )
            results[case["name"]] = {
                "image_id": case["image_id"],
                "input_sha256": actual_asset_sha,
                "detections": detections,
            }
            labels = ", ".join(item.get("label", str(item["class_id"])) for item in compared)
            print(f"{case['name']}: {len(compared)} compared detections ({labels})")

    payload = {
        "schema_version": 1,
        "contract_sha256": file_sha256(args.contract),
        "reference": {
            "repo": source["repo"],
            "revision": source["revision"],
            "filename": source["filename"],
            "checkpoint_sha256": actual_checkpoint_sha,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "agreement": contract["agreement"],
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
