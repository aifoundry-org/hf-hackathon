# YOLOv10n End-to-End

| Field | Value |
| --- | --- |
| Model | YOLOv10n detector |
| Input | `uint8[480][640][3]` RGB, HWC |
| Preprocess | Bilinear resize to 288x512, scale by 1/255, HWC to CHW |
| Output | `{count, class_id, score, x1, y1, x2, y2}` detections |
| CI metric | Mean end-to-end kernel wait seconds across five images |
| CI accuracy | Agreement with the pinned host model on five public COCO images |

This is the canonical `yolo` leaderboard benchmark. The host runs the pinned
Hugging Face YOLOv10n checkpoint and derives the reference classes, scores, and
boxes for every case. ET-SoC1 output must pass that correctness gate before its
mean end-to-end latency is eligible for the leaderboard.
