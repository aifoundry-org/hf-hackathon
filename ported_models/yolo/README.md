# YOLO End-to-End Detector

This port runs the real YOLOv10n detector path on ET-SoC1:

- raw uint8 RGB input at `0x04A00000`
- on-chip resize/normalization/transposition to 288x512 CHW FP32
- YOLOv10n backbone, neck, and detection heads
- on-chip DFL decode, class sigmoid, thresholding, and class-aware NMS
- compact detections at `0x01D00000`

The CI benchmark uses five committed COCO `val2017` samples. On each run, the
host categorizes them with the pinned YOLOv10n checkpoint; the board must match
the host classes with the configured precision, recall, IoU, and score-error
bounds. Implementations may fuse, quantize, or repack weights and scales, but
only a correctness-passing result is ranked by mean end-to-end latency.
