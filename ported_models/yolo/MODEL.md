# YOLO Model Card

- Reference family: YOLOv10n object detection.
- Hugging Face base: `onnx-community/yolov10n` at
  `57657320425ee34056408a57ad9d29c4d4815bd8`.
- Reference file: `onnx/model.onnx`.
- Main source: `src/yolo_vpu_argbuf.c`.
- Smoke manifest: `manifests/yolo_10_variants.txt`.
- Sweep manifest: `manifests/yolo_100_variants.txt`.
- Key docs: `docs/optimizations.md`.

Use the pinned Hugging Face model config/preprocessor metadata for model I/O and
keep large ONNX/input artifacts outside git.
