# YOLO Reference Model

This folder contains the YOLO kernel source, variant manifests, parse scripts,
and optimization notes used for ET-SoC1 sweeps.

The Hugging Face model package should be treated as the external reference
source. Keep ONNX/input blobs and generated ELFs out of git. The current base is
`onnx-community/yolov10n` pinned in
[`docs/HF_REFERENCES.md`](../../docs/HF_REFERENCES.md).

Start with `docs/optimizations.md`.
