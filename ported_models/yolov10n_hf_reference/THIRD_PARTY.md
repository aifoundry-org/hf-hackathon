# Third-party model artifact

The model weights and ONNX graph are not part of this repository's Apache-2.0
licensed source and are not committed.

| Component | Source | Revision | File | SHA-256 | License |
|---|---|---|---|---|---|
| YOLOv10n ONNX | `onnx-community/yolov10n` | `57657320425ee34056408a57ad9d29c4d4815bd8` | `onnx/model.onnx` | `a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b` | AGPL-3.0 |

Resolve URL:
`https://huggingface.co/onnx-community/yolov10n/resolve/57657320425ee34056408a57ad9d29c4d4815bd8/onnx/model.onnx?download=true`.

`tools/download_model.py` downloads the pinned artifact into the ignored
`local-artifacts/` cache and verifies its 9,386,116-byte size and checksum
before it is used. The port does not re-export from PyTorch: this ONNX file is
the sole source of topology, weights, tensor names, shapes, and golden outputs.
