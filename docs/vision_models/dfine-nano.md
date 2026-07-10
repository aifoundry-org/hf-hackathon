# Porting Plan: D-FINE Nano & Small (ONNX Object Detection)

**Path:** ONNX (ggonnx bridge) — Pure vision (object detection)
**Priority:** #5 — Easiest ONNX port, fills detection gap in ggonnx
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | D-FINE Nano | D-FINE Small |
|----------|-------------|-------------|
| **Parameters** | 3.79M | 10.33M |
| **Architecture** | DETR-based (DEtection TRansformer) | Same family, larger |
| **Task** | Object detection (COCO 80-class) | Same |
| **Input** | Image (dynamic resolution) | Same |
| **Output** | Bounding boxes + class labels | Same |
| **License** | **Apache-2.0** | **Apache-2.0** |
| **HF Repo** | `keras/dfine_nano_coco` | `keras/dfine_small_coco` |
| **Key feature** | Fine-grained Distribution Refinement (FDR) | Same |

## Architecture

D-FINE is a DETR-family model that redefines bounding box regression as **Fine-grained Distribution Refinement (FDR)** — instead of predicting fixed coordinates, it iteratively refines probability distributions for precise localization.

Components:
- **Backbone**: HGNetV2 (lightweight CNN backbone)
- **Encoder**: Transformer encoder with multi-scale features
- **Decoder**: Transformer decoder with FDR refinement heads
- **Head**: Distribution-based bounding box regression

## ONNX Export

### Availability
- Keras provides these models — need to check if they ship pre-exported ONNX
- If not, export via:
  ```bash
  optimum-cli export onnx --model keras/dfine_nano_coco --task object-detection dfine_nano_onnx/
  ```
- Alternative: Convert Keras model → PyTorch → ONNX (if needed)

### Estimated ONNX Sizes
- D-FINE Nano: ~15 MB (3.79M params × 4 bytes FP32)
- D-FINE Small: ~40 MB (10.33M params × 4 bytes FP32)

### ONNX Op Compatibility with ggonnx
- **Transformer ops**: MatMul, Add, Softmax, LayerNorm — standard, likely supported
- **CNN backbone**: Conv2d, BatchNorm, ReLU — standard, likely supported
- **FDR heads**: Custom distribution refinement — may use Gather, Scatter, or custom ops
- **Risk**: DETR models use bipartite matching and Hungarian loss during training, but inference is simpler (just feed-forward + NMS)

---

## Porting Steps

### Step 1: Export to ONNX

```bash
# Install optimum
pip install optimum[exporters]

# Export D-FINE Nano
optimum-cli export onnx \
  --model keras/dfine_nano_coco \
  --task object-detection \
  dfine_nano_onnx/

# Validate
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('dfine_nano_onnx/model.onnx')
inputs = {inp.name: np.random.randn(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32) for inp in sess.get_inputs()}
outputs = sess.run(None, inputs)
print('Output shapes:', [o.shape for o in outputs])
"
```

### Step 2: Check ONNX Op Compatibility

```bash
# Use ggonnx tools to probe supported ops
python ported_models/ggonnx/src/ggonnx/tools/node_discovery.py \
  dfine_nano_onnx/model.onnx

# Compare against supported ops
python ported_models/ggonnx/src/ggonnx/tools/op_report.py \
  dfine_nano_onnx/model.onnx
```

### Step 3: Pin References & Add to `artifacts.json`

```json
"dfine_nano": {
  "kind": "model",
  "framework": "ggonnx",
  "filename": "dfine_nano.onnx",
  "source_url": "<direct-download-url or HF resolve link>",
  "size_bytes": 15000000,
  "validation": "compare_outputs_to_onnxruntime_cpu"
},
"dfine_small": {
  "kind": "model",
  "framework": "ggonnx",
  "filename": "dfine_small.onnx",
  "source_url": "<direct-download-url>",
  "size_bytes": 40000000,
  "validation": "compare_outputs_to_onnxruntime_cpu"
}
```

### Step 4: Update `docs/HF_REFERENCES.md`

```markdown
| `dfine_nano` | `keras/dfine_nano_coco` | `<SHA>` | `apache-2.0` | `model.onnx` |
| `dfine_small` | `keras/dfine_small_coco` | `<SHA>` | `apache-2.0` | `model.onnx` |
```

### Step 5: ggonnx Runner (Prerequisite — Not Yet Built)

The ggonnx runner infrastructure is not yet wired to CI. This port depends on:
1. Building ggonnx EP against board's GGML-ET
2. Adding a ggonnx runner (analogous to `llama_server`)
3. Registering models in `benchmark_config.json`

**Strategy**: Port the model and add it to `artifacts.json` now; the runner work is shared across all ONNX models.

---

## Comparison to Existing ggonnx Models

| Model | Params | Task | Status |
|-------|--------|------|--------|
| tiny-yolov3-11 | ~8M | Detection | Registered |
| tinyyolov2-8 | ~16M | Detection | Registered |
| resnet18-v2-7 | 11M | Classification | Registered |
| mobilenetv2-12 | 3.5M | Classification | Registered |
| **D-FINE Nano** | **3.79M** | **Detection** | **NEW — modern DETR arch** |
| **D-FINE Small** | **10.33M** | **Detection** | **NEW — better accuracy** |

D-FINE brings **modern transformer-based detection** (DETR family) to complement the existing CNN-based YOLO variants.

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Keras model may not export cleanly to ONNX | **Medium** | May need Keras → TF SavedModel → ONNX path, or PyTorch conversion |
| DETR ops (bipartite matching) in ONNX | **Low** | Matching is only for training; inference is feed-forward |
| ggonnx runner not yet built | **High** | Shared prerequisite — all ONNX models depend on this |
| FDR distribution refinement may use unsupported ops | **Medium** | Check op report; may need op-level workaround |

## Why Include This

- **Smallest detection model**: 3.79M params — tiny footprint
- **Modern architecture**: DETR family vs existing CNN-based YOLO
- **FDR innovation**: Distribution-based box regression is more precise
- **Apache-2.0**: Clean license
- **Fills detection diversity gap**: Complements existing YOLO variants
