# Porting Plan: TinyViT (ONNX Image Classification)

**Path:** ONNX (ggonnx bridge) — Pure vision (image classification)
**Priority:** #6 — Easiest ggonnx port, already has ONNX with INT8
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 21M |
| **Architecture** | TinyViT (Microsoft Cream) |
| **Task** | Image classification (ImageNet-1K, 1000 classes) |
| **Input** | 224×224 RGB image |
| **Output** | 1000-class logits |
| **License** | **MIT** |
| **HF Repo** | `Blaize-AI/TinyVit_ImageNet` (ONNX format) |
| **Source** | `github.com/microsoft/Cream/tree/main/TinyViT` |

## Available Formats

The `Blaize-AI/TinyVit_ImageNet` repo already provides:

| Format | Description | Use |
|--------|-------------|-----|
| **ONNX (FP32)** | Full precision ONNX model | Baseline |
| **INT8** | Integer quantized | Fastest inference, smallest size |
| **BF16** | BFloat16 | High precision |
| **AMP** | Mixed INT8 + BF16 | Balance |

This means **no conversion needed** — the ONNX model is ready to download and use.

## ONNX Sizes (estimated)

| Format | Estimated Size |
|--------|---------------|
| FP32 | ~84 MB (21M × 4 bytes) |
| INT8 | ~21 MB (21M × 1 byte) |
| BF16 | ~42 MB (21M × 2 bytes) |

## Architecture

TinyViT is a lightweight Vision Transformer from Microsoft's Cream (Conditional Regression and Attention Model) framework:

- **Patch embedding**: Conv-based stem (efficient)
- **Transformer blocks**: MobileViT-style with efficient attention
- **Classification head**: Linear layer
- **Design**: Specifically optimized for edge/mobile deployment

---

## Porting Steps

### Step 1: Download ONNX Model

```bash
# From HuggingFace
huggingface-cli download Blaize-AI/TinyVit_ImageNet \
  model.onnx --local-dir tinyvit_onnx/

# Or from the model zoo directly
wget "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth"
# (would need PyTorch → ONNX conversion if using .pth)
```

### Step 2: Validate with ONNX Runtime

```bash
python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('tinyvit_onnx/model.onnx')
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f'Input: {input_name}, shape: {input_shape}')

# Test inference
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = sess.run(None, {input_name: dummy})
print(f'Output shape: {outputs[0].shape}')
print(f'Top-5 classes: {np.argsort(outputs[0][0])[-5:]}')
"
```

### Step 3: Check ggonnx Op Compatibility

```bash
python ported_models/ggonnx/src/ggonnx/tools/op_report.py \
  tinyvit_onnx/model.onnx
```

Expected ops: Conv, MatMul, Add, Softmax, LayerNorm, GELU/ReLU, Reshape, Transpose — all standard.

### Step 4: Add to `artifacts.json`

```json
"tinyvit_21m": {
  "kind": "model",
  "framework": "ggonnx",
  "filename": "tinyvit_21m_22kto1k.onnx",
  "source_url": "https://huggingface.co/Blaize-AI/TinyVit_ImageNet/resolve/main/model.onnx",
  "size_bytes": 84000000,
  "validation": "compare_outputs_to_onnxruntime_cpu"
}
```

### Step 5: Update `docs/HF_REFERENCES.md`

```markdown
| `tinyvit` | `Blaize-AI/TinyVit_ImageNet` | `<SHA>` | `mit` | `model.onnx` |
```

### Step 6: Depends on ggonnx runner

Same as all ONNX models — add to `benchmark_config.json` once the runner is wired.

---

## Comparison to Existing ggonnx Classification Models

| Model | Params | Architecture | ImageNet Top-1 |
|-------|--------|-------------|----------------|
| mobilenetv2-12 | 3.5M | CNN (MobileNetV2) | 71.8% |
| resnet18-v2-7 | 11M | CNN (ResNet18) | 71.5% |
| shufflenet-v2-10 | 2.3M | CNN (ShuffleNetV2) | 69.4% |
| **TinyViT 21M** | **21M** | **ViT (transformer)** | **~83%** |

TinyViT would be the **highest-accuracy classification model** in the ggonnx zoo, and the first transformer-based classifier.

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Blaize-specific ONNX optimizations may not be standard | **Low** | The model is a standard TinyViT; any non-standard ops would show in op report |
| 21M params is larger than existing classification models | **Low** | Still very manageable; ~84 MB ONNX is small |
| ggonnx runner not yet built | **High** | Shared prerequisite |

## Why Include This

- **Easiest ONNX port**: Already in ONNX format, no conversion needed
- **Highest classification accuracy**: ~83% ImageNet Top-1 vs ~72% for existing CNN models
- **MIT license**: Most permissive
- **First ViT in ggonnx**: Brings transformer-based classification to the zoo
- **INT8 variant available**: Can demonstrate quantized inference on ET-SoC1
- **Well-documented**: Microsoft Cream project, academic paper
