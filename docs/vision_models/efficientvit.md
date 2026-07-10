# Porting Plan: EfficientViT (ONNX Classification + Segmentation)

**Path:** ONNX (ggonnx bridge) — Pure vision (classification + segmentation)
**Priority:** #11 — MIT Han Lab, dual-purpose (classification + segmentation via SAM)
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | Varies by variant (see below) |
| **Architecture** | EfficientViT (lightweight multi-scale attention) |
| **Tasks** | Image classification, semantic segmentation, SAM acceleration |
| **License** | **Apache-2.0** |
| **Source** | `github.com/mit-han-lab/efficientvit` |
| **HF Repos** | Various community repos |

## Available Variants

### Classification Models

| Variant | Params | ImageNet Top-1 | Latency (A100) |
|---------|--------|----------------|-----------------|
| EfficientViT-M0 | ~7M | 72.1% | ~1.5 ms |
| EfficientViT-M1 | ~10M | 73.5% | ~1.8 ms |
| EfficientViT-M2 | ~14M | 74.3% | ~2.1 ms |
| EfficientViT-M3 | ~20M | 75.2% | ~2.7 ms |
| EfficientViT-M4 | ~30M | 76.1% | ~3.5 ms |
| EfficientViT-M5 | ~42M | 76.8% | ~4.8 ms |

### EfficientViT-SAM (Segmentation)

| Variant | Speedup vs SAM-ViT-H | COCO mAP |
|---------|---------------------|----------|
| L0 | 84× | ~39 |
| L1 | 58× | ~41 |
| L2 | 48.9× | ~42.7 (matched SAM accuracy) |
| XL0 | 21× | ~43 |
| XL1 | 16.5× | ~43.5 |

## Architecture

EfficientViT uses **lightweight multi-scale attention** designed for high-resolution dense prediction:

- **Backbone**: Cascaded efficient blocks with multi-scale feature extraction
- **Attention**: Lightweight multi-head attention with reduced computational cost
- **Head**: Task-specific (classification, segmentation, or SAM encoder replacement)
- **Key innovation**: Achieves ViT-level accuracy with CNN-level efficiency

## ONNX Export

### Classification

```bash
# From PyTorch (mit-han-lab)
pip install efficientvit
python -c "
from efficientvit.models.efficientvit import efficientvit_m0
model = efficientvit_m0(pretrained=True)
import torch
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'efficientvit_m0.onnx',
                  input_names=['image'], output_names=['logits'],
                  dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}})
"
```

### Segmentation (EfficientViT-SAM)

```bash
# Export SAM encoder + mask decoder
from efficientvit.sam_model.efficientvit_sam import build_efficientvit_sam_l0
model = build_efficientvit_sam_l0(pretrained=True)
# Export encoder and decoder separately for ONNX compatibility
```

### Estimated ONNX Sizes

| Variant | FP32 Size | INT8 Size |
|---------|-----------|-----------|
| M0 (7M) | ~28 MB | ~7 MB |
| M2 (14M) | ~56 MB | ~14 MB |
| M4 (30M) | ~120 MB | ~30 MB |

---

## Porting Steps

### Phase 1: Classification (Easier)

1. **Export M0 or M2 to ONNX** (smallest, most ET-SoC1-friendly)
2. **Validate with ONNX Runtime**
3. **Check ggonnx op compatibility**
4. **Add to `artifacts.json`**

### Phase 2: Segmentation (Stretch Goal)

1. **Export EfficientViT-SAM L2** (balanced speed/accuracy)
2. **Split into encoder + decoder ONNX models** (SAM has two components)
3. **Validate each component separately**
4. **Add both to `artifacts.json`**

### Step 1: Export Classification Model

```bash
pip install efficientvit torch onnx

python -c "
from efficientvit.models.efficientvit import efficientvit_m0
import torch
model = efficientvit_m0(pretrained=True).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'efficientvit_m0.onnx',
                  input_names=['image'], output_names=['logits'],
                  opset_version=17)
"
```

### Step 2: Validate

```bash
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('efficientvit_m0.onnx')
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f'Output shape: {outputs[0].shape}')
print(f'Top-5: {np.argsort(outputs[0][0])[-5:]}')
"
```

### Step 3: Add to `artifacts.json`

```json
"efficientvit_m0": {
  "kind": "model",
  "framework": "ggonnx",
  "filename": "efficientvit_m0.onnx",
  "source_url": "<upload-to-HF-or-direct-link>",
  "size_bytes": 28000000,
  "validation": "compare_outputs_to_onnxruntime_cpu"
}
```

---

## Comparison to Existing ggonnx Models

| Model | Params | Task | Notes |
|-------|--------|------|-------|
| mobilenetv2-12 | 3.5M | Classification (CNN) | Existing |
| resnet18-v2-7 | 11M | Classification (CNN) | Existing |
| shufflenet-v2-10 | 2.3M | Classification (CNN) | Existing |
| **EfficientViT-M0** | **7M** | **Classification (ViT)** | **NEW** |
| **EfficientViT-SAM-L2** | **~30M** | **Segmentation** | **NEW — first segmentation model** |

EfficientViT would be:
1. The first **transformer-based classifier** in ggonnx (alongside TinyViT)
2. The first **segmentation model** in ggonnx (EfficientViT-SAM)

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| No pre-exported ONNX on HF | **Medium** | Export from PyTorch (mit-han-lab provides clean code) |
| SAM export complexity (two-component model) | **Medium** | Phase 2 stretch goal; start with classification |
| ggonnx runner not yet built | **High** | Shared prerequisite |
| EfficientViT attention mechanism may use uncommon ONNX ops | **Medium** | Check op report after export |

## Why Include This

- **Dual-purpose**: Classification AND segmentation from one architecture family
- **MIT Han Lab pedigree**: Well-known efficient vision research group
- **Apache-2.0**: Clean license
- **SAM acceleration**: EfficientViT-SAM brings segment-anything capability at 48.9× speedup
- **First segmentation model**: Fills a gap in the ggonnx zoo (no UNet, DeepLab, etc.)
- **Edge-optimized**: Designed specifically for efficient deployment
- **Scalable**: M0-M5 variants let you choose size/accuracy tradeoff
