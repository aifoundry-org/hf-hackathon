# EfficientViT — Porting Notes

## Hugging Face / GitHub Reference

| Field | Value |
|-------|-------|
| **Source** | https://github.com/mit-han-lab/efficientvit |
| **License** | Apache-2.0 |
| **Paper** | https://arxiv.org/abs/2205.14756 |
| **HF Models** | `timm/efficientvit_b0.r224_in1k`, `timm/efficientvit_b1.r224_in1k`, etc. |
| **Note** | No pre-exported ONNX on HuggingFace; export required from PyTorch |

## Architecture

EfficientViT is a family of lightweight vision transformers with **multi-scale attention** and **cascaded efficient blocks**. Designed for edge deployment with high accuracy-efficiency trade-offs.

### Classification Variants (M-series)

| Variant | Params | ImageNet Top-1 | Est. ONNX Size |
|---------|--------|----------------|----------------|
| **M0** | ~7M | 72.1% | ~28 MB |
| M2 | ~14M | 74.3% | ~56 MB |
| M4 | ~30M | 76.1% | ~120 MB |

### Segmentation Variants (SAM-series)

EfficientViT-SAM accelerates Segment Anything Model (SAM) with efficient encoders:

| Variant | Speedup vs SAM-ViT-H | Accuracy | Est. ONNX Size |
|---------|---------------------|----------|----------------|
| L0 | 84× | Matched | ~80 MB |
| **L2** | 48.9× | Matched | ~128 MB |
| XL0 | 21× | Matched | ~200 MB |

**Key innovations:**
- Lightweight multi-scale attention mechanism
- Cascaded efficient blocks for progressive feature refinement
- Hardware-friendly operations (standard Conv, MatMul, LayerNorm)
- Two-component SAM architecture: image encoder + mask decoder

## ONNX Export Status

**Status: ONNX export required from PyTorch**

No pre-built ONNX files are available. Export must be performed from the official PyTorch implementation.

### Classification Export (M0)

```python
import torch
from efficientvit.models.efficientvit import efficientvit_m0

# Load pretrained model
model = efficientvit_m0(pretrained=True)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'efficientvit_m0.onnx',
    opset_version=17,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}
)
```

### SAM Export (Two-Component)

EfficientViT-SAM requires exporting two separate models:

```python
import torch
from efficientvit.models.efficientvit import efficientvit_sam_l2

# Load pretrained SAM model
model = efficientvit_sam_l2(pretrained=True)
model.eval()

# Export image encoder
encoder = model.image_encoder
dummy_image = torch.randn(1, 3, 1024, 1024)
torch.onnx.export(
    encoder,
    dummy_image,
    'efficientvit_sam_l2_encoder.onnx',
    opset_version=17,
    input_names=['image'],
    output_names=['image_embeddings']
)

# Export mask decoder
decoder = model.mask_decoder
# Dummy inputs for decoder (prompt embeddings)
dummy_embeddings = torch.randn(1, 256, 64, 64)
dummy_points = torch.randn(1, 2, 2)  # point prompts
torch.onnx.export(
    decoder,
    (dummy_embeddings, dummy_points),
    'efficientvit_sam_l2_decoder.onnx',
    opset_version=17,
    input_names=['image_embeddings', 'point_coords'],
    output_names=['masks', 'iou_predictions']
)
```

**Note**: SAM export is more complex due to prompt-based decoding. Verify input/output signatures match your use case.

## Expected ONNX Characteristics

### Input Tensor (Classification)

- **Shape**: `[1, 3, 224, 224]` (NCHW format)
- **Dtype**: `float32`
- **Content**: RGB image normalized with ImageNet mean/std

### Output Tensor (Classification)

- **Shape**: `[1, 1000]` (ImageNet 1K classes)
- **Dtype**: `float32`
- **Content**: Raw logits (apply softmax externally)

### Expected ONNX Ops

EfficientViT uses standard operations well-supported by ggonnx:
- **Conv**: Depthwise and pointwise convolutions
- **MatMul**: Attention projections
- **Softmax**: Attention weights
- **LayerNorm**: Normalization layers
- **Reshape/Transpose**: Tensor manipulations
- **Add**: Residual connections

**Recommendation**: Inspect exported ONNX with Netron to verify all ops are supported by ggonnx ET backend.

## Comparison to Existing Classification Models

| Model | Params | Architecture | Task | ONNX Size | Status |
|-------|--------|--------------|------|-----------|--------|
| `resnet18-v2-7` | ~11.7M | ResNet | Classification | ~47 MB | Supported |
| `mobilenetv2-12` | ~3.5M | MobileNetV2 | Classification | ~14 MB | Supported |
| `shufflenet-v2-10` | ~2.3M | ShuffleNetV2 | Classification | ~9 MB | Supported |
| **efficientvit_m0** | ~7M | EfficientViT | Classification | ~28 MB (est.) | Export pending |

EfficientViT-M0 fills a gap between lightweight CNNs (MobileNet, ShuffleNet) and heavier transformers, offering better accuracy at moderate compute cost.

## ggonnx Runner Integration

**Status: ggonnx runner not yet wired to CI**

This is a **preparatory registration**. The ggonnx framework is vendored in `ported_models/ggonnx/src/ggonnx`, but CI integration is pending.

Next steps:
1. Complete ONNX export from PyTorch
2. Validate exported ONNX against ONNX Runtime CPU
3. Add EfficientViT to ggonnx test suite
4. Wire ggonnx runner to CI pipeline
5. Benchmark on ET-SoC1 hardware

## Validation Strategy

```json
"validation": "compare_outputs_to_onnxruntime_cpu"
```

Once ONNX is exported:
1. Run inference with ONNX Runtime on CPU as baseline
2. Run inference with ggonnx on ET-SoC1
3. Compare outputs: logits (classification) or masks (segmentation) should match within tolerance
4. Metrics: `latency_ms`, `provider_coverage`, `max_abs_drift`, `max_rel_drift`

### Accuracy Validation

For classification:
- Run ImageNet validation set (50K images)
- Verify Top-1 accuracy matches PyTorch baseline (~72.1% for M0)

For segmentation:
- Run SA-1B or COCO validation subset
- Verify IoU matches SAM-ViT-H baseline

## Recommendation

**Start with EfficientViT-M0 (classification)**:
- Smallest variant, easiest to export and validate
- Standard single-model inference pipeline
- Fills accuracy gap in ggonnx classification zoo

**Stretch goal: EfficientViT-SAM-L2 (segmentation)**:
- First segmentation model in ggonnx
- Two-component architecture adds complexity
- Demonstrates SAM acceleration on edge hardware
- Higher impact if successful

## Next Steps

- [ ] Export EfficientViT-M0 ONNX from PyTorch
- [ ] Validate exported ONNX with ONNX Runtime CPU
- [ ] Inspect ONNX graph for unsupported ops
- [ ] Add EfficientViT-M0 to ggonnx test fixtures
- [ ] Benchmark latency on ET-SoC1
- [ ] Verify ImageNet Top-1 accuracy (~72.1%)
- [ ] (Stretch) Export EfficientViT-SAM-L2 encoder + decoder
- [ ] (Stretch) Validate segmentation accuracy

## References

- GitHub: https://github.com/mit-han-lab/efficientvit
- Paper: https://arxiv.org/abs/2205.14756
- HuggingFace (timm): https://huggingface.co/timm/efficientvit_b0.r224_in1k
- ggonnx upstream: https://github.com/marty1885/ggonnx
