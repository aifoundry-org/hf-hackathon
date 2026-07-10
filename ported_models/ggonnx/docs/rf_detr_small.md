# Porting Notes: RF-DETR Small (ONNX Object Detection)

**Path:** ONNX (ggonnx bridge) — Pure vision (object detection)
**Status:** ONNX export pending — preparatory registration

---

## Model Overview

| Property | Value |
|----------|-------|
| **HF Repo** | `Roboflow/rf-detr-small` |
| **Revision** | `3bdc465063270f99769da5a1b4c00c68bd2d439d` |
| **License** | **Apache-2.0** |
| **Parameters** | 32.1M |
| **Architecture** | Real-Time Detection Transformer (NAS-optimized) |
| **Task** | Object detection (COCO 80-class) |
| **Input** | Image (RGB) |
| **Output** | Bounding boxes + class labels + confidence scores |
| **Paper** | arXiv:2511.09554 (November 2025) |
| **Estimated ONNX size** | ~120 MB (FP32) |
| **Transformers integration** | PR #36895 |

## HF Repo Files

| File | Size |
|------|------|
| `config.json` | 5,771 B |
| `model.safetensors` | 128,504,496 B (~122 MB) |
| `preprocessor_config.json` | 442 B |
| `README.md` | 4,718 B |

**No pre-exported ONNX in the upstream repo.** A community export exists at `onnx-community/rfdetr_small-ONNX` (transformers.js target, 24 downloads) — not suitable for ggonnx ingestion but useful for opset reference.

## Architecture

RF-DETR is a **NAS-optimized** real-time detection transformer that represents a significant architectural advance over D-FINE and classical DETR:

- **Backbone**: DINOv2-with-registers style ViT with alternating windowed/full attention
  - Self-supervised pretraining (DINOv2) with register tokens to stabilize attention maps
  - Windowed attention in early layers for efficiency, full attention in deeper layers
- **Multi-scale projector**: C2f-style blocks (YOLOv8-inspired) that produce multi-scale feature maps
- **Decoder**: Deformable DETR with multi-scale deformable cross-attention
  - Object queries attend to multi-scale encoder outputs via deformable attention
  - Iterative bounding box refinement across decoder layers
- **Object queries**: Learned queries with bipartite matching (Hungarian algorithm at training; simple feed-forward at inference)

## ONNX Export

### Export Command

```bash
pip install optimum[exporters]

optimum-cli export onnx \
  --model Roboflow/rf-detr-small \
  --task object-detection \
  rf_detr_small_onnx/
```

If `optimum-cli` fails (model class may not be fully registered in transformers yet), alternatives:

1. **PyTorch direct export**: Load via `AutoModelForObjectDetection`, use `torch.onnx.export()` with dummy input
2. **Community ONNX**: Download from `onnx-community/rfdetr_small-ONNX` and adapt

### Expected ONNX Ops

| Op Category | Ops | ggonnx Support Risk |
|-------------|-----|---------------------|
| Standard transformer | MatMul, Add, Softmax, LayerNorm, GELU | Low — standard ops |
| Convolution (backbone) | Conv, BatchNorm, ReLU | Low — standard ops |
| Attention | Reshape, Transpose, Gather, Split | Medium |
| **Deformable attention** | GridSample / custom bilinear interpolation | **High** — may need op decomposition |
| Multi-scale projection | Resize, Concat, Slice | Medium |
| Box decoding | Add, Mul, Exp (for anchor-free decoding) | Low |

### Key Concern: Deformable Attention

Multi-scale deformable cross-attention (MSDeformAttn) samples feature maps at learned offsets using bilinear interpolation. In ONNX, this typically decomposes into:

- `GridSample` (opset 16+) — may not be supported by ggonnx
- Or a manual decomposition using `GatherND`, `Mul`, `Add` — feasible but verbose

**Mitigation**: If `GridSample` is unsupported, decompose the deformable attention module at the PyTorch level before export, or patch the ONNX graph post-export.

## Comparison to Existing Detection Models

| Model | Params | Architecture | mAP (COCO) | Status |
|-------|--------|-------------|------------|--------|
| tiny-yolov3 | ~8M | CNN (YOLO) | ~33 mAP | Registered in ggonnx |
| tinyyolov2 | ~16M | CNN (YOLO) | ~30 mAP | Registered in ggonnx |
| D-FINE Nano | 3.79M | DETR (CNN backbone) | ~42 mAP | Registered, export pending |
| D-FINE Small | 10.33M | DETR (CNN backbone) | ~48 mAP | Registered, export pending |
| **RF-DETR Small** | **32.1M** | **DETR (ViT backbone)** | **~50+ mAP** | **NEW — state-of-the-art tier** |

RF-DETR Small is the **most capable detection model** in this collection, offering transformer backbone features and NAS-optimized architecture at the cost of larger parameter count.

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deformable attention ops unsupported in ggonnx | **High** | Op decomposition or graph patching post-export |
| Model class not in transformers yet (PR #36895 pending) | **Medium** | Use community ONNX or direct PyTorch export |
| Large model (~120 MB FP32) | **Medium** | Consider FP16 or INT8 quantization post-export |
| ggonnx runner not yet built | **High** | Shared prerequisite — all ONNX models depend on this |

## Why Include This

- **State-of-the-art detection**: ~50+ mAP on COCO — significantly outperforms existing YOLO and D-FINE models
- **Modern architecture**: First ViT-backbone detection model in ggonnx — showcases transformer capabilities on ET-SoC1
- **NAS-optimized**: Demonstrates that neural architecture search produces competitive real-time models
- **Apache-2.0**: Clean license
- **Complementary to D-FINE**: D-FINE uses CNN backbone (HGNetV2); RF-DETR uses ViT — tests different op coverage

## CI Integration Note

The ggonnx runner infrastructure is **not yet wired to CI**. This is a preparatory registration — the model entry in `artifacts.json` and this document establish the porting plan. Actual board execution depends on:

1. Building ggonnx EP against board's GGML-ET
2. Adding a ggonnx runner (analogous to `llama_server`)
3. Registering models in `benchmark_config.json`
4. Completing ONNX export and validating outputs against onnxruntime CPU
