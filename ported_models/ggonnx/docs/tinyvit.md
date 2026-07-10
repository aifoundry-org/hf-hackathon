# TinyViT 21M — ImageNet-1K Classification

## HF Pin

- **Repo:** `onnx-community/tiny_vit_21m_384.dist_in22k_ft_in1k-ONNX`
- **Revision:** `1e084c5313b5ee8089a34ffaeea87dc28a742b36`
- **License:** Apache-2.0
- **Original timm model:** `timm/tiny_vit_21m_384.dist_in22k_ft_in1k`
- **Alternative:** `Blaize-AI/TinyVit_ImageNet` (Blaize `.bm` format only, revision `8eaba2e6df3e7c926a64e30076e1dfce9edb9f61`, MIT license)

## Architecture

TinyViT from [Microsoft Cream](https://github.com/microsoft/Cream/tree/main/TinyViT) is a compact
Vision Transformer that distills knowledge from larger ViT models trained on ImageNet-22k.
Key architectural features:

- **Params:** 21M
- **Distillation:** 22k-to-1k distillation for high accuracy with small footprint
- **Mechanism:** MobileViTv2-style local-enhanced Transformer blocks with efficient attention
- **Input:** `[1, 3, 384, 384]` (384×384 RGB image)
- **Output:** `[1, 1000]` logits (ImageNet-1K classes)
- **Preprocessing:** Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Available ONNX Formats

The `onnx-community` repo provides pre-exported ONNX files:

| Format | File | Size | Notes |
|--------|------|------|-------|
| FP32 | `onnx/model.onnx` | 184.5 MB | Full precision — use for initial validation |
| INT8 | `onnx/model_int8.onnx` | 121.5 MB | Quantized for board performance |
| FP16 | `onnx/model_fp16.onnx` | 92.4 MB | Half precision |
| BNB4 | `onnx/model_bnb4.onnx` | 117.6 MB | BitsAndBytes 4-bit |
| Q4 | `onnx/model_q4.onnx` | 118.8 MB | 4-bit quantized |
| Q4F16 | `onnx/model_q4f16.onnx` | 64.4 MB | 4-bit with FP16 scales |

**Recommendation:** Start with FP32 (`model.onnx`) for validation against ONNX Runtime CPU,
then INT8 (`model_int8.onnx`) for board performance benchmarking.

## Expected ONNX Ops

TinyViT uses standard ONNX ops:

- **Conv** — patch embedding and convolutions
- **MatMul** — attention and MLP layers
- **LayerNormalization** — transformer blocks
- **GELU** — activation function
- **Softmax** — attention and final classification
- **Add/Multiply** — residual connections and scaling

All ops are standard and should be supported by ggonnx without custom node implementations.

## Comparison to Existing Classification Models

| Model | Top-1 Acc | Params | Input Size |
|-------|-----------|--------|------------|
| `resnet18` | 71.5% | 11.7M | 224×224 |
| `mobilenet_v2` | 71.8% | 3.5M | 224×224 |
| `shufflenet_v2` | 69.4% | 2.3M | 224×224 |
| **TinyViT 21M** | **~83%** | **21M** | **384×384** |
| `efficientvit_m0` | 72.1% | 7M | 224×224 |

**TinyViT at ~83% Top-1 would be the highest-accuracy classifier in the ggonnx zoo**, a significant
improvement over the CNN-based models currently registered.

## CI Status

The ggonnx runner is not yet wired to CI. This is a **preparatory registration** — the artifact
entry documents the source and validation method for when CI integration is complete.

## Notes

- Input resolution is 384×384 (not 224×224 as in the Blaize variant). This is the standard
  `tiny_vit_21m_384` timm variant.
- The Blaize-AI HF repo only contains `.bm` (Blaize binary format) files, not ONNX. Use the
  `onnx-community` repo for actual ONNX artifacts.
- The model was distilled from ImageNet-22k, giving it higher accuracy than models trained
  only on ImageNet-1K.
