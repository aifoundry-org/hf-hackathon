# Porting Plan: RF-DETR Small (ONNX Object Detection)

**Path:** ONNX (ggonnx bridge) — Pure vision (object detection)
**Priority:** #10 — Modern DETR architecture, NAS-optimized
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 32.1M |
| **Architecture** | Real-Time Detection Transformer (NAS-optimized) |
| **Backbone** | DINOv2-with-registers style ViT (windowed/full attention alternation) |
| **Decoder** | Deformable DETR with multi-scale deformable cross-attention |
| **Task** | Object detection (COCO 80-class) |
| **License** | **Apache-2.0** |
| **HF Repo** | `Roboflow/rf-detr-small` |
| **Paper** | arXiv:2511.09554 (Nov 2025) |

## Architecture Detail

RF-DETR combines ideas from LW-DETR and Deformable DETR:

1. **Backbone**: DINOv2-with-registers style ViT
   - Alternates between windowed attention and full attention (RF-DETR windowing pattern)
   - More efficient than purely full attention

2. **Multi-scale projector**: C2f-style blocks (LW-DETR lineage) between encoder and decoder

3. **Decoder**: Deformable DETR-style with multi-scale deformable cross-attention
   - Depth and resolution vary by checkpoint (NAS frontier)

4. **Queries**: DETR-style object queries with bipartite matching

## ONNX Export

### Direct Support
- RF-DETR is integrated in 🤗 Transformers (PR #36895)
- Should support `optimum-cli export onnx`

```bash
optimum-cli export onnx \
  --model Roboflow/rf-detr-small \
  --task object-detection \
  rf_detr_small_onnx/
```

### Estimated ONNX Size
- ~120 MB (32.1M params × 4 bytes FP32)
- INT8 quantized: ~32 MB

### ONNX Op Compatibility Concerns
- **Deformable attention**: Uses `MultiScaleDeformableAttention` — custom CUDA op in PyTorch. ONNX export may use a decomposed version with standard ops (Gather, Scatter, bilinear interpolation).
- **ViT backbone**: Standard self-attention, LayerNorm, FFN — well-supported
- **Risk**: Deformable attention decomposition may produce a large ONNX graph with many nodes

---

## Porting Steps

### Step 1: Export to ONNX

```bash
pip install transformers optimum
optimum-cli export onnx \
  --model Roboflow/rf-detr-small \
  --task object-detection \
  rf_detr_small_onnx/

# Check model
python -c "
import onnx
model = onnx.load('rf_detr_small_onnx/model.onnx')
print(f'Nodes: {len(model.graph.node)}')
ops = set(n.op_type for n in model.graph.node)
print(f'Op types: {sorted(ops)}')
print(f'Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]}')
"
```

### Step 2: Check ggonnx Compatibility

```bash
python ported_models/ggonnx/src/ggonnx/tools/op_report.py \
  rf_detr_small_onnx/model.onnx
```

Focus on:
- Are deformable attention ops supported?
- How many unique op types?
- Any custom/non-standard ops?

### Step 3: Add to `artifacts.json`

```json
"rf_detr_small": {
  "kind": "model",
  "framework": "ggonnx",
  "filename": "rf-detr-small.onnx",
  "source_url": "<direct-download-url>",
  "size_bytes": 120000000,
  "validation": "compare_outputs_to_onnxruntime_cpu"
}
```

### Step 4-5: Standard registration (depends on ggonnx runner)

---

## Comparison to Existing Detection Models

| Model | Params | Architecture | COCO mAP (est.) |
|-------|--------|-------------|-----------------|
| tiny-yolov2-8 | ~16M | CNN (YOLO v2) | ~29 |
| tiny-yolov3-11 | ~8M | CNN (YOLO v3) | ~33 |
| yolov10n (hand-written C) | 2.76M | CNN (YOLOv10) | ~38 |
| **D-FINE Nano** | **3.79M** | **DETR** | **~42** |
| **RF-DETR Small** | **32.1M** | **DETR + ViT** | **~50+** |

RF-DETR would be the highest-accuracy detection model in the ggonnx zoo, but also the largest.

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deformable attention ONNX export issues | **High** | Check if Transformers PR #36895 handles ONNX export; may need manual decomposition |
| Large model (32M params, ~120 MB ONNX) | **Medium** | Consider INT8 quantization to reduce to ~32 MB |
| Complex ONNX graph (many nodes) | **Medium** | ggonnx may have performance issues with large graphs |
| ViT backbone with windowed attention | **Low** | Standard ops, should work |

## Why Include This

- **State-of-the-art detection**: NAS-optimized DETR with strong COCO performance
- **Modern architecture**: ViT backbone + deformable attention — pushes ggonnx capabilities
- **Roboflow backing**: Active maintenance, well-documented
- **Apache-2.0**: Clean license
- **Complements D-FINE**: D-FINE Nano for tiny/fast, RF-DETR for accuracy
