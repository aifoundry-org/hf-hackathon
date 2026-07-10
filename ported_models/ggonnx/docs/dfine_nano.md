# D-FINE Nano / Small — Porting Notes

## Hugging Face Reference

| Field | Value |
|-------|-------|
| **Model** | `keras/dfine_nano_coco` |
| **Revision** | `6faa0a036f923847b1053cd7c29b391c098bde2c` |
| **License** | Apache-2.0 |
| **Params** | 3.79M |
| **Weights file** | `model.weights.h5` (16.7 MB) |

| Field | Value |
|-------|-------|
| **Model** | `keras/dfine_small_coco` |
| **Revision** | `6bed36f52e1b03bdee6698c710b123f4f5ea8972` |
| **License** | Apache-2.0 |
| **Params** | 10.33M |
| **Weights file** | `model.weights.h5` (43.1 MB) |

## Architecture

D-FINE is a DETR-based object detection model with **Fine-grained Distribution Refinement (FDR)**. It uses the **HGNetV2** backbone for feature extraction.

- **Task**: Object detection (COCO 80-class)
- **Family**: D-FINE (DETR variant)
- **Backbone**: HGNetV2
- **Key innovation**: FDR improves bounding box regression by refining the distribution of predictions

## ONNX Export Status

**Status: ONNX export required**

The Hugging Face repos ship Keras `.h5` format only. No pre-built ONNX files are available. ONNX export must be performed using `optimum-cli` or custom Keras-to-ONNX conversion.

### Export Instructions

```bash
# Install dependencies
pip install optimum[exporters] tensorflow

# Export D-FINE Nano
optimum-cli export onnx \
  --model keras/dfine_nano_coco \
  --task object-detection \
  dfine_nano_onnx/

# Export D-FINE Small
optimum-cli export onnx \
  --model keras/dfine_small_coco \
  --task object-detection \
  dfine_small_onnx/
```

**Note**: If `optimum-cli` does not support D-FINE models directly, a custom export script may be needed to:
1. Load the Keras model
2. Define the forward pass with dummy inputs
3. Use `tf2onnx` or `keras2onnx` to convert

## Expected ONNX Characteristics

### Input Tensor

- **Shape**: `[1, 3, 640, 640]` (NCHW format, typical for DETR models)
- **Dtype**: `float32`
- **Content**: RGB image normalized to [0, 1] or standard ImageNet normalization

### Output Tensors

DETR-based models typically output:
- **boxes**: `[1, N, 4]` — bounding box coordinates (xyxy format)
- **scores**: `[1, N]` — confidence scores
- **labels**: `[1, N]` — class labels (0-79 for COCO)

Where `N` is the number of predicted boxes (e.g., 300 for standard DETR).

### Potential ONNX Compatibility Concerns

D-FINE uses transformer attention mechanisms and distribution refinement layers. Potential issues for ggonnx:

1. **Multi-head attention**: May require specialized ops (MatMul, Softmax, Reshape)
2. **Distribution refinement**: Custom ops or complex tensor manipulations
3. **Hungarian matching**: Post-processing may not be in ONNX graph
4. **HGNetV2 backbone**: Standard convolutions, but verify all ops are supported

**Recommendation**: After export, inspect the ONNX graph with Netron to identify unsupported ops. Compare against existing ggonnx-supported models (e.g., `tiny-yolov3`, `resnet18`) to estimate compatibility.

## Comparison to Existing Detection Models

| Model | Params | Architecture | ONNX size | Status |
|-------|--------|--------------|-----------|--------|
| `tiny-yolov3-11` | ~8.9M | YOLOv3-tiny | ~34 MB | Supported |
| `tinyyolov2-8` | ~15.5M | YOLOv2-tiny | ~62 MB | Supported |
| **dfine_nano** | 3.79M | DETR+FDR | ~17 MB (est.) | Export pending |
| **dfine_small** | 10.33M | DETR+FDR | ~43 MB (est.) | Export pending |

D-FINE Nano is significantly smaller than existing detection models, making it an attractive candidate for ET-SoC1 edge deployment.

## ggonnx Runner Integration

**Status: ggonnx runner not yet wired to CI**

This is a **preparatory registration**. The ggonnx framework is vendored in `ported_models/ggonnx/src/ggonnx`, but CI integration is pending. Next steps:

1. Complete ONNX export from Keras format
2. Validate exported ONNX against ONNX Runtime CPU
3. Add D-FINE to ggonnx test suite
4. Wire ggonnx runner to CI pipeline
5. Benchmark on ET-SoC1 hardware

## Validation Strategy

```json
"validation": "compare_outputs_to_onnxruntime_cpu"
```

Once ONNX is exported:
1. Run inference with ONNX Runtime on CPU as baseline
2. Run inference with ggonnx on ET-SoC1
3. Compare outputs: boxes, scores, labels should match within tolerance
4. Metrics: `latency_ms`, `provider_coverage`, `max_abs_drift`, `max_rel_drift`

## Next Steps

- [ ] Export ONNX from Keras using `optimum-cli` or custom script
- [ ] Validate exported ONNX with ONNX Runtime CPU
- [ ] Inspect ONNX graph for unsupported ops
- [ ] Add D-FINE to ggonnx test fixtures
- [ ] Benchmark latency on ET-SoC1
- [ ] Compare accuracy (mAP) against baseline

## References

- Hugging Face: https://huggingface.co/keras/dfine_nano_coco
- Hugging Face: https://huggingface.co/keras/dfine_small_coco
- D-FINE paper: (link to arXiv if available)
- ggonnx upstream: https://github.com/marty1885/ggonnx
