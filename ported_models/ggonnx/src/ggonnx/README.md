# ggonnx

Out-of-tree ONNX Runtime Execution Provider that lowers ONNX graph partitions onto GGML.

> [!NOTE]
> Running on GGML backends does not automatically mean high performance. Espicially as GGML is focused on transformers and less on Conv nets. However, this porject still provides a viable path for custom hardware already with GGML support to run ONNX models without modification.

## Build and testing

```bash
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_ROOT=/path/to/your/local/llama.cpp
make -j
```

To test

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 # Makes things faster
python -m pytest tests/test_runtime.py --ep-library build/libggonnx_ep.so
python -m pytest tests/test_integration.py --ep-library build/libggonnx_ep.so
python -m pytest tests/test_onnx_node_conformance.py --ep-library build/libggonnx_ep.so
```

By default the Execution Provider will attempt to initialize and use GGML GPU devices before falling back to GGML's CPU backend and ONNX's other backends.

## Developer notes

Convolutional networks is really a 2nd order concern in GGML. And GGML has different semantics then ONNX. Namely quantization works differently.. ONNX wants input and output be the same, both FP32 or both FP16. GGML almost always wants input and output be FP32 and weight be whatever. Making supporting FP16 models on `ggonnx` quite annoyning and hard.

On the same note, lots of GGML backends does not have a `CONV_2D` operator implementation and will either fallback to IM2COL (very slow) or fallback to the CPU for IM2COL and device for MUL_MAT. This is very slow. For `ggonnx` to work well, backends must implement their own `CONV_2D` and `PAD` operators.. and certain fusions that is common in CNNs like `CONV_2D + UNARY`.

## Roadmap

Grunt work (upstream work in GGML needed):

- [ ] Native Erf operator (Swin UNet needes them, right now implemented as `ggml_custom_map1`)
- [ ] Map `Attention` (Opset 23) to GGML_FLASH_ATTN_EXT
- [ ] Test Clip models

Experiment needed:

- [ ] Automated quantization for `Matmul` and `Gemm` operator on device
- [ ] Test newer YOLO models
