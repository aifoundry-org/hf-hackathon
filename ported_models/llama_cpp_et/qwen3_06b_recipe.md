# Qwen3-0.6B Porting Recipe

## Overview
This recipe documents the addition of the `Qwen3-0.6B` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `ggml-org/Qwen3-0.6B-GGUF`
- **Revision**: `a41486f827d17edd055fe6b3b0ba3f8d427c0519`
- **Filename**: `Qwen3-0.6B-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Identified Base Model**: We selected the newly optimized Qwen3 0.6B model. Its compact architecture and lightweight footprint make it perfect for efficient deployment and benchmarking under the ET-SoC1 constraints.
2. **Updated `artifacts.json`**:
   Verified that `qwen3_06b_q8_gguf` is properly defined in `ported_models/llama_cpp_et/artifacts.json` with the exact upstream source, revision, and SHA256 checksum.
3. **Created Benchmark Configuration**:
   Added `ported_models/llama_cpp_et/benchmarks/qwen3_06b.json` to define the decoding performance test parameters using `llama-server`.
4. **Registered Benchmark**:
   Added the benchmark config mapping to `.github/ci/benchmark_config.json` under the `"models"` block.

## Instructions for Reproduction
No custom model packing or quantization was required, as the model was already provided in the Q8_0 GGUF format via the `ggml-org` Hugging Face repository. The board CI will automatically download the GGUF asset based on the URL and hash requirements in `artifacts.json` and execute the benchmark natively on the hardware target.