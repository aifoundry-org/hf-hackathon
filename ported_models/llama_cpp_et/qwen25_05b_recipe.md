# Qwen2.5-0.5B-Instruct Porting Recipe

## Overview
This recipe documents the addition of the `Qwen2.5-0.5B-Instruct` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- **Revision**: `main`
- **Filename**: `qwen2.5-0.5b-instruct-q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Identified Base Model**: We chose the lightweight and highly performant Qwen2.5 0.5B Instruct model. Its extremely small size makes it an excellent fit for the ET-SoC1 memory constraints while retaining strong generative capabilities.
2. **Updated `artifacts.json`**:
   Added `qwen25_05b_q8_gguf` to `ported_models/llama_cpp_et/artifacts.json`, supplying the exact URL and SHA256 checksum.
3. **Created Benchmark Configuration**:
   Added `ported_models/llama_cpp_et/benchmarks/qwen25_05b.json` to define the decoding performance test using `llama-server`.
4. **Registered Benchmark**:
   Added the benchmark config mapping to `.github/ci/benchmark_config.json` under the `"models"` block.

## Instructions for Reproduction
No custom model packing or quantization was required, as the model was already available in Q8_0 GGUF directly from the Qwen Hugging Face organization. The board CI will automatically download the GGUF file from Hugging Face based on the SHA256 and URL in `artifacts.json` and run the `llama-server` binary using the provided configuration.
