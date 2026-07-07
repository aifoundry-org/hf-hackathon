# DeepSeek-R1-Distill-Qwen-1.5B Porting Recipe

## Overview
This recipe documents the addition of the `DeepSeek-R1-Distill-Qwen-1.5B` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`
- **Revision**: `3cb4d15544a2a5e07439592b9a0965b6445fbd34` (main branch at the time of addition)
- **Filename**: `DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Identified Base Model**: We chose a highly popular distilled variant of DeepSeek R1 (1.5B parameters), which fits well within the ET-SoC1 memory constraints and relies on standard Qwen architecture.
2. **Updated `artifacts.json`**:
   Added `deepseek_r1_15b_q8_gguf` to `ported_models/llama_cpp_et/artifacts.json`, supplying the exact URL, SHA256 checksum (`068a721e47419ccfc94b6420118f772478544e1a0d4fad7118212774b3f9ba9e`), and byte size.
3. **Created Benchmark Configuration**:
   Added `ported_models/llama_cpp_et/benchmarks/deepseek_r1_15b.json` to define the decoding performance test. We used standard Llama.cpp backend settings (gpu_layers: 99, ubatch_size: 128) which match the other models on the board.
4. **Registered Benchmark**:
   Added the benchmark config mapping to `.github/ci/benchmark_config.json` under the `"models"` block.
5. **Validation**:
   Ran `.github/ci/scripts/ci_preflight.sh` to ensure all JSON schemas are valid and CI workflows parse correctly.

## Instructions for Reproduction
No custom model packing or quantization was required, as the model was already available in Q8_0 GGUF on Hugging Face. The board CI will automatically download the GGUF file from Hugging Face based on the SHA256 and URL in `artifacts.json` and run the `llama-server` binary using the provided prompt configuration.
