## Overview
This recipe documents the addition of the `smollm2_135m` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `unsloth/SmolLM2-135M-Instruct-GGUF`
- **Revision**: `9e6855bc4be717fca1ef21360a1db4b29d5c559a`
- **Filename**: `SmolLM2-135M-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Steps**: As smollm2_135m_q8_gguf is already present in the ported models artifacts, just ran the preflight and soc3 benchmark. Detailed steps below

## Steps to reproduce
1. From the hackathon root repo .github/ci/platform/deploy/install-soc3-ssh-key.sh
2. MODELS="smollm2_135m"   .github/ci/platform/deploy/soc3-benchmark.sh