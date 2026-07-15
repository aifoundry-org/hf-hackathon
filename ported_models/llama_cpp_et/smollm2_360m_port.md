## Overview
This recipe documents the addition of the `smollm2_360m` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `unsloth/SmolLM2-360M-Instruct-GGUF`
- **Revision**: `391ed11137586e383b1be0fab9acf01d282c2e11`
- **Filename**: `SmolLM2-360M-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Steps**: As smollm2_360m_q8_gguf is already present in the ported models artifacts, just ran the preflight and soc3 benchmark. Detailed steps below

## Steps to reproduce
1. From the hackathon root repo .github/ci/platform/deploy/install-soc3-ssh-key.sh
2. MODELS="smollm2_360m"   .github/ci/platform/deploy/soc3-benchmark.sh