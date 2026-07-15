## Overview
This recipe documents the addition of the `smollm2_17b` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `unsloth/SmolLM2-1.7B-Instruct-GGUF`
- **Revision**: `e933f1cdf73cc87cb67915bf5dd6ea81d36080ca`
- **Filename**: `SmolLM2-1.7B-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Steps**: As smollm2_17b_q8_gguf is already present in the ported models artifacts, just ran the preflight and soc3 benchmark. Detailed steps below

## Steps to reproduce
1. From the hackathon root repo .github/ci/platform/deploy/install-soc3-ssh-key.sh
2. MODELS="smollm2_17b"   .github/ci/platform/deploy/soc3-benchmark.sh