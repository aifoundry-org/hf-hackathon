# Edge-Optimized 7-Model Fleet Porting Recipe

## Overview
This recipe documents the promotion of 7 highly efficient models (all under 2B parameters) to default benchmark entries in the \llama.cpp-et\ framework on ET-SoC1 boards.

Models included:
- SmolLM2 (135M, 360M, 1.7B)
- Qwen2.5 (0.5B)
- Qwen3 (0.6B)
- TinyLlama (1.1B)
- Llama 3.2 (1B)

All models were already registered as candidate artifacts in \rtifacts.json\ and had benchmark configs under \enchmarks/\. This PR promotes them to the default CI sweep.

## Why this Fleet on ET-SoC1
These models represent the absolute best-in-class performance for their parameter size. They were specifically chosen because their small memory footprint (Q8_0 quantization) ensures they can fit entirely in the ET-SoC1 DRAM with full offload (\-ngl 99\). Expected decode throughput is significantly higher than larger models.

## Verification Steps
We utilized a custom Python validator script (\conveyor_smoke.py\) to automate the QA process prior to submission:
1. Pinged all HuggingFace URLs in \rtifacts.json\ to verify availability.
2. Downloaded the GGUF models locally.
3. Verified the SHA256 hashes of the downloaded models against the hashes in \rtifacts.json\.

All models successfully passed the hash verification, ensuring they are safe for the automated ET-SoC1 CI pipeline to process.

## Steps Executed
1. Verified all 7 artifacts via SHA256 hash matching.
2. Promoted the benchmark configs to \enchmark_default: true\.
3. Wrote this recipe documenting the workflow.
4. Preflight passed via \ash .github/ci/scripts/ci_preflight.sh\.
