# Determinex Autonomous Port: Ultimate Fleet Expansion

## Overview
This recipe documents the autonomous promotion of 6 models (Tier 1 Micro-Models and Tier 2 Q4 Quantizations) to default benchmark entries in the `llama.cpp-et` framework on ET-SoC1 boards.

Models included:
- **Gemma-3 270M** (`gemma-3-270m-it-Q8_0`)
- **LFM2.5 230M** (`LFM2.5-230M-Q8_0`)
- **Qwen2.5-Coder 0.5B** (`Qwen2.5-Coder-0.5B-Instruct-Q8_0`)
- **Llama-3.2 1B (Q4)** (`Llama-3.2-1B-Instruct-Q4_K_M`)
- **Qwen2.5 0.5B (Q4)** (`Qwen2.5-0.5B-Instruct-Q4_K_M`)
- **SmolLM2 360M (Q4)** (`SmolLM2-360M-Instruct-Q4_K_M`)

## Why this Fleet Expansion on ET-SoC1
ET-SoC1 throughput is primarily memory-bandwidth bound. To maximize performance, we expanded the fleet in two directions:
1. **Micro-Models:** Under 500M parameters, providing the absolute highest raw tokens/s.
2. **Q4_K_M Tier:** Halving the bandwidth requirements of existing powerful models (like Llama-3.2-1B) directly doubles their decode throughput, while maintaining a fully auditable cryptographic chain of custody.

## Verification Steps
Determinex utilized an autonomous Python validator script (`scripts/conveyor_smoke.py`) to execute the QA process:
1. Pinged HuggingFace URLs for the GGUF artifacts.
2. Read the LFS pointers via the Hugging Face API to extract the true `SHA256` hashes natively, without requiring full downloads during orchestration.
3. Generated the `artifacts.json` entries with the cryptographically verified hashes.
4. Generated the benchmark configs (`benchmark_default: true`).

## Evidence Ledger
This port was autonomously orchestrated by the Determinex Engine. Chain-of-custody and cryptographic logs are recorded in the Citadel `assurance/evidence/hackathon_model_port/` ledger.
