# Granite-4.0-H-Micro Porting Recipe

## Overview

Adds `ibm-granite/granite-4.0-h-micro` (IBM's fourth-generation Granite,
hybrid attention+Mamba architecture) to the `llama.cpp-et` framework.
Confirmed via local GGUF metadata inspection: `general.architecture =
granitehybrid` -- distinct from both `granite` (dense, already ported
elsewhere on this board) and `granitemoe` (MoE, already ported earlier in
this campaign as `granite_3_1b_a400m`). A third, genuinely different
Granite execution family.

## Model Reference

- **Source**: `ibm-granite/granite-4.0-h-micro` (Hugging Face), revision
  `d5f01a3ea75f088947be3aae039f4ad52837dfde`
- **License**: Apache-2.0
- **GGUF source**: `ibm-granite/granite-4.0-h-micro-GGUF` (official), file
  `granite-4.0-h-micro-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=a009111abf2865b7aad1e66326a6c772cddc29bccd22898f470292068b27bb59`
  (verified locally against the downloaded file)
- **Architecture**: `arch = granitehybrid` per GGUF metadata, 506 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly, allocating **both** a small transformer KV cache (`llama_kv_cache`,
only 4 layers) **and** a much larger recurrent SSM state cache
(`llama_memory_recurrent`, 40 layers) -- a hybrid weighted heavily toward
recurrent layers (4 attention : 40 recurrent), a different ratio from
`falcon_h1_1_5b`'s hybrid mix earlier in this campaign:

```
Final estimate: PPL = 13.2058 +/- 2.78765
```

This is a third data point (after `mamba_1_4b`, `falcon_h1_1_5b`) on
`SSM_CONV`/`SSM_SCAN` working correctly on `ggml-cpu`.

## Why this port's ET-SoC1 kernel support is a real, open question

Same caveat as `mamba_1_4b`/`falcon_h1_1_5b`: CPU success doesn't prove
ET-SoC1 support for `SSM_CONV`/`SSM_SCAN`. Three independent hybrid/SSM
models now share this same open question in this campaign.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/granite4_h_micro.json`,
  and `.github/ci/benchmark_config.json` (port 18142) -- board-testable
  now, independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/granite4_h_micro.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- SSM ops confirmed on CPU, NOT live-verified against ET sysemu
  specifically.
