# MiniCPM3-4B Porting Recipe

## Overview

Adds `openbmb/MiniCPM3-4B` (4B-parameter causal LM) to the `llama.cpp-et`
framework. Confirmed via local GGUF metadata inspection:
`general.architecture = minicpm3` -- distinct from `minicpm` (already claimed
elsewhere on this board, MiniCPM5-1B in this campaign's earlier batch),
llama.cpp registers MiniCPM3 as its own execution family (different
attention mechanism: MiniCPM3 uses a compressed KV / latent-attention
scheme related to DeepSeek-V2's MLA, unlike MiniCPM/MiniCPM5's plain GQA).

## Model Reference

- **Source**: `openbmb/MiniCPM3-4B` (Hugging Face), revision
  `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`
- **License**: Apache-2.0
- **GGUF source**: `QuantFactory/MiniCPM3-4B-GGUF` (official repo ships only
  fp16/q4_k_m; used this third-party Q8_0 mirror instead), file
  `MiniCPM3-4B.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=a540961f131d6ab9616b6cb1c266be86735afe84907a445ad68e1e3e9422d122`
  (verified locally against the downloaded file)
- **Architecture**: `arch = minicpm3` per GGUF metadata, 748 tensors (high
  tensor count consistent with MLA's extra compression/decompression
  projection matrices per layer).

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly on `ggml-cpu`:

```
Final estimate: PPL = 11.8752 +/- 2.29542
```

This confirms MiniCPM3's MLA-style compressed attention (low-rank KV
down/up-projection) works correctly through the CPU backend.

## Why this port's ET-SoC1 kernel support is still an open question

CPU-backend success does not prove the ET-SoC1 backend's own `MUL_MAT` chain
handles the same low-rank-compression graph shape correctly -- that has not
been checked against the real ET sysemu backend or board. Flagging as
genuinely unconfirmed on ET specifically, not a formality.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/minicpm3_4b.json`,
  and `.github/ci/benchmark_config.json` (port 18134) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/minicpm3_4b.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- MLA attention path confirmed on CPU, NOT live-verified against ET sysemu
  specifically.
