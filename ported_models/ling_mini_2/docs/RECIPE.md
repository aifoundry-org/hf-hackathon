# Ling-mini-2.0 Porting Recipe

## Overview

Adds `inclusionAI/Ling-mini-2.0` (16B-total / ~1.4B-active-parameter
sparse MoE causal LM, Ant Group's Ling series) to the `llama.cpp-et`
framework. Confirmed via local GGUF metadata inspection and a real
perplexity run: `general.architecture = bailingmoe2` -- a genuinely
distinct execution family, the second generation of Ant Group's
BailingMoE architecture.

## Model Reference

- **Source**: `inclusionAI/Ling-mini-2.0` (Hugging Face), revision
  `920c3fd9916e3d5e543fc4f609e827cad8a32983`
- **License**: MIT
- **GGUF source**: `inclusionAI/Ling-mini-2.0-GGUF` (official), file
  `Ling-mini-2.0-Q8_0.gguf`
- **Quantization**: Q8_0, 17,307,656,736 bytes,
  `sha256=a87cc4911217c4ec24734959f547666d5c8d970bff3da55a14fdb818963b5f88`
  (verified locally against the downloaded file -- the first download
  attempt was truncated at ~4.05 GB by the same disk-space exhaustion
  issue documented on `lfm25_8b_a1b`; re-downloaded and re-verified
  complete and correct after fixing that)
- **Architecture**: `arch = bailingmoe2` per GGUF metadata, 278 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 12.5899 +/- 2.99657
```

This confirms MoE routing (`GGML_OP_MUL_MAT_ID`) works correctly on
`ggml-cpu` for a third distinct MoE architecture in this campaign (after
`granite_3_1b_a400m`, `lfm25_8b_a1b`).

## Why this port's ET-SoC1 kernel support is a real, open question

Same open question as every other MoE model in this campaign: CPU-backend
success does not prove the ET-SoC1 backend's `MUL_MAT_ID` implementation
handles this model's specific routing/expert-count configuration.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/ling_mini_2.json`,
  and `.github/ci/benchmark_config.json` (port 18146) -- board-testable
  now, independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/ling_mini_2.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
