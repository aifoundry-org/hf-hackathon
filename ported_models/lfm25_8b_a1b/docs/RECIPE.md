# LFM2.5-8B-A1B Porting Recipe

## Overview

Adds `LiquidAI/LFM2.5-8B-A1B` (8B-total / ~1B-active-parameter sparse MoE,
Liquid AI's hybrid conv+attention+MoE architecture) to the `llama.cpp-et`
framework. Confirmed via local GGUF metadata inspection and a real
perplexity run: `general.architecture = lfm2moe` -- distinct from `lfm2`
(already a seed identity on this board), the MoE variant of Liquid's
architecture.

## Model Reference

- **Source**: `LiquidAI/LFM2.5-8B-A1B` (Hugging Face), revision
  `5673e0de372b64331504de73bbbc33b0dde71903`
- **License**: LFM Open License v1.0 (custom, non-standard-OSI license
  from the source repo's own tag; flagged honestly)
- **GGUF source**: `LiquidAI/LFM2.5-8B-A1B-GGUF` (official), file
  `LFM2.5-8B-A1B-Q8_0.gguf`
- **Quantization**: Q8_0, 9,010,195,680 bytes,
  `sha256=33ab3b8ce6a964fb8ebac89360c9b3cf72c4fa418d5e4c0a94d46883124d5c02`
  (verified locally against the downloaded file -- the first download
  attempt was truncated at ~2.9 GB by a disk-space exhaustion issue
  discovered and fixed this session; re-verified complete and correct
  after fixing that and re-downloading)
- **Architecture**: `arch = lfm2moe` per GGUF metadata, 256 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly, allocating both a small transformer KV cache (6 layers) and a
recurrent conv-state cache (24 layers) simultaneously -- confirms this is
a genuine hybrid conv+attention+MoE model, not mislabeled:

```
Final estimate: PPL = 29.8771 +/- 6.77437
```

This is the second sparse-MoE model in this campaign (after
`granite_3_1b_a400m`) and the fourth hybrid-recurrent model (after
`mamba_1_4b`, `falcon_h1_1_5b`, `granite4_h_micro`) to confirm its
respective op set works on `ggml-cpu`.

## Why this port's ET-SoC1 kernel support is a real, open question

Combines two independently-open questions from this campaign: MoE routing
(`MUL_MAT_ID`, open since `granite_3_1b_a400m`) and recurrent/conv state
ops (`SSM_CONV`-family, open since `mamba_1_4b`). Neither has been checked
against the real ET sysemu backend or board.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/lfm25_8b_a1b.json`,
  and `.github/ci/benchmark_config.json` (port 18144) -- board-testable
  now, independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/lfm25_8b_a1b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- Custom (non-OSI) license -- flagged for maintainer awareness.
