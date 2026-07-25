# Granite-3.0-1B-A400M Porting Recipe

## Overview

Adds `ibm-granite/granite-3.0-1b-a400m-instruct` (1B-total-parameter /
400M-active-parameter sparse Mixture-of-Experts causal LM) to the
`llama.cpp-et` framework. Confirmed via local GGUF metadata inspection:
`general.architecture = granitemoe` -- distinct from IBM's dense `granite`
architecture (already claimed elsewhere on this board), and the first MoE
model attempted in this specific porting campaign.

## Model Reference

- **Source**: `ibm-granite/granite-3.0-1b-a400m-instruct` (Hugging Face),
  revision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`
- **License**: Apache-2.0
- **GGUF source**: `bartowski/granite-3.0-1b-a400m-instruct-GGUF`, file
  `granite-3.0-1b-a400m-instruct-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=8c37dd0c10b73e9304b98a242be4adcd3050c09e9042c4862d49e2cfccf35411`
  (verified locally against the downloaded file)
- **Architecture**: `arch = granitemoe` per GGUF metadata, 242 tensors
  (roughly double a similarly-sized dense model's tensor count, consistent
  with per-expert FFN weight tensors).

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly on the CPU backend:

```
Final estimate: PPL = 5.7635 +/- 0.84793
```

This confirms the MoE routing path (`GGML_OP_MUL_MAT_ID`, indexed/batched
matmul against a per-token-selected expert weight subset) works correctly
on `ggml-cpu` -- the first MoE model actually exercised in this campaign.

## Open question: MoE routing on the ET backend specifically (not confirmed)

CPU-backend success does not prove the ET-SoC1 backend's own `MUL_MAT_ID`
implementation works -- `ggml-et.cpp` lists it as supported (per the
op-coverage audit in `falcon7b_recipe.md` from earlier in this campaign),
but that has never been exercised against a real MoE model on ET sysemu or
board. Flagging as a genuinely open verification item, not a formality.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/granite_3_1b_a400m.json`,
  and `.github/ci/benchmark_config.json` (port 18132) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/granite_3_1b_a400m.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- MoE routing (`MUL_MAT_ID`) confirmed on CPU, NOT live-verified against ET
  sysemu specifically -- see note above. Genuinely unproven on ET, not a hedge.
