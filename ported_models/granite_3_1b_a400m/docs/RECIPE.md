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

GGUF-metadata-level only (downloaded exact file, verified sha256, confirmed
`general.architecture` + tensor count) -- not a full ET sysemu load/offload
test this round. This is the first MoE architecture in this campaign, so the
ET-SoC1 kernel-support question below is a real open question, not a
formality like the dense models in this same PR.

## Open question: MoE routing op support (not confirmed either way)

MoE inference needs `GGML_OP_MUL_MAT_ID` (indexed/batched matmul against a
per-token-selected expert weight subset) in addition to the standard dense
ops. Earlier this session's op-coverage audit of `ggml-et.cpp` (see
`falcon7b_recipe.md`'s ET-op list from earlier in this campaign) found
`MUL_MAT_ID` already listed as supported -- but that finding was never
exercised against a REAL MoE model until now. This port has not yet been
loaded against the ET sysemu backend to confirm the routing path actually
works end-to-end; flagging as an open verification item rather than
asserting success.

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/granite_3_1b_a400m.json`
  is the model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- MoE routing (`MUL_MAT_ID`) not live-verified against ET sysemu -- see note
  above. Genuinely unproven, not a hedge.
