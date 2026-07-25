# OpenELM-1.1B Porting Recipe

## Overview

Adds `apple/OpenELM-1_1B` (1.1B-parameter causal LM, Apple's efficient
layer-wise scaling architecture) to the `llama.cpp-et` framework. Confirmed
via local GGUF metadata inspection (`gguf.GGUFReader`): `general.architecture
= openelm` -- a genuinely distinct execution family, not registered by any
existing board identity.

## Model Reference

- **Source**: `apple/OpenELM-1_1B` (Hugging Face), revision
  `ee559a10b14895dde9f8cfde3fdc77b3ff0dbc0f`
- **License**: Apple Sample Code License (research/non-commercial terms;
  flagged honestly, matches upstream)
- **GGUF source**: `LiteLLMs/OpenELM-1_1B-Instruct-GGUF`, file
  `Q8_0/Q8_0-00001-of-00001.gguf`
- **Quantization**: Q8_0, 1,148,477,504 bytes on disk,
  `sha256=13dc6676b2355d0356d7892d8b34d8518233d79a28f9ea1d14f94e81d7decad5`
  (verified locally against the downloaded file)
- **Architecture**: `arch = openelm` per GGUF metadata (`general.architecture`),
  254 tensors, layer-wise scaled transformer (per-layer varying head count /
  FFN width -- Apple's DeLighT-style efficient scaling, distinct from
  uniform-width transformers).

## Verification performed this round

Given the scale of this porting batch, verification for this model is
**GGUF-metadata-level only**: downloaded the exact Q8_0 file, verified its
sha256, and confirmed `general.architecture = openelm` + tensor count via
`gguf.GGUFReader` in Python -- not a full ET sysemu load/offload test (no
`--device ET` run, no compute-graph confirmation, no decode output). This is
a lighter verification bar than most other ports in this campaign, noted
honestly rather than implied as equivalent.

## Why this port likely needs no new ET-SoC1 kernel work

llama.cpp's existing `LLM_ARCH_OPENELM` graph builder uses standard ops
already proven on the ET backend elsewhere in this suite (`GGML_OP_MUL_MAT`,
`GGML_OP_NORM`/RMSNorm, `GGML_OP_ROPE`, `GGML_OP_SOFT_MAX`, `GGML_OP_ADD`,
`GGML_UNARY_OP_SILU`) -- the layer-wise scaling changes tensor *shapes* per
layer, not the *op types* used, so it should not require new kernel work.
Not confirmed live this round (see verification note above).

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/openelm_1_1b.json`
  is the model-ports track claim, pending identity approval (see issue #200
  follow-ups).
- No changes to any protected file or the vendored submodule.
- License is non-commercial -- flagging per established practice this session.
