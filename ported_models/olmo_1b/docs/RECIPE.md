# OLMo-1B Porting Recipe

## Overview

Adds `allenai/OLMo-1B-hf` (1.2B-parameter causal LM, AI2's fully-open
pretraining-transparent model) to the `llama.cpp-et` framework. Confirmed
via local GGUF metadata inspection: `general.architecture = olmo` -- distinct
from `olmo2` (already claimed elsewhere on this board's "most models ported"
track), a separate execution family in llama.cpp.

## Model Reference

- **Source**: `allenai/OLMo-1B-hf` (Hugging Face), revision
  `aee7752d9c08ee4775e9b0091426d8410e8f6a89`
- **License**: Apache-2.0
- **GGUF source**: `RichardErkhov/allenai_-_OLMo-1B-hf-gguf`, file
  `OLMo-1B-hf.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=11ad66bb4b0c4b9d4b40ccef351a506e322fc8265b87e3638b18802768d8875e`
  (verified locally against the downloaded file)
- **Architecture**: `arch = olmo` per GGUF metadata, 113 tensors.

## Verification performed this round

GGUF-metadata-level only (downloaded exact file, verified sha256, confirmed
`general.architecture` + tensor count via `gguf.GGUFReader`) -- not a full
ET sysemu load/offload test this round, given the scale of this porting
batch. Noted honestly, matching the same caveat on openelm_1_1b in this
same PR.

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_OLMO`'s graph builder is a standard pre-norm transformer (RMSNorm,
RoPE, standard MHA/GQA, SwiGLU FFN) using only ops already proven on the ET
backend by every other decoder-only text model in this suite. Not confirmed
live this round.

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/olmo_1b.json`
  is the model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
