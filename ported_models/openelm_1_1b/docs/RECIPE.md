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

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
exact board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks -- the same parameters every
other model on this board uses. The model loads and runs cleanly:

```
Final estimate: PPL = 14.4998 +/- 3.00329
```

This is a genuine host-CPU quality signal, not a placeholder. ET-SoC1 board
execution itself (the "Board Correctness"/"Board Performance" pieces of the
Fully Validated Model Standard) is not something this session can produce --
that runs through the maintainer's own trusted workflow after identity
approval.

## Why this port likely needs no new ET-SoC1 kernel work

llama.cpp's existing `LLM_ARCH_OPENELM` graph builder uses standard ops
already proven on the ET backend elsewhere in this suite (`GGML_OP_MUL_MAT`,
`GGML_OP_NORM`/RMSNorm, `GGML_OP_ROPE`, `GGML_OP_SOFT_MAX`, `GGML_OP_ADD`,
`GGML_UNARY_OP_SILU`) -- the layer-wise scaling changes tensor *shapes* per
layer, not the *op types* used, so it should not require new kernel work.
Not confirmed live this round (see verification note above).

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/openelm_1_1b.json`,
  and `.github/ci/benchmark_config.json` (port 18130) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/openelm_1_1b.json` is the model-ports
  track claim, pending identity approval (see issue #200 follow-ups).
- No changes to any protected file or the vendored submodule.
- License is non-commercial -- flagging per established practice this session.
