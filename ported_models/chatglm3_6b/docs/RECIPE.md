# ChatGLM3-6B Porting Recipe

## Overview

Adds `zai-org/chatglm3-6b` (6B-parameter causal LM, Zhipu AI's ChatGLM3;
this model was previously published under `THUDM/chatglm3-6b`, which now
redirects to `zai-org`) to the `llama.cpp-et` framework. Confirmed via
local GGUF metadata inspection: `general.architecture = chatglm` --
distinct from `glm4` (already ported earlier in this campaign), a
separate, older execution family in llama.cpp (ChatGLM uses a distinct
attention-bias/RoPE convention from GLM4).

## Model Reference

- **Source**: `zai-org/chatglm3-6b` (Hugging Face), revision
  `e9e0406d062cdb887444fe5bd546833920abd4ac`
- **License**: Apache-2.0
- **GGUF source**: `hellork/chatglm3-6b-128k-Q8_0-GGUF` (a 128k-context
  fine-tune/extension of the base model, same architecture), file
  `chatglm3-6b-128k-q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=805e3761a997486eb08f45e0f74efa5c1cc0be8475afa2b170263b490a3ba8c8`
  (verified locally against the downloaded file)
- **Architecture**: `arch = chatglm` per GGUF metadata, 199 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 15.9100 +/- 3.37450
```

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_CHATGLM`'s graph builder uses standard ops already proven on the
ET backend (RMSNorm, RoPE, MHA/GQA with a QKV bias term, GELU-based FFN) --
no new op type. ET-SoC1 board execution itself is not something this
session can produce.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/chatglm3_6b.json`,
  and `.github/ci/benchmark_config.json` (port 18140) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/chatglm3_6b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- GGUF used is a 128k-context fine-tune, not a base-model quant -- same
  architecture and weights lineage, flagged for transparency.
