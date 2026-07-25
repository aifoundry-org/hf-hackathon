# GLM-4-9B-0414 Porting Recipe

## Overview

Adds `zai-org/GLM-4-9B-0414` (9B-parameter causal LM, Zhipu AI's GLM-4
series) to the `llama.cpp-et` framework. Confirmed via local GGUF metadata
inspection: `general.architecture = glm4` -- a genuinely distinct execution
family, not registered by any existing board identity (`chatglm`/`glm4` are
separate archs in llama.cpp; this is `glm4` specifically).

## Model Reference

- **Source**: `zai-org/GLM-4-9B-0414` (Hugging Face; this model was
  previously published under `THUDM/GLM-4-9B-0414`, which now redirects to
  `zai-org`), revision `645b8482494e31b6b752272bf7f7f273ef0f3caf`
- **License**: MIT
- **GGUF source**: `bartowski/THUDM_GLM-4-9B-0414-GGUF`, file
  `THUDM_GLM-4-9B-0414-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=7b4ea2795934ca05dc409251dddd289160a194a4d920b575d1de4516808bb50d`
  (verified locally against the downloaded file)
- **Architecture**: `arch = glm4` per GGUF metadata, 523 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 12.2894 +/- 2.42663
```

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_GLM4`'s graph builder uses RMSNorm (with an extra post-attention/
post-FFN norm pair GLM4 adds -- still `GGML_OP_NORM`, no new op type), RoPE,
GQA, and a gated FFN -- all already proven on the ET backend. ET-SoC1 board
execution itself is not something this session can produce.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/glm4_9b.json`,
  and `.github/ci/benchmark_config.json` (port 18138) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/glm4_9b.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- GGUF quantization used (`bartowski/THUDM_GLM-4-9B-0414-GGUF`) predates
  the org rename and was built from the `THUDM` namespace; the sha256 above
  is the artifact actually verified, independent of the naming change.
