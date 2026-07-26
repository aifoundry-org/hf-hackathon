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

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 11.5101 +/- 1.96062
```

ET-SoC1 board execution itself is not something this session can produce --
deferred to the maintainer's trusted workflow after identity approval.

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_OLMO`'s graph builder is a standard pre-norm transformer (RMSNorm,
RoPE, standard MHA/GQA, SwiGLU FFN) using only ops already proven on the ET
backend by every other decoder-only text model in this suite. Not confirmed
live this round.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/olmo_1b.json`,
  and `.github/ci/benchmark_config.json` (port 18131) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/olmo_1b.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
