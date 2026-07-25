# Apertus-8B-Instruct Porting Recipe

## Overview

Adds `swiss-ai/Apertus-8B-Instruct-2509` (8B-parameter causal LM, the Swiss
AI Initiative's fully-open, transparency-focused model) to the
`llama.cpp-et` framework. Confirmed via local GGUF metadata inspection:
`general.architecture = apertus` -- a genuinely distinct execution family,
not registered by any existing board identity.

## Model Reference

- **Source**: `swiss-ai/Apertus-8B-Instruct-2509` (Hugging Face), revision
  `b946d40447b2b597999b9c86d44bee0b452c919f`
- **License**: Apache-2.0
- **GGUF source**: `DevQuasar/swiss-ai.Apertus-8B-Instruct-2509-GGUF`, file
  `swiss-ai.Apertus-8B-Instruct-2509.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=823a2ce1b89f8e31c197167e144fb4e5220f0adb44132b8e54ccdb2dd5bedae7`
  (verified locally against the downloaded file)
- **Architecture**: `arch = apertus` per GGUF metadata, 324 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 7.8747 +/- 1.46638
```

## Why this port likely needs no new ET-SoC1 kernel work

Apertus uses a standard pre-norm transformer with RMSNorm, RoPE, GQA, and a
gated FFN (xIELU activation -- a parameterized variant of an existing
elementwise unary op, not a new op type) -- every primitive is already
proven on the ET backend by other decoders in this suite. ET-SoC1 board
execution itself is not something this session can produce.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/apertus_8b.json`,
  and `.github/ci/benchmark_config.json` (port 18137) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/apertus_8b.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
