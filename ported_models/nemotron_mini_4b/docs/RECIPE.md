# Nemotron-Mini-4B-Instruct Porting Recipe

## Overview

Adds `nvidia/Nemotron-Mini-4B-Instruct` (4B-parameter causal LM, NVIDIA's
distilled/pruned Minitron family) to the `llama.cpp-et` framework. Confirmed
via local GGUF metadata inspection: `general.architecture = nemotron` --
a genuinely distinct execution family, not registered by any existing board
identity.

## Model Reference

- **Source**: `nvidia/Nemotron-Mini-4B-Instruct` (Hugging Face), revision
  `791833e92ebddb0bc2c1007f6d2b6764f886a2ae`
- **License**: NVIDIA Community Model License
- **GGUF source**: `bartowski/Nemotron-Mini-4B-Instruct-GGUF`, file
  `Nemotron-Mini-4B-Instruct-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=8ed6d148ded733d4401495e44dc834ffa79e8cfee6f44d8fb60808dddd78b8eb`
  (verified locally against the downloaded file)
- **Architecture**: `arch = nemotron` per GGUF metadata, 324 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 10.5717 +/- 2.16348
```

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_NEMOTRON`'s graph builder is a standard pre-norm transformer
(squared-ReLU FFN activation instead of SiLU/GELU is its main quirk, but
that's an existing `GGML_UNARY_OP`, not a new op type) using RMSNorm,
standard MHA/GQA, and RoPE -- all already proven on the ET backend. ET-SoC1
board execution itself is not something this session can produce.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/nemotron_mini_4b.json`,
  and `.github/ci/benchmark_config.json` (port 18136) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/nemotron_mini_4b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
