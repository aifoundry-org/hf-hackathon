# Qwen3-0.6B Reproducibility Recipe

## Overview
This recipe documents the already-registered `qwen3_06b` model (`Qwen3-0.6B`,
Q8_0 GGUF) on the `llama.cpp-et` framework for the AIFoundry CORE-ET hackathon.
The model is already scaffolded in the repo (pinned artifact in `artifacts.json`
plus the benchmark config in `benchmarks/qwen3_06b.json`); this submission adds
only the reproducibility recipe. It does **not** port the model or establish a
board baseline, and it does not by itself establish leaderboard or model-port
credit.

## Model Reference
- **Base model**: `Qwen/Qwen3-0.6B` (Apache-2.0)
- **GGUF source repo**: `ggml-org/Qwen3-0.6B-GGUF`
- **Revision (pinned)**: `a41486f827d17edd055fe6b3b0ba3f8d427c0519`
- **Filename**: `Qwen3-0.6B-Q8_0.gguf`
- **Format**: Q8_0 GGUF
- **SHA256**: `84c0dbe606526d5907251d88ea88b41457f46ce456e9a333d5d2b6245a95cafe`
- **Size**: 804,753,504 bytes
- **License**: Apache-2.0

## Why this model
Qwen3-0.6B is a current, strong sub-1B instruction/reasoning model. Its Q8_0
footprint (~0.8 GB) fits comfortably within ET-SoC1 constraints while still
producing coherent generations. The ET backend documentation lists non-MoE
Qwen3 Q8_0 models as fully supported (see the `qwen3_06b_q8_gguf` artifact note
in `artifacts.json`).

## How the registered model is defined
1. **Provenance**: the `ggml-org` Q8_0 GGUF is pinned by commit revision,
   filename, SHA256, and byte size in `artifacts.json` (`qwen3_06b_q8_gguf`).
   No floating `main` revision is used for model identity.
2. **No custom quantization required**: the Q8_0 GGUF is published upstream, so
   there is no local conversion/packing step; the file is content-addressed by
   URL + SHA256.
3. **Shared runner + scorer**: the registered benchmark config
   `ported_models/llama_cpp_et/benchmarks/qwen3_06b.json` uses the shared
   `llama_server` runner on `device: ET` with full offload (`gpu_layers: 99`)
   and the standard decode-throughput workload, plus the built-in WikiText-2
   perplexity quality gate (`min_ppl 1.0`, `max_ppl 1000.0`).

## Scope
This is a **recipe-only** documentation change. It adds no `benchmark_default`
or other config change, selects no model in the changed-file board selector
(`.github/ci/scripts/changed_benchmark_models.py`), and consumes no ET-SoC1
board run. It does not port the model (the port already exists on `main`) and
does not by itself establish a leaderboard entry or model-port credit.

## Reproduction
Fetch the pinned GGUF (content-addressed by SHA256):

```bash
huggingface-cli download ggml-org/Qwen3-0.6B-GGUF \
  Qwen3-0.6B-Q8_0.gguf \
  --revision a41486f827d17edd055fe6b3b0ba3f8d427c0519
```

The registered benchmark config
`ported_models/llama_cpp_et/benchmarks/qwen3_06b.json` defines the ET-SoC1
workload (llama.cpp-et `llama-server` / `llama-perplexity`) that a
maintainer-driven board run would execute; this recipe does not trigger that run.
