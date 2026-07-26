# Llama-3.1-8B-Instruct Reproducibility Recipe

## Overview
This recipe documents the already-registered `llama31_8b` model
(`meta-llama/Llama-3.1-8B-Instruct`, Q8_0 GGUF) on the `llama.cpp-et` framework
for the AIFoundry CORE-ET hackathon. The model is already scaffolded in the repo
(pinned artifact in `artifacts.json` plus the benchmark config in
`benchmarks/llama31_8b.json`); this submission adds only the reproducibility
recipe. It does **not** port the model or establish a board baseline, and it does
not by itself establish leaderboard or model-port credit.

## Model Reference
- **Base model**: `meta-llama/Llama-3.1-8B-Instruct` (llama3.1)
- **GGUF source repo**: `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF`
- **Revision (pinned)**: `8601e6db71269a2b12255ebdf09ab75becf22cc8`
- **Filename**: `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF
- **SHA256**: `f6597a0fbdf50364457ed7a077999a1dc45a8cb8d779e12d6fb4be16a52ec31a`
- **Size**: 8,540,775,776 bytes
- **License**: llama3.1

## Why this model
meta-llama/Llama-3.1-8B-Instruct is a strong open instruction model whose Q8_0
GGUF is supported by the ET backend. The ET backend docs list the Llama 3.1
family as fully supported.

Note: this 8B Q8 model (~8.5 GB) is excluded from the default CI set upstream
(`benchmark_default: false` on `main`) because it runs slower on ET-SoC1. This
recipe does not change that; it only documents the registered model.

## How the registered model is defined
1. **Provenance**: the `lmstudio-community` Q8_0 GGUF is pinned by commit
   revision, filename, SHA256, and byte size in `artifacts.json`
   (`llama31_8b_q8_gguf`). No floating `main` revision is used for model identity.
2. **No custom quantization required**: the Q8_0 GGUF is published upstream, so
   there is no local conversion/packing step; the file is content-addressed by
   URL + SHA256.
3. **Shared runner + scorer**: the registered benchmark config
   `ported_models/llama_cpp_et/benchmarks/llama31_8b.json` uses the shared
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
huggingface-cli download lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --revision 8601e6db71269a2b12255ebdf09ab75becf22cc8
```

The registered benchmark config
`ported_models/llama_cpp_et/benchmarks/llama31_8b.json` defines the ET-SoC1
workload (llama.cpp-et `llama-server` / `llama-perplexity`) that a
maintainer-driven board run would execute; this recipe does not trigger that run.
