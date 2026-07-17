# Llama-3.2-3B-Instruct Reproducibility Recipe

## Overview
This recipe documents the already-registered `llama32_3b` model
(`meta-llama/Llama-3.2-3B-Instruct`, Q8_0 GGUF) on the `llama.cpp-et` framework
for the AIFoundry CORE-ET hackathon. The model is already scaffolded in the repo
(pinned artifact in `artifacts.json` plus the benchmark config in
`benchmarks/llama32_3b.json`); this submission adds only the reproducibility
recipe. It does **not** port the model or establish a board baseline, and it does
not by itself establish leaderboard or model-port credit.

## Model Reference
- **Base model**: `meta-llama/Llama-3.2-3B-Instruct` (llama3.2)
- **GGUF source repo**: `lmstudio-community/Llama-3.2-3B-Instruct-GGUF`
- **Revision (pinned)**: `c91307b5cf18c8106b1f8a6218c26ae4dbfee472`
- **Filename**: `Llama-3.2-3B-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF
- **SHA256**: `94f22b7231df5cd1907ff48dba54497b2d7912a4ce60d914f3dcfc0347fa8f21`
- **Size**: 3,421,899,040 bytes
- **License**: llama3.2

## Why this model
meta-llama/Llama-3.2-3B-Instruct is a strong open instruction model whose Q8_0
GGUF is supported by the ET backend. The ET backend docs list Llama 3.2 1B/3B
Q8_0 models as fully supported, making it a useful mid-scale entry alongside the
other `llama.cpp-et` models.

## How the registered model is defined
1. **Provenance**: the `lmstudio-community` Q8_0 GGUF is pinned by commit
   revision, filename, SHA256, and byte size in `artifacts.json`
   (`llama32_3b_q8_gguf`). No floating `main` revision is used for model identity.
2. **No custom quantization required**: the Q8_0 GGUF is published upstream, so
   there is no local conversion/packing step; the file is content-addressed by
   URL + SHA256.
3. **Shared runner + scorer**: the registered benchmark config
   `ported_models/llama_cpp_et/benchmarks/llama32_3b.json` uses the shared
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
huggingface-cli download lmstudio-community/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q8_0.gguf \
  --revision c91307b5cf18c8106b1f8a6218c26ae4dbfee472
```

The registered benchmark config
`ported_models/llama_cpp_et/benchmarks/llama32_3b.json` defines the ET-SoC1
workload (llama.cpp-et `llama-server` / `llama-perplexity`) that a
maintainer-driven board run would execute; this recipe does not trigger that run.
