# Qwen3-4B Reproducibility Recipe

## Overview
This recipe documents the already-registered `qwen3_4b` model (`Qwen/Qwen3-4B`,
Q8_0 GGUF) on the `llama.cpp-et` framework for the AIFoundry CORE-ET hackathon.
The model is already scaffolded in the repo (pinned artifact in `artifacts.json`
plus the benchmark config in `benchmarks/qwen3_4b.json`); this submission adds
only the reproducibility recipe. It does **not** port the model or establish a
board baseline, and it does not by itself establish leaderboard or model-port
credit.

## Model Reference
- **Base model**: `Qwen/Qwen3-4B` (Apache-2.0)
- **GGUF source repo**: `ggml-org/Qwen3-4B-GGUF`
- **Revision (pinned)**: `2f3b082b1356a6123f7ed71e65aea340da25d53c`
- **Filename**: `Qwen3-4B-Q8_0.gguf`
- **Format**: Q8_0 GGUF
- **SHA256**: `865fbdcb8119c5cd5ba557b139d49351aa2c4e33bf93bdd7b08e24b573dd1dd4`
- **Size**: 4,280,405,120 bytes
- **License**: Apache-2.0

## Why this model
Qwen/Qwen3-4B is a strong open instruction model whose Q8_0 GGUF is supported by
the ET backend. The ET backend docs list non-MoE Qwen3 Q8_0 models as fully
supported, making it a useful mid-scale entry alongside the other `llama.cpp-et`
models.

## How the registered model is defined
1. **Provenance**: the `ggml-org` Q8_0 GGUF is pinned by commit revision,
   filename, SHA256, and byte size in `artifacts.json` (`qwen3_4b_q8_gguf`).
   No floating `main` revision is used for model identity.
2. **No custom quantization required**: the Q8_0 GGUF is published upstream, so
   there is no local conversion/packing step; the file is content-addressed by
   URL + SHA256.
3. **Shared runner + scorer**: the registered benchmark config
   `ported_models/llama_cpp_et/benchmarks/qwen3_4b.json` uses the shared
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
huggingface-cli download ggml-org/Qwen3-4B-GGUF \
  Qwen3-4B-Q8_0.gguf \
  --revision 2f3b082b1356a6123f7ed71e65aea340da25d53c
```

The registered benchmark config
`ported_models/llama_cpp_et/benchmarks/qwen3_4b.json` defines the ET-SoC1
workload (llama.cpp-et `llama-server` / `llama-perplexity`) that a
maintainer-driven board run would execute; this recipe does not trigger that run.
