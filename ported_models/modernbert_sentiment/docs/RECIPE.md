# ModernBERT-base Multilingual Sentiment Porting Recipe

## Overview

Adds `clapAI/modernBERT-base-multilingual-sentiment` (149M-parameter
encoder-only sequence-classification model, 3-way sentiment) to the
`llama.cpp-et` framework. Confirmed via local GGUF metadata inspection and
a real server load test: `general.architecture = modern-bert` -- distinct
from `bert` (already ported earlier in this campaign as `distilbert_sst2`)
and from `nomic-bert`/`jina-bert-v2`/`jina-bert-v3`/`neo-bert`/`eurobert`
(all separate, unclaimed arches) -- ModernBERT is architecturally its own
family: alternating global/local sliding-window attention (`n_swa=128`,
pattern every 3 layers), RoPE instead of learned position embeddings,
GeGLU FFN.

This artifact was self-converted earlier this session (before this
specific porting campaign) and hosted as a GitHub Release
(`model-port-modernbert-sentiment-v1`), matching the same precedent as
`distilbert_sst2`/`roberta_sst2` -- it had not yet had a model-ports claim
filed until now.

## Model Reference

- **Source**: `clapAI/modernBERT-base-multilingual-sentiment` (Hugging
  Face), revision `baa7f26fee4eca5aec35538caa5fdda7b86d83c6`
- **License**: Apache-2.0
- **Artifact**: self-converted F32 GGUF (no quantization -- encoder
  classification heads are small; F32 avoids any quantization-accuracy
  question for a 3-way classifier), produced via this repo's own
  `convert_hf_to_gguf.py`, hosted at
  `https://github.com/DarthCeltic/hf-hackathon/releases/download/model-port-modernbert-sentiment-v1/modernbert_sentiment_f32.gguf`,
  `sha256=7896d82f2548762409157ae4e68732cd2c78796b5c8cea5e0145b964e916ab4c`
  (verified locally against the downloaded file), 600,207,808 bytes
- **Architecture**: `arch = modern-bert` per GGUF metadata, 138 tensors,
  22 layers, 768 embedding, 12 heads, sliding-window attention (window
  128, every 3rd layer is global), `n_cls_out=3`
  (`cls_label = [negative, neutral, positive]`)

## Verification performed this round

Built `llama-server` from a plain CPU-only (`GGML_ET=OFF`) configuration
of the vendored `llama.cpp-et` source, loaded the GGUF, and issued a real
`POST /rerank` request. The server loaded all metadata correctly (3-way
classifier head, sliding-window pattern) and returned real relevance
scores for both documents -- confirmed live, not just GGUF-metadata
inspection:

```
{"model":"modernbert_sentiment_f32.gguf","results":[
  {"index":1,"relevance_score":2.3337833881378174},
  {"index":0,"relevance_score":0.5375099182128906}]}
```

This is the same verification tier as `distilbert_sst2`/`roberta_sst2`
earlier in this campaign (real server load + real endpoint response), not
just a metadata read.

## Why this port likely needs no new ET-SoC1 kernel work

Sliding-window attention is already proven on this backend by Gemma2/
Cohere2/Starcoder2/PLaMo-3 earlier in this campaign (an ordinary masked
`GGML_OP_SOFT_MAX`, not a new op). RoPE, LayerNorm-family norm, and GeGLU
FFN (an existing gated-unary op) are all already proven elsewhere on this
board. Not confirmed live against ET sysemu specifically this round --
only the CPU backend was exercised.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/modernbert_sentiment.json`,
  and `.github/ci/benchmark_config.json` -- but see the same caveat as
  `distilbert_sst2`: the shared runner (`run_llama_server_benchmark.py`)
  only speaks `/completion`/`/v1/chat/completions`, not `/rerank`, so this
  entry documents the port but cannot be scored by automated per-PR CI
  without a maintainer adding `/rerank` support to that script (a protected
  file we cannot edit ourselves).
- `ported_models/submissions/model_ports/modernbert_sentiment.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
