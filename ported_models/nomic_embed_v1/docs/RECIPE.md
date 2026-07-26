# Nomic-Embed-Text-v1 Porting Recipe

## Overview

Adds `nomic-ai/nomic-embed-text-v1` (137M-parameter text-embedding model)
to the `llama.cpp-et` framework. Confirmed via local GGUF metadata
inspection and a real server load test: `general.architecture =
nomic-bert` -- distinct from `bert` (already ported as `distilbert_sst2`)
and from `modern-bert` (ported earlier in this campaign as
`modernbert_sentiment`). Nomic-BERT uses rotary position embeddings
instead of learned absolute positions, unlike either.

## Model Reference

- **Source**: `nomic-ai/nomic-embed-text-v1` (Hugging Face), revision
  `3ac47f125a41961d13b397d0332866be2f9152e1`
- **License**: Apache-2.0
- **GGUF source**: `nomic-ai/nomic-embed-text-v1-GGUF` (official), file
  `nomic-embed-text-v1.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=194206fcf0e77681bd2eaa3517b6fce880e1a3e15ef89935a77a1892574d15f7`
  (verified locally against the downloaded file)
- **Architecture**: `arch = nomic-bert` per GGUF metadata, 112 tensors.

## Verification performed this round

Built `llama-server` from a plain CPU-only (`GGML_ET=OFF`) configuration
of the vendored `llama.cpp-et` source, loaded the GGUF with
`--embedding --pooling mean`, and issued a real `POST /embedding` request.
Got back a real, well-formed embedding vector -- confirmed live, not just
a metadata read:

```
[{"index":0,"embedding":[[0.0108737...
```

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_NOMIC_BERT`'s graph builder uses RoPE, standard bidirectional
attention (`causal_attn=false`, already proven by every other BERT-family
port in this campaign), and a GEGLU FFN -- all ops already proven
elsewhere on this backend.

## Open items for maintainer review

- Registered in `artifacts.json` only (source: official HF GGUF repo,
  direct URL) -- not in `.github/ci/benchmark_config.json`, since the
  shared runner (`run_llama_server_benchmark.py`) only speaks
  `/completion`/`/v1/chat/completions`, not `/embedding`, same limitation
  already documented for `distilbert_sst2`/`modernbert_sentiment`.
- `ported_models/submissions/model_ports/nomic_embed_v1.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
