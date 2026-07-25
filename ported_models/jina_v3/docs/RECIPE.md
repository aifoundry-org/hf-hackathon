# Jina-Embeddings-v3 Porting Recipe

## Overview

Adds `jinaai/jina-embeddings-v3` (572M-parameter multilingual
text-embedding model, XLM-RoBERTa-based with task-specific LoRA adapters)
to the `llama.cpp-et` framework. Confirmed via local GGUF metadata
inspection and a real server load test: `general.architecture =
jina-bert-v3` -- distinct from `jina-bert-v2` and every other BERT-family
port in this campaign.

## Model Reference

- **Source**: `jinaai/jina-embeddings-v3` (Hugging Face), revision
  `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`
- **License**: CC-BY-NC-4.0 -- non-commercial, flagged honestly (same
  practice as `cohere2_7b`/`stablelm2_16b`/`plamo3_nict_8b` earlier in
  this campaign)
- **GGUF source**: `second-state/jina-embeddings-v3-GGUF`, file
  `jina-embeddings-v3-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=da95bb315ec9766aabfdfa920124a6997a5d9617bd7c9708c4195557136864e1`
  (verified locally against the downloaded file)
- **Architecture**: `arch = jina-bert-v3` per GGUF metadata, 292 tensors.

## Verification performed this round

Built `llama-server` from a plain CPU-only (`GGML_ET=OFF`) configuration
of the vendored `llama.cpp-et` source, loaded the GGUF with
`--embedding --pooling mean`, and issued a real `POST /embedding` request
(needed a longer load wait than the other, smaller embedding models --
this GGUF bundles multiple task-specific LoRA adapters). Got back a real,
well-formed embedding vector.

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_JINA_BERT_V3`'s graph builder uses RoPE, standard bidirectional
attention, and a GEGLU FFN -- the same op set already proven by
`nomic_embed_v1`/`jina_v2_base_en` earlier in this same PR. The bundled
LoRA adapters are applied as ordinary additional `MUL_MAT` terms, not a
new op type.

## Open items for maintainer review

- Registered in `artifacts.json` only, not `.github/ci/benchmark_config.json`
  -- same `/embedding`-not-supported-by-the-shared-runner limitation as
  the other embedding models in this PR.
- `ported_models/submissions/model_ports/jina_v3.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- Non-commercial license -- flagged for maintainer awareness.
