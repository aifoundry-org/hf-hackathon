# Jina-Embeddings-v2-Base-EN Porting Recipe

## Overview

Adds `jinaai/jina-embeddings-v2-base-en` (137M-parameter text-embedding
model, 8192-token context via ALiBi) to the `llama.cpp-et` framework.
Confirmed via local GGUF metadata inspection and a real server load test:
`general.architecture = jina-bert-v2` -- distinct from `bert`,
`modern-bert`, and `nomic-bert` already ported in this campaign. Uses
ALiBi (like `mpt7b`/`bloom560m` earlier in this campaign) instead of RoPE
or learned positions, enabling its long 8192-token native context.

## Model Reference

- **Source**: `jinaai/jina-embeddings-v2-base-en` (Hugging Face), revision
  `322d4d7e2f35e84137961a65af894fda0385eb7a`
- **License**: Apache-2.0
- **GGUF source**: `gpustack/jina-embeddings-v2-base-en-GGUF`, file
  `jina-embeddings-v2-base-en-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=123b13ec105de8e0d2becd7dca4f4c46cef3a535f6de9ff1b6ade3fbbf6f5184`
  (verified locally against the downloaded file)
- **Architecture**: `arch = jina-bert-v2` per GGUF metadata, 196 tensors.

## Verification performed this round

Built `llama-server` from a plain CPU-only (`GGML_ET=OFF`) configuration
of the vendored `llama.cpp-et` source, loaded the GGUF with
`--embedding --pooling mean`, and issued a real `POST /embedding` request.
Got back a real, well-formed embedding vector.

## Why this port likely needs no new ET-SoC1 kernel work

ALiBi is an ordinary elementwise `GGML_OP_ADD` bias on attention scores
(already proven by `mpt7b`/`bloom560m` earlier in this campaign) --
combined with standard bidirectional attention and a GELU FFN, no new op
type is needed.

## Open items for maintainer review

- Registered in `artifacts.json` only, not `.github/ci/benchmark_config.json`
  -- same `/embedding`-not-supported-by-the-shared-runner limitation as
  `nomic_embed_v1`/`distilbert_sst2`/`modernbert_sentiment`.
- `ported_models/submissions/model_ports/jina_v2_base_en.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
