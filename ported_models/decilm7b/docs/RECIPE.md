# DeciLM-7B-instruct Porting Recipe

## Overview

Adds `Deci/DeciLM-7B-instruct` (7.11B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using the official `Deci/DeciLM-7B-instruct-GGUF` Q8_0 GGUF
(published by the model's own authors). Same rationale as the other ports
in this session: decoder-only causal LM, uses the existing, unmodified
`"api": "completion"` benchmark path -- no protected file touched.

This introduces the **DeciLM** execution family to the board -- notable
for using variable Grouped-Query Attention (a different number of KV heads
per layer, chosen per-layer via neural architecture search rather than a
single fixed GQA ratio for the whole model), which is a materially
different structural pattern from every fixed-GQA model already on this
board.

## Why this port needed no new ET-SoC1 kernel work

Despite the variable-GQA design being architecturally novel, it resolves
to the exact same graph primitives as any other GQA model at the ggml
level: each layer's attention block still repeats/broadcasts its (smaller)
set of KV heads across the (larger) set of query heads via ordinary
`GGML_OP_MUL_MAT`/`GGML_OP_GET_ROWS`/`GGML_OP_SOFT_MAX` -- the "variable"
part is a per-layer config choice baked into the GGUF's tensor shapes at
conversion time, not a new op the runtime needs to know about. Combined
with standard RoPE, RMSNorm, and SwiGLU FFN (all already proven on this
board), this port needed no new ET kernel work.
`convert_hf_to_gguf.py` already registers `DeciLMForCausalLM`, and the
GGUF used here is published by the model's own authors, so this port is
pure configuration wiring.

## Model Reference

- **GGUF source**: `Deci/DeciLM-7B-instruct-GGUF` (Hugging Face, official)
- **File**: `decilm-7b-uniform-gqa-q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `Deci/DeciLM-7B-instruct`)
- **Architecture**: `DeciLMForCausalLM`, 32 transformer layers, variable
  grouped-query attention (per-layer NAS-chosen KV head count), RoPE,
  RMSNorm, SwiGLU FFN, 8192-token native context, 32000-token SentencePiece
  vocab.
- **Quantization**: Q8_0 (~7.55 GB on disk).

## Steps Taken

1. **Downloaded** `decilm-7b-uniform-gqa-q8_0.gguf` from the Hugging Face
   repo above; verified `sha256=1b60beae4c9d5d7131f846f7545cb386e20d8f2c613844a0142ee701537dbfe3`,
   `size=7554187712` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18105`. Confirmed the model correctly
   identifies as `7.11 B` params / `7B` model type, full `33/33` layer ET
   offload (31 repeating layers + output layer), and a clean 1158-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- sysemu's cycle-accurate simulation makes interactive
   end-to-end generation impractical to observe, especially at 7B scale.
   The real per-PR board score for `"board": true` models comes from the
   self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `decilm7b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/decilm7b.json` -- new
     benchmark config, `ctx_size: 2048` (kept comparable to other models
     rather than the full native 8192), port `18105` (next free port after
     this session's earlier additions), `ready_timeout_s: 300` /
     `request_timeout_s: 420` (more generous, given the 7B size).
   - `.github/ci/benchmark_config.json` -- new `"decilm7b"` entry pointing
     at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download Deci/DeciLM-7B-instruct-GGUF decilm-7b-uniform-gqa-q8_0.gguf --local-dir .
./bin/llama-server -m decilm-7b-uniform-gqa-q8_0.gguf --device ET -ngl 99 --port 18105
curl -s http://127.0.0.1:18105/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- This is the largest model added this session (7.11B params); happy to
  see how it scores relative to the other large models already on the
  board (`qwen3_8b`, `llama31_8b`).
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
