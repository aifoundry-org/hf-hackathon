# Cohere2 (C4AI Command R7B) Porting Recipe

## Overview

Adds `CohereForAI/c4ai-command-r7b-12-2024` (8.03B-parameter
instruction-tuned causal LM) to the `llama_cpp_et` benchmark suite as a
fully automated, board-scored model port, using
`bartowski/c4ai-command-r7b-12-2024-GGUF`'s Q8_0 GGUF. Same rationale as
the other ports in this session: decoder-only causal LM, uses the
existing, unmodified `"api": "completion"` benchmark path -- no protected
file touched.

This introduces the **Cohere2** execution family to the board (distinct
from the older, non-sliding-window `Cohere`/`CohereForCausalLM` class).

## Why this port needed no new ET-SoC1 kernel work

Cohere2 (Command R7B's architecture) uses grouped-query attention (32
query heads / 8 KV heads) with a 4096-token sliding-window mask on most
layers (`n_swa=4096`, `is_swa_any=1` -- the same alternating local/global
attention masking mechanism already proven by `gemma2_2b` and
`starcoder2_3b` earlier this session), RMSNorm, a logit softcap
(`f_logit_scale`, applied via ordinary scale/tanh ops), and a standard
FFN. Every op involved (`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`,
`GGML_OP_SCALE`) is already proven on the ET backend by the existing
decoder-only models on the board. `convert_hf_to_gguf.py` already
registers `Cohere2ForCausalLM`, and the GGUF used here is a pre-made
community conversion, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `bartowski/c4ai-command-r7b-12-2024-GGUF` (Hugging
  Face)
- **File**: `c4ai-command-r7b-12-2024-Q8_0.gguf`
- **License**: CC-BY-NC-4.0 (matches upstream
  `CohereForAI/c4ai-command-r7b-12-2024`)
- **Architecture**: `Cohere2ForCausalLM`, 32 transformer layers, GQA (32
  query heads / 8 KV heads), 4096-token sliding-window attention on most
  layers, 256000-token BPE vocab, 8192-token native context.
- **Quantization**: Q8_0 (~7.94 GiB per the model's own reported file
  size; ~8.54 GB on disk).

## Steps Taken

1. **Downloaded** `c4ai-command-r7b-12-2024-Q8_0.gguf` from the Hugging
   Face repo above; verified `sha256=cd7281ec7974bb5810d8f6c922801d6fff3e8bd761073942757dcef990ca9b4d`,
   `size=8541100160` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18114`. Confirmed the model correctly
   identifies as `arch = cohere2`, `8.03 B` params, full `33/33` layer ET
   offload (31 repeating layers + output layer), and a clean 1080-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- the real per-PR board score for `"board": true` models
   comes from the self-hosted **real ET-SoC1 hardware** runner, not local
   sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `cohere2_7b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/cohere2_7b.json` -- new
     benchmark config, port `18114` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~8B size).
   - `.github/ci/benchmark_config.json` -- new `"cohere2_7b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download bartowski/c4ai-command-r7b-12-2024-GGUF c4ai-command-r7b-12-2024-Q8_0.gguf --local-dir .
./bin/llama-server -m c4ai-command-r7b-12-2024-Q8_0.gguf --device ET -ngl 99 --port 18114
curl -s http://127.0.0.1:18114/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- The upstream model's license is CC-BY-NC-4.0 (non-commercial). Flagging
  explicitly since this is more restrictive than the Apache-2.0/MIT
  licenses on most other ports this session -- worth a maintainer
  sanity-check on whether that's acceptable for this board.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
