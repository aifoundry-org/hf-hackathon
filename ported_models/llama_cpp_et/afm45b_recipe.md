# Arcee AFM-4.5B Porting Recipe

## Overview

Adds `arcee-ai/AFM-4.5B` (4.62B-parameter instruction-tuned causal LM) to
the `llama_cpp_et` benchmark suite as a fully automated, board-scored
model port, using the official `arcee-ai/AFM-4.5B-GGUF` Q8_0 GGUF
(published by the model's own authors). Same rationale as the other ports
in this session: decoder-only causal LM, uses the existing, unmodified
`"api": "completion"` benchmark path -- no protected file touched.

This introduces the **Arcee AFM** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

AFM-4.5B uses grouped-query attention (20 query heads / 4 KV heads),
YaRN-scaled RoPE, RMSNorm, and a standard FFN. Every op involved
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `ArceeForCausalLM`, and the GGUF
used here is published by the model's own authors, so this port is pure
configuration wiring.

## Model Reference

- **GGUF source**: `arcee-ai/AFM-4.5B-GGUF` (Hugging Face, official)
- **File**: `AFM-4.5B-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `arcee-ai/AFM-4.5B`)
- **Architecture**: `ArceeForCausalLM`, 36 transformer layers, GQA (20
  query heads / 4 KV heads), YaRN RoPE scaling, 65536-token native
  context.
- **Quantization**: Q8_0 (~4.57 GiB per the model's own reported file
  size; ~4.92 GB on disk).

## Steps Taken

1. **Downloaded** `AFM-4.5B-Q8_0.gguf` from the Hugging Face repo above;
   verified `sha256=2de4692fc3404ba302701753f4110610d3d8f88cf49294d5ce0f176e01cc8871`,
   `size=4916265216` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18109`. Confirmed the model correctly
   identifies as `arch = arcee`, `4.62 B` params, full `37/37` layer ET
   offload (35 repeating layers + output layer), and a clean 1302-node /
   74-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- sysemu's cycle-accurate simulation makes interactive
   end-to-end generation impractical to observe at this model size. The
   real per-PR board score for `"board": true` models comes from the
   self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `afm45b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/afm45b.json` -- new benchmark
     config, `ctx_size: 2048` (kept comparable to other models rather than
     the full native 65536), port `18109` (next free port after this
     session's earlier additions), `ready_timeout_s: 300` /
     `request_timeout_s: 420`.
   - `.github/ci/benchmark_config.json` -- new `"afm45b"` entry pointing at
     the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download arcee-ai/AFM-4.5B-GGUF AFM-4.5B-Q8_0.gguf --local-dir .
./bin/llama-server -m AFM-4.5B-Q8_0.gguf --device ET -ngl 99 --port 18109
curl -s http://127.0.0.1:18109/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- `ctx_size` is set to 2048 (not the model's full 65536) to keep sysemu
  load time and KV cache footprint comparable to other models on the
  board.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
