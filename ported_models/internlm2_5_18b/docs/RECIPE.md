# InternLM2.5-1.8b-chat Porting Recipe

## Overview

Adds `internlm/internlm2_5-1_8b-chat` (1.8B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using the official `internlm/internlm2_5-1_8b-chat-gguf` Q8_0 GGUF
(quantized and published by the model's own authors, not a third-party
community conversion). Same rationale as the other ports in this session:
decoder-only causal LM, uses the existing, unmodified `"api": "completion"`
benchmark path -- no protected file touched.

This introduces the **InternLM2** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

InternLM2 uses grouped-query attention, RoPE, RMSNorm, and a SwiGLU-gated
FFN -- the same op set already proven on the ET backend by every
Llama-family model on this board (`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_GLU` for SwiGLU,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`). `convert_hf_to_gguf.py` already
registers `InternLM2ForCausalLM`, and the GGUF used here is published by
the model's own authors, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `internlm/internlm2_5-1_8b-chat-gguf` (Hugging Face,
  official)
- **File**: `internlm2_5-1_8b-chat-q8_0.gguf`
- **License**: Apache-2.0 (matches upstream
  `internlm/internlm2_5-1_8b-chat`)
- **Architecture**: `InternLM2ForCausalLM`, 24 transformer layers, GQA,
  RMSNorm, SwiGLU FFN, 32768-token native context.
- **Quantization**: Q8_0 (~2.01 GB on disk).

## Steps Taken

1. **Downloaded** `internlm2_5-1_8b-chat-q8_0.gguf` from the Hugging Face
   repo above; verified `sha256=8526cc24717fcab32b20540c546f8c23a6ea3ff40b86f421a0cd060c8123e8b2`,
   `size=2009613056` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18102`. Confirmed full `25/25` layer ET
   offload (23 repeating layers + output layer), the model's native
   32768-token context correctly detected at load time, and a clean
   870-node / 2-split compute graph. As with the other ports in this
   session, full multi-token `/completion` decode output was not captured
   locally within this session -- sysemu's cycle-accurate simulation makes
   interactive end-to-end generation impractical to observe at this model
   size. The real per-PR board score for `"board": true` models comes from
   the self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `internlm2_5_18b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/internlm2_5_18b.json` -- new
     benchmark config, `ctx_size: 2048` (kept comparable to other
     mid-size models rather than the full native 32768), port `18102`
     (next free port after this session's earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"internlm2_5_18b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download internlm/internlm2_5-1_8b-chat-gguf internlm2_5-1_8b-chat-q8_0.gguf --local-dir .
./bin/llama-server -m internlm2_5-1_8b-chat-q8_0.gguf --device ET -ngl 99 --port 18102
curl -s http://127.0.0.1:18102/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- `ctx_size` is set to 2048 (not the model's full 32768) to keep sysemu
  load time and KV cache footprint comparable to other models on the
  board. Happy to raise it if preferred.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
