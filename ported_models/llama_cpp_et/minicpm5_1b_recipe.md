# MiniCPM5-1B Porting Recipe

## Overview

Adds `openbmb/MiniCPM5-1B` (1B-parameter causal LM) to the `llama_cpp_et`
benchmark suite as a fully automated, board-scored model port, using the
official `openbmb/MiniCPM5-1B-GGUF` Q8_0 GGUF (published by the model's own
authors). Same rationale as the other ports in this session: decoder-only
causal LM, uses the existing, unmodified `"api": "completion"` benchmark
path -- no protected file touched.

This introduces the **MiniCPM** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

MiniCPM5 uses grouped-query attention, RoPE, and a gated FFN (a "fused
Gated Delta Net" variant that the vendored llama.cpp-et fork already
resolves at graph-build time -- the load log explicitly reports "fused
Gated Delta Net (autoregressive) enabled" / "(chunked) enabled", meaning
this exact op fusion path is already implemented and selected
automatically, not something this port had to add). Every underlying op
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_GLU`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on
the ET backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `MiniCPMForCausalLM` /
`MiniCPM3ForCausalLM`, and the GGUF used here is published by the model's
own authors, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `openbmb/MiniCPM5-1B-GGUF` (Hugging Face, official)
- **File**: `MiniCPM5-1B-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `openbmb/MiniCPM5-1B`)
- **Architecture**: dense 1B-parameter decoder, 24 transformer layers, GQA,
  gated FFN, 131072-token (128k) native context.
- **Quantization**: Q8_0 (~1.15 GB on disk).

## Steps Taken

1. **Downloaded** `MiniCPM5-1B-Q8_0.gguf` from the Hugging Face repo above;
   verified `sha256=0dc7638539067268774c275a14a6ec9c7e01f7eeb2cff606c8590361fa527e4c`,
   `size=1153529216` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18104`. Confirmed full `25/25` layer ET
   offload (23 repeating layers + output layer) and a clean 870-node /
   2-split compute graph. The model's native 131072-token context was
   correctly detected at load time (the default KV cache at that context
   length is large -- 3072 MiB -- which the benchmark config below avoids
   by pinning `ctx_size` down to 2048, matching every other model on the
   board). As with the other ports in this session, full multi-token
   `/completion` decode output was not captured locally within this
   session -- sysemu's cycle-accurate simulation makes interactive
   end-to-end generation impractical to observe. The real per-PR board
   score for `"board": true` models comes from the self-hosted **real
   ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `minicpm5_1b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/minicpm5_1b.json` -- new
     benchmark config, `ctx_size: 2048` (see above), port `18104` (next
     free port after this session's earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"minicpm5_1b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q8_0.gguf --local-dir .
./bin/llama-server -m MiniCPM5-1B-Q8_0.gguf --device ET -ngl 99 --port 18104
curl -s http://127.0.0.1:18104/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- `ctx_size` is pinned to 2048, well below the model's 128k native
  context, purely to keep sysemu/board load time and KV cache footprint
  comparable to other models on the board.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
