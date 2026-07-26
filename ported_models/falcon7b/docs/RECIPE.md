# Falcon-7B Porting Recipe

## Overview

Adds `tiiuae/falcon-7b` (7.22B-parameter causal LM, the original classic
Falcon architecture -- not Falcon-Mamba or Falcon-H1) to the `llama_cpp_et`
benchmark suite as a fully automated, board-scored model port, using
`maddes8cht/tiiuae-falcon-7b-gguf`'s Q8_0 GGUF. Same rationale as the other
ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

This introduces the **Falcon** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

Classic Falcon uses multi-query attention (71 query heads sharing a single
KV head -- an extreme MQA ratio), parallel attention+FFN (both branches
read the same normed input and sum, like Phi), LayerNorm, and a GELU FFN.
Every op involved (`GGML_OP_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`,
`GGML_OP_ADD`, `GGML_OP_UNARY` for GELU, `GGML_OP_SOFT_MAX`,
`GGML_OP_GET_ROWS`) is already proven on the ET backend by the existing
decoder-only models on the board. `convert_hf_to_gguf.py` already registers
`FalconForCausalLM`/`RWForCausalLM`, and the GGUF used here is a pre-made
community conversion, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `maddes8cht/tiiuae-falcon-7b-gguf` (Hugging Face)
- **File**: `ggml-tiiuae-falcon-7b-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `tiiuae/falcon-7b`)
- **Architecture**: `FalconForCausalLM`, 32 transformer layers, 71 query
  heads / 1 KV head (multi-query attention), parallel attention+FFN,
  2048-token native context.
- **Quantization**: Q8_0 (~7.14 GiB per the model's own reported file
  size; ~7.67 GB on disk).

## Steps Taken

1. **Downloaded** `ggml-tiiuae-falcon-7b-Q8_0.gguf` from the Hugging Face
   repo above; verified `sha256=47e0a8ef8eb3d55765f7d412a1d9b816159fece3e9f87ce79caa7e6d9b2341a8`,
   `size=7671704512` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18108`. Confirmed the model correctly
   identifies as `7.22 B` params / `7B` model type with `n_head_kv=1`
   (the extreme MQA ratio), full `33/33` layer ET offload (31 repeating
   layers + output layer), and a clean 1032-node / 66-split compute graph.
   As with the other ports in this session, full multi-token `/completion`
   decode output was not captured locally within this session -- sysemu's
   cycle-accurate simulation makes interactive end-to-end generation
   impractical to observe at 7B scale. The real per-PR board score for
   `"board": true` models comes from the self-hosted **real ET-SoC1
   hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `falcon7b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/falcon7b.json` -- new
     benchmark config. `ctx_size: 2048` here matches the model's own
     native context exactly (`n_ctx_train=2048`), so there is no
     truncation relative to native, unlike some of the other large models
     added this session. Port `18108` (next free port after this
     session's earlier additions), `ready_timeout_s: 300` /
     `request_timeout_s: 420` (generous, given the 7B size).
   - `.github/ci/benchmark_config.json` -- new `"falcon7b"` entry pointing
     at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download maddes8cht/tiiuae-falcon-7b-gguf ggml-tiiuae-falcon-7b-Q8_0.gguf --local-dir .
./bin/llama-server -m ggml-tiiuae-falcon-7b-Q8_0.gguf --device ET -ngl 99 --port 18108
curl -s http://127.0.0.1:18108/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.

## Negative result also from this investigation (not included in this PR)

While looking for more Falcon-family/small-model candidates alongside this
one, RWKV-6-World-3B (`mradermacher/rwkv-6-world-3b-v2.1-GGUF`, Q8_0) was
also loaded against this same ET backend and **crashed at graph-reserve
time**: `pre-allocated tensor (cache_r_l0 (view) (copy of )) in a buffer
(ET) that cannot run the operation (CPY)`. RWKV-6's recurrent state cache
requires a `GGML_OP_CPY` the ET backend does not implement -- unlike
RWKV-7 (already on this board), which apparently uses a different,
supported state-update mechanism. This is a genuine architecture
incompatibility, not a configuration issue, so RWKV-6 is not included in
this PR or any other. Flagging in case it's useful signal for future ET
backend op coverage work.
