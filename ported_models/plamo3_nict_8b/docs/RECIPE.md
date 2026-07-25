## Registration note

upstream repo is gated (auto) and non-commercial (contact Preferred Networks).

# PLaMo-3-NICT-8B-base Porting Recipe

## Overview

Adds `pfnet/plamo-3-nict-8b-base` (8.09B-parameter causal LM, from
Preferred Networks Inc.) to the `llama_cpp_et` benchmark suite as a fully
automated, board-scored model port, using
`mmnga-o/plamo-3-nict-8b-base-gguf`'s Q8_0 GGUF. Same rationale as the
other ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

This introduces the **PLaMo-3** execution family to the board -- notable
because a sibling architecture, PLaMo-2, was tried earlier in this
session and found **incompatible** with this ET backend (its hybrid
Mamba-style recurrent layer needs an unsupported `GGML_OP_CPY`, the same
failure documented for RWKV-6). PLaMo-3 is architecturally different: it
uses standard alternating sliding-window/full attention (the same pattern
already proven by `gemma2_2b` and `cohere2_7b` earlier this session), not
a recurrent state cache, so it loads and runs cleanly here.

## Why this port needed no new ET-SoC1 kernel work

PLaMo-3 uses grouped-query attention with alternating 4096-token
sliding-window and full-attention layers (confirmed at load time: a
4-layer non-SWA KV cache and a 28-layer SWA KV cache were both created
successfully), RoPE (with a separate frequency base for SWA vs full
layers), and a standard FFN. Every op involved (`GGML_OP_ROPE`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`)
is already proven on the ET backend by the existing decoder-only models
on the board. `convert_hf_to_gguf.py` registers `Plamo3ForCausalLM`, and
the GGUF used here is a pre-made community conversion, so this port is
pure configuration wiring.

## Model Reference

- **GGUF source**: `mmnga-o/plamo-3-nict-8b-base-gguf` (Hugging Face)
- **File**: `plamo-3-nict-8b-base-Q8_0.gguf`
- **License**: PLaMo Community License (matches upstream
  `pfnet/plamo-3-nict-8b-base`; non-commercial use requires contacting
  Preferred Networks -- flagged below)
- **Architecture**: `Plamo3ForCausalLM`, 32 transformer layers (4 full
  attention + 28 sliding-window, alternating), 4096-token native context
  window for the SWA layers, PLaMo2-family tokenizer (107520 vocab).
- **Quantization**: Q8_0 (~8.6 GB on disk).

## Steps Taken

1. **Downloaded** `plamo-3-nict-8b-base-Q8_0.gguf` from the Hugging Face
   repo above; verified `sha256=339e0c93206fbf495dd35d47ec2d942286c26a0cb70294546a75588eb5129bbb`,
   `size=8601238080` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18116`. Confirmed `8.09 B` params, full
   `33/33` layer ET offload (31 repeating layers + output layer), both
   the non-SWA (4-layer) and SWA (28-layer) KV caches allocated
   correctly, and a clean 1318-node / 194-split compute graph (the high
   split count is consistent with the frequent attention-type transitions
   between SWA and full layers). As with the other ports in this session,
   full multi-token `/completion` decode output was not captured locally
   within this session -- the real per-PR board score for `"board": true`
   models comes from the self-hosted **real ET-SoC1 hardware** runner, not
   local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `plamo3_nict_8b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/plamo3_nict_8b.json` -- new
     benchmark config, port `18116` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~8B size).
   - `.github/ci/benchmark_config.json` -- new `"plamo3_nict_8b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download mmnga-o/plamo-3-nict-8b-base-gguf plamo-3-nict-8b-base-Q8_0.gguf --local-dir .
./bin/llama-server -m plamo-3-nict-8b-base-Q8_0.gguf --device ET -ngl 99 --port 18116
curl -s http://127.0.0.1:18116/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- The upstream PLaMo Community License requires contacting Preferred
  Networks for commercial use -- flagging in case that affects
  eligibility, same as the CC-BY-NC-4.0 note on `cohere2_7b` in the prior
  PR.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
- Worth noting for the record: PLaMo-2 (a sibling architecture, tried
  earlier this session) is **not** portable to this backend -- its hybrid
  Mamba recurrent layer needs an unsupported `GGML_OP_CPY`. PLaMo-3 avoids
  that entirely by using standard SWA/full attention instead.
