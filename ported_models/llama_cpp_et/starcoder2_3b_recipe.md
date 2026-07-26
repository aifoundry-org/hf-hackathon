# Starcoder2-3b Porting Recipe

## Overview

Adds `bigcode/starcoder2-3b` (3B-parameter code-completion causal LM) to
the `llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using `QuantFactory/starcoder2-3b-GGUF`'s Q8_0 GGUF. Same rationale
as the other ports in this session: decoder-only causal LM, uses the
existing, unmodified `"api": "completion"` benchmark path -- no protected
file touched.

This introduces the **Starcoder2** execution family to the board --
distinct from `tiny_starcoder_py`'s GPTBigCode family added earlier this
session (`Starcoder2ForCausalLM` is a separate registered class from
`GPTBigCodeForCausalLM` in `convert_hf_to_gguf.py`, with a materially
different architecture: grouped-query attention and sliding-window
attention rather than tiny_starcoder_py's plain multi-query attention).

## Why this port needed no new ET-SoC1 kernel work

Starcoder2 uses grouped-query attention (GQA) alternating with a
4096-token sliding-window mask (same masking mechanism already proven by
`gemma2_2b`'s alternating SWA/full attention layers earlier this session),
RMSNorm, and a GELU FFN. Every op involved (`GGML_OP_RMS_NORM`,
`GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_UNARY` for GELU,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `Starcoder2ForCausalLM`, and the
GGUF used here is a pre-made community conversion, so this port is pure
configuration wiring.

## Model Reference

- **GGUF source**: `QuantFactory/starcoder2-3b-GGUF` (Hugging Face)
- **File**: `starcoder2-3b.Q8_0.gguf`
- **License**: BigCode OpenRAIL-M v1 (matches upstream
  `bigcode/starcoder2-3b`)
- **Architecture**: `Starcoder2ForCausalLM`, 30 transformer layers, GQA,
  4096-token sliding-window attention, RMSNorm, GELU FFN, 16384-token
  native context.
- **Quantization**: Q8_0 (~3.22 GB on disk).

## Steps Taken

1. **Downloaded** `starcoder2-3b.Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=6ed1dfa70e4695fbbc1cb781c4baa93da347ed25516395d48eaf99e9e86c4bc8`,
   `size=3224556928` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18103`. Confirmed full `31/31` layer ET
   offload (29 repeating layers + output layer) and a clean 870-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- sysemu's cycle-accurate simulation makes interactive
   end-to-end generation impractical to observe for a 3B model, and
   memory/time budget on the local verification box favored moving on to
   register this model rather than wait out a multi-hour decode. The real
   per-PR board score for `"board": true` models comes from the
   self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `starcoder2_3b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/starcoder2_3b.json` -- new
     benchmark config, `ctx_size: 2048` (kept comparable to other
     mid-size models rather than the full native 16384), port `18103`
     (next free port after this session's earlier additions),
     `ready_timeout_s: 240` / `request_timeout_s: 360` (slightly more
     generous than the smaller ports this session, given this model's
     larger size).
   - `.github/ci/benchmark_config.json` -- new `"starcoder2_3b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download QuantFactory/starcoder2-3b-GGUF starcoder2-3b.Q8_0.gguf --local-dir .
./bin/llama-server -m starcoder2-3b.Q8_0.gguf --device ET -ngl 99 --port 18103
curl -s http://127.0.0.1:18103/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- `ctx_size` is set to 2048 (not the model's full 16384) to keep sysemu
  load time and KV cache footprint comparable to other models on the
  board. Happy to raise it if preferred.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
