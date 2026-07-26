# Refact-1.6B-fim Porting Recipe

## Overview

Adds `smallcloudai/Refact-1_6B-fim` (1.59B-parameter code-completion
causal LM) to the `llama_cpp_et` benchmark suite as a fully automated,
board-scored model port, using `oblivious/Refact-1.6B-fim-GGUF`'s Q8_0
GGUF. Same rationale as the other ports in this session: decoder-only
causal LM, uses the existing, unmodified `"api": "completion"` benchmark
path -- no protected file touched.

This introduces the **Refact** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

Refact uses multi-query attention (a single shared KV head across all
query heads, the same MQA pattern already proven by `gpt2` and
`falcon7b` earlier this session), ALiBi-style/RoPE positional handling,
LayerNorm, and a GELU FFN, plus fill-in-the-middle (FIM) special tokens
for code infilling (tokenizer-level, not graph-level). Every op involved
(`GGML_OP_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_UNARY` for GELU, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is
already proven on the ET backend by the existing decoder-only models on
the board. `convert_hf_to_gguf.py` registers this as `arch = refact`
(`GPTRefactForCausalLM`), and the GGUF used here is a pre-made community
conversion, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `oblivious/Refact-1.6B-fim-GGUF` (Hugging Face)
- **File**: `refact-1_6b-Q8_0.gguf`
- **License**: BigScience OpenRAIL-M (matches upstream
  `smallcloudai/Refact-1_6B-fim`)
- **Architecture**: `refact` (`GPTRefactForCausalLM`), 32 transformer
  layers, multi-query attention, FIM-aware BPE tokenizer (49216 vocab),
  4096-token native context.
- **Quantization**: Q8_0 (~1.69 GB on disk).

## Steps Taken

1. **Downloaded** `refact-1_6b-Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=2298c7e2ad48d17e9db5c46e783b55dfe9e8ed99a00364b9c1f3bd14c0f6519f`,
   `size=1687086368` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18115`. Confirmed the model correctly
   identifies as `arch = refact`, `1.59 B` params, full `33/33` layer ET
   offload (31 repeating layers + output layer), and a clean 1094-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- the real per-PR board score for `"board": true` models
   comes from the self-hosted **real ET-SoC1 hardware** runner, not local
   sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `refact16b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/refact16b.json` -- new
     benchmark config, port `18115` (next free port after this session's
     earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"refact16b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download oblivious/Refact-1.6B-fim-GGUF refact-1_6b-Q8_0.gguf --local-dir .
./bin/llama-server -m refact-1_6b-Q8_0.gguf --device ET -ngl 99 --port 18115
curl -s http://127.0.0.1:18115/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
