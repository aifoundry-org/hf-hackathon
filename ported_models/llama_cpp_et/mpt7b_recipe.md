# MPT-7B Porting Recipe

## Overview

Adds `mosaicml/mpt-7b` (6.86B-parameter causal LM, MosaicML's original
MPT architecture) to the `llama_cpp_et` benchmark suite as a fully
automated, board-scored model port, using
`maddes8cht/mosaicml-mpt-7b-gguf`'s Q8_0 GGUF. Same rationale as the
other ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

This introduces the **MPT** execution family to the board -- notable for
using **ALiBi (Attention with Linear Biases)** instead of RoPE for
positional encoding, the first port this session that doesn't use
rotary embeddings at all (`rope type = -1`, `f_max_alibi_bias = 8.0`).

## Why this port needed no new ET-SoC1 kernel work

ALiBi adds a fixed linear bias to the raw attention scores before the
softmax, scaled by the query/key distance -- an ordinary elementwise
`GGML_OP_ADD` on the score matrix, computed at graph-build time by
llama.cpp's existing `LLM_ARCH_MPT` builder, not a new runtime op. Beyond
that, MPT uses standard multi-head attention (no GQA, `n_head_kv ==
n_head`), LayerNorm, and a GELU FFN. Every op involved (`GGML_OP_NORM`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_UNARY` for GELU,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `MPTForCausalLM`, and the GGUF
used here is a pre-made community conversion, so this port is pure
configuration wiring.

## Model Reference

- **GGUF source**: `maddes8cht/mosaicml-mpt-7b-gguf` (Hugging Face)
- **File**: `mosaicml-mpt-7b-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `mosaicml/mpt-7b`)
- **Architecture**: `arch = mpt`, 32 transformer layers, standard MHA (no
  GQA), ALiBi positional bias (no RoPE), LayerNorm, GELU FFN, 2048-token
  native context.
- **Quantization**: Q8_0 (~6.78 GiB per the model's own reported file
  size; ~7.29 GB on disk).

## Steps Taken

1. **Downloaded** `mosaicml-mpt-7b-Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=075b2cb838694a8a03b026c88d1a485190ad771d7fa141a97bec9c858369d79b`,
   `size=7286912224` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18119`. Confirmed the model correctly
   identifies as `arch = mpt`, `6.86 B` params, ALiBi enabled
   (`f_max_alibi_bias = 8.0`, `rope type = -1`), full `33/33` layer ET
   offload (31 repeating layers + output layer), and a clean 998-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally
   within this session -- the real per-PR board score for `"board": true`
   models comes from the self-hosted **real ET-SoC1 hardware** runner,
   not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `mpt7b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/mpt7b.json` -- new benchmark
     config, port `18119` (next free port after this session's earlier
     additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~7B size).
   - `.github/ci/benchmark_config.json` -- new `"mpt7b"` entry pointing at
     the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download maddes8cht/mosaicml-mpt-7b-gguf mosaicml-mpt-7b-Q8_0.gguf --local-dir .
./bin/llama-server -m mosaicml-mpt-7b-Q8_0.gguf --device ET -ngl 99 --port 18119
curl -s http://127.0.0.1:18119/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- The GGUF's own loader emits a "missing pre-tokenizer type, using
  'default'" warning at load time (a known llama.cpp quirk for older MPT
  conversions -- the tokenizer still loads and tokenizes correctly, but
  generation quality could be marginally affected versus a from-scratch
  reconversion with an explicit pre-tokenizer type). Flagging for
  visibility; did not attempt to fix since that would mean
  re-converting from the original safetensors rather than using this
  pre-made GGUF.
- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
