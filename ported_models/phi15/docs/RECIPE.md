# Phi-1.5 Porting Recipe

## Overview

Adds `microsoft/phi-1_5` (1.3B-parameter causal LM) to the `llama_cpp_et`
benchmark suite as a fully automated, board-scored model port, using
`TKDKid1000/phi-1_5-GGUF`'s Q8_0 GGUF. Same rationale as the other ports in
this session: decoder-only causal LM, uses the existing, unmodified
`"api": "completion"` benchmark path -- no protected file touched.

This introduces the **Phi** execution family to the board, distinct from
every other family currently scored here.

## Why this port needed no new ET-SoC1 kernel work

Phi-1.5 uses `LLM_ARCH_PHI2`-style construction: LayerNorm (not RMSNorm),
partial rotary position embeddings (rotary applied to a subset of head
dimensions, still expressed as ordinary `GGML_OP_ROPE` over a narrower
tensor slice), parallel attention+FFN blocks (both branches read the same
normed input and sum), and a GELU FFN. Every op involved
(`GGML_OP_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_UNARY` for GELU, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already
proven on the ET backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `PhiForCausalLM`, and the GGUF
used here is a pre-made community conversion, so this port is pure
configuration wiring.

## Model Reference

- **GGUF source**: `TKDKid1000/phi-1_5-GGUF` (Hugging Face)
- **File**: `phi-1_5-Q8_0.gguf`
- **License**: Microsoft Research License (matches upstream
  `microsoft/phi-1_5`)
- **Architecture**: `PhiForCausalLM`, 24 transformer layers, 2048 hidden,
  32 attention heads, parallel attention+FFN, partial rotary embeddings,
  GELU activation, 2048-token native context.
- **Quantization**: Q8_0 (~1.51 GB on disk).

## Steps Taken

1. **Downloaded** `phi-1_5-Q8_0.gguf` from the Hugging Face repo above;
   verified `sha256=e8c26615319e1141348b8534641da54d58e02b7baa01ee611b9c69cc07bf43fd`,
   `size=1510464928` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18099`. Confirmed full `25/25` layer ET
   offload (23 repeating layers + output layer) and a valid 897-node
   compute graph (98 splits -- notably higher than the other ports in this
   session, consistent with Phi's parallel attention+FFN structure and
   partial-rotary slicing requiring more scheduler boundaries between ET
   and CPU-fallback ops than a standard sequential transformer block).
   As with the other ports in this session, full multi-token `/completion`
   decode output was not captured locally within this session -- sysemu's
   cycle-accurate simulation makes interactive end-to-end generation
   impractical to observe for a 1.3B model. The real per-PR board score
   for `"board": true` models comes from the self-hosted **real ET-SoC1
   hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `phi15_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/phi15.json` -- new benchmark
     config, port `18099` (next free port after this session's earlier
     additions).
   - `.github/ci/benchmark_config.json` -- new `"phi15"` entry pointing at
     the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download TKDKid1000/phi-1_5-GGUF phi-1_5-Q8_0.gguf --local-dir .
./bin/llama-server -m phi-1_5-Q8_0.gguf --device ET -ngl 99 --port 18099
curl -s http://127.0.0.1:18099/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
