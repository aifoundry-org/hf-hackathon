# StableLM-2-1.6b Porting Recipe

## Overview

Adds `stabilityai/stablelm-2-1_6b` (1.6B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using `afrideva/stablelm-2-1_6b-GGUF`'s Q8_0 GGUF. Same rationale as
the other ports in this session: decoder-only causal LM, uses the
existing, unmodified `"api": "completion"` benchmark path -- no protected
file touched.

This introduces the **StableLM-2** execution family to the board, distinct
from every other family currently scored here.

## Why this port needed no new ET-SoC1 kernel work

StableLM-2 uses `LLM_ARCH_STABLELM`-style construction: parallel
LayerNorm-per-head-group (multi-query-ish normalization applied per
attention/FFN branch), partial rotary position embeddings, and a
SwiGLU-gated FFN. Every op involved (`GGML_OP_NORM`, `GGML_OP_ROPE`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_GLU` for SwiGLU gating,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET backend
by the existing decoder-only models on the board (the `GLU` op in
particular is already exercised by every Llama-family/Gemma model's gated
FFN). `convert_hf_to_gguf.py` already registers `StableLmForCausalLM`, and
the GGUF used here is a pre-made community conversion, so this port is
pure configuration wiring.

## Model Reference

- **GGUF source**: `afrideva/stablelm-2-1_6b-GGUF` (Hugging Face)
- **File**: `stablelm-2-1_6b.q8_0.gguf`
- **License**: `stabilityai-stablelm-2-1_6b` (StabilityAI's non-commercial
  research license -- matches upstream `stabilityai/stablelm-2-1_6b`)
- **Architecture**: `StableLmForCausalLM`, 24 transformer layers, 2048
  hidden, 32 attention heads, SwiGLU FFN, partial rotary embeddings,
  4096-token native context.
- **Quantization**: Q8_0 (~1.75 GB on disk).

## Steps Taken

1. **Downloaded** `stablelm-2-1_6b.q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=fcdeda8182b99b007cb2c69b216e31efd21b78a4588188b076fa92821f3fee20`,
   `size=1751879200` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18101`. Confirmed full `25/25` layer ET
   offload (23 repeating layers + output layer) and a clean 991-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- sysemu's cycle-accurate simulation makes interactive
   end-to-end generation impractical to observe for a 1.6B model. The real
   per-PR board score for `"board": true` models comes from the
   self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `stablelm2_16b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/stablelm2_16b.json` -- new
     benchmark config, port `18101` (next free port after this session's
     earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"stablelm2_16b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download afrideva/stablelm-2-1_6b-GGUF stablelm-2-1_6b.q8_0.gguf --local-dir .
./bin/llama-server -m stablelm-2-1_6b.q8_0.gguf --device ET -ngl 99 --port 18101
curl -s http://127.0.0.1:18101/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
