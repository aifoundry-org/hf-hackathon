# Maincoder-1B Porting Recipe

## Overview

Adds `Maincode/Maincoder-1B` (1.03B-parameter code-focused causal LM) to
the `llama_cpp_et` benchmark suite as a fully automated, board-scored
model port, using the official `Maincode/Maincoder-1B-GGUF` Q8_0 GGUF
(published by the model's own authors). Same rationale as the other
ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

This introduces the **Maincoder** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

Maincoder uses grouped-query attention (16 query heads / 4 KV heads),
RoPE, RMSNorm, and a standard FFN, plus a Qwen2-style BPE tokenizer with
FIM (fill-in-the-middle) special tokens for code infilling
(tokenizer-level, not graph-level). Every op involved (`GGML_OP_RMS_NORM`,
`GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_SOFT_MAX`,
`GGML_OP_GET_ROWS`) is already proven on the ET backend by the existing
decoder-only models on the board. `convert_hf_to_gguf.py` registers this
as `arch = maincoder` (`MaincoderForCausalLM`), and the GGUF used here is
published by the model's own authors, so this port is pure configuration
wiring.

## Model Reference

- **GGUF source**: `Maincode/Maincoder-1B-GGUF` (Hugging Face, official)
- **File**: `Maincoder-1B-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `Maincode/Maincoder-1B`)
- **Architecture**: `arch = maincoder`, 32 transformer layers, GQA (16
  query heads / 4 KV heads), Qwen2-style BPE tokenizer with FIM tokens
  (151936 vocab), 2048-token native context.
- **Quantization**: Q8_0 (~1.02 GiB per the model's own reported file
  size; ~1.10 GB on disk).

## Steps Taken

1. **Downloaded** `Maincoder-1B-Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=43cd43d942de46327ff82a77e63717ef9a34d97787a3c5dcd309a9da3978a668`,
   `size=1096604672` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18118`. Confirmed the model correctly
   identifies as `arch = maincoder`, `1.03 B` params, full `33/33` layer
   ET offload (31 repeating layers + output layer), and a clean 1286-node
   / 2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- the real per-PR board score for `"board": true` models
   comes from the self-hosted **real ET-SoC1 hardware** runner, not local
   sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `maincoder_1b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/maincoder_1b.json` -- new
     benchmark config, port `18118` (next free port after this session's
     earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"maincoder_1b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download Maincode/Maincoder-1B-GGUF Maincoder-1B-Q8_0.gguf --local-dir .
./bin/llama-server -m Maincoder-1B-Q8_0.gguf --device ET -ngl 99 --port 18118
curl -s http://127.0.0.1:18118/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
