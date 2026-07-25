# InternLM3-8B-Instruct Porting Recipe

## Overview

Adds `internlm/internlm3-8b-instruct` (8.80B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using the official `internlm/internlm3-8b-instruct-gguf` Q8_0 GGUF
(published by the model's own authors). Same rationale as the other
ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

**Like `baichuan2_7b` earlier this session, this does not introduce a new
execution family.** The load log reports `arch = llama` -- this
checkpoint's GGUF conversion resolves to the plain `LLM_ARCH_LLAMA` graph
(InternLM3 uses standard GQA transformer blocks, architecturally very
close to Llama/Qwen at the graph level), not a distinct
`InternLM2ForCausalLM`/`InternLM3ForCausalLM` path (the `internlm2_5_18b`
port added earlier this session *did* register as its own distinct
`internlm2`-tagged architecture in llama.cpp's internals; this one did
not). Flagged explicitly so the board-diversity claim stays honest: this
is one more *model*, not one more *architecture*.

## Why this port needed no new ET-SoC1 kernel work

Since it resolves to `LLM_ARCH_LLAMA`, every op it needs
(`GGML_OP_RMS_NORM` via norm epsilon config, `GGML_OP_ROPE`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`)
is already proven by every existing Llama-family model on this board.

## Model Reference

- **GGUF source**: `internlm/internlm3-8b-instruct-gguf` (Hugging Face,
  official)
- **File**: `internlm3-8b-instruct-q8_0.gguf`
- **License**: Apache-2.0 (matches upstream
  `internlm/internlm3-8b-instruct`)
- **Architecture**: converts to `LLM_ARCH_LLAMA` (see above), 48
  transformer layers, GQA (32 query heads / 2 KV heads), 32768-token
  native context, 128512-token SentencePiece vocab.
- **Quantization**: Q8_0 (~8.71 GiB per the model's own reported file
  size; ~9.36 GB on disk).

## Steps Taken

1. **Downloaded** `internlm3-8b-instruct-q8_0.gguf` from the Hugging Face
   repo above; verified `sha256=4b6cd620c74ff56aa465d31f30b9e54a0a73defe18dd979de84e82d6ef54174b`,
   `size=9358826688` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18117`. Confirmed the model correctly
   identifies as `8.80 B` params, full `49/49` layer ET offload (47
   repeating layers + output layer), and a clean 1734-node / 2-split
   compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally
   within this session -- the real per-PR board score for `"board": true`
   models comes from the self-hosted **real ET-SoC1 hardware** runner,
   not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `internlm3_8b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/internlm3_8b.json` -- new
     benchmark config, port `18117` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~9B size).
   - `.github/ci/benchmark_config.json` -- new `"internlm3_8b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download internlm/internlm3-8b-instruct-gguf internlm3-8b-instruct-q8_0.gguf --local-dir .
./bin/llama-server -m internlm3-8b-instruct-q8_0.gguf --device ET -ngl 99 --port 18117
curl -s http://127.0.0.1:18117/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
