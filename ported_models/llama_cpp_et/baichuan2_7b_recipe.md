# Baichuan2-7B-Chat Porting Recipe

## Overview

Adds `baichuan-inc/Baichuan2-7B-Chat` (7.51B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using `shaowenchen/baichuan2-7b-chat-gguf`'s Q8_0 GGUF. Same
rationale as the other ports in this session: decoder-only causal LM,
uses the existing, unmodified `"api": "completion"` benchmark path -- no
protected file touched.

**Unlike the other ports added this session, this does not introduce a
new execution family.** The load log reports `arch = llama`, `general.name
= LLaMA v2` -- Baichuan2-7B is structurally a Llama-2 clone (same RoPE,
RMSNorm, SwiGLU FFN, tensor layout) with its own tokenizer/vocab
(125696-token SentencePiece) and pretraining data. `convert_hf_to_gguf.py`
registers a distinct `BaichuanForCausalLM` class, but for this checkpoint
the conversion resolved to the plain `LLM_ARCH_LLAMA` graph, identical to
every other Llama-family model already on this board (qwen2/3, smollm2,
llama32, tinyllama, deepseek_r1). This is flagged explicitly so the board
diversity claim stays honest: this PR adds one more *model*, not one more
*architecture family*.

## Why this port needed no new ET-SoC1 kernel work

Since it resolves to `LLM_ARCH_LLAMA`, every op it needs
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_GLU`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven
by every existing Llama-family model on this board.

## Model Reference

- **GGUF source**: `shaowenchen/baichuan2-7b-chat-gguf` (Hugging Face)
- **File**: `baichuan2-7b-chat.Q8_0.gguf`
- **License**: Baichuan2 Community License (matches upstream
  `baichuan-inc/Baichuan2-7B-Chat`)
- **Architecture**: converts to `LLM_ARCH_LLAMA` (see above), 32
  transformer layers, standard MHA (`n_head_kv == n_head`, no GQA),
  4096-token native context, 125696-token SentencePiece vocab.
- **Quantization**: Q8_0 (~7.98 GB on disk).

## Steps Taken

1. **Downloaded** `baichuan2-7b-chat.Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=7388d78f8b68209106aeb8b591e6404d07a66153cb8c134678bb833e2baf5c46`,
   `size=7978757472` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18110`. Confirmed the model correctly
   identifies as `7.51 B` params, full `33/33` layer ET offload (31
   repeating layers + output layer), and a clean 1158-node / 2-split
   compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally
   within this session -- sysemu's cycle-accurate simulation makes
   interactive end-to-end generation impractical to observe at 7B scale.
   The real per-PR board score for `"board": true` models comes from the
   self-hosted **real ET-SoC1 hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `baichuan2_7b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/baichuan2_7b.json` -- new
     benchmark config, port `18110` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the 7B size).
   - `.github/ci/benchmark_config.json` -- new `"baichuan2_7b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download shaowenchen/baichuan2-7b-chat-gguf baichuan2-7b-chat.Q8_0.gguf --local-dir .
./bin/llama-server -m baichuan2-7b-chat.Q8_0.gguf --device ET -ngl 99 --port 18110
curl -s http://127.0.0.1:18110/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
