# CodeShell-7B-Chat Porting Recipe

## Overview

Adds `WisdomShell/CodeShell-7B-Chat` (7.98B-parameter code-focused causal
LM, from Peking University's Knowledge Computing Lab) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using `DevQuasar/WisdomShell.Shell-7B-Chat-GGUF`'s Q8_0 GGUF. Same
rationale as the other ports in this session: decoder-only causal LM,
uses the existing, unmodified `"api": "completion"` benchmark path -- no
protected file touched.

This introduces the **CodeShell** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

CodeShell uses grouped-query attention (32 query heads / 8 KV heads),
RoPE, LayerNorm, and a standard FFN, plus fill-in-the-middle (FIM) special
tokens for code infilling (a tokenizer-level feature, not a graph-level
one). Every op involved (`GGML_OP_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`,
`GGML_OP_ADD`, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven
on the ET backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `CodeShellForCausalLM`, and the
GGUF used here is a pre-made community conversion, so this port is pure
configuration wiring.

## Model Reference

- **GGUF source**: `DevQuasar/WisdomShell.Shell-7B-Chat-GGUF` (Hugging
  Face)
- **File**: `WisdomShell.Shell-7B-Chat.Q8_0.gguf`
- **License**: CodeShell License Agreement (matches upstream
  `WisdomShell/CodeShell-7B-Chat`)
- **Architecture**: `CodeShellForCausalLM`, 42 transformer layers, GQA (32
  query heads / 8 KV heads), FIM-aware BPE tokenizer (70144 vocab),
  8192-token native context.
- **Quantization**: Q8_0 (~8.48 GB on disk).

## Steps Taken

1. **Downloaded** `WisdomShell.Shell-7B-Chat.Q8_0.gguf` from the Hugging
   Face repo above; verified `sha256=207e1ee777455b17d3e58f3174dead52f1c8d007690c96679a7deba355284876`,
   `size=8482151296` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18113`. Confirmed the model correctly
   identifies as `arch = codeshell`, `7.98 B` params, full `43/43` layer
   ET offload (41 repeating layers + output layer), and a clean 1645-node
   / 86-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- the real per-PR board score for `"board": true` models
   comes from the self-hosted **real ET-SoC1 hardware** runner, not local
   sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `codeshell_7b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/codeshell_7b.json` -- new
     benchmark config, port `18113` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~8B size).
   - `.github/ci/benchmark_config.json` -- new `"codeshell_7b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download DevQuasar/WisdomShell.Shell-7B-Chat-GGUF WisdomShell.Shell-7B-Chat.Q8_0.gguf --local-dir .
./bin/llama-server -m WisdomShell.Shell-7B-Chat.Q8_0.gguf --device ET -ngl 99 --port 18113
curl -s http://127.0.0.1:18113/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
