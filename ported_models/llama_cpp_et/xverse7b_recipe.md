# XVERSE-7B-Chat Porting Recipe

## Overview

Adds `xverse-ai/XVERSE-7B-Chat` (7.30B-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using the official `xverse/XVERSE-7B-Chat-GGUF` Q8_0 GGUF
(published by the model's own authors). Same rationale as the other
ports in this session: decoder-only causal LM, uses the existing,
unmodified `"api": "completion"` benchmark path -- no protected file
touched.

This introduces the **XVERSE** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

XVERSE uses standard multi-head attention (no GQA, `n_head_kv ==
n_head`), RoPE, RMSNorm, and a standard FFN. Every op involved
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` already registers `XverseForCausalLM`, and the
GGUF used here is published by the model's own authors, so this port is
pure configuration wiring.

## Model Reference

- **GGUF source**: `xverse/XVERSE-7B-Chat-GGUF` (Hugging Face, official)
- **File**: `xverse-7b-chat-q8_0.gguf`
- **License**: Apache-2.0 code license + XVERSE model license (free for
  commercial use per upstream; matches `xverse-ai/XVERSE-7B-Chat`)
- **Architecture**: `arch = xverse`, 32 transformer layers, standard MHA
  (no GQA), 100534-token SentencePiece vocab, 8192-token native context.
- **Quantization**: Q8_0 (~7.22 GiB per the model's own reported file
  size; ~7.76 GB on disk).

## Steps Taken

1. **Downloaded** `xverse-7b-chat-q8_0.gguf` from the Hugging Face repo
   above. The first download attempt **silently truncated at ~1.3 GB**
   (curl exited 0 without `--fail`, masking the incomplete transfer) --
   caught by comparing the downloaded size against the real remote
   `Content-Length` (`7758437632` bytes, confirmed via a redirect-following
   `HEAD` request), not just trusting a non-error exit code. Re-downloaded
   with `curl -L --fail`; the corrected file matches the expected size
   exactly and verified `sha256=1f2cdfeedcf3e2fe37ba3436d2a879b3f5f5a71b9ebc2ba2e888f8a3e046388d`.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18120`. Confirmed the model correctly
   identifies as `arch = xverse`, `7.30 B` params, full `33/33` layer ET
   offload (31 repeating layers + output layer), and a clean 1158-node /
   2-split compute graph. As with the other ports in this session, full
   multi-token `/completion` decode output was not captured locally within
   this session -- the real per-PR board score for `"board": true` models
   comes from the self-hosted **real ET-SoC1 hardware** runner, not local
   sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `xverse7b_q8_gguf`
     artifact entry, following the `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/xverse7b.json` -- new
     benchmark config, port `18120` (next free port after this session's
     earlier additions), `ready_timeout_s: 300` / `request_timeout_s: 420`
     (generous, given the ~7B size).
   - `.github/ci/benchmark_config.json` -- new `"xverse7b"` entry pointing
     at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download xverse/XVERSE-7B-Chat-GGUF xverse-7b-chat-q8_0.gguf --local-dir .
./bin/llama-server -m xverse-7b-chat-q8_0.gguf --device ET -ngl 99 --port 18120
curl -s http://127.0.0.1:18120/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.
