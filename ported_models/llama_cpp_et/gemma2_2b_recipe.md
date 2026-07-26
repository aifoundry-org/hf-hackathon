# Gemma-2-2b-it Porting Recipe

## Overview

Adds `google/gemma-2-2b-it` (2B-parameter instruction-tuned causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model port.
Same rationale as the `gpt2` port added alongside it in this PR: decoder-only
causal LM, scored via the shared runner's existing, unmodified
`"api": "completion"` path -- no changes to any protected file.

## Why this port needed no new ET-SoC1 kernel work

Gemma-2 uses `LLM_ARCH_GEMMA2` in this repo's vendored `llama.cpp-et` fork:
RMSNorm (pre- and post-layer, plus pre/post-FFN), GeGLU-activated FFN,
grouped-query attention with alternating sliding-window (SWA) and full
attention layers, and a logit soft-cap on the final projection and attention
scores. Every op involved (`GGML_OP_RMS_NORM`, `GGML_OP_SOFT_MAX`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_UNARY_OP_GELU`/`GGML_OP_SIGMOID` for
GeGLU, `GGML_OP_GET_ROWS`, `GGML_OP_TANH` for logit soft-capping) is already
proven on the ET backend by the existing `gemma3n_e2b` port on this same
board. `convert_hf_to_gguf.py` already registers `Gemma2ForCausalLM`, and the
GGUF used here is a pre-made community conversion (see Model Reference), so
this port is pure configuration wiring, mirroring `gemma3n_e2b.json` closely
(same family, one generation apart) and `qwen25_05b.json` structurally.

## Model Reference

- **GGUF source**: `bartowski/gemma-2-2b-it-GGUF` (Hugging Face)
- **File**: `gemma-2-2b-it-Q8_0.gguf`
- **License**: Gemma (Google's Gemma Terms of Use -- accepted upstream by
  the quantizer; this is a public, ungated GGUF repackaging of the public
  `google/gemma-2-2b-it` weights)
- **Architecture**: `Gemma2ForCausalLM`, 26 transformer layers, 2304 hidden,
  8 attention heads / 4 KV heads (GQA), 9216 FFN, alternating local
  (sliding-window) and global attention, logit soft-capping, 8192-token
  native context.
- **Quantization**: Q8_0 (~2.78 GB on disk).

## Steps Taken

1. **Downloaded** `gemma-2-2b-it-Q8_0.gguf` from the Hugging Face repo above;
   verified `sha256=2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`,
   `size=2784495456` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18086`. Confirmed full `27/27` layer ET
   offload (25 repeating layers + output layer), both the SWA and non-SWA
   KV caches allocating correctly (416 MiB each, 13 layers per cache type,
   matching the model's alternating attention pattern), and a clean
   1154-node / 56-split compute graph. As with the `gpt2` port added
   alongside this one, full `/completion` decode output was **not**
   captured locally within this session -- sysemu's cycle-accurate
   simulation is far too slow to observe end-to-end generation
   interactively for a 2B model (the model's own warmup pass alone ran
   for over 50 minutes of sustained ~95% CPU before this recipe was
   written). This is a sysemu-speed limitation only; the real per-PR
   board score for `"board": true` models comes from the self-hosted
   **real ET-SoC1 hardware** runner
   (`.github/workflows/benchmark-board.yml`, `runs-on: [self-hosted, Linux,
   X64, et-soc1, board-ci, single-board]`), not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `gemma2_2b_q8_gguf`
     artifact entry, following the exact `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/gemma2_2b.json` -- new benchmark
     config, following the `gemma3n_e2b.json`/`qwen25_05b.json` template:
     `"api": "completion"`, `tokens_per_second` score metric, perplexity
     quality gate against `wikitext2_raw_test`, `ctx_size: 2048` (matching
     every other mid-size model on the board rather than the model's full
     8192 native context, to keep KV cache size and eval time comparable
     to peers), port `18096` (next free port after the existing
     `18080`-`18096` allocations), `ready_timeout_s: 300` /
     `request_timeout_s: 420` (matching `gemma3n_e2b.json`'s generous
     timeouts, since this is a 2B model on sysemu, not the small models'
     faster defaults).
   - `.github/ci/benchmark_config.json` -- new `"gemma2_2b"` entry pointing
     at the benchmark config above.

## Instructions for Reproduction

```bash
# Download
huggingface-cli download bartowski/gemma-2-2b-it-GGUF gemma-2-2b-it-Q8_0.gguf --local-dir .

# Board verification (manual, matches what CI will run automatically)
./bin/llama-server -m gemma-2-2b-it-Q8_0.gguf --device ET -ngl 99 --port 18096
curl -s http://127.0.0.1:18096/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- `ctx_size` is set to 2048 (not the model's full 8192) to keep sysemu load
  time and KV cache footprint comparable to the other mid-size models
  already on the board (`gemma3n_e2b`, `llama32_1b`). Happy to raise it if
  the board prefers testing at native context length.
- No changes were made to any protected file. This PR only adds new
  artifact/benchmark-config entries and a new benchmark JSON file, in the
  same shape as every prior `llama_server`-family model addition.
