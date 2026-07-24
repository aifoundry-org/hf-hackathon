# GPT-2 Porting Recipe

## Overview

Adds `gpt2` (OpenAI's original 124M-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model port.
Unlike the `distilbert_sst2`/`roberta_sst2` classification ports in this
same tree, this is a decoder-only causal LM, so it uses the shared runner's
existing, unmodified `"api": "completion"` path -- the exact same scoring
mechanism already used by `qwen25_05b`, `smollm2_135m`, and every other
`llama_server`-family model on the board. No changes to any protected file
were needed or made.

## Why this port needed no new ET-SoC1 kernel work

GPT-2 uses the standard `LLM_ARCH_GPT2` graph in this repo's vendored
`llama.cpp-et` fork: learned absolute position embeddings, LayerNorm,
multi-head causal self-attention, GELU-activated FFN. Every op involved
(`GGML_OP_NORM`, `GGML_OP_SOFT_MAX`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_UNARY_OP_GELU`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board. `convert_hf_to_gguf.py`
already registers `GPT2LMHeadModel`, and the GGUF used here is a pre-made
community conversion (see Model Reference), so this port is pure
configuration wiring: one `artifacts.json` entry, one
`ported_models/llama_cpp_et/benchmarks/gpt2.json` benchmark config, one
`.github/ci/benchmark_config.json` registration line.

## Model Reference

- **GGUF source**: `igorbkz/gpt2-Q8_0-GGUF` (Hugging Face)
- **File**: `gpt2.Q8_0.gguf`
- **License**: MIT (matches upstream `openai-community/gpt2`)
- **Architecture**: `GPT2LMHeadModel`, 12 transformer layers, 768 hidden,
  12 heads, 3072 FFN, GELU activation, 50257-token BPE vocab, 1024 native
  context length.
- **Quantization**: Q8_0 (~176 MB on disk).

## Steps Taken

1. **Downloaded** `gpt2.Q8_0.gguf` from the Hugging Face repo above;
   verified `sha256=4d1df054e273ac10b3a31a2f90660426212d033156bb0b530e7517662fbe0c32`,
   `size=176095840` bytes against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18085`. Confirmed full `13/13` layer ET
   offload (12 repeating layers + output layer) and a clean 453-node
   compute graph. Sent the same gate prompt used by every other
   `llama_server`-family model on the board ("Repeat this token sequence
   without commentary: OK OK OK OK OK OK OK OK OK OK") to `/completion`;
   the server correctly tokenized it (17 tokens) and completed prompt
   processing with no errors on each of three separate requests. Multi-token
   decode output was **not** captured locally within this session: sysemu is
   a cycle-accurate software simulator of the ET-SoC1, and even a single
   decoded token took long enough that a 40-minute client timeout elapsed
   without a response -- consistent with this repo's own `ci_smoke` config
   budgeting a 3-hour (`10800`s) launcher timeout for exactly this reason.
   This is a sysemu-speed limitation, not a correctness signal either way;
   the actual per-PR board score for `"board": true` models is produced by
   the self-hosted **real ET-SoC1 hardware** runner
   (`.github/workflows/benchmark-board.yml`, `runs-on: [self-hosted, Linux,
   X64, et-soc1, board-ci, single-board]`), not local sysemu, and real
   hardware is expected to be dramatically faster than a cycle-accurate
   simulator.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new `gpt2_q8_gguf`
     artifact entry, following the exact `qwen25_05b_q8_gguf` pattern
     (Hugging Face `source`, pinned `sha256`, `size`, `local_cache`,
     `board_path`).
   - `ported_models/llama_cpp_et/benchmarks/gpt2.json` -- new benchmark
     config, following the `qwen25_05b.json`/`smollm2_135m.json` template:
     `"api": "completion"`, `tokens_per_second` score metric, perplexity
     quality gate against `wikitext2_raw_test` (same corpus every other
     text model on the board uses), `ctx_size: 1024` (GPT-2's native
     context length -- verified from the model's own printed
     `n_ctx_orig_yarn` at load time), port `18095` (next free port after
     the existing `18080`-`18094`/`18100`/`18107` allocations).
   - `.github/ci/benchmark_config.json` -- new `"gpt2"` entry pointing at
     the benchmark config above. This file is not on the protected-path
     list; only the shared runner scripts and reference/oracle files are
     protected, per `.github/workflows/benchmark-board.yml`.

## Instructions for Reproduction

```bash
# Download
huggingface-cli download igorbkz/gpt2-Q8_0-GGUF gpt2.Q8_0.gguf --local-dir .

# Board verification (manual, matches what CI will run automatically)
./bin/llama-server -m gpt2.Q8_0.gguf --device ET -ngl 99 --port 18095
curl -s http://127.0.0.1:18095/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- This is a small (124M param) model chosen to validate the causal-LM
  onboarding path end-to-end before larger ports; happy to see it scored
  alongside the existing small models (`smollm2_135m`, `smollm2_360m`) for
  context.
- No changes were made to any protected file. This PR only adds new
  artifact/benchmark-config entries and a new benchmark JSON file, in the
  same shape as every prior `llama_server`-family model addition.
