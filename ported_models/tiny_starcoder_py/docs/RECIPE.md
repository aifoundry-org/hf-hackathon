# tiny_starcoder_py Porting Recipe

## Overview

Adds `bigcode/tiny_starcoder_py` (164M-parameter code-completion causal LM,
same architecture family as StarCoder) to the `llama_cpp_et` benchmark suite
as a fully automated, board-scored model port. Same rationale as the
`gpt2`/`gemma2_2b` ports added in the sibling PR: decoder-only causal LM
using the shared runner's existing, unmodified `"api": "completion"` path --
no changes to any protected file.

This introduces a new execution family to the board: `LLM_ARCH_STARCODER`
(GPTBigCode), distinct from the already-present `LLM_ARCH_LLAMA`-family
(qwen2/qwen3/smollm2/llama), `LLM_ARCH_GEMMA2`/`GEMMA3N`, `LLM_ARCH_GPT2`,
and `LLM_ARCH_BERT` (distilbert/roberta) families.

## Why this port needed no new ET-SoC1 kernel work

GPTBigCode uses learned absolute position embeddings, LayerNorm, multi-query
attention (MQA -- one shared KV head across all query heads, still expressed
as ordinary `GGML_OP_MUL_MAT`/`GGML_OP_SOFT_MAX` at the graph level), and a
GELU-activated FFN. Every op involved is already proven on the ET backend by
the existing decoder-only models on the board (see `gpt2_recipe.md` for the
identical op list). `convert_hf_to_gguf.py` already registers
`GPTBigCodeForCausalLM`, and the GGUF used here is a pre-made community
conversion, so this port is pure configuration wiring.

## Model Reference

- **GGUF source**: `RichardErkhov/bigcode_-_tiny_starcoder_py-gguf`
  (Hugging Face)
- **File**: `tiny_starcoder_py.Q8_0.gguf`
- **License**: BigCode OpenRAIL-M v1 (matches upstream
  `bigcode/tiny_starcoder_py`)
- **Architecture**: `GPTBigCodeForCausalLM`, 20 transformer layers, MQA,
  GELU FFN, 8192-token native context, trained on Python data from
  StarCoderData (~100B tokens).
- **Quantization**: Q8_0 (~195 MB on disk).

## Steps Taken

1. **Downloaded** `tiny_starcoder_py.Q8_0.gguf` from the Hugging Face repo
   above; verified `sha256=ffa756aaa62050ffeea4f5b8897c54aa1a4d9c9ee0783e6ccc07f10eff8b077a`,
   `size=195125568` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18097`. Confirmed full `21/21` layer ET
   offload (19 repeating layers + output layer), the model's native 8192
   context correctly detected at load time, and a clean 749-node / 2-split
   compute graph. As with `gpt2`/`gemma2_2b`, full multi-token `/completion`
   decode output was not captured locally within this session -- sysemu's
   cycle-accurate simulation makes interactive end-to-end generation
   impractical to observe; the real per-PR board score for `"board": true`
   models comes from the self-hosted **real ET-SoC1 hardware** runner, not
   local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `tiny_starcoder_py_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/tiny_starcoder_py.json` -- new
     benchmark config, following the `qwen25_05b.json`/`gpt2.json`
     template: `"api": "completion"`, `tokens_per_second` score metric,
     perplexity quality gate against `wikitext2_raw_test`, `ctx_size: 2048`
     (kept comparable to the other small models on the board rather than
     the full native 8192, matching `smollm2_135m`'s convention), port
     `18098` (next free port after the `18080`-`18096`/`18100`/`18107`
     allocations, including this session's `gpt2`/`gemma2_2b` additions on
     18095/18096).
   - `.github/ci/benchmark_config.json` -- new `"tiny_starcoder_py"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
# Download
huggingface-cli download RichardErkhov/bigcode_-_tiny_starcoder_py-gguf tiny_starcoder_py.Q8_0.gguf --local-dir .

# Board verification (manual, matches what CI will run automatically)
./bin/llama-server -m tiny_starcoder_py.Q8_0.gguf --device ET -ngl 99 --port 18098
curl -s http://127.0.0.1:18098/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file. This PR only adds new
  artifact/benchmark-config entries and a new benchmark JSON file, in the
  same shape as every prior `llama_server`-family model addition.
- As with `gpt2`/`gemma2_2b`, this port has not been claimed against the
  "most models ported" track: per `docs/SUBMISSION_GUIDE.md`, that requires
  a maintainer to first register an `identity_id` for this model on `main`
  (a separate, non-credited step), after which a follow-up PR could add a
  standalone `ported_models/tiny_starcoder_py/` root and the corresponding
  `ported_models/submissions/model_ports/tiny_starcoder_py.json` claim.
  This PR is a general board/leaderboard addition only.
