# DistilBERT-SST2 Porting Recipe

## Overview

Adds `distilbert-base-uncased-finetuned-sst-2-english` (encoder-only
sequence-classification, sentiment analysis) to the `llama.cpp-et` framework.
This is a new execution family for the board: every existing `llama_cpp_et`
text model (llama, qwen2, qwen3, smollm2, gemma3n, rwkv7, lfm2) is a
decoder-only causal LM scored on decode `tokens_per_second`. DistilBERT is a
bidirectional encoder with a fixed 2-way classification head (`n_cls_out=2`,
`causal_attn=0`) -- it has no next-token decode metric, so it is scored on
classification correctness against a pinned host oracle instead, per the
"Model Quality" requirement in `docs/SUBMISSION_GUIDE.md` ("another
task-appropriate metric... a faster result that materially degrades quality
is not a valid leaderboard improvement" -- the inverse also holds: a
classification model's validity gate is correctness, not speed).

## Why this port needed no new ET-SoC1 kernel work

Checked every op DistilBERT's forward pass needs against
`ggml_backend_et_device_supports_op` in
`ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et/ggml-et.cpp`:
`GGML_OP_NORM` (real LayerNorm, not RMSNorm), `GGML_OP_SOFT_MAX`,
`GGML_OP_MUL_MAT`, `GGML_OP_ADD` (residual connections), `GGML_UNARY_OP_GELU`
(FFN activation), and `GGML_OP_GET_ROWS` (token + positional embeddings) are
all already implemented and proven on this backend (the same primitives
power every existing text model's attention/FFN). The upstream `llama.cpp`
fork this project vendors already has native `LLM_ARCH_BERT` graph
construction and `convert_hf_to_gguf.py` already registers
`DistilBertForSequenceClassification` (see that script,
`class DistilBertModel(BertModel)`). This port is therefore config/build
wiring, not new silicon kernel engineering -- the first case in this repo
where an entirely new model architecture required zero new ET kernel code.

## Model Reference

- **Hugging Face Repository**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Revision**: `714eb0fa89d2f80546fda750413ed43d93601a13`
- **File**: `model.safetensors`
- **License**: `apache-2.0`
- **Architecture**: `DistilBertForSequenceClassification`, 6 transformer
  layers, 768 hidden, 12 heads, 3072 FFN, GELU activation, 2-way
  NEGATIVE/POSITIVE classification.

## Steps Taken

1. **Host oracle**: `ported_models/llama_cpp_et/scripts/gen_distilbert_sst2_oracle.py`
   runs the real `transformers` `DistilBertForSequenceClassification` on the
   pinned revision against a fixed, self-contained gate prompt ("This board
   runs faster than anything the competition has shipped.") and records the
   reference logits/label. Real run (2026-07-24): `label=POSITIVE,
   logit=2.5304`.
2. **GGUF conversion**: `python3 convert_hf_to_gguf.py <hf-cache-snapshot>
   --outfile distilbert_sst2_f32.gguf --outtype f32`, using this repo's own
   vendored `llama.cpp-et` submodule's convert script (no separate tool).
   Produced a clean 104-tensor, ~268 MB F32 GGUF; the converter correctly
   emitted `n_cls_out=2` and the `NEGATIVE`/`POSITIVE` class labels from the
   HF `config.json`.
3. **Hosting the converted artifact**: we do not have a Hugging Face upload
   token, so the converted GGUF is hosted as a GitHub Release asset on this
   fork (`DarthCeltic/hf-hackathon`, tag `model-port-distilbert-sst2-v1`) and
   referenced by direct URL + `sha256` in `artifacts.json`, exactly like any
   other `materialize_artifact`-downloaded model artifact. It is a build
   output (converted, not a verbatim redistribution of the original
   safetensors format) reproducible from steps 1-2 above.
4. **Board-verified via sysemu (2026-07-24)**: built `llama-server` from the
   committed submodule, loaded the GGUF against `--device ET -ngl 99
   --rerank`, confirmed full `7/7` layer ET offload and a clean 224-node
   compute graph, and queried `/rerank` with the same gate prompt. Real
   result: `relevance_score=2.4789`, ~2% relative difference from the host
   oracle's `2.5304` (consistent with correct computation across host-F32 vs
   ET-simulated-F32 execution paths, not a bit-exact claim).

   **Correction (found while porting a second model, roberta_sst2)**:
   `llama.cpp`'s `/rerank` always returns `embd[0]` -- the raw first logit of
   the classifier head (`tools/server/server-context.cpp`, `send_rerank()`:
   `res->score = embd[0];`), with no label-aware selection. For this specific
   checkpoint's GGUF conversion, `embd[0]` happens to land on the POSITIVE
   logit (verified: oracle logits = `[NEGATIVE=-2.5035, POSITIVE=+2.5304]`).
   That is a numeric fact about *this model's* conversion, not a general
   "index 0 is negative" or "index 0 is the top class" rule -- a different
   architecture's conversion can put a different class at index 0 (see
   `roberta_sst2_recipe.md`). The original wording here implied the score
   confirmed the *predicted label*; the more precise claim is that the board
   score matches the oracle's index-0 logit within tolerance.
5. **Runner extension**: `run_llama_server_benchmark.py` only spoke
   `/completion` and `/v1/chat/completions`. Added a third `api: "rerank"`
   mode (surgical, additive -- existing model configs are untouched) that
   issues `--rerank`, POSTs `/rerank`, and scores classification correctness
   + numeric closeness against an `oracle` block in the model's benchmark
   JSON instead of `tokens_per_second`.
6. **Registered**: `ported_models/llama_cpp_et/benchmarks/distilbert_sst2.json`
   (new) + `.github/ci/benchmark_config.json` (new `distilbert_sst2` model
   entry) + `ported_models/llama_cpp_et/artifacts.json` (new
   `distilbert_sst2_f32_gguf` artifact entry).

## Instructions for Reproduction

```bash
pip install torch transformers numpy
python3 ported_models/llama_cpp_et/scripts/gen_distilbert_sst2_oracle.py

# from the llama.cpp-et submodule root, with the HF snapshot cached locally:
python3 convert_hf_to_gguf.py <path-to-hf-snapshot> \
  --outfile distilbert_sst2_f32.gguf --outtype f32
```

Board CI downloads the hosted GGUF via `artifacts.json`'s pinned URL +
`sha256`, builds `llama-server` from the committed submodule, and runs it
against `/rerank` per `benchmarks/distilbert_sst2.json`.

## Open items for maintainer review

- We are not certain this is the intended long-term hosting pattern for a
  converted (not pre-existing) GGUF artifact -- happy to move it to an
  Hugging Face repo under an official/maintainer account, or any other
  location, if preferred.
- This is a `llama_server`-runner model-suite addition, not a submission
  under `ported_models/submissions/model_ports/` (that path's identity
  registry currently only has `smolvlm` open, already claimed) -- if this is
  meant to also count toward the "most models ported" track, it would need a
  registered identity per `docs/SUBMISSION_GUIDE.md`'s two-step process;
  we are asking about that separately and are not assuming it here.
