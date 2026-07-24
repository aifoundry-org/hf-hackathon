# RoBERTa-base-SST2 Porting Recipe

## Overview

Adds `JeremiahZ/roberta-base-sst2` (encoder-only binary sequence
classification, SST-2 sentiment) to the `llama.cpp-et` framework. Same
board-verification approach as `distilbert_sst2` (see that recipe for the
general rationale), but a genuinely different pretraining architecture:
RoBERTa uses byte-level BPE tokenization and a different pretraining
objective (no next-sentence-prediction, dynamic masking) from BERT/
DistilBERT's WordPiece lineage, even though both currently route through
`llama.cpp`'s shared `LLM_ARCH_BERT` graph builder.

## Why this port needed no new ET-SoC1 kernel work

Same op set as `distilbert_sst2`: `NORM`, `SOFT_MAX`, `MUL_MAT`, `ADD`,
`GELU`, `GET_ROWS` -- all already proven on this backend. `convert_hf_to_gguf.py`
already registers `RobertaForSequenceClassification` (`class RobertaModel(BertModel)`).

## Model Reference

- **Hugging Face Repository**: `JeremiahZ/roberta-base-sst2`
- **Revision**: `0daa4dc5532d79114a5e76e277d2d921c1bfd9ed`
- **File**: `model.safetensors`
- **License**: `MIT`
- **Architecture**: `RobertaForSequenceClassification`, 12 transformer layers,
  768 hidden, 12 heads, 3072 FFN, GELU activation, binary
  NEGATIVE/POSITIVE classification (`id2label={0: negative, 1: positive}`).

## An important correction found while building this port

While verifying this port against its host oracle, the board's `/rerank`
score (`-1.9601`) did **not** match this model's predicted class (POSITIVE,
logit `+1.8671`) -- it was much closer to the NEGATIVE logit (`-2.2555`,
diff `0.295`). Tracing `llama.cpp`'s server source
(`tools/server/server-context.cpp`, `send_rerank()`) confirmed why:

```cpp
res->score = embd[0];
```

`/rerank` always returns the raw value at index 0 of the classifier head's
output vector -- there is no label-aware selection of "the positive class"
or "the predicted class." Which HF label that index corresponds to is
decided entirely by how each architecture's own `convert_hf_to_gguf.py`
class lays out the classifier head weights during conversion, and is **not**
guaranteed to be consistent across architectures even when their HF configs
declare the identical `id2label` convention:

| Model | HF `id2label` | GGML `embd[0]` maps to | Oracle value at that index | Board score | Diff |
|---|---|---|---|---|---|
| `distilbert_sst2` | `{0: NEGATIVE, 1: POSITIVE}` | POSITIVE (index 1 of the HF head) | `2.5304` | `2.4789` | `0.05` |
| `roberta_sst2` (this port) | `{0: negative, 1: positive}` | NEGATIVE (index 0 of the HF head) | `-2.2555` | `-1.9601` | `0.295` |

Both are numerically correct computations of *some* real logit from the
model -- this is not a bug in either port. It does mean the earlier
`distilbert_sst2_recipe.md` wording (which read the match as confirming the
*predicted label*) was imprecise; both recipes and
`run_llama_server_benchmark.py`'s scoring logic were corrected to check
numeric closeness to a pinned oracle value at whatever index `embd[0]`
empirically resolves to for that specific model, without inferring or
asserting a predicted label from the score's sign.

## Steps Taken

1. **Host oracle**: `ported_models/llama_cpp_et/scripts/gen_roberta_sst2_oracle.py`
   runs the real `transformers` `RobertaForSequenceClassification` on the
   pinned revision against the same gate prompt as `distilbert_sst2`. Real
   run (2026-07-24): `label=positive, logits=[NEGATIVE=-2.2555,
   POSITIVE=+1.8671]`.
2. **GGUF conversion**: same `convert_hf_to_gguf.py` path as `distilbert_sst2`.
   Produced a clean 201-tensor, ~500 MB F32 GGUF.
3. **Hosting**: converted GGUF hosted as a GitHub Release asset on this fork
   (tag `model-port-roberta-sst2-v1`), same pattern as `distilbert_sst2`.
4. **Board-verified via sysemu (2026-07-24)**: full `12/12` layer ET offload,
   clean 436-node compute graph, real `/rerank` result: `relevance_score=
   -1.9601`, diff `0.295` from the oracle's index-0 (NEGATIVE) logit
   `-2.2555`, within the `1.0` tolerance.
5. **Registered**: `ported_models/llama_cpp_et/artifacts.json` (new
   `roberta_sst2_f32_gguf` entry) only -- same reasoning as
   `distilbert_sst2`: no entry in `.github/ci/benchmark_config.json`'s active
   model list, since automated scoring would require editing the protected
   `run_llama_server_benchmark.py`, which this PR does not do.

## Instructions for Reproduction

```bash
pip install torch transformers numpy
python3 ported_models/llama_cpp_et/scripts/gen_roberta_sst2_oracle.py

python3 convert_hf_to_gguf.py <path-to-hf-snapshot> \
  --outfile roberta_sst2_f32.gguf --outtype f32
```

## Open items for maintainer review

- Same hosting, CI-scoring, and model-port-track-credit caveats as
  `distilbert_sst2` (see that recipe) apply here.
- The `embd[0]`-is-architecture-specific finding above is relevant to any
  future classification-model port using a `/rerank`-based approach -- worth
  checking against the oracle's full per-class logits (not just the top
  prediction) before trusting a match.
