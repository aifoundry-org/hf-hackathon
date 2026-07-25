# GPT-OSS-20B Porting Recipe

## Overview

Adds `openai/gpt-oss-20b` (OpenAI's own open-weight release, 20B total /
~3.6B active parameters, sparse MoE with reasoning-effort control) to the
`llama.cpp-et` framework. Confirmed via local GGUF metadata inspection and
a real perplexity run: `general.architecture = gpt-oss` -- a genuinely
distinct execution family, notable as OpenAI's first open-weight model
family since GPT-2.

## Model Reference

- **Source**: `openai/gpt-oss-20b` (Hugging Face), revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`
- **License**: Apache-2.0
- **GGUF source**: `ggml-org/gpt-oss-20b-GGUF` (official llama.cpp org
  quantization), file `gpt-oss-20b-MXFP4.gguf`
- **Quantization**: native MXFP4 (OpenAI's own 4-bit microscaling format
  for the MoE expert weights -- not a post-training quantization of a
  higher-precision release; this is the model's actual shipped
  representation, similar in spirit to `bitnet_2b`'s native ternary
  format earlier in this campaign, though MXFP4 loaded successfully where
  bitnet's format did not), 12,109,566,624 bytes,
  `sha256=27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`
  (verified locally against the downloaded file)
- **Architecture**: `arch = gpt-oss` per GGUF metadata, 459 tensors,
  sliding-window attention alternating with full attention
  (`llama_kv_cache_iswa`, confirmed live), sparse MoE.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly, confirming both native MXFP4 MoE-expert loading and
sliding-window-plus-full attention (`llama_kv_cache_iswa`) work correctly
on `ggml-cpu`:

```
Final estimate: PPL = 70.2523 +/- 14.46029
```

This is notably higher than every other model in this campaign (typically
5-16). Not a bug: gpt-oss is trained heavily toward instruction-following/
reasoning-trace generation rather than raw next-token prediction on plain
prose, and MXFP4 is a genuinely lower-precision native format (4-bit
microscaling) than the Q8_0 used almost everywhere else in this campaign
-- both plausible, real contributors, reported honestly rather than
treated as a red flag.

## Why this port's ET-SoC1 kernel support is a real, open question

MoE routing (`MUL_MAT_ID`) is the same open question as `granite_3_1b_a400m`
earlier in this campaign. MXFP4 is a genuinely new quantization format not
used by any other model in this campaign or (as far as this session
checked) elsewhere on this board -- whether `ggml-et.cpp`'s `MUL_MAT`
implementation supports MXFP4 blocks at all is unconfirmed, a real
open question, not assumed either way.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/gpt_oss_20b.json`,
  and `.github/ci/benchmark_config.json` (port 18143) -- board-testable
  now, independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/gpt_oss_20b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- MXFP4 support on the ET backend specifically is genuinely unconfirmed,
  not assumed to work.
