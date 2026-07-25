# PLaMo-13B Porting Recipe

## Overview

Adds `pfnet/plamo-13b` (13B-parameter causal LM, Preferred Networks' first
PLaMo release) to the `llama.cpp-et` framework. Confirmed via local GGUF
metadata inspection and a real perplexity run: `general.architecture =
plamo` -- distinct from `plamo2` (a confirmed negative result earlier in
this campaign, blocked on missing `GGML_OP_CPY`) and `plamo3` (already
claimed). The original, first-generation PLaMo architecture.

## Model Reference

- **Source**: `pfnet/plamo-13b` (Hugging Face), revision
  `88237e8483cdf6672faf3144f76f73f89b96d30c`
- **License**: Apache-2.0
- **GGUF source**: `RichardErkhov/pfnet_-_plamo-13b-gguf`, file
  `plamo-13b.Q8_0.gguf`
- **Quantization**: Q8_0, 13,920,140,960 bytes,
  `sha256=7f3f26fd8eb337e82f4e1a5916eccf2b96a7d0e2bd20af71b4c52946c327c66f`
  (verified locally against the downloaded file -- the first download
  attempt was truncated at ~8.5 GB by the same disk-space exhaustion
  issue documented on `lfm25_8b_a1b`; re-downloaded and re-verified
  complete and correct after fixing that)
- **Architecture**: `arch = plamo` per GGUF metadata, 323 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 31.5102 +/- 6.73369
```

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_PLAMO`'s graph builder is a standard pre-norm transformer
(RMSNorm, RoPE, standard MHA, SwiGLU FFN) -- unlike `plamo2`'s Mamba-hybrid
recurrent mechanism (which needs `GGML_OP_CPY`, unsupported on this ET
backend), this generation uses only ops already proven elsewhere on this
board.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/plamo_13b.json`,
  and `.github/ci/benchmark_config.json` (port 18145) -- board-testable
  now, independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/plamo_13b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
