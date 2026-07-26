# Mamba-1.4B Porting Recipe

## Overview

Adds `state-spaces/mamba-1.4b-hf` (1.4B-parameter State Space Model, not a
transformer) to the `llama.cpp-et` framework. Confirmed via local GGUF
metadata inspection: `general.architecture = mamba` -- a completely
different compute paradigm from every other model in this campaign
(no attention, no KV cache; a recurrent SSM state updated per token via
selective scan).

## Model Reference

- **Source**: `state-spaces/mamba-1.4b-hf` (Hugging Face), revision
  `6e46eae61c27280517feef46f536d16b91076f08`. HF repo metadata has no
  `license` tag and the README does not state one -- flagging honestly
  rather than assuming Apache-2.0 just because the reference Mamba GitHub
  repo uses that license.
- **GGUF source**: `RichardErkhov/state-spaces_-_mamba-1.4b-hf-gguf`, file
  `mamba-1.4b-hf.Q6_K.gguf` (no Q8_0 available at this source -- highest
  quant offered is Q6_K)
- **Quantization**: Q6_K,
  `sha256=a5dde81ce41c34213bf26bcaf2ba26cd1c6fde20fd20b5af6a4a76a5272478bc`
  (verified locally against the downloaded file)
- **Architecture**: `arch = mamba` per GGUF metadata, 482 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly, using a **recurrent memory cache** instead of the usual
transformer KV cache (`llama_memory_recurrent`, 1 cell instead of 256 --
SSM state is a fixed-size running summary, not a growing per-token cache):

```
Final estimate: PPL = 11.7985 +/- 1.97189
```

This confirms `GGML_OP_SSM_CONV`/`GGML_OP_SSM_SCAN` (the ops Mamba's
selective-scan recurrence needs) work correctly on the CPU backend -- the
first genuinely non-attention architecture exercised in this campaign.

## Why this port's ET-SoC1 kernel support is a real, open question

CPU-backend success does not prove the ET-SoC1 backend supports
`SSM_CONV`/`SSM_SCAN` at all. Unlike the MoE routing question for
granitemoe (where `MUL_MAT_ID` was at least listed in `ggml-et.cpp`'s
op-coverage audit from earlier in this campaign), these SSM-specific ops
were not checked against `ggml-et.cpp` this round. Flagging as completely
open, not a formality -- this may be a genuine ET-SoC1 blocker similar to
this campaign's RWKV-6/PLaMo-2 findings (both blocked on a different
missing op, `GGML_OP_CPY`).

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/mamba_1_4b.json`,
  and `.github/ci/benchmark_config.json` (port 18139) -- board-testable now
  (this will directly answer the SSM-op-support question above),
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/mamba_1_4b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- License unspecified in the HF repo metadata -- do not assume Apache-2.0.
