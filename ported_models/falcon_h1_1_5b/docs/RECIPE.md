# Falcon-H1-1.5B-Instruct Porting Recipe

## Overview

Adds `tiiuae/Falcon-H1-1.5B-Instruct` (1.5B-parameter causal LM, TII's
newer hybrid attention+SSM architecture) to the `llama.cpp-et` framework.
Confirmed via local GGUF metadata inspection: `general.architecture =
falcon-h1` -- distinct from `falcon` (already ported earlier in this
campaign), a genuinely different architecture, not just a size variant:
Falcon-H1 interleaves standard attention layers with Mamba-style recurrent
layers in the same model (hybrid, not pure transformer or pure SSM).

## Model Reference

- **Source**: `tiiuae/Falcon-H1-1.5B-Instruct` (Hugging Face), revision
  `80ebc50d7799a440b96c93bb6686a3924a09b0cb`
- **License**: Falcon LLM License (`falcon-llm-license`,
  https://falconllm.tii.ae/falcon-terms-and-conditions.html) -- a custom
  TII license, not a standard OSI license; same family as falcon7b's
  license in this campaign but a distinct license id
- **GGUF source**: `tiiuae/Falcon-H1-1.5B-Instruct-GGUF` (official), file
  `Falcon-H1-1.5B-Instruct-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=0044322ad5aaea5ccc2264f83f430f4ab2d20acc6db71a12d79b3f5acb0f69d5`
  (verified locally against the downloaded file)
- **Architecture**: `arch = falcon-h1` per GGUF metadata, 411 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly, allocating **both** a standard transformer KV cache (`llama_kv_cache`,
256 cells / 24 layers) **and** a recurrent SSM state cache
(`llama_memory_recurrent`, 1 cell / 24 layers) simultaneously -- direct
confirmation this is a genuine hybrid, not mislabeled:

```
Final estimate: PPL = 11.6708 +/- 2.60807
```

This is the second SSM-adjacent model in this campaign (after
`mamba_1_4b`) to confirm `SSM_CONV`/`SSM_SCAN` work on `ggml-cpu`, and the
first to combine them with standard attention in one graph.

## Why this port's ET-SoC1 kernel support is a real, open question

Same caveat as `mamba_1_4b`: CPU-backend success does not prove the
ET-SoC1 backend supports `SSM_CONV`/`SSM_SCAN`. This is a second, mostly
independent data point on the same open question (different model, same
underlying ops) -- if either mamba_1_4b or this port ever gets a real
board run, it directly answers whether ET-SoC1 can run recurrent-state
architectures at all.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/falcon_h1_1_5b.json`,
  and `.github/ci/benchmark_config.json` (port 18141) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/falcon_h1_1_5b.json` is the
  model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- SSM ops (`SSM_CONV`/`SSM_SCAN`) confirmed on CPU, NOT live-verified
  against ET sysemu specifically.
