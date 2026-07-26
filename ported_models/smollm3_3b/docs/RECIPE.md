# SmolLM3-3B Porting Recipe

## Overview

Adds `HuggingFaceTB/SmolLM3-3B` (3B-parameter causal LM, Hugging Face's
third-generation Smol series) to the `llama.cpp-et` framework. Confirmed via
local GGUF metadata inspection: `general.architecture = smollm3` -- distinct
from `smollm2` (already a seed identity on this board's identity registry,
ineligible for credit as a size/checkpoint variant), llama.cpp registers
SmolLM3 as its own execution family (adds NoPE layers -- a subset of layers
skip RoPE entirely -- and native long-context/reasoning-mode support absent
from SmolLM2).

## Model Reference

- **Source**: `HuggingFaceTB/SmolLM3-3B` (Hugging Face), revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`
- **License**: Apache-2.0
- **GGUF source**: `ggml-org/SmolLM3-3B-GGUF` (official llama.cpp org
  quantization), file `SmolLM3-Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248`
  (verified locally against the downloaded file)
- **Architecture**: `arch = smollm3` per GGUF metadata, 326 tensors.

## Verification performed this round

Host reference: built a plain CPU-only (`GGML_ET=OFF`) configuration of the
same vendored `llama.cpp-et` source and ran `llama-perplexity` against the
board-pinned WikiText-2 corpus (`wikitext2_raw_test`,
`sha256=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`),
context 128 / batch 128 / ubatch 128 / 4 chunks. The model loads and runs
cleanly:

```
Final estimate: PPL = 12.3653 +/- 2.40724
```

## Why this port likely needs no new ET-SoC1 kernel work

SmolLM3's distinguishing feature (NoPE on a subset of layers) means those
layers simply skip the `GGML_OP_ROPE` call rather than needing a new op --
the graph builder conditionally omits an already-proven op per layer. Every
other op (RMSNorm, GQA attention, SwiGLU FFN) is identical to every other
Smol-family/Llama-family decoder already proven on this backend. Not
confirmed live this round.

## Open items for maintainer review

- Registered in `artifacts.json`, `ported_models/llama_cpp_et/benchmarks/smollm3_3b.json`,
  and `.github/ci/benchmark_config.json` (port 18135) -- board-testable now,
  independent of the model-ports track claim below.
- `ported_models/submissions/model_ports/smollm3_3b.json` is the model-ports
  track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
