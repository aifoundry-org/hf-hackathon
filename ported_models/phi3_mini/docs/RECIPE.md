# Phi-3-mini-4k-instruct Porting Recipe

## Overview

Adds `microsoft/Phi-3-mini-4k-instruct` (3.8B-parameter causal LM) to the
`llama.cpp-et` framework. Confirmed via local GGUF metadata inspection:
`general.architecture = phi3` -- distinct from `phi2` (Phi-1.5, already
claimed on this board's "most models ported" track), a separate execution
family in llama.cpp (longrope-based RoPE scaling, different QKV/gate-up
weight packing than phi2).

## Model Reference

- **Source**: `microsoft/Phi-3-mini-4k-instruct` (Hugging Face), revision
  `f39ac1d28e925b323eae81227eaba4464caced4e`
- **License**: MIT
- **GGUF source**: `microsoft/Phi-3-mini-4k-instruct-gguf` (official), file
  `Phi-3-mini-4k-instruct-q4.gguf`
- **Quantization**: Q4 (the official Microsoft repo ships only `fp16` and
  `q4` -- no Q8_0 available at the source; used the smaller of the two for
  this batch), `sha256=8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef`
  (verified locally against the downloaded file)
- **Architecture**: `arch = phi3` per GGUF metadata, 195 tensors.

## Verification performed this round

GGUF-metadata-level only (downloaded exact file, verified sha256, confirmed
`general.architecture` + tensor count) -- not a full ET sysemu load/offload
test this round, given the scale of this porting batch.

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_PHI3`'s graph builder shares the same op set as phi2/falcon/other
already-proven decoders on this backend (RMSNorm, RoPE with longrope
scaling, standard MHA/GQA, SwiGLU-style gated FFN via a fused up/gate
weight) -- no new op type, just a different weight layout the graph builder
already handles. Not confirmed live this round.

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/phi3_mini.json`
  is the model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- Quantization is Q4 not Q8_0 (source constraint, not a choice) -- flagged
  honestly since Q8_0 has been this campaign's default elsewhere.
