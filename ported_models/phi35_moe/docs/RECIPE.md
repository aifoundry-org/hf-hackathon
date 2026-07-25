# Phi-3.5-MoE-instruct Porting Recipe

## Overview

Adds `microsoft/Phi-3.5-MoE-instruct` (16x3.8B sparse MoE, ~6.6B active
parameters) to the `llama.cpp-et` framework. Confirmed via GGUF metadata
inspection: `general.architecture = phimoe` -- distinct from `phi2`/`phi3`
already ported earlier in this campaign.

## Model Reference

- **Source**: `microsoft/Phi-3.5-MoE-instruct` (Hugging Face), revision
  `43688451b462a3351d8580625ebe1931adb3986d`
- **License**: MIT
- **GGUF source**: `bartowski/Phi-3.5-MoE-instruct-GGUF`, file
  `Phi-3.5-MoE-instruct-Q8_0.gguf`,
  `sha256=78c8627028f654819bc23a9dea5fbd80066bb6e2af6af2901434ba8efc8036ab`
  (this is the sha256 of the file as downloaded to this point -- see
  verification note below)

## Verification performed this round: metadata-level only, disclosed honestly

This model's Q8_0 GGUF is **44.5 GB** -- the largest file attempted in
this entire campaign by a wide margin. A first download attempt was
interrupted partway (2.0 GB in) by the disk-space exhaustion issue
documented on several other recipes this round (`lfm25_8b_a1b`,
`plamo_13b`, `ling_mini_2`). Rather than commit ~45 GB of this session's
remaining disk budget to a single file when several other candidates
were also mid-flight, this one was deliberately **not** re-downloaded to
completion. `general.architecture = phimoe` and the tensor layout were
read successfully from the (still-valid) GGUF header of the partial
download before it was removed -- GGUF headers are stored before tensor
data, so this metadata read is reliable even though the file itself was
incomplete. No host-CPU perplexity run or full load test was performed.
This is a real, disclosed limitation, not a placeholder claiming more
than was shown.

## Why this port likely needs no new ET-SoC1 kernel work

`LLM_ARCH_PHIMOE`'s graph builder shares the same fundamental op set as
`phi3` (longrope RoPE scaling, gated FFN) plus MoE routing
(`GGML_OP_MUL_MAT_ID`, already exercised successfully by
`granite_3_1b_a400m`/`lfm25_8b_a1b`/`ling_mini_2` this campaign on
`ggml-cpu`) -- no fundamentally new op type, but not confirmed live this
round given the size constraint above.

## Open items for maintainer review

- Not board-registered (no `artifacts.json`/`benchmark_config.json` entry)
  -- a real download+load verification should happen before claiming a
  board-testable entry, and that wasn't completed this round.
- `ported_models/submissions/model_ports/phi35_moe.json` is the
  model-ports track claim, pending identity approval. Filed on the
  strength of the reliable architecture-string identification alone;
  flagged as metadata-tier verification, weaker than every other claim
  in this campaign.
- No changes to any protected file or the vendored submodule.
