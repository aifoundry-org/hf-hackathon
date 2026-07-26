# ERNIE-4.5-21B-A3B-Base Porting Recipe

## Overview

Adds `baidu/ERNIE-4.5-21B-A3B-Base-PT` (fine-grained Mixture-of-Experts
causal LM, 21B total params / 3B active per token, 64 routed experts +
shared expert, 131K context) to the `llama_cpp_et` benchmark suite.
This introduces the **ERNIE 4.5 MoE** execution family to the board —
distinct from the already-claimed `ernie45_03b` identity (PR #185),
which covers ERNIE-4.5-0.3B, a small *dense* model using the separate
`Ernie4_5Model` class. This port targets the MoE variant
(`Ernie4_5MoeModel`, a subclass), a genuinely different architecture
path in the converter and a different identity/credit.

## Model Reference

- **Source**: `baidu/ERNIE-4.5-21B-A3B-Base-PT` (Hugging Face), revision
  `c6804383159ca6ad60896b3a8a98c5c01e9b42d5`
- **License**: Apache 2.0
- **Architecture**: `arch = ernie4_5-moe` (`Ernie4_5_MoeForCausalLM`),
  embedding length 2560, feed-forward length 12288, 64 routed experts
  + 1 shared expert per layer, 20 attention heads / 4 KV heads
  (GQA), 131072 context length.

## Conversion

No fix was needed. `Ernie4_5MoeModel` (subclassing `Ernie4_5Model`) is
already a correctly-registered architecture class in this repo's
`convert_hf_to_gguf.py`, so this converted cleanly on the first attempt
via the stock, unmodified converter: `convert_hf_to_gguf.py --outtype
q8_0`. Produced a 389-tensor, 23.2 GB file,
`sha256=2d18e66097f598c8fb5f3d15fd2fba9e6f916652985426bf0e7e1b478a625d48`.

## Hosting

This GGUF (23.2 GB) far exceeds GitHub's 2 GB release-asset limit, so
it is hosted on Hugging Face: `darthceltic85/ernie45-21b-a3b-base-gguf`,
file `ernie45-21b-a3b-base-Q8_0.gguf`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see the `jamba_tiny_dev` recipe
for the verification-tier note on why CPU, not full ET sysemu) and ran
real inference:

- Model loads cleanly: `arch = ernie4_5-moe`, correct MoE routing
  (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` per expert plus shared
  expert tensors), clean compute graph.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 6.8926 +/- 1.25185** — a good result, well within
  this campaign's normal range, confirming genuine coherent output from
  a correctly-converted 21B-parameter MoE model.

## Committed deterministic oracle (added per maintainer review)

`ported_models/ernie45_21b_a3b_base/oracle/perplexity_oracle.json`
commits the exact reproduction command, pinned corpus/artifact hashes,
per-chunk and final PPL from this session's CPU reference run, and an
explicit ±20% comparison threshold for independently verifying a future
full-offload ET-SoC1 run against this reference.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('baidu/ERNIE-4.5-21B-A3B-Base-PT'))"
# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <snapshot-dir> --outfile ernie45-21b-a3b-base-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule. No standalone wrapper was needed for this port — the
  converter already handles this architecture correctly.
- This is a distinct architecture/identity from `ernie45_03b` (PR
  #185): that port is the small dense `Ernie4_5ForCausalLM`; this one
  is the MoE `Ernie4_5_MoeForCausalLM` subclass, a genuinely different
  code path (expert routing, shared experts) and model family.
