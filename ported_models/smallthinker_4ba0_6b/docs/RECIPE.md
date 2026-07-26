# SmallThinker-4BA0.6B-Instruct Porting Recipe

## Overview

Adds `PowerInfer/SmallThinker-4BA0.6B-Instruct` (4B-total/0.6B-active
Mixture-of-Experts causal LM, 32 layers, 32 experts, 4 active per token,
32K context) to the `llama_cpp_et` benchmark suite. This introduces the
**SmallThinker** execution family to the board — a MoE architecture
purpose-built for resource-constrained, on-device local deployment
(co-developed by IPADS/Shanghai Jiao Tong University and Zenergize AI).

## Model Reference

- **Source**: `PowerInfer/SmallThinker-4BA0.6B-Instruct` (Hugging Face),
  revision `a24932bb880c85c04c794283202a9194b1b42709`
- **License**: Apache 2.0
- **Architecture**: `arch = smallthinker` (`SmallThinkerForCausalLM`), 32
  layers, 32 experts, 4 active per token.

## Conversion

No usable pre-made Q8_0 GGUF exists for this checkpoint at the time of
this port, so it was self-converted directly from safetensors using this
repo's own `convert_hf_to_gguf.py --outtype q8_0`, no fixes needed —
converted cleanly on the first attempt. Produced a 4.55 GB file,
`sha256=6a352d7554a736c738cbef069aa1bb2fbf95c929567d48e9c4eb3b24f5942f16`.

## Hosting

This GGUF (4.55 GB) exceeds GitHub's 2 GB release-asset limit, so unlike
the earlier `jamba_tiny_dev` self-converted port (hosted as a GitHub
Release), this one is hosted on Hugging Face instead:
`darthceltic85/smallthinker-4ba0.6b-gguf`, file
`smallthinker-4ba0.6b-Q8_0.gguf`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see the `jamba_tiny_dev` recipe
for the verification-tier note on why CPU, not full ET sysemu) and ran
real inference:

- Model loads cleanly: `arch = smallthinker`, clean 1640-node compute
  graph, 1 split, "fused Gated Delta Net" MoE routing paths enabled.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 14.2763 +/- 2.92502** — solidly within this
  campaign's normal range, confirming genuine, coherent-quality output.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('PowerInfer/SmallThinker-4BA0.6B-Instruct'))"
# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <snapshot-dir> --outfile smallthinker-4b-a0.6b-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule.
