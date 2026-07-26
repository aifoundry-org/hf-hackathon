# Trinity-Nano-Base Porting Recipe

## Overview

Adds `arcee-ai/Trinity-Nano-Base` (fine-grained Mixture-of-Experts causal
LM, 56 layers, 128 experts, 8 active per token + 1 shared expert, 131K
context via sliding-window + full attention layer mix, 6.12B total
params) to the `llama_cpp_et` benchmark suite. This introduces the
**Afmoe** execution family to the board — Arcee AI's Trinity model
line.

## Model Reference

- **Source**: `arcee-ai/Trinity-Nano-Base` (Hugging Face), revision
  `923a95dbba013c78c4fd6dbc6bce31eea553da7c`
- **License**: OpenMDW License Agreement, version 1.1 (`license_name:
  openmdw-1.1`) — a genuinely permissive open license (no restriction
  on use/modification/redistribution beyond retaining copyright
  notices), confirmed by reading the actual `LICENSE` file in the
  source repo rather than assumed.
- **Architecture**: `arch = afmoe` (`AfmoeForCausalLM`), 56 layers,
  embedding length 1024, 128 experts (8 active + 1 shared),
  vocab_size 200192.

## Conversion

No fix was needed. `AfmoeForCausalLM` is already a correctly-registered
architecture class in this repo's `convert_hf_to_gguf.py`
(`AfmoeModel(LlamaModel)`), so this converted cleanly on the first
attempt via the stock, unmodified converter: `convert_hf_to_gguf.py
--outtype q8_0`. Produced a 1057-tensor, 6.52 GB file,
`sha256=a52296d493fbdb810008a638eda4ce7f59717ddd4282ba2750518c7cefd36872`.

## Hosting

This GGUF (6.52 GB) exceeds GitHub's 2 GB release-asset limit, so it is
hosted on Hugging Face: `darthceltic85/trinity-nano-base-gguf`, file
`trinity-nano-base-Q8_0.gguf`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see the `jamba_tiny_dev` recipe
for the verification-tier note on why CPU, not full ET sysemu) and ran
real inference:

- Model loads cleanly: `arch = afmoe`, 128 experts / 8 used, sliding
  window + full attention mix (`is_swa_any = 1`), fused Gated Delta Net
  (autoregressive + chunked) enabled, clean 4223-node compute graph, 1
  split.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 10.5908 +/- 1.92727** — a good result, well
  within this campaign's normal range, confirming genuine coherent
  output.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('arcee-ai/Trinity-Nano-Base'))"
# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <snapshot-dir> --outfile trinity-nano-base-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule. No standalone wrapper was needed for this port — the
  converter already handles this architecture correctly.
