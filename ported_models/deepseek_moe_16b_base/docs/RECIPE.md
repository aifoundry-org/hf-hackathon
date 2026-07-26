# DeepSeek-MoE-16B-Base Porting Recipe

## Overview

Adds `deepseek-ai/deepseek-moe-16b-base` (fine-grained Mixture-of-Experts
causal LM, 28 layers, 64 routed experts + 2 shared experts, 6 active per
token, 16.38B total params) to the `llama_cpp_et` benchmark suite. This
introduces the **DeepSeek MoE** execution family to the board — the
architecture that predates and informed the later DeepSeek-V2/V3 MoE
designs.

## Model Reference

- **Source**: `deepseek-ai/deepseek-moe-16b-base` (Hugging Face),
  revision `521d2bc4fb69a3f3ae565310fcc3b65f97af2580`
- **License**: DeepSeek's custom Model License
  (`license_name: deepseek`, `license_link:
  https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/LICENSE-MODEL`)
  — recorded accurately rather than assumed to be MIT/Apache.
- **Architecture**: `arch = deepseek` (`DeepseekForCausalLM`), 28 layers,
  64 routed experts (6 active), 2 shared experts, embedding length 2048.

## Real Bugs Found and Fixed (converter-side, not model-side)

Two distinct, layered bugs in `convert_hf_to_gguf.py`'s tokenizer
handling, both hit while converting this checkpoint:

**Bug 1 — unrecognized tokenizer chkhsh.** The converter's
`TextModel.get_vocab_base_pre()` identifies a model's BPE
pre-tokenizer family by hashing a fixed check string encoded through
the model's own tokenizer, then matching against a table of known
hashes. `deepseek-moe-16b-base`'s tokenizer produces a chkhsh
(`93105512fde79bc726022fe3cbfb7efef9738465b988d0600beba1296e3a91d8`)
that isn't in that table at all, so conversion fails outright with an
unrecognized-tokenizer error.

**Bug 2 — the fix's own first attempt was incomplete.** The first fix
registered a new, made-up pre-tokenizer name (`'deepseek-moe'`) for
that hash. This let the Python-side conversion complete successfully,
producing a seemingly-valid GGUF — but the **separate, independent**
hardcoded pre-tokenizer registry in the C++ runtime (`llama-vocab.cpp`)
only recognizes `'deepseek-llm'` / `'deepseek-coder'` / `'deepseek-v3'`
/ `'deepseek-r1-qwen'`, not `'deepseek-moe'`. A GGUF built with the
invented name loads and *appears* fine in the Python tooling but is
silently wrong at inference time in the real runtime. Caught this by
actually loading the produced GGUF with `llama-perplexity`, not just by
trusting a clean Python conversion.

**Final fix**: register the known hash to the existing name
`'deepseek-llm'` instead of inventing a new one, since DeepSeek's early
dense (`deepseek-llm`) and MoE (`deepseek-moe`) models share the same
base BPE tokenizer family. Confirmed live: the final GGUF loads with
`tokenizer.ggml.pre = deepseek-llm` and produces coherent, in-range
perplexity (see below).

Both fixes applied via a standalone monkeypatch wrapper
(`convert_wrapper.py`, alongside this recipe), which extracts the
original function's exact `chktxt` check-string via source
introspection (`ast`/`inspect`) rather than retyping it by hand, to
guarantee byte-identical hashing versus the real function. The vendored
`convert_hf_to_gguf.py` itself is never touched, matching the
established `pythia410m` precedent.

## Conversion

Converted from safetensors via the patched `get_vocab_base_pre()` using
`convert_hf_to_gguf.py --outtype q8_0`. Produced a 363-tensor, 17.4 GB
file, `sha256=63eb27478a35ec36a3fd7c50e704ef6752ca235a59491d7673909da6fd8f0971`.
Required `trust_remote_code` prompt handling for this repo's custom
modeling code (config-loading only; standard for this model family).

## Hosting

This GGUF (17.4 GB) far exceeds GitHub's 2 GB release-asset limit, so
it is hosted on Hugging Face: `darthceltic85/deepseek-moe-16b-base-gguf`,
file `deepseek-moe-16b-base-Q8_0.gguf`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see the `jamba_tiny_dev` recipe
for the verification-tier note on why CPU, not full ET sysemu) and ran
real inference:

- Model loads cleanly: `arch = deepseek`, `tokenizer.ggml.pre =
  deepseek-llm` (confirming the C++ runtime correctly recognizes the
  corrected pre-tokenizer name), 64 experts / 6 used, clean compute
  graph, 1 split.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 7.2909 +/- 1.30187** — a good result, solidly
  within this campaign's normal range, confirming genuine coherent
  output from a correctly-tokenized model.

## Committed deterministic oracle (added per maintainer review)

`ported_models/deepseek_moe_16b_base/oracle/perplexity_oracle.json`
commits the exact reproduction command, pinned corpus/artifact hashes,
the final PPL from this session's CPU reference run, and an explicit
±20% comparison threshold (matching this repo's own leaderboard-gate
policy) for independently verifying a future full-offload ET-SoC1 run
against this reference. No ET-SoC1 hardware was available to this
session to perform that run directly.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('deepseek-ai/deepseek-moe-16b-base'))"
# from the llama.cpp-et submodule root, with the standalone wrapper applied:
python3 convert_wrapper.py <snapshot-dir> --outfile deepseek-moe-16b-base-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule — the tokenizer fix is a standalone wrapper script, same
  pattern as the existing pythia410m/nemotron_h fixes.
- The `'deepseek-moe'`-named intermediate fix attempt (Bug 2 above) is
  documented here specifically because it's a trap: it looks correct
  from the Python side alone. Future ports of DeepSeek-family MoE
  checkpoints with unrecognized chkhsh should register against an
  *existing* C++-recognized pre-tokenizer name, not a new one, unless
  the C++ registry itself is also being patched (out of scope here).
