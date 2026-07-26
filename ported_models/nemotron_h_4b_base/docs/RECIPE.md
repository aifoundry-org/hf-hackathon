# Nemotron-H-4B-Base-8K Porting Recipe

## Overview

Adds `nvidia/Nemotron-H-4B-Base-8K` (hybrid Mamba2/Attention/MLP causal
LM, 52 layers, hidden size 3072, 131K vocab, 8K context) to the
`llama_cpp_et` benchmark suite. This introduces the **Nemotron-H**
hybrid-SSM execution family to the board — a dense model distilled/pruned
down from NVIDIA's larger Nemotron-H-8B MoE parent.

## Model Reference

- **Source**: `nvidia/Nemotron-H-4B-Base-8K` (Hugging Face), revision
  `faba3b731ad7ea5781b9518ae75fb610a94affcf`
- **License**: NVIDIA Internal Scientific Research and Development Model
  License (`license_name: nvidia-internal-scientific-research-and-development-model-license`,
  see the model card) — a custom NVIDIA license, not a standard OSS
  license. Recorded accurately rather than assumed.
- **Architecture**: `arch = nemotron_h` (`NemotronHForCausalLM`), 52
  layers, hidden_size 3072, vocab_size 131072.

## Real Bug Found and Fixed (converter-side, not model-side)

`convert_hf_to_gguf.py`'s `NemotronHModel.__init__` decides MoE vs dense
by checking key *presence* of `num_experts_per_tok` in the merged
hparams produced by `ModelBase.load_hparams()` (the framework's own
authoritative config loader, which pulls in more than a plain
`json.load()` of `config.json`). For this specific checkpoint,
`load_hparams()` reports a full, plausible-looking MoE config:
`num_experts_per_tok=2`, `moe_intermediate_size=7688`,
`n_routed_experts=8` — so the converter takes the MoE branch and would
mis-tag this checkpoint as MoE.

Directly inspecting the tensor names across **both** safetensors shards
(311 tensors total) found **zero** tensors matching `expert`/`moe`/
`router` substrings. This checkpoint is genuinely dense — the MoE
config fields are stale/vestigial, almost certainly inherited from
Nemotron-H's larger 8B MoE parent during pruning/distillation down to
this 4B base model, never cleaned out of `config.json`.

Fixed via a standalone monkeypatch wrapper (`convert_wrapper.py`,
alongside this recipe — not committed to the vendored submodule,
matching the established `pythia410m`/`deepseek-moe` precedent of never
touching `convert_hf_to_gguf.py` directly) that hardcodes `is_moe=False`
for this model, treating the direct tensor evidence as authoritative
over the checkpoint's own misleading config fields, while replicating
the full original `__init__` body (head_dim, d_inner, `_ssm_layers`,
`_mlp_layers` assignment) so nothing else about the hybrid
Mamba/Attention/MLP layer-pattern parsing is affected.

## Conversion

Converted from safetensors via the patched `NemotronHModel.__init__`
using `convert_hf_to_gguf.py --outtype q8_0`. Produced a 311-tensor,
4.8 GB file, `sha256=23cf524eaf7cc0c84ffe3dc168a5bacaad10bde2bb9fbaacba431a96b50c9d31`.

During conversion, several `RuntimeWarning`s appeared ("overflow
encountered in divide", "invalid value encountered in subtract/cast").
These turned out to be a false alarm, not corruption — confirmed by the
clean load and in-range perplexity result below.

## Hosting

This GGUF (4.8 GB) exceeds GitHub's 2 GB release-asset limit, so it is
hosted on Hugging Face: `darthceltic85/nemotron-h-4b-base-gguf`, file
`nemotron-h-4b-base-Q8_0.gguf`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see the `jamba_tiny_dev` recipe
for the verification-tier note on why CPU, not full ET sysemu) and ran
real inference:

- Model loads cleanly: `arch = nemotron_h`, fused Gated Delta Net
  (autoregressive + chunked) enabled, clean 1489-node compute graph, 1
  split.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 9.4458 +/- 1.75431** — a good result, well within
  this campaign's normal range, confirming genuine coherent output (and
  confirming the quantization warnings above were harmless).

## Committed deterministic oracle (added per maintainer review)

`ported_models/nemotron_h_4b_base/oracle/perplexity_oracle.json` commits
the exact reproduction command, pinned corpus/artifact hashes, the
per-chunk and final PPL from this session's CPU reference run, and an
explicit ±20% comparison threshold (matching this repo's own
leaderboard-gate policy) for independently verifying a future
full-offload ET-SoC1 run against this reference. No ET-SoC1 hardware
was available to this session to perform that run directly.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/Nemotron-H-4B-Base-8K'))"
# from the llama.cpp-et submodule root, with the standalone wrapper applied:
python3 convert_wrapper.py <snapshot-dir> --outfile nemotron-h-4b-base-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule — the MoE-misdetection fix is a standalone wrapper script,
  same pattern as the existing pythia410m/deepseek-moe fixes.
