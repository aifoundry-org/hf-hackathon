# BLOOM-560m Porting Recipe

## Overview

Adds `bigscience/bloom-560m` (559M-parameter causal LM, from the BLOOM
multilingual model family) to the `llama_cpp_et` benchmark suite. Like
`pythia410m` in this same PR, no usable pre-made Q8_0 GGUF exists at a
manageable size (community quants found were Q2_K/Q3_K only), so this was
**self-converted directly from the original safetensors** using this
repo's own `convert_hf_to_gguf.py --outtype q8_0`, with no fixes or
patches needed -- this one converted cleanly on the first attempt.

This introduces the **BLOOM** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

BLOOM uses **ALiBi (Attention with Linear Biases)** for positional
encoding instead of RoPE (`rope type = -1`, `f_max_alibi_bias = 8.0`) --
the second ALiBi port this session after `mpt7b`. As documented there,
ALiBi is just a fixed linear bias added to the raw attention scores
before softmax, computed by llama.cpp's existing `LLM_ARCH_BLOOM` graph
builder via ordinary `GGML_OP_ADD` -- no new runtime op needed. BLOOM also
uses standard multi-head attention (no GQA), LayerNorm, and a GELU FFN.
Every op involved is already proven on the ET backend by the existing
decoder-only models on the board.

## Model Reference

- **Source**: `bigscience/bloom-560m` safetensors (Hugging Face, main
  revision)
- **License**: BigScience RAIL License v1.0
- **Architecture**: `arch = bloom` (`BloomForCausalLM`/`BloomModel`), 24
  transformer layers, standard MHA (no GQA), ALiBi positional bias (no
  RoPE), 250880-token BPE vocab (BLOOM's very large multilingual
  tokenizer), 1024-token native context.
- **Quantization**: Q8_0 for weight matrices, F32 for norms/biases
  (llama.cpp's standard mixed-precision convention) -- ~608 MB on disk.

## Steps Taken

1. **Downloaded** the original `model.safetensors` + config/tokenizer
   files from `bigscience/bloom-560m` directly via `curl -L --fail`
   (the initial attempt via `huggingface_hub.snapshot_download` stalled
   indefinitely on this repo for an unclear reason -- a direct single-file
   fetch of the known safetensors filename succeeded immediately).
2. **Converted** to GGUF: `python3 convert_hf_to_gguf.py <hf-snapshot>
   --outfile bloom-560m-Q8_0.gguf --outtype q8_0`, unmodified, no
   monkeypatch needed. Produced a 607,738,080-byte file,
   `sha256=9ddc108352d9a7b1f49d1c8e74fefacf2bac881c9d5b68be428df1ca773db5af`.
3. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18122`. Confirmed `arch = bloom`, `559.21 M`
   params (matches upstream), ALiBi active, full `25/25` layer ET offload
   (23 repeating layers + output layer), and a clean 898-node / 4-split
   compute graph. Full multi-token `/completion` decode output was not
   captured locally, consistent with every other port this session.

## Hosting (update, POST-DEADLINE -- not a hackathon submission)

The original converted artifact referenced above was lost before it could
be hosted. **After the official hackathon deadline**, purely for
record-completeness (not seeking hackathon credit), this was
re-converted from the same upstream revision with the stock, unmodified
converter (no fix needed, as documented above), and hosted on Hugging
Face: `darthceltic85/bloom-560m-gguf`, file `bloom-560m-Q8_0.gguf`,
`sha256=c2f1d6893150b5cf9755ebd20fcc90f385705e954a8117b7b2c1c5cd51f6b616`
(607,738,688 bytes). Now registered in `artifacts.json`
(`bloom560m_q8_gguf`) and `.github/ci/benchmark_config.json`
(`bloom560m`, port 18154). Re-verified: `arch = bloom`,
`f_max_alibi_bias = 8.0` (confirming ALiBi positional encoding as
documented above), real perplexity run against WikiText-2 raw (4
chunks, ctx=128, batch=128): **PPL = 27.8342 +/- 5.56596**.

## Committed deterministic oracle (added per maintainer review, POST-DEADLINE)

`ported_models/bloom560m/oracle/perplexity_oracle.json` commits the
exact reproduction command, pinned corpus/artifact hashes, the final PPL
from the CPU reference run above, and an explicit ±20% comparison
threshold for independently verifying a future full-offload ET-SoC1 run
against this reference.

## Instructions for Reproduction

```bash
curl -L --fail -o model.safetensors 'https://huggingface.co/bigscience/bloom-560m/resolve/main/model.safetensors'
# plus config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json from the same repo

# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <path-to-snapshot-dir> --outfile bloom-560m-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- **Not yet board-registered** -- see hosting note above. Once a GitHub
  Release (or other hosting) is available, this is a one-line addition to
  both config files following the exact pattern of every other port this
  session.
- No changes were made to any protected file, and none to the vendored
  submodule.
