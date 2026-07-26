# EXAONE-4.0-1.2B -- Negative Result (Loads, Produces Broken Output)

## Summary

Attempted to add `LGAI-EXAONE/EXAONE-4.0-1.2B` (1.2B-parameter causal LM)
to the `llama.cpp-et` framework. Confirmed distinct from `exaone`
(EXAONE-3.5, already claimed elsewhere on this board) via GGUF metadata:
`general.architecture = exaone4`. **The model loads and runs without
crashing, but produces clearly degenerate output** -- same failure
category as `hunyuan_0_5b` earlier in this campaign. No model-ports claim
is filed.

## Model Reference

- **Source**: `LGAI-EXAONE/EXAONE-4.0-1.2B` (Hugging Face), revision
  `3abf2810673c7c0778df64a73c2d52eab32d91c4`
- **License**: EXAONE AI Model License Agreement 1.2 (custom, non-standard
  license)
- **GGUF source**: `LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF` (official), file
  `EXAONE-4.0-1.2B-Q8_0.gguf`, 1,363,939,616 bytes,
  `sha256=cc0b2a3f447e134cafd2853104d06227122cc280f4c9fee8c90172066174ef04`
  (verified locally against the downloaded file -- size matches the real
  remote `Content-Length` exactly, ruling out truncation)
- **Architecture**: `arch = exaone4` per GGUF metadata, 333 tensors.

## What actually happens (confirmed live, not speculative)

The model loads cleanly (compute graph builds, 1057 nodes) and
`llama-perplexity` runs to completion with no error or crash. The output
is degenerate:

```
[1]116.6505,[2]216.6080,[3]315.7759,[4]276.6599,
Final estimate: PPL = 276.6599 +/- 85.51548
```

Every working model in this campaign scores roughly 5-70 on this same
corpus/parameters -- 277 is far outside that range and, combined with the
identical failure pattern already seen on `hunyuan_0_5b` (official
first-party GGUF, size-verified, still degenerate), points to a genuine
`exaone4` graph-builder issue in this repo's vendored `llama.cpp-et`
rather than a bad third-party quantization (this is the model author's
own official GGUF release, not a community requant).

## Why no claim is filed

Same reasoning as `hunyuan_0_5b`: a PPL this far outside every other
measured model's range is itself evidence the model is not producing
correct output, regardless of whether the process crashes. Filing a claim
on "it loaded" would overstate what was actually shown.

## What would need to change

An update to this repo's vendored `llama.cpp-et`'s `exaone4` graph-builder
code (a submodule change, out of scope for a model-port PR) -- since this
is the model's own official first-party GGUF, a bad third-party
quantization is unlikely to be the explanation here, unlike the residual
uncertainty left open on `hunyuan_0_5b`.
