# Persimmon-8B -- Negative Result

## Summary

Attempted to add `adept/persimmon-8b-base` (8B-parameter causal LM, Adept's
Persimmon architecture) to the `llama.cpp-et` framework. **This repo's
vendored `llama.cpp-et` submodule has no `persimmon` architecture support
compiled in at all -- the model fails to load with `unknown model
architecture: 'persimmon'`.** No model-ports claim is filed.

## Model Reference

- **Source**: `adept/persimmon-8b-base` (Hugging Face), revision
  `94dc4e0bb7eeb26ec521eb3f78c36c91f6fe866b`
- **License**: Apache-2.0
- **GGUF source**: `maddes8cht/adept-persimmon-8b-base-gguf`, file
  `adept-persimmon-8b-base-Q8_0.gguf`
- **Quantization**: Q8_0, 9.30 GiB,
  `sha256=25eb5e4901cd45ae03c9a447bb49c7716a71a26562d1a5a3b291de232c27b0c4`
  (verified locally against the downloaded file)
- GGUF metadata (read successfully -- the failure is architecture support,
  not file corruption): `general.architecture = persimmon`, 580 tensors,
  36 layers, 4096 embedding, 64 attention heads (no GQA -- head_count ==
  head_count_kv), 16384-token context, RoPE freq_base 25000.

## Root cause: confirmed, not speculative

Built a plain CPU-only (`GGML_ET=OFF`) configuration of the exact vendored
`llama.cpp-et` source in this repo and tried to load the file with
`llama-perplexity`. The GGUF header parses completely (llama.cpp's own
metadata dump above proves the file itself is well-formed), but model
construction fails immediately after:

```
llama_model_load: error loading model: error loading model architecture:
unknown model architecture: 'persimmon'
llama_model_load_from_file_impl: failed to load model
```

Upstream llama.cpp has historically had Persimmon support (it was one of
the earlier non-Llama architectures added), but this repo's vendored fork
either predates that support or has since dropped it. Either way, the
architecture registry `llama_model_load()` dispatches on simply does not
contain an entry for `persimmon` in this exact vendored commit.

## Why no claim is filed

The failure is unambiguous and reproducible -- not a tooling gap like
`flan_t5_base` (where the model loads but the CLI can't drive it) and not
an exotic-quant-type collision like `bitnet_2b` (where the file itself
gets rejected). This is the simplest and most direct kind of negative
result: the architecture is not implemented in this vendored source at
all.

## What would need to change

Persimmon support would need to be added to (or restored in) this repo's
vendored `llama.cpp-et` fork -- a submodule content change, out of scope
for a model-port PR.
