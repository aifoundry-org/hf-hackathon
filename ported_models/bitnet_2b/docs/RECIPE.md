# BitNet-b1.58-2B-4T -- Negative Result

## Summary

Attempted to add `microsoft/bitnet-b1.58-2B-4T` (2B-parameter, native
1.58-bit ternary weight causal LM) to the `llama.cpp-et` framework. **Does
not load with this repo's vendored `llama.cpp-et` submodule.** No
model-ports claim is filed for this model; this recipe documents the
failure rather than a working port.

## Model Reference

- **Source**: `microsoft/bitnet-b1.58-2B-4T` (Hugging Face), revision
  `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- **License**: MIT
- **GGUF source**: `microsoft/bitnet-b1.58-2B-4T-gguf` (official), file
  `ggml-model-i2_s.gguf`,
  `sha256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
  (verified locally against the downloaded file)
- GGUF metadata (read via a minimal custom KV-only parser, since the
  installed `gguf` PyPI package's tensor-dtype enum doesn't recognize this
  file's dtype either): `general.architecture = bitnet-b1.58`, 332 tensors.

## Root cause: confirmed, not speculative

Built a plain CPU-only (`GGML_ET=OFF`) configuration of the exact vendored
`llama.cpp-et` source in this repo and tried to load the file with
`llama-perplexity`. It fails at model load, before any ET-specific code
runs at all:

```
gguf_init_from_file_impl: tensor 'blk.0.ffn_down.weight' of type 36
(TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking) has 6912
elements per row, not a multiple of block size (0)
gguf_init_from_file_impl: failed to read tensor info
llama_model_load: error loading model: llama_model_loader: failed to load
model from ggml-model-i2_s.gguf
```

This repo's vendored `llama.cpp-et` fork is from a point in llama.cpp's
history where GGML tensor-type id 36 was `TYPE_IQ4_NL_4_4`, since removed
upstream. BitNet's official GGUF export uses type id 36 for its native
`i2_s` ternary format. The two collide: this fork's tensor-type enum
interprets BitNet's ternary tensors as a stale, incompatible legacy type,
and rejects the file outright at the format-parsing stage -- before
reaching any question of whether the ET backend's `MUL_MAT` could handle
ternary matmul.

## Why no claim is filed

The port genuinely does not run, on CPU or otherwise, with this repo's
current vendored source. Filing a model-ports claim for something that
doesn't load would be exactly the kind of unsubstantiated submission this
process is meant to prevent. This is the same category as this campaign's
other documented negative results (RWKV-6, PLaMo-2 -- both blocked on
missing `GGML_OP_CPY` support) -- a real, reproducible incompatibility,
not a hedge or a "maybe."

## What would need to change

Either this repo's vendored `llama.cpp-et` needs to be updated past the
point where type id 36 was reused for ternary support (a submodule bump,
out of scope for a model-port PR), or BitNet's GGUF would need
re-exporting against an older, non-conflicting tensor-type scheme (not
attempted -- would change the model's actual on-disk representation, not
just add board wiring).
