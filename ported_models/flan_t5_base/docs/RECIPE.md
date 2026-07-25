# FLAN-T5-base -- Inconclusive Result (No Claim Filed)

## Summary

Attempted to add `google/flan-t5-base` (250M-parameter encoder-decoder
causal LM, an instruction-tuned T5) to the `llama.cpp-et` framework.
**The model loads and its compute graph builds successfully, but neither of
the two standard tools in this vendored `llama.cpp-et` submodule
(`llama-perplexity`, `llama-cli`) can actually drive it to produce output.**
No model-ports claim is filed -- this recipe documents the finding rather
than a working port.

## Model Reference

- **Source**: `google/flan-t5-base` (Hugging Face), revision
  `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`
- **License**: Apache-2.0
- **GGUF source**: `Felladrin/gguf-flan-t5-base`, file
  `flan-t5-base.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=73eb18e2341b28f40e9596c3740ef46941075a57c79af1ebffab177fd66cbe50`
  (verified locally against the downloaded file)
- **Architecture**: `arch = t5` per GGUF metadata, 282 tensors.

## What actually happens (both confirmed live, not speculative)

The model loads cleanly and reserves a compute graph (`llama_kv_cache`
allocated, 704 graph nodes) -- so far identical to every working port in
this campaign. Then:

1. **`llama-perplexity`** crashes with
   `GGML_ASSERT(!llama_vocab_get_add_eos(vocab)) failed`. T5's tokenizer
   always appends an EOS token (a fixed property of its seq2seq
   vocabulary) -- the perplexity tool's raw-continuation PPL method
   explicitly asserts the OPPOSITE for every model it evaluates, since
   that method assumes a plain causal-LM tokenizer. This is a tool/model
   mismatch, not a model defect.
2. **`llama-cli`** crashes with
   `GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called
   first") failed`. T5 is encoder-decoder: its cross-attention layers need
   the encoder pass (`llama_encode()`) run once before any decoder step
   (`llama_decode()`). `llama-cli` goes straight to `llama_decode()` --
   it never calls `llama_encode()` at all. The underlying library
   (`libllama`) clearly supports the encode/decode split (the assert
   exists specifically to catch exactly this ordering mistake), but
   neither CLI frontend in this vendored build orchestrates it.

## Why no claim is filed

Both failures happen in the harness/tooling, not the model itself -- the
GGUF is valid and the graph builds. But this campaign's standard is
"prove the model actually runs and produces output," and neither available
tool can demonstrate that for an encoder-decoder model. Filing a claim on
"it loads" alone, with the same evidence bar as this campaign's actually-
completed ports, would overstate what was actually shown. This is a
different category from `bitnet_2b` (which fails at model load) --
here the model format is fine, but this repo's tooling has no
encoder-decoder driver.

## What would need to change

A custom driver that calls `llama_encode()` once against the input, then
loops `llama_decode()` for the output tokens (this is exactly what
llama.cpp's upstream examples do for T5/text2text tasks in some example
programs, but that driver is not present in the ported subset of tools
built in this repo's vendored submodule). Out of scope for a model-port
recipe -- would be new tooling work, not board wiring.
