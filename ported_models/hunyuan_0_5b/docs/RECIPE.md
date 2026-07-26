# Hunyuan-0.5B-Instruct -- Negative Result (Loads, Produces Broken Output)

## Summary

Attempted to add `tencent/Hunyuan-0.5B-Instruct` (0.5B-parameter causal LM)
to the `llama.cpp-et` framework. **The model loads and runs without
crashing, but produces clearly degenerate output** -- perplexity orders
of magnitude outside the normal range for every other model in this
campaign. No model-ports claim is filed.

## Model Reference

- **Source**: `tencent/Hunyuan-0.5B-Instruct` (Hugging Face), revision
  `2359fb220c010e9d6d62c62d466f0eda179c2cf3`
- **License**: Tencent Hunyuan Community License
- **GGUF source**: `Disya/Hunyuan-0.5B-Instruct-Q8_0-GGUF`, file
  `hunyuan-0.5b-instruct-q8_0.gguf`, 577,952,992 bytes,
  `sha256=4a5f9d81e4add0e58befa330f064143eaa13de61a249879ced7acc7dca791292`
  (verified locally against the downloaded file -- size matches the real
  remote `Content-Length` exactly, ruling out truncation as the cause of
  what follows)
- **Architecture**: `arch = hunyuan-dense` per GGUF metadata, 266 tensors.

## What actually happens (confirmed live, not speculative)

The model loads cleanly (KV cache allocates, compute graph builds, 847
nodes) and `llama-perplexity` runs to completion without any error or
crash. But the output itself is degenerate:

```
[1]2376.7956,[2]4508.7212,[3]5080.0506,[4]5444.0753,
Final estimate: PPL = 5444.0753 +/- 1456.73009
```

Every working model in this entire campaign scores between roughly 5 and
70 on this same corpus/parameters. 5444 is not "a worse model" -- it is
consistent with the model predicting close to random tokens, meaning
something in this GGUF's conversion (RoPE scaling, tokenizer mapping, or
tensor layout for `hunyuan-dense` specifically) does not match what this
vendored `llama.cpp-et`'s graph builder expects, even though it doesn't
hit any assertion that would catch the mismatch.

A second, independently-downloaded Q8_0 quant from a different
third-party quantizer (`bartowski/tencent_Hunyuan-0.5B-Instruct-GGUF`)
was also attempted as a cross-check, but that specific download was
truncated by this session's disk-space issue and not re-verified before
time ran out on this finding -- the negative result above stands on the
Disya quant alone, which is confirmed complete and correct.

## Why no claim is filed

A leaderboard submission should prove the intended model actually runs,
not just that a same-named file loads without crashing. A PPL this far
outside every other measured model's range is itself evidence the model
is not doing what it's supposed to -- filing a claim on "it loaded" alone
would be exactly the kind of unsubstantiated submission this process
exists to prevent.

## What would need to change

Either a different GGUF quantization of this exact checkpoint (to rule
out a bad conversion in this specific file), or an update to this repo's
vendored `llama.cpp-et`'s `hunyuan-dense` graph-builder code (a submodule
change, out of scope for a model-port PR) would be needed to determine
whether this is a conversion-specific or implementation-specific problem.
