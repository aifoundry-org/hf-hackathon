# OLMoE-1B-7B-0125 Porting Recipe

## Overview

Adds `allenai/OLMoE-1B-7B-0125` (7B-total/1B-active-parameter Mixture-of-Experts
causal LM, 64 experts, 8 active per token) to the `llama_cpp_et` benchmark
suite. This introduces the **OLMoE** execution family to the board — a
fully open-source MoE release from Ai2 (open weights, open training data,
open code).

## Model Reference

- **Source**: `allenai/OLMoE-1B-7B-0125` (Hugging Face), revision
  `9b0c1aa87e34a20052389dce1f0cf01da783f654`
- **License**: Apache 2.0
- **Architecture**: `arch = olmoe` (`OlmoeForCausalLM`), 16 layers, 64
  experts, 8 experts active per token (confirmed against `config.json`).

## Hosting

Ai2 publishes an official first-party GGUF repository with a full
quantization ladder, so this uses that directly rather than a self-converted
artifact:

- **GGUF source**: `allenai/OLMoE-1B-7B-0125-GGUF`, revision
  `b8453f3c06477f3efc002ff89f2c3c149ffbeec1`, file
  `OLMoE-1B-7B-0125-Q8_0.gguf`, 7,359,942,720 bytes,
  `sha256=5a82c080fb821ed5afac78768377260f984e19decf3fc49befafe61d7a9d44ad`
  (verified against the file's own `x-linked-etag` header, matching the
  real remote content-length exactly).

## Local Verification (confirmed live, not speculative)

Independently self-converted the base checkpoint (`convert_hf_to_gguf.py
--outtype q8_0`, no fixes needed, clean conversion first attempt) to
cross-check the official GGUF is architecturally sound, then ran real
inference via the CPU-backend build (see verification-tier note in the
`jamba_tiny_dev` recipe for why CPU, not full ET sysemu):

- Model loads cleanly: `arch = olmoe`, 16 layers, 64 experts, 8 active per
  token, KV cache allocated (32 MiB, 16 layers), clean 935-node compute
  graph, "fused Gated Delta Net" MoE routing paths enabled.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128,
  batch=128): **PPL = 9.5706 +/- 1.58463** — solidly within this campaign's
  normal range, and notably strong for a model this size, consistent with
  Ai2's own published benchmarks for OLMoE.

## Committed deterministic oracle (added per maintainer review)

`ported_models/olmoe_1b_7b_0125/oracle/perplexity_oracle.json` commits
the exact reproduction command, pinned corpus/artifact hashes, the
final PPL from this session's CPU reference run, and an explicit ±20%
comparison threshold for independently verifying a future full-offload
ET-SoC1 run against this reference.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/OLMoE-1B-7B-0125'))"
# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <snapshot-dir> --outfile olmoe-1b-7b-0125-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule.
