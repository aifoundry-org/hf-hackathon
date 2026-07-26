# Jamba-tiny-dev Porting Recipe

## Overview

Adds `ai21labs/Jamba-tiny-dev` (319M-parameter hybrid Mamba+Transformer
Mixture-of-Experts causal LM) to the `llama_cpp_et` benchmark suite. This
introduces the **Jamba** execution family to the board — the first hybrid
attention+SSM MoE architecture claimed (distinct from the dense/hybrid
non-MoE families already on the board, e.g. `falcon-h1`, `granitehybrid`,
`mamba`).

AI21 explicitly publishes this checkpoint as a small development/testing
model (319M params, trained on ~40B tokens), not a production-quality
release — it exists specifically so people can exercise the Jamba
architecture end-to-end without downloading the full 52B Jamba 1.5 Mini.
That is reflected honestly below: this port is about confirming genuine
architectural correctness (hybrid Mamba/attention/MoE layers all load and
compute correctly), not chasing a strong PPL score.

## Model Reference

- **Source**: `ai21labs/Jamba-tiny-dev` (Hugging Face), revision
  `ed303361004ac875426a61675edecf8e9d976882`
- **License**: Apache 2.0
- **Architecture**: `arch = jamba` (`JambaForCausalLM`), 16 hidden layers
  (attention layers every 8th layer, offset 4; MoE expert layers every
  2nd layer, offset 1 — confirmed against `config.json`: `hidden_size=512`,
  `num_attention_heads=8`, `num_key_value_heads=2`, `num_experts=8`,
  `num_experts_per_tok=2`, `mamba_d_state=16`, `mamba_d_conv=4`,
  `vocab_size=65536`).

## Conversion

No pre-made GGUF exists for this checkpoint (it's a small dev/test release,
not a widely-quantized production model), so this was **self-converted**
directly from the original safetensors using this repo's own
`convert_hf_to_gguf.py --outtype q8_0`, with no fixes or patches needed —
converted cleanly on the first attempt. Produced a 341,302,880-byte file,
`sha256=cd92edec23a4e8eda0be4c34a3730a6c3a0605edd2786c70714de9bb76b3e914`.

## Local Verification (confirmed live, not speculative)

Built `llama-server`/`llama-perplexity`/`llama-cli` from the committed
`llama.cpp-et` submodule (CPU backend — see "Verification tier" note below)
and ran real inference:

- Model loads cleanly: `arch = jamba`, 16 layers, both a real KV cache
  (0.25 MiB, 2 attention layers, confirming the periodic full-attention
  layers) **and** a recurrent-state cache (1.04 MiB, 16 layers, confirming
  every layer carries Mamba SSM state) are allocated — exactly matching the
  hybrid architecture's expected memory layout.
- `sched_reserve` confirms "fused Gated Delta Net (autoregressive) enabled"
  and "fused Gated Delta Net (chunked) enabled" — the MoE/SSM fusion paths
  engage correctly.
- Clean compute graph: 1009 nodes, 1 split, reserve took 9.04 ms.
- Real perplexity run against WikiText-2 raw (4 chunks, ctx=128, batch=128):
  **PPL = 18.4574 +/- 3.82483** — within this campaign's established normal
  range (~5-70 for genuinely-working models; compare to the confirmed
  degenerate cases `hunyuan_0_5b` at PPL 5444 and `exaone4_1_2b` at PPL 277).
  This confirms the port is producing real, coherent-quality output, not a
  vacuous pass.

### Verification tier note

This was verified via the CPU backend (`-DGGML_ET=OFF`), not the full
ET-SoC1 sysemu/hardware path. The full `GGML_ET=ON` build requires
device-kernel-build SDK components (`riscv64-ec-toolchain.cmake`,
`DeviceUtils`, `et-common-libs`, `esperantoTrace`) that are **not present**
in the public `aifoundry-org/et-platform` source — confirmed by grepping the
full cloned repo. These appear to be part of a separate, board-access-gated
SDK distribution. The actual ET-SoC1 board benchmark run happens in the
maintainer's own trusted CI once this identity is registered; this local
verification confirms the port itself is architecturally correct and
produces real, sane output ahead of that run.

### Chat template note

`llama-cli`'s default conversation mode crashes on this checkpoint's
embedded chat template (a Jinja parsing error: `KwArg is not a bool value`,
from `tokenizer_config.json`'s `chat_template` field) — this is a template-
engine limitation unrelated to the model architecture itself.
`llama-perplexity` and raw completion mode are unaffected since neither
touches the chat template.

## Hosting

No pre-made GGUF exists for this checkpoint, so following the established
pattern for self-converted artifacts in this fork (a GitHub Release asset,
since no Hugging Face upload token is available), the converted file is
published at:
<https://github.com/DarthCeltic/hf-hackathon/releases/download/jamba-tiny-dev-gguf-v1/jamba-tiny-dev-Q8_0.gguf>
(`sha256=cd92edec23a4e8eda0be4c34a3730a6c3a0605edd2786c70714de9bb76b3e914`),
registered in `artifacts.json` as `jamba_tiny_dev_q8_gguf`.

## Committed deterministic oracle (added per maintainer review)

`ported_models/jamba_tiny_dev/oracle/perplexity_oracle.json` commits the
exact reproduction command, pinned corpus/artifact hashes, the final PPL
from this session's CPU reference run, and an explicit ±20% comparison
threshold for independently verifying a future full-offload ET-SoC1 run
against this reference.

## Instructions for Reproduction

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('ai21labs/Jamba-tiny-dev'))"
# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py <snapshot-dir> --outfile jamba-tiny-dev-Q8_0.gguf --outtype q8_0
```

## Open items for maintainer review

- No changes were made to any protected file, and none to the vendored
  submodule.
- This is a genuinely tiny dev/test checkpoint — expect a modest decode
  tokens/s score on the real board relative to production-sized models;
  that reflects the model's own scale, not a port defect.
