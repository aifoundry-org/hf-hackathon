# BLOOMZ-560M Q8_0 — optimization recipe

## Architecture notes

BLOOMZ-560M is BigScience's multilingual BLOOM architecture model:
- **ALiBi position encodings** (linear biases)
- **LayerNorm** normalization
- **GeLU** activation function
- Pre-trained on 46+ languages and multi-task prompt instructions

## Config changes vs default

| Setting | Value | Rationale |
|---------|-------|-----------|
| `ctx_size` | 256 | Small KV cache — only ~128 tokens needed for PP256/TG128 |
| `flash_attn` | true | Hardware-accelerated flash attention reduces memory bandwidth |
| `gpu_layers` | 99 | Full ET offload |
| `extra_args` | `-nkvo` | No KV offload — tiny KV cache fits on-device |

## Status

Fully supported in llama.cpp-et with perplexity validation enabled.
