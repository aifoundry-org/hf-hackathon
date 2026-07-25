# Pythia-410M Q8_0 — optimization recipe

## Architecture notes

Pythia-410M is EleutherAI's GPT-NeoX architecture model:
- **Parallel attention + FFN** computation
- **Rotary embeddings (RoPE)**
- **LayerNorm** normalization
- Untied embedding and output heads

## Config changes vs default

| Setting | Value | Rationale |
|---------|-------|-----------|
| `ctx_size` | 256 | Small KV cache — only ~128 tokens needed for PP256/TG128 |
| `flash_attn` | true | Hardware-accelerated flash attention reduces memory bandwidth |
| `gpu_layers` | 99 | Full ET offload |
| `extra_args` | `-nkvo` | No KV offload — tiny KV cache fits on-device |

## Status

Fully supported in llama.cpp-et with perplexity validation enabled.
