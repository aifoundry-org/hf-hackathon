# StableLM-2-Zephyr-1.6B Q8_0 — optimization recipe

## Architecture notes

StableLM-2-Zephyr-1.6B is Stability AI's optimized compact language model:
- **Parallel attention + SwiGLU** MLP layers
- **RoPE** position encodings
- **LayerNorm** normalization

## Config changes vs default

| Setting | Value | Rationale |
|---------|-------|-----------|
| `ctx_size` | 256 | Small KV cache — only ~128 tokens needed for PP256/TG128 |
| `flash_attn` | true | Hardware-accelerated flash attention reduces memory bandwidth |
| `gpu_layers` | 99 | Full ET offload |
| `extra_args` | `-nkvo` | No KV offload — tiny KV cache fits on-device |

## Status

Fully supported in llama.cpp-et with perplexity validation enabled.
