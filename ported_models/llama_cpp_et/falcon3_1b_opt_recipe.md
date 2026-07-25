# Falcon3-1B-Instruct Q8_0 — optimization recipe

## Architecture notes

Falcon3-1B is architecturally distinct from the Llama-family models on the board:
- **ALiBi position encoding** (not RoPE)
- **Parallel attention + MLP** computation (not sequential)
- Uses GeLU activation in the MLP (not SiLU)

## Config changes vs default

| Setting | Value | Rationale |
|---------|-------|-----------|
| `ctx_size` | 256 | Small KV cache — only ~128 tokens needed for PP256/TG128 |
| `flash_attn` | true | Flash attention reduces memory bandwidth |
| `gpu_layers` | 99 | Full ET offload (unsupported ops fall back to CPU) |
| `extra_args` | `-nkvo` | No KV offload — tiny cache fits on-device |

## Status

**Experimental.** Falcon3 uses ALiBi which may not be fully accelerated by the ET backend. 
Unsupported ops fall back to CPU. Perplexity validation enabled.
