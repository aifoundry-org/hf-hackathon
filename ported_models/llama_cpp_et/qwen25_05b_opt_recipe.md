# Qwen2.5-0.5B Q8_0 — optimization recipe

## Changes

| Setting | Before | After | Benefit |
|---------|--------|-------|---------|
| `ctx_size` | 2048 | 256 | Smaller KV cache |
| `flash_attn` | false | true | Hardware-accelerated flash attention |
| `extra_args` | — | `-nkvo` | No KV offload |

Runtime flags only — no submodule changes.
