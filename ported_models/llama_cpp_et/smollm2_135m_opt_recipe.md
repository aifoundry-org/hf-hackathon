# SmolLM2-135M Q8_0 — optimization recipe

## Changes from upstream benchmark config

| Setting | Before | After | Benefit |
|---------|--------|-------|---------|
| `ctx_size` | 2048 | 256 | Smaller KV cache; only ~128 tokens needed for PP256/TG128 |
| `flash_attn` | false | true | Hardware-accelerated flash attention reduces memory bandwidth |
| `extra_args` | — | `-nkvo` | No KV offload — tiny KV cache fits on-device |

## Runtime (submodule)

Repins to `CodeDoes/llama.cpp` (`8982212`) which includes:
- Q8_0 dot product fix (`fg32b.ps` → `fgb.ps` — fixes stream error crash)
- Uberkernel enabled by default — batches small ops, critical for small models
- K-split disabled (avoids FCC semaphore synchronization bug)

## Verification

- Submodule compiles via ET toolchain
- Model artifact: `unsloth/SmolLM2-135M-Instruct-GGUF` @ `9e6855b` (Q8_0)
