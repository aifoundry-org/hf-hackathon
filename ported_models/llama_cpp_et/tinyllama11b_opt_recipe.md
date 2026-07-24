# TinyLlama 1.1B Q8_0 — optimization recipe

## Changes from upstream benchmark config

| Setting | Before | After | Benefit |
|---------|--------|-------|---------|
| `ctx_size` | 2048 | 256 | Smaller KV cache |
| `flash_attn` | false | true | Hardware-accelerated flash attention |
| `extra_args` | — | `-nkvo` | No KV offload |

## Runtime (submodule)

Repins to `CodeDoes/llama.cpp` (`8982212`) which includes:
- Q8_0 dot product fix (`fg32b.ps` → `fgb.ps`)
- Uberkernel enabled by default
- K-split disabled

## Verification

- Submodule compiles via ET toolchain
- Model artifact: TinyLlama/TinyLlama-1.1B-Chat-v1.0 GGUF (Q8_0)
