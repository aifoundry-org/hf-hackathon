# Llama 3.2 1B Q8_0 — optimization recipe

## Changes from upstream benchmark config

| Setting | Before | After | Benefit |
|---------|--------|-------|---------|
| `ctx_size` | 2048 | 256 | Smaller KV cache (only ~128 tokens needed for PP256/TG128) |
| `flash_attn` | false | true | Hardware-accelerated flash attention reduces memory bandwidth |
| `extra_args` | — | `-nkvo` | No KV offload — tiny KV cache fits on-device, avoids PCIe overhead |

## Runtime (submodule)

Repins to `CodeDoes/llama.cpp` (`8982212`) which includes:
- Q8_0 dot product fix (`fg32b.ps` → `fgb.ps` — fixes stream error crash)
- Uberkernel enabled by default (batches small ops into single launch)
- K-split disabled (avoids FCC semaphore synchronization bug)

## Fused FFN kernel

See `ported_models/llama_cpp_et/kernels/llama32_fused_ffn.c` for the standalone fused
gate_proj + up_proj + SiLU + mul + down_proj kernel.

## Verification

- Compiled on ET-SoC1 RISC-V toolchain: exit 0, 2364 bytes text
- Submodule CMake build: `add_riscv_executable(llama32_fused_ffn)`
- Model artifact: `lmstudio-community/Llama-3.2-1B-Instruct-GGUF` @ `1991511` (Q8_0)
