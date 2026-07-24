# SmolVLM-256M — optimization recipes

## Q8_0 config changes

| Setting | Before | After | Benefit |
|---------|--------|-------|---------|
| `ctx_size` | 2048 | 256 | Smaller KV cache |
| `flash_attn` | false | true | Hardware-accelerated flash attention |
| `extra_args` | — | `-nkvo` | No KV offload |

## Q4_K_M variant (additional)

Switches to imatrix-quantized Q4_K_M GGUF from `mradermacher/SmolVLM-256M-Instruct-i1-GGUF`:
- ~71% of Q8_0 memory bandwidth → proportionally faster decode
- File: `SmolVLM-256M-Instruct.i1-Q4_K_M.gguf` (125 MB vs 175 MB Q8_0)
- SHA256: `e2c68191ea88b166206f37afbae73c66423619550119d9c51e0e7aabcc725113`

## Runtime (submodule)

Repins to `CodeDoes/llama.cpp` (`8982212`) which includes:
- Q8_0 dot product fix (`fg32b.ps` → `fgb.ps`)
- Uberkernel enabled by default
- K-split disabled
- Flash attention support

## Verification

- Submodule compiles via ET toolchain
- BBLLM backbone is SmolLM2-135M-Instruct (already proven on ET-SoC1)
