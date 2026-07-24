# Llama 3.2 1B Kernels for ET-SoC1

Two custom kernels for optimizing Llama 3.2 1B inference on ET-SoC1, bypassing
the per-op GGML dispatch to fuse operations and keep data on-chip.

## Kernels

### `llama32_fused_ffn.c` — Fused FFN (gate × up × SiLU × down)

Combines 5 operations into one kernel launch:

| Before (5 launches) | After (1 launch) |
|---|---|
| `gate = gate_proj @ input` | All 3 matmuls + SiLU + multiply |
| `up = up_proj @ input` | in one pass |
| `gate_s = SiLU(gate)` | Intermediate results stay |
| `gated = gate_s * up` | in L2 scratchpad, never |
| `output = down_proj @ gated` | touch DRAM |

**Savings per FFN layer (M=1 token generation):**
- 4 fewer kernel launches
- ~64 KB less DRAM write traffic (gate + up)
- ~64 KB less DRAM read traffic (gate + up readback + gated readback)
- Total: ~128 KB DRAM traffic eliminated per FFN layer

**Code size:** 1540 bytes (vs 9064 for a single existing Q8_0 matmul)

### `llama32_q80_matmul.c` — Q8_0 × f32 standalone matmul

A minimal single-tile Q8_0 matmul using 8 T0 harts (YOLO-style). Mostly useful
as a reference or for testing; the existing `mul_mat_Q8_0.c` (2048 harts,
multiple dispatch strategies) is more optimized for general use.

## Build

```bash
cd ported_models/llama_cpp_et/kernels/

# Build fused FFN
export LLAMA_CPP_ET=/path/to/llama.cpp-et
export ET_PLATFORM=/opt/et-platform
./build.sh

# Output: build/llama32_fused_ffn.elf
```

Requires `riscv64-unknown-elf-gcc` on `$PATH`.

## Integration

### Option A — Add to the uberkernel build (for PR/CI)

1. Copy the kernel to the ET backend kernel source:
   ```bash
   cp kernels/llama32_fused_ffn.c \
     llama.cpp-et/ggml/src/ggml-et/et-kernels/src/
   ```
2. Add to `et-kernels/CMakeLists.txt` KERNELS list
3. Modify `llama.cpp-et` FFN inference code (e.g., in `llama.cpp` or
   `ggml-et.cpp`) to detect the FFN pattern and call this kernel
   instead of dispatching `ggml_mul_mat` × 3 + `ggml_silu` + `ggml_mul`
4. Repin the submodule in the hackathon repo

### Option B — Runtime load (for local iteration)

Place the `.elf` where `GGML_ET_KERNELS_PATH` points, then modify the same
FFN dispatch code to load and call this kernel.

## Shapes (Llama 3.2 1B)

| Parameter | Value |
|---|---|
| `hidden` | 2048 |
| `intermediate` | 8192 |
| `H_blocks` | 64 |
| `I_blocks` | 256 |
| Weight memory (3 matrices) | ~53 MB |
| Scratch memory (gate + up → gated) | 64 KB → 32 KB |

## Next optimizations

- **Attention fusion**: Fuse QKV projection + attention + output projection
- **Weight prepacking**: Repack Q8_0 blocks into OC-tile-major layout for
  better cache behavior (like YOLO's `repack_1x1_weights`)
- **Scratchpad tiling**: Stream K tiles through L1 scratchpad instead of
  loading all weights from DRAM per row
- **KV cache reuse**: Keep KV cache on-chip across decode steps
