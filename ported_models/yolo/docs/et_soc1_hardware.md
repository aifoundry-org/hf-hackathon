# ET-SoC1 Hardware Capabilities

## Architecture Overview

**ET-SoC1** (Esperanto Technologies) is a RISC-V AI accelerator chip. Each
compute cluster (called a **shire**) contains:

| Unit | Count per Shire | Details |
|------|----------------|---------|
| **Minions** | 8 | RISC-V cores, each with 2 hardware threads |
| **T0 threads** | 8 | Even harts (0,2,4,...14) — have VPU + Tensor unit access |
| **T1 threads** | 8 | Odd harts (1,3,5,...15) — helper/drain threads |
| **Total harts** | 16 | Per shire |
| **VPU** | 1 per minion | 8-wide FP32 SIMD (vector processing unit) |
| **Tensor Unit** | 1 per minion | Hardware matrix multiply-accumulate + convolution |
| **L1D** | per minion | 64-byte cache lines, configurable as shared/split/SCP |
| **MRAM** | 16 MB | Main memory (shared across shires) |

**Full chip**: 32+ shires = 256+ T0 (compute) harts + 256 T1 (helper) harts.

---

## Compute Units

### 1. VPU (Vector Processing Unit) — **currently used by YOLO**

The VPU provides 8-wide FP32 SIMD via these instructions:

| Instruction | Operation |
|-------------|-----------|
| `flq2` | Load 8 floats (2×128-bit quads) |
| `fsq2` | Store 8 floats |
| `fmadd.ps` | Fused multiply-add: `acc += v * w` (8 lanes) |
| `fbcx.ps` | Broadcast scalar to 8 lanes |

Current YOLO uses `CONV_1x1` (OC8/OC16 blocked VPU), `CONV_3x3_P1` (OC4 VPU),
`CONV_3x3_S2_P1` (stride-2 VPU), and `CONV_DW3x3_*` (depthwise VPU).

### 2. Tensor Unit — **not used by YOLO**

The tensor unit is a hardware matrix multiply-accumulate engine. It operates
on data in the **SCP scratchpad** (Streaming Compute Path — a portion of L1D
reconfigured as tightly-coupled memory).

#### Pipeline

```
tensor_load()     — Load data from MRAM → SCP scratchpad
tensor_fma()      — C[m,n] += A[m,k] * B[k,n]  (hardware matmul)
tensor_store()    — Store results from RF → MRAM
tensor_quant()    — Optional: SiLU/ReLU/bias/quantization on the fly
tensor_reduce()   — Cross-hart sum/max/min reductions
```

#### Key API

| Function | Purpose |
|----------|---------|
| `tensor_load(use_tmask, use_coop, dst_start, transform, use_tenb, addr, offset, num_lines, stride, id)` | Load data to SCP |
| `tensor_fma(use_tmask, b_num_col, a_num_rows, a_num_cols, offset, tenc_loc, tenb_unsigned, tena_unsigned, tenb_loc, scp_loc_b, scp_loc_a, opcode, first_pass)` | Matmul: C += A × B |
| `tensor_store(reg_stride, start_reg, cols, Arows, addr, coop_store, stride)` | Store RF to memory |
| `tensor_store_scp(entry_stride, start_scp_entry, Arows, addr, stride)` | Store SCP to memory |
| `tensor_reduce_auto(start_reg, operation, num_reg, tree_depth)` | Cross-hart reduction |
| `tensor_quant(start_reg, col, row, scp_loc, transf0..9)` | Quantization pipeline |
| `convolution_ctrl(row_start, col_start)` | Convolution control |
| `convolution_size(srow, nrow, scol, ncol)` | Convolution dimensions |
| `tensor_coop(val)` | Cooperative load/store setup |
| `tensor_wait(id)` | Wait for tensor ops to complete |

### 3. SCP (Streaming Compute Path)

The SCP scratchpad is carved from L1D cache when in split+SCP mode:

```c
// Enable split L1D + SCP mode (requires M-mode syscall on soc1sim)
set_l1_cache_control(/*d1_split=*/1, /*scp_en=*/1);

// U-mode control
ucache_control(/*scp_en=*/1, /*cacheop_rate=*/0, /*cacheop_max=*/0);

// Check current mode
enum l1d_mode mode = get_l1d_mode();  // l1d_shared, l1d_split, l1d_scp
```

### 4. T1 Helper Threads

T1 threads (odd harts) can independently:
- Load weights into SCP via `tensor_load` while T0 runs FMA
- Drain results from SCP via `tensor_store_scp`
- Double-buffer data across two SCP regions

---

## Optimization Opportunities for YOLO

### Tier 1: 1×1 Convolutions → Tensor FMA (biggest win)

All ~30 1×1 convs in YOLOv10n are matrix multiplies:
```
C[OC, HW] = W[OC, IC] @ X[IC, HW]
```

The tensor FMA processes this directly in hardware. A tile-based approach:

```
for each OC tile (16 or 32 channels):
    for each HW tile (spatial positions):
        tensor_load(weights[OC_tile, IC], into SCP)
        tensor_load(activations[IC, HW_tile], into SCP)
        tensor_fma(C += A × B)
        tensor_quant(bias + SiLU)   // optional hardware activation
        tensor_store(C[OC_tile, HW_tile], to MRAM)
```

### Tier 2: 3×3 Convolutions → Convolution Accelerator

```
convolution_ctrl(row_start, col_start);
convolution_size(srow, nrow, scol, ncol);
// Then tensor loads + FMA for the im2col'd data
```

### Tier 3: PSA Attention → Tensor FMA + Reduce

The two matmuls in PSA attention:
- `Q @ K^T`: `[2, 144, 32] @ [2, 32, 144]` → `[2, 144, 144]`
- `V @ softmax^T`: `[2, 64, 144] @ [2, 144, 144]` → `[2, 64, 144]`

The softmax can use `tensor_reduce_auto` for cross-hart FMAX/FADD reduction.

### Tier 4: Cross-Shire Parallelism

The full chip has 32 shires. YOLO uses 1.
- **Data parallelism**: Run 32 images simultaneously
- **Model parallelism**: Split layers across shires, sync via `tensor_reduce`

### Tier 5: T1 Double-Buffering

Use T1 threads to prefetch the next tile's weights while T0 computes:
```
T0: tensor_fma(tile_N)
T1: tensor_load(tile_N+1 weights, into SCP_B)
T0: tensor_fma(tile_N+1 from SCP_B)
T1: tensor_load(tile_N+2 weights, into SCP_A)
...ping-pong between SCP_A and SCP_B
```

---

## Reference

- Tensor unit tests: `et-platform/test-compute-kernels/src/tl_tfma_tstore_fc/`
- MLP with tensor FMA: `et-platform/test-compute-kernels/src/mlp/`
- Tensor header: `<erbium/isa/tensors.h>`
- SCP/cache control: `<erbium/isa/cacheops-umode.h>`
