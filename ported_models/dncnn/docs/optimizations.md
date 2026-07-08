# DnCNN3 on Erbium ETSOC1 - Optimization Journey

A complete engineering log of every optimization tried for the DnCNN3
denoiser on a single shire of Erbium ETSOC1, the silent silicon bugs we
hit on the way, and an honest assessment of which approaches paid off.

## Final state

| metric | value |
|---|---:|
| Per-image kernel time (best, n=5+ runs) | **2.43 s** (sigma < 2%, bit-exact deterministic) |
| Per-image steady-state (3-image internal loop) | **2.26 s** |
| fps single-shot | 0.41 |
| fps steady-state | 0.443 |
| Total MACs in network | 51.05 GMAC |
| Effective throughput | 21 GMAC/s |
| Audit max_abs vs FP32 ORT | 4.31e-02 (PASS 5e-2 gate) |
| Audit reproducibility | bit-exact across runs and across DMA-streamed inputs |
| Speedup vs v1 scalar (106.7 s) | **43.8x** single-shot, **47.2x** steady-state |
| Speedup vs prior FP32 VPU baseline (29.57 s) | **12.2x** |

---

## Optimization timeline (ordered by when we did them)

```
v1 scalar INT8                             106.70 s    1.0x    baseline
v2 TFMA INT8 (initial, with bugs)          [debug]    -        2 silent silicon bugs found
v2 TFMA INT8 working                         4.01 s   26.6x    Tier 1.1 done
v2 + VPU fmadd.ps marshalling                2.77 s   38.5x    
v2 + padded layout + non-volatile pack       2.43 s   43.8x    <- current best single-shot
v2 with SMT T1 producer                    [hang]      -       blocked, see sec 3.5
v2 with VPU fscw.ps scatter pack             2.62 s   no help  reverted
v2 with k0-outer pack loop                   3.27 s   slower   reverted
```

---

## 1. Approaches that paid off

### 1.1 TFMA INT8 hardware dispatch - **biggest single win**

**What**: Replace scalar int8xint8->int32 inner loop in the 18 hidden conv
layers with `tensor_fma(type=3, ...)` dispatches on Erbium's TFMA matrix
unit.

**Setup**: Per output tile (16 OC x 16 spatial), issue 9 dispatches
(one per `(ky, kx)` tap of the 3x3 conv), each computing 16 OC x 64 IC x 16 spatial =
16,384 INT8 MACs in ~318 cycles. `first_pass=1` only on tap 0,
`tenc_loc=1` only on tap 8 (copies TenC accumulator -> FREGs for `tensor_store`).

**Numbers**:
- Cycles per MAC went from 13.4 (scalar) -> 0.044 (TFMA) = **305x tighter**
- Wall time: 106.70 s -> 4.01 s = **26.6x speedup**

**Why it worked**: TFMA's INT8 path widens the multiplier array 4x compared
to FP32 (same silicon area, narrower lanes). Combined with a shorter
pipeline, it delivers 6.9x more MACs/cycle than FP32 on the same hardware,
plus 4x more compute per dispatch from the wider `acols`.

**Gotchas hit (see sec 2 silent bugs)**:
- Linker placed weight blobs at non-cache-aligned addresses; `tensor_load`
  silently rounded down by up to 60 bytes
- `tensor_fma(tenc_loc=1)` clobbers FREGs without GCC knowing - required
  explicit FREG clobber barrier

**Verdict**: [yes] **Essential.** The single change worth all the effort. If
you're doing CNN inference on Erbium, this is non-negotiable.

---

### 1.2 VPU `fmadd.ps` marshalling - **second-biggest win**

**What**: Replace the scalar FP marshalling (`int32 -> fp32 -> xscale + bias ->
ReLU -> xinv_scale -> sat_int8`) with 8-lane packed FP32 SIMD using
`fmadd.ps`, `fmax.ps`, `fmin.ps`, `fcvt.pw.ps`.

**Setup**: Per OC, 16 lanes processed as 2 x 8-lane vectors:
```
fcvt.ps.pw (int32 -> fp32) -> fmadd.ps (x combined_scale + combined_bias)
                          -> fmax.ps (ReLU at 0) -> fmin.ps (clamp at 127)
                          -> fcvt.pw.ps (round-nearest-even back to int32)
                          -> scalar lane-wise int8 store
```

Pre-folded the math: `combined_scale[oc] = w_scale x a_scale_in x inv_a_scale_out`,
`combined_bias[oc] = bias x inv_a_scale_out`. Single `fmadd` covers
dequant+bias.

**Numbers**:
- Marshalling cost per layer: 100 ms -> 32 ms (3.1x faster on that block)
- Wall time: 4.01 s -> 2.77 s, saving 1.24 s (45% -> 32% of layer time)
- Total speedup vs v1: 26.6x -> 38.5x

**Why it worked**: 8-lane SIMD is 8x the throughput of scalar FP. The
remaining 32 ms is dominated by the lane-wise int<->fp conversions and the
scattered int8 dst stores (NHWC stride 64), which can't be vectorized
without changing the output layout.

**Verdict**: [yes] **Worth doing.** The VPU was already there; using it on
the FP-shaped portion of the int8 pipeline is free leverage.

---

### 1.3 Padded activation buffer + branch-free `pack_b_tile` - **modest win**

**What**: Pad activation buffers from 240x320x64 (4.69 MB) to 242x322x64
(4.99 MB, +1.5%) with zero halo on all sides. `pack_b_tile` reads from
`act_pad[(oy+ky)-W_PAD + (ox+kx+j)]` - no bounds checks needed because the
halo provides the zero fallback for out-of-image taps.

Also dropped `volatile` from `static_bpack` so the compiler can fuse byte
stores into 4-byte `sw` instructions, and added explicit `__asm__("" :::
"memory")` barriers to preserve ordering around `evict()`.

**Numbers**:
- Wall time: 2.77 s -> 2.43 s, saving 0.34 s
- pack_B cost per layer: 62 ms -> 47 ms (1.3x faster)
- Speedup vs v1: 38.5x -> 43.8x

**Why it worked**: Eliminating the `if (y_in valid && x_in valid)` branch
removed a stall in the inner loop. Combined with wider stores, total
pack_B IPC improved meaningfully.

**Verdict**: [yes] **Easy win.** Low effort, real benefit. Should be
considered any time a kernel has boundary-conditional accesses inside a
hot loop.

---

### 1.4 64-byte aligned A-pack via runtime copy - **silent-bug fix**

**What**: At kernel startup, all 8 minions cooperatively copy the
INT8 weight blob from its linker-placed (4-byte aligned) address to a
64-byte-aligned static buffer. Subsequent `tensor_load` calls target
the aligned buffer.

**Numbers**:
- Without this fix, kernel produced max_abs=2.19 (FAIL) - TFMA was reading
  60 bytes of stale prior memory + 4 bytes of the actual weight blob due
  to silent address rounding
- After fix: max_abs=4.31e-02 (PASS), one-time copy cost ~1.6 ms
  (cooperative, amortized across 8 harts)

**Why we needed it**: `tensor_load` encodes its addr field as
`addr & 0xFFFFFFFFFFC0ULL`, silently dropping the low 6 bits. The linker
sections were 4-byte aligned (header noted size in bytes was 663,552;
loaded at offset `0x...3c`).

**Verdict**: [fix] **Mandatory bug fix**, not really an "optimization."
Documented as gotcha #17 in the project journal.

---

### 1.5 FREG clobber barrier around `tensor_fma`/`tensor_store` - **silent-bug fix**

**What**: After the dispatch chain that ends with `tenc_loc=1` (which
copies the int32 TenC accumulator into the floating-point register file),
add an explicit clobber barrier:
```c
__asm__ __volatile__("" ::: "memory",
    "f0",  "f1",  ..., "f30", "f31");
```

**Why we needed it**: `tensor_fma` is a CSR write asm with no FP-register
clobber list. GCC happily hoists FP scalars (like `a_scale_in`,
`inv_a_scale_out`) into FREGs that TFMA *silently overwrites* with int32
output. The marshalling math then runs with garbage scales.

**Symptoms**: First-tile outbuf was bit-exact correct (TFMA computed
correctly), but final image diverged by max_abs=1.59. The compromised
scales propagated through the marshalling chain.

**Verdict**: [fix] **Mandatory bug fix.** Documented as gotcha #18.

---

## 2. Silent silicon bugs encountered

These deserve their own section because each one looked like "the kernel
just produces wrong output" with no diagnostic signal.

### 2.1 Linker alignment + `tensor_load` silent address rounding

**Symptom**: `tensor_fma` produces values that look like noise. Output
audits at max_abs=2.19 instead of expected ~0.04.

**Diagnosis**: 8 hours of bisection across the kernel pipeline. Per-tile
debug dump showed scalar-reference and TFMA agreed bit-exact for tile 0
of hart 0 layer 0... yet the final 18-layer output diverged by huge
amounts. Eventually noticed all the linker-placed binary blobs landed at
addresses with low 6 bits = `0x3c`. Cross-checked the `tensor_load`
encoding: `(addr & 0xFFFFFFFFFFC0)`. Silent rounding by up to 60 bytes.

**Fix**: cooperative runtime copy to a 64-byte-aligned static buffer.
See sec 1.4.

### 2.2 `tensor_fma(tenc_loc=1)` writes FREGs invisibly to GCC

**Symptom**: After the sec 2.1 fix, the *first* output tile was bit-exact
correct, but the final image still diverged at max_abs=1.59. All 8 harts'
*first-tile* outbufs validated, all 20 tiles of hart-0 layer-0 validated.

**Diagnosis**: The TFMA dispatch with `tenc_loc=1` copies the int32 TenC
accumulator into the floating-point register file. The marshalling code
that follows does FP math. GCC had hoisted `a_scale_in` and
`inv_a_scale_out` into FREGs that TFMA was overwriting with int32 garbage.
The first iteration's marshalling worked because GCC reloaded the FREGs
from memory (no stale value yet); subsequent iterations ran with the
clobbered values.

**Fix**: explicit `__asm__("" ::: "memory", "f0".."f31")` after every
`tensor_store`. See sec 1.5.

### 2.3 BSS not auto-zeroed in this U-mode runtime path

**Symptom**: Halo activations had random values from prior memory state,
producing wrong outputs at image boundaries.

**Fix**: Cooperative zero-init of activation buffers at kernel startup,
across all active harts.

### 2.4 `fdiv.s` hangs in U-mode

**Symptom**: A `1.0f / a_scale` expression in the kernel hung the kernel
indefinitely.

**Fix**: Precompute reciprocals on the host, link as `.rodata`. Or
replace `x /= constant` with `x *= 1.0f/constant` (where `1.0f/constant`
is computed at host-link time as a constant).

---

## 3. Approaches that didn't pay off

### 3.1 INT4 weights - **rejected with reasoning**

**Hypothesis**: 2x weight memory savings might help.

**Why rejected**:
- TFMA on Erbium has only `fp32`, `fp16`, `int8` paths; the four other
  enum slots throw "illegal type" in sysemu. INT4 has no hardware path.
- DRAM is at 0.001% utilization during compute (PMC measurement). Halving
  weight memory saves nothing we care about.
- Software emulation would unpack INT4 to INT8 at dispatch time, *adding*
  CPU work to a CPU-bound kernel.

**Verdict**: [no] Confirmed not applicable on this hardware for this workload.

---

### 3.2 Block-sparse activations + TFMA zero-skip - **rejected after measurement**

**Hypothesis**: TFMA's INT8 dispatch has a zero-skip path that elides
compute when 4-IC quartets of activation are all zero. ReLU activations
have ~38% element-wise zeros - could we exploit this for speedup?

**What we did**: Measured the actual fraction of 4-IC quartets in the
activation tensors (post-ReLU, post-quantization) that happen to be all
zero across all 18 hidden layers.

**Result**:
- Element-wise zero rate: 38.5% average (lots of dead ReLU outputs)
- **4-IC-quartet zero rate: 2.89%** (the granularity TFMA actually skips)
- Even with PERFECT skip on every quartet-zero block: ~9 ms wall savings = 0.4%

**Why so low**: The TFMA zero-skip needs 4 *contiguous* IC values at the
same spatial position to be all zero. Natural denoising activations don't
cluster that way - element-wise zeros are scattered, not block-shaped.

**Verdict**: [no] Even with QAT to maximize block-sparsity, the ceiling
gain is too small to justify a retraining effort.

---

### 3.3 VPU `fscw.ps` scatter for `pack_b_tile` - **didn't help**

**Hypothesis**: `pack_b_tile` writes 9216 scattered bytes per output tile.
Vectorizing the scatter via the VPU's 8-lane `fscw.ps` (scatter word)
should be faster than scalar 4-byte stores.

**Numbers**:
- Scalar 4-byte stores (j-outer, padded layout): 47 ms / layer
- VPU scatter via `fscw.ps`: 49 ms / layer (slightly worse)

**Why it didn't help**: `fscw.ps` issues 8 individual cache-line stores
per call under the hood. The scatter doesn't compress them; it's the
same total store traffic with extra setup overhead. The scalar version's
tighter unrolled loop was actually slightly faster.

**Verdict**: [no] Reverted.

---

### 3.4 `k0`-outer pack loop - **slower than `j`-outer**

**Hypothesis**: Reordering the pack inner loop so each iteration writes
contiguous 4-byte chunks within ONE cache line (16 stores total) should
be cache-friendlier than scattering 16 stores across 16 cache lines.

**Numbers**:
- Original `j`-outer / `k0`-inner: 47 ms / layer
- New `k0`-outer / `j`-inner: 85 ms / layer (1.8x slower)

**Why it didn't help**: The new order required 16 different source-cache-line
reads per (k0) iteration to gather 16 spatial positions' worth of 4-byte
chunks. The original order had 16 cache lines warm in L1D the whole time
(write-combining), with a single 64-byte source read per j shared across
all 16 k0 destinations. Read pressure dominated.

**Verdict**: [no] Reverted. The original loop order is better.

---

### 3.5 SMT T1 producer for pack_B - **blocked by hardware**

**Hypothesis**: T0 spends 50% of each tile in `tensor_wait(TENSOR_FMA_WAIT)`
where the integer pipeline is idle. T1 (the SMT sibling) could pack B
for tile N+1 during this window, hiding the 47 ms/layer pack_B cost.

**What we did**:
1. Activated T1 across all 8 minions, included it in `shire_barrier`
   masks (validated: T1 attends barriers, kernel completes when T1 does
   no work)
2. Built a producer/consumer protocol: T1 writes `pipe->produced`,
   T0 polls; double-buffered bpack[2]
3. Added explicit `evict + WAIT_CACHEOPS + FENCE` around flag updates
   to handle Erbium's split-mode L1D between SMT siblings

**Result**: Kernel hangs (timeout, launcher segfaults). Deadlock in the
producer/consumer flag exchange.

**Diagnosis** (incomplete): Erbium has 3 L1D modes (`l1d_shared`,
`l1d_split`, `l1d_scp`). In split mode, T0 and T1 use disjoint cache sets;
flag updates from T1 don't surface in T0's L1D without going through L2.
The cacheops umode header notes: *"On real Erbium the same encoding
resolves to 'memory' (since erbium has no L2/L3 cache), matching native
erbium's hardcoded `CACHEOP_DST_MEM`."* So evict goes to DRAM with
larger latency than expected. The exact failure point is unclear without
instrumentation we couldn't reach (kernel hangs prevent dump).

**Workaround attempted**: per-iteration `evict + invalidate-and-reload`
of flag addresses. Still hangs.

**Verdict**: [paused] **Blocked.** The optimization could probably work with
the right primitive, but figuring out the right primitive on real silicon
needs Esperanto guidance. The scaffolding is in place behind a
`DNCNN_SMT_PIPELINE` define; can be revisited when we have an answer.

**Projected gain if it worked**: ~0.85 s wall savings (47 ms x 18
layers), bringing wait to ~1.5 s = 0.66 fps. Real but not transformative.

---

### 3.6 Cooperative weight loads (`tensor_coop`) - **not pursued (small win)**

**Hypothesis**: `tensor_coop` lets one minion fetch a weight chunk and
broadcast to all 8 minions sharing the load. Reduces L2 read pressure
for A operands.

**Why not pursued**: The TFMA dispatch chain itself is only 18 ms /
layer (10% of total). Even halving that saves ~0.16 s wall - small
compared to pack_B (47 ms) and marshalling (32 ms) which were our
priority. After the SMT effort failed, we ran out of session budget.

**Verdict**: [paused] **Pending.** Worth doing for its own sake; estimated
~5-7% additional speedup. Pattern is proven in `coop_tl_tfma_fc.c` so
implementation should be straightforward when revisited.

---

### 3.7 Multi-shire scaling - **policy-blocked**

**Hypothesis**: 32 shires x current per-shire throughput = 32x linear
speedup -> ~76 ms / image = 13 fps.

**Why not pursued**: Project constraint specifies single-shire only
(per `CLAUDE.md`).

**Verdict**: [paused] **Eventual ceiling-breaker** when policy permits. The
per-shire kernel as built should drop in trivially via `shire_mask =
0xFFFFFFFF`.

---

### 3.8 Network surgery (channel/depth prune + retrain) - **out of scope**

**Hypothesis**: Halving channels (64 -> 32) gives 4x MAC reduction.
Halving depth (18 -> 9 layers) gives 2x reduction. Combined with QAT to
recover accuracy: ~3-5x speedup.

**Why not pursued**: This is a model-redesign + retraining project, not
a kernel optimization. Requires PyTorch DnCNN3 training pipeline,
calibration data, multiple training epochs, accuracy validation. Out of
scope for the silicon-side work.

**Verdict**: [wip] **Big lever, separate project.** Worth flagging if the
team wants to push fps significantly higher than the kernel-side ceiling.

---

## 4. Architectural decisions and tradeoffs

These are not "optimizations" per se - they're the structural choices
that shaped the kernel and weren't explicitly evaluated against
alternatives. Recording them for future-us.

### 4.1 Mixed-precision quantization

**Choice**: FP32 for `conv_first` (1->64) and `conv_final` (64->1) +
residual subtract; INT8 per-channel symmetric for the 18 hidden layers.

**Tradeoff considered**: All-INT8 vs mixed-precision.

**Reasoning**:
- `conv_first` and `conv_final` together are 0.87% of total MACs. Their
  speedup contribution if quantized would be negligible.
- They sit at the I/O boundaries where quantization noise is most visible
  in the output.
- `conv_final` does the residual subtract producing the user-facing image;
  any noise here is directly visible.

**Verdict**: [yes] Right call. INT8 hidden + FP32 boundary is the standard
mixed-precision PTQ pattern for image-to-image networks.

---

### 4.2 Per-channel weight scales, per-tensor activation scales

**Choice**: Each output channel of each weight tensor gets its own scale
factor (1152 scales total for 18 layers x 64 OC). Each activation tensor
gets one scale per layer (19 scales total).

**Tradeoff considered**: Per-tensor for everything (simpler) vs
per-channel for both (more accurate, more storage).

**Reasoning**:
- Per-channel weights catch the wide range of magnitudes between
  different output filters (some channels have weights ~10x larger than
  others).
- Per-tensor activations are cheaper and PTQ converges fine here because
  ReLU bounds activation distribution and 32 calibration images cover
  the variation.

**Verdict**: [yes] Good balance. Per-channel activations would have been
overkill and complicated the marshalling.

---

### 4.3 NHWC activation layout

**Choice**: Activations stored as `[y][x][ic=0..63]` int8, row-major.
64 IC fits in exactly one 64-byte cache line.

**Tradeoff considered**: NCHW (channel-first), `[k0][y][x][4_ic]`
(quartet-grouped), or various tile-blocked layouts.

**Reasoning**:
- 64 IC x 1 byte = exactly one cache line - the hardware was practically
  designed for this. One contiguous read fetches all channels.
- Matches the existing scalar reference code so we could reuse `conv_first`,
  `conv_final` unchanged.

**Cost**: TFMA's INT8 B-operand layout is "4-IC-quartet interleaved"
(different from NHWC) - forced our `pack_b_tile` step which became 47%
of every hidden layer's time. We tried twice to eliminate it (HW
interleave-load, k0-outer-major layout) but alignment and cache-thrash
issues defeated both.

**Verdict**: [partial] Probably the right starting layout, but `pack_b_tile`
is our biggest remaining cost. Future work might revisit a `[k0][y][x]
[4_ic]` layout with 21% memory bloat in exchange for eliminating
pack_B entirely.

---

### 4.4 Row-stripe partition across 8 minions

**Choice**: Each of 8 minions owns 30 rows (240/8). Minions process their
stripes independently, with cross-hart halo invalidation between layers.

**Tradeoff considered**: OC-stripe (each minion owns 8 of 64 channels),
spatial-stripe (each minion owns a quadrant), cooperative tiling.

**Reasoning**:
- Row-stripe has the simplest balanced split with minimal cross-hart
  data exchange (only 1 row above + 1 row below per minion).
- Each minion's working set per layer fits in a small fraction of L1D /
  L2, so memory locality is excellent.
- Can always go to OC-stripe or 2D tiling later if we need finer-grain
  control.

**Verdict**: [yes] Right starting choice. The kernel would have been much
more complex with any other partition.

---

## 5. Lessons learned

### 5.1 Real silicon has silent failure modes

Two of our biggest bugs (linker alignment, FREG clobbering) had **no
runtime error signal** - kernel ran to completion, just produced wrong
numbers. Both were caught only by audit failure + days of bisection.

**Takeaway**: For TFMA-class hardware, every external blob touched by
`tensor_load` must be 64-byte aligned, and every compiler-visible interface
to a tensor instruction must declare an FP-register clobber list.

### 5.2 Don't optimize without measuring first

We almost spent multiple days on QAT+block-sparse-training to exploit
TFMA zero-skip - until we measured the existing block-zero rate
(2.89%) and found the ceiling was 0.4% wall savings.

**Takeaway**: When an optimization "should work in theory," measure the
actual exploitable headroom before committing to it.

### 5.3 SIMD-able vs compute-bound vs memory-bound diagnosis matters

PMC analysis told us we were compute-bound on the scalar pipeline (13.4
cyc/MAC vs ideal 3 - most cycles in mul-add chain). That immediately
narrowed the search to "use the TFMA," which produced 26.6x from a
single change. Without PMC we might have chased memory optimizations
(which the same analysis showed were at <0.01% utilization).

**Takeaway**: Spend the time to capture and read PMC counters early.
The ratio of compute / DRAM / L2 traffic / L1D miss rate determines
which optimization category is even *possible*.

### 5.4 Loop reordering can go either way

We tried two pack_B loop orders. The "obviously better" cache-line-batched
write order turned out to be 1.8x *slower* because read pressure on the
source cache lines dominated. The original "obviously bad" scattered-write
order won.

**Takeaway**: Microarchitectural intuition is useful for hypothesis
generation but only measurement decides. Reorders are cheap to try; just
try both and pick the faster one.

### 5.5 The right SIMD tool depends on the operation type

VPU `fmadd.ps` worked great for the marshalling FP chain (3x speedup on
that block). VPU `fscw.ps` (scatter) didn't help for pack_B (silicon
issues 16 stores either way regardless of which SIMD instruction emitted
them). The architectural advantage of SIMD comes from compute lanes, not
from store lanes.

**Takeaway**: For VPU on Erbium, SIMD wins are concentrated in the
arithmetic ops (`fmadd`, `fmax`, `fmin`, `fcvt`), not in the load/store
ops. Applies broadly to in-order CPUs with separate SIMD compute lanes
and store buffers.

---

## 6. Where the remaining time goes (after all optimizations)

Per hidden layer (100 ms x 18 layers = 1800 ms = 74% of total):

| sub-phase | ms | % |
|---|---:|---:|
| pack_B (NHWC -> TFMA layout, scalar 4-byte writes) | 47 | 47% |
| TFMA dispatch chain (load A + load B + 9 x fma) | 18 | 18% |
| tensor_store + evict outbuf | 1 | 1% |
| VPU marshalling (dequant + bias + ReLU + requant) | 32 | 32% |
| barriers + halo evict | 2 | 2% |

Plus the bookend phases:

| phase | ms | % |
|---|---:|---:|
| conv_first + quantize (scalar FP32) | 290 | 12% |
| 18 hidden layers | 1800 | 74% |
| conv_final (scalar FP32, residual subtract) | 100 | 4% |
| barriers + setup | 245 | 10% |
| **total** | **2435** | 100% |

The TFMA dispatch is now only 18% of each layer. The bottleneck has
shifted entirely to the data-marshalling glue (pack_B + marshal = 79%).

---

## 7. Future levers (ranked)

| optimization | expected wall savings | risk | status |
|---|---:|---|---|
| Multi-shire (32x) | 32x linear -> 76 ms/img = 13 fps | low | policy-blocked |
| SMT pipelining for pack_B | ~0.85 s | medium | blocked, needs Esperanto guidance on right sync primitive |
| Cooperative weight loads (`tensor_coop`) | ~0.15 s (~5-7%) | low | not pursued, recommend revisiting |
| Network surgery (prune + QAT) | 2-4x | medium | out-of-scope, separate project |
| L2 SCP (4 MB shire-shared) caching | unclear | medium | unused so far; could replace some DRAM traffic |
| Larger spatial tile + B reuse across neighbors | ~0.2 s | medium | fiddly indexing |

The single biggest leftover lever is **multi-shire** (32x linear), which
needs only `shire_mask` configuration but is policy-gated.

---

## 8. Reproduction

### Build the current best kernel

```bash
bash $ARTIFACT_ROOT/erbium_amp_probe/hf-real-dncnn/build_tfma_int8_v2.sh
# produces /tmp/dncnn_tfma/int8_tfma_v2.elf
```

### Run + audit

```bash
TIMEOUT=180 $ARTIFACT_ROOT/erbium_amp_probe/hf-real-dncnn/run_int8_tfma_v2.sh
# Prints rc=0 wait_s=2.43...
# Dump at /tmp/dncnn_tfma/int8_tfma_v2-<stamp>/dump.bin
# Output FP32 image at offset 0x300000, 240x320x4 bytes
```

### Audit against ORT

```bash
python3 -c "
import os, numpy as np
runs = sorted([d for d in os.listdir('/tmp/dncnn_tfma') if d.startswith('int8_tfma_v2-2026')])
data = open('/tmp/dncnn_tfma/' + runs[-1] + '/dump.bin', 'rb').read()
out = np.frombuffer(data[0x300000:0x300000+320*240*4], dtype=np.float32).reshape(240,320)
ref = np.fromfile('$ARTIFACT_ROOT/erbium_amp_probe/hf-real-dncnn/dncnn3_hf_240x320_ort_output_f32.bin', dtype=np.float32).reshape(240,320)
print(f'max_abs={np.abs(out-ref).max():.4e}  PASS={np.abs(out-ref).max()<5e-2}')
"
```

### Visualize

```bash
python3 $ARTIFACT_ROOT/erbium_amp_probe/hf-real-dncnn/tools/build_viz_html.py
# Open /tmp/dncnn_visualization.html in any browser
```

### Run streaming with custom input

```bash
LABEL=my_test /tmp/run_streaming_test.sh /path/to/input_240x320_fp32.bin
# Expects 307200-byte FP32 binary file (240x320 x 4)
```

### Key files

- `dncnn3_tfma_int8_v2.c` - the kernel
- `tools/quantize_dncnn_perchannel.py` - generates per-channel INT8 weights
  + per-tensor activation scales from the ONNX model
- `tools/pack_tfma_int8_a.py` - repacks weights into TFMA A-side layout
- `tools/build_viz_html.py` - generates the standalone HTML visualization
- `tools/render_dncnn.py` - single-pair PNG renderer
- `build_tfma_int8_v2.sh` - kernel build script
- `run_int8_tfma_v2.sh` - stage + run on `<board-host>`
- `audit_master_tfma_fp32.tsv` (in `etsoc1-hf-experiments/results/`)
  - every run we made with wait_s + max_abs

---

## 9. Summary table - every approach, ranked by impact

| # | approach | type | wall impact | verdict |
|---|---|---|---:|---|
| 1 | TFMA INT8 hardware dispatch | speedup | -102.7 s | [yes] essential |
| 2 | VPU `fmadd.ps` marshalling | speedup | -1.24 s | [yes] worth doing |
| 3 | Padded layout + non-volatile pack | speedup | -0.34 s | [yes] easy win |
| 4 | 64-byte aligned A-pack copy | bug fix | required for correctness | [fix] mandatory |
| 5 | FREG clobber barrier | bug fix | required for correctness | [fix] mandatory |
| 6 | Cooperative buffer zero-init | bug fix | required for correctness | [fix] mandatory |
| 7 | Precomputed reciprocals (no fdiv) | bug fix | required for correctness | [fix] mandatory |
| 8 | Streaming mode (DMA per frame) | demo | per-frame DMA validated | [yes] infra works |
| 9 | N-image internal loop | demo | shows steady-state 2.26 s | [yes] confirms scaling |
| 10 | VPU `fscw.ps` scatter pack | speedup attempt | +2 ms (slightly worse) | [no] reverted |
| 11 | k0-outer pack loop reorder | speedup attempt | +38 ms / layer | [no] reverted |
| 12 | INT4 weights | speedup attempt | n/a - no HW path | [no] rejected with reasoning |
| 13 | Block-sparse activations / QAT | speedup attempt | ceiling 0.4% wall savings | [no] rejected after measurement |
| 14 | SMT T1 pack_B producer | speedup attempt | hang on real silicon | [paused] blocked |
| 15 | Cooperative weight loads | speedup attempt | ~5% (estimated) | [paused] pending |
| 16 | Multi-shire (32x) | speedup attempt | 32x linear (estimated) | [paused] policy-blocked |

Total: 9 things tried that worked, 4 that didn't (1 reverted, 1 reverted,
1 rejected on theory, 1 rejected on measurement), 3 still on the bench.

End-to-end: from 106.7 s (v1 scalar baseline) to 2.43 s (best single-shot)
to 2.26 s (steady-state in a multi-image loop) - 47.2x speedup over v1
with bit-exact reproducibility and 4.31e-02 max_abs accuracy vs FP32 ORT.
