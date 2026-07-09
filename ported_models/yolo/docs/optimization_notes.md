# YOLO end-to-end detector — optimization notes

This documents two optimizations applied to the **real YOLOv10n end-to-end
detector** (`ported_models/yolo/src/yolo_m30_argbuf.c`).

**Variant:** `yolo_m30_vpus2` (combined: VPU stride-2 convs + skip final evict)

## Optimization 1: VPU stride-2 convolutions (conv0, conv1, conv3)

**What it does:** Replaces 3 scalar 3×3 stride-2 convolutions with VPU-vectorized
versions using the existing `conv2d_3x3_s2_p1_fp32_mh_vpu` function.

| Conv | Input | Output | Scalar MACs |
|------|-------|--------|------------:|
| conv0 | 3×288×512 → 16×144×256 | 3×3 s=2 | 15.9M |
| conv1 | 16×144×256 → 32×72×128 | 3×3 s=2 | 42.5M |
| conv3 | 32×72×128 → 64×36×64  | 3×3 s=2 | 42.5M |
| **Total** | | | **100.9M** |

**Why it's faster:** The VPU `fmadd.ps` instruction processes 8 lanes of
floating-point multiply-add in 1 cycle. For stride-2, 4 of 8 lanes carry valid
output (the other 4 compute discarded garbage), giving a ~2-4× effective
speedup over the scalar per-output-pixel loop.

These 3 convs are at the largest spatial dimensions (288×512 → 144×256 → 72×128)
and account for ~29% of the model's total multiply-accumulate operations.
The VPU version was already implemented in the source (function exists) but
not wired into the main pipeline for these 3 convs — likely an oversight or
the original author prioritized correctness over optimization for the
initial end-to-end port.

## Optimization 2: `YOLO_SKIP_FINAL_EVICT`

**What it does:** Removes 20 redundant `EVICT_AND_FENCE` calls — 14 at the end
of the backbone and 6 at the end of the detection head. These calls evicted the
**full tensor** even though each conv function already evicted its own output
slice via the per-hart eviction at the end of `conv2d_*_mh`. The `MH_BARRIER()`
after each conv ensures the eviction is complete before the next stage.

Tensors are much larger than L1D (up to 2.4 MB vs 32 KB per-hart L1D), so
only a tiny fraction was cached by any single hart. For cache lines never in
the local L1D, `evict` is a tag-lookup no-op — but the `WAIT_CACHEOPS + FENCE`
overhead is still paid. Removing these saves ~877 kcycles per invocation.

## Performance estimate

| Optimization | Est. improvement | Notes |
|:-------------|:-----------------|:------|
| VPU stride-2 convs | ~14% | 3 convs → VPU, ~2× faster each |
| YOLO_SKIP_FINAL_EVICT | ~0.9% | 20 redundant eviction sequences |
| **Combined** | **~15%** | Both optimizations in one variant |

> **Caveat:** These estimates assume a realistic 2× speedup from VPU stride-2
> (the ideal 4× is limited by edge-path scalar fallback and memory bandwidth).
> The total YOLOv10n MAC count is estimated at ~350M. Board CI is the
> authoritative score.

## How it works (for future optimizers)

The kernel partitions by **output channel** (OC) across 8 T0 harts using
`yolo_range(N, cidx, ...)`. Each hart computes a contiguous range of output
channels for all spatial positions. Every conv function evicts only its own
OC slice and follows with `MH_BARRIER()`.

The VPU stride-2 conv (`conv2d_3x3_s2_p1_fp32_mh_vpu`) processes 4 output
columns per VPU iteration. Input columns are loaded 8-wide via `flq2`; only
even lanes (0, 2, 4, 6) carry valid output-aligned data. Lane 1/3/5/7 compute
garbage and are discarded on store. Edge columns near the border fall back to
scalar per-lane update.

## Other options considered

| Option | Verdict |
|--------|---------|
| `YOLO_BOUNDARY_ONLY_EVICT` (old synthetic kernel approach) | Doesn't apply — kernel uses OC partition for convs (not row-stripe), so all tensor data must be visible to all harts |
| `YOLO_USE_16HART` | Incompatible with `BENCH_THREAD0_ONLY` (work-slicing mismatch). Would need to drop `BENCH_THREAD0_ONLY` and use `shire_barrier`, a larger change. |
| VPU depthwise stride-2 | No VPU depthwise stride-2 function exists; would need to write one. |
| OC16 VPU 1×1 | Marked as "hung silicon (register pressure?)" in source — not usable. |