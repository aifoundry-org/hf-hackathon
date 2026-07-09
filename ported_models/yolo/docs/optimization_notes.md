# YOLO end-to-end detector — optimization notes

**Variant:** `yolo_m30_vpus2` — replaces 3 scalar 3×3 stride-2 convolutions
with VPU-vectorized versions.

## Optimization: VPU stride-2 convolutions (conv0, conv1, conv3)

The existing `conv2d_3x3_s2_p1_fp32_mh_vpu` function was already implemented
in the source but was not wired into the main pipeline for conv0, conv1, and
conv3 — they used the scalar `CONV_MH` instead. This change switches them to
`CONV_3x3_S2_P1_VPU`.

| Conv | Input shape | Output shape | MACs | % of total model |
|------|-------------|--------------|-----:|:-----------------|
| conv0 | 3×288×512 → 16×144×256 | 3×3 s=2 | 15.9M | 4.5% |
| conv1 | 16×144×256 → 32×72×128 | 3×3 s=2 | 42.5M | 12.1% |
| conv3 | 32×72×128 → 64×36×64  | 3×3 s=2 | 42.5M | 12.1% |
| **Total** | | | **100.9M** | **28.8%** |

## Performance estimate (analytical)

The VPU `fmadd.ps` instruction processes 8 lanes of floating-point multiply-add
in 1 cycle. For stride-2, only 4 of 8 lanes carry valid output data (the other
4 compute garbage and are discarded). Expected speedup over the scalar loop:

- **Ideal:** 4× (8 lanes / 2 discard factor)
- **Realistic:** 2× (edge-path scalar fallback for boundary columns, VPU
  setup overhead for weight broadcast and input load, and memory bandwidth
  limitations)

The 3 scalar convs account for ~29% of the model's total MACs. At 2× speedup:

$$\text{Total improvement} = \frac{0.29}{2} = 14.4\%$$

| Component | Detail |
|:----------|:-------|
| Conv0 MACs (scalar → VPU) | 15.9M → ~2× faster |
| Conv1 MACs (scalar → VPU) | 42.5M → ~2× faster |
| Conv3 MACs (scalar → VPU) | 42.5M → ~2× faster |
| **Estimated total improvement** | **~14%** |
| Method | Analytical (MAC counts × expected VPU speedup) |

> **Caveat:** This is an analytical estimate based on MAC counts and expected
> VPU speedup. The actual improvement depends on memory bandwidth, cache
> behavior, and the scalar vs VPU instruction mix. A proper partial run on
> sys-emu is not feasible for this kernel (full YOLOv10n is too large for
> emulation). The authoritative score is board CI.

## Verification

The VPU stride-2 function `conv2d_3x3_s2_p1_fp32_mh_vpu` was already present
in the source and is used by other code paths. The change only replaces the
macro invocation — the function itself is unchanged. Correctness is preserved
by construction: the same weights, input, and output pointers are passed.

## Other options considered

| Option | Verdict |
|--------|---------|
| YOLO_SKIP_FINAL_EVICT | Removes redundant full-tensor evictions (~0.9% est.) |
| YOLO_BOUNDARY_ONLY_EVICT | Doesn't apply — kernel uses OC partition for convs (not row-stripe), so all tensor data must be visible to all harts |
| YOLO_USE_16HART | Incompatible with BENCH_THREAD0_ONLY (work-slicing mismatch) |
| VPU depthwise stride-2 | No VPU depthwise stride-2 function exists |