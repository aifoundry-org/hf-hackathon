# YOLO end-to-end detector — optimization notes

This documents the optimization applied to the **real YOLOv10n end-to-end
detector** (`ported_models/yolo/src/yolo_m30_argbuf.c`). This is a full
480×640 RGB → detection‑list pipeline (preprocessing, backbone, neck, PSA
attention, head, DFL decode, NMS) running on 8 T0 harts with VPU‑accelerated
convolutions.

## Optimization: `YOLO_SKIP_FINAL_EVICT`

**What it does:** Removes 20 redundant `EVICT_AND_FENCE` calls — 14 at the end
of the backbone and 6 at the end of the detection head. These calls evicted the
**full tensor** (all channels, all spatial positions) even though:

1. Each conv function (`conv2d_fp32_mh`, `conv2d_1x1_fp32_mh_vpu_oc8`,
   `conv2d_3x3_p1_fp32_mh_vpu_oc4`, etc.) **already** evicts the writing
   hart's own output slice at the end of the function.
2. The `MH_BARRIER()` macro (`FENCE; WAIT_CACHEOPS; atomic_barrier`) follows
   every conv, ensuring the eviction is globally visible before the next stage.
3. Tensors are much larger than L1D (up to 2.4 MB vs 32 KB per-hart L1D), so
   only a tiny fraction of the tensor was cached by any single hart.
4. For cache lines never in the local L1D, `evict` is a tag‑lookup no‑op —
   but the `WAIT_CACHEOPS + FENCE` serialization overhead is still paid.

**Safety:** write‑eviction of output slices is unchanged (each conv still
evicts what it wrote). The removed evictions only targeted data that was
either already in L2 or never in the local cache. No cross‑hart coherence
path is affected.

## Performance estimate

| Metric | Value |
|--------|------:|
| Redundant evictions removed | 20 calls |
| Total address range iterated | ~10.7 MB (174,672 cache lines) |
| Est. cycles saved per invocation | ~877 kcycles |
| Est. time saved (1 GHz) | ~0.88 ms |
| Est. time saved (1.2 GHz) | ~0.73 ms |
| Est. improvement on ~100 ms kernel | ~0.9 % |
| ELF size reduction | 195 KB → 177 KB (−9 %) |

> **Caveat**: The cycle estimate assumes most tensor data IS in L1D (worst
> case for evict overhead). If data is primarily in L2, the `evict` tag
> lookup is faster (~1 cycle per line vs ~5 cycles for a cache hit). The
> board CI is the authoritative score.

## How the kernel works (for future optimisers)

The kernel partitions by **output channel** (OC) across 8 T0 harts using
`yolo_range(N, cidx, ...)`. Each hart computes a contiguous range of output
channels for all spatial positions:

- **Scalar convs** (`conv2d_fp32_mh`): all harts, partition by OC
- **VPU 1×1 convs** (`conv2d_1x1_fp32_mh_vpu_oc8`): T0 harts only, OC tiles
  of 8
- **VPU 3×3 convs** (`conv2d_3x3_p1_fp32_mh_vpu_oc4`): T0 harts only, OC
  tiles of 4 (OC4 silicon workaround for M18 hang)
- **Depthwise convs**: partition by channel
- **PSA attention**: single‑hart (hart 0) using `H0_RUN` macro
- **MaxPool / concat / add**: multi‑hart helpers with per‑hart eviction

Every multi‑hart helper evicts its own output slice and follows with
`MH_BARRIER()`. The barrier uses an atomic spinlock (since
`BENCH_THREAD0_ONLY=1`).

## Other options considered

| Option | Verdict |
|--------|---------|
| `YOLO_USE_16HART` | Incompatible with `BENCH_THREAD0_ONLY` (work‑slicing mismatch). Would need to drop `BENCH_THREAD0_ONLY` and use `shire_barrier`, which is a larger change. |
| VPU weight prefetch | Weights are ~40 KB total and accessed sequentially — little gain. |
| OC16 VPU 1×1 | Marked as "hung silicon (register pressure?)" — not usable. |
| Loop interchange / tiling | Would require major rewrite; existing layout is NHWC which is reasonable for VPU. |