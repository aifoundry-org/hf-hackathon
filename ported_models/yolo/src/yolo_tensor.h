/*
 * Tensor-unit accelerated 1x1 conv for YOLO inference on ET-SoC1.
 *
 * Tries tensor_fma for compatible 1x1 convs; falls through to VPU
 * when dimensions don't align or SCP is unavailable.
 *
 * Reference: erbium/isa/tensors.h, et-platform/test-compute-kernels/
 */
#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "erbium/isa/tensors.h"
#include "erbium/isa/cacheops.h"

/* ------------------------------------------------------------------ */
/* SCP init — call once at startup                                     */
/* ------------------------------------------------------------------ */
static inline int tensor_scp_enable(void)
{
    if (get_l1d_mode() == l1d_scp)
        return 0;
    int64_t ret = set_l1_cache_control(/*d1_split=*/1, /*scp_en=*/1);
    if (ret != 0) return (int)ret;
    ucache_control(/*scp_en=*/1, /*cacheop_rate=*/0, /*cacheop_max=*/0);
    return (get_l1d_mode() == l1d_scp) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Tile sizes — must fit in SCP (256 lines available).                 */
/* 16-OC x 16-IC weights = 16 lines; 16-IC x 16-HW acts = 16 lines.   */
/* Result registers: (16/4)*(16/16)*4 = 4 registers total.             */
/* ------------------------------------------------------------------ */
#define T_TILE_OC  16u
#define T_TILE_IC  16u
#define T_TILE_HW  16u
#define T_SCP_A    0u
#define T_SCP_B    32u

static inline bool tensor_can_handle(uint32_t IC, uint32_t OC,
                                     uint32_t H, uint32_t W_)
{
    /* SCP must be enabled (checked at startup) */
    if (get_l1d_mode() != l1d_scp) return false;
    /* Dimensions must tile evenly */
    if (IC % T_TILE_IC != 0u) return false;
    if (OC % T_TILE_OC != 0u) return false;
    if ((H * W_) % T_TILE_HW != 0u) return false;
    return true;
}

static inline void conv2d_1x1_fp32_mh_tensor(uint32_t hid,
                                             const float *in, float *out,
                                             const float *W, const float *B,
                                             uint32_t IC, uint32_t H, uint32_t W_,
                                             uint32_t OC,
                                             uint32_t act)
{
    if (!mh_is_t0(hid) || !tensor_can_handle(IC, OC, H, W_)) {
        conv2d_1x1_disp(hid, in, out, W, B, IC, H, W_, OC, act);
        return;
    }

    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t HW = H * W_;
    const uint32_t OC_tiles = OC / T_TILE_OC;
    const uint32_t IC_tiles = IC / T_TILE_IC;
    const uint32_t HW_tiles = HW / T_TILE_HW;

    uint32_t t_lo, t_hi;
    mh_range(OC_tiles, cidx, &t_lo, &t_hi);

    for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
        const uint32_t oc0 = t_oc * T_TILE_OC;

        for (uint32_t t_hw = 0; t_hw < HW_tiles; t_hw++) {
            const uint32_t hw0 = t_hw * T_TILE_HW;

            for (uint32_t t_ic = 0; t_ic < IC_tiles; t_ic++) {
                const uint32_t ic0 = t_ic * T_TILE_IC;
                const float *wtile = W + oc0 * IC + ic0;
                const float *atile = in + ic0 * HW + hw0;
                const bool first = (t_ic == 0u);

                /* Load weights (tenA) and activations (tenB) to SCP */
                tensor_load(0, 0, T_SCP_A, 0, 0, (uint64_t)wtile, 0,
                            T_TILE_OC * T_TILE_IC / 16u - 1u, 0x40, 0);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_load(0, 0, T_SCP_B, 0, 1, (uint64_t)atile, 0,
                            T_TILE_IC * T_TILE_HW / 16u - 1u, 0x40, 1);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                /* C[OC_tile, HW_tile] += A[OC_tile, IC_tile] @ B[IC_tile, HW_tile]
                 * first_pass clears the accumulator (t_ic==0), subsequent calls add. */
                tensor_fma(0, T_TILE_HW/16u-1, T_TILE_OC/4u-1, T_TILE_IC/4u-1,
                           0, 0, 0, 0, 1, T_SCP_B, T_SCP_A, 0, first);
                tensor_wait(TENSOR_FMA_WAIT);
            }

            /* Store accumulated C[OC_tile, HW_tile] for this HW tile */
            float *ctile = out + oc0 * HW + hw0;
            tensor_store(0, 0, T_TILE_HW/4u - 1u, T_TILE_OC - 1u,
                         (uint64_t)ctile, 0, (uint64_t)(HW) * sizeof(float));
            tensor_wait(TENSOR_STORE_WAIT);
        }

        /* Bias + activation per OC */
        if (B || act) {
            for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                const float bias = B ? B[oc] : 0.0f;
                for (uint32_t hw = 0; hw < HW; hw++) {
                    float v = out[oc * HW + hw] + bias;
                    if (act == 1u) v = silu(v);
                    out[oc * HW + hw] = v;
                }
            }
        }
    }

    uint32_t oc_lo = t_lo * T_TILE_OC;
    uint32_t oc_hi = t_hi * T_TILE_OC;
    if (oc_hi > oc_lo)
        evict((const void *)(out + oc_lo * H * W_), (oc_hi - oc_lo) * H * W_ * sizeof(float));
}

/* Override CONV_1x1 to route through tensor dispatcher */
#undef CONV_1x1
#define CONV_1x1(...) do { \
    conv2d_1x1_fp32_mh_tensor(hid, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)

#endif /* YOLO_TENSOR_H */
