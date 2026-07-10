/*
 * Tensor-unit accelerated 1x1 conv for YOLO inference on ET-SoC1.
 *
 * When YOLO_USE_TENSOR is defined:
 *   - SCP scratchpad is enabled at kernel startup
 *   - conv2d_1x1_fp32_mh_tensor tries tensor_fma for compatible tiles
 *   - Falls through to VPU for irregular dimensions
 *
 * The tensor unit computes: C += A @ B where:
 *   A = weights [OC, IC], loaded into SCP via tenb=1 side
 *   B = activations [IC, HW], loaded into SCP via tenb=0 side
 *   C = output [OC, HW], accumulated in RF then stored to memory
 *
 * Each FMA call processes (a_num_rows+1)*4 OC rows x (b_num_col+1)*16 HW cols.
 * The RF holds the result across one FMA; tensor_store writes it to MRAM.
 *
 * Reference:
 *   erbium/isa/tensors.h  (real API signatures)
 *   et-platform/test-compute-kernels/src/tl_tfma_tstore_fc/
 *   docs/et_soc1_hardware.md
 */
#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#ifdef YOLO_USE_TENSOR

#include <stdint.h>
#include <stdbool.h>
#include "erbium/isa/tensors.h"
#include "erbium/isa/cacheops-umode.h"

/* ------------------------------------------------------------------ */
/* SCP scratchpad init                                                 */
/* ------------------------------------------------------------------ */

/* Enable SCP mode.  Returns 0 on success. */
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
/* Tensor matmul for one tile: C += A @ B                              */
/*                                                                     */
/* Dimensions: OC rows, IC cols in A; IC rows, HW cols in B;           */
/*             C is OC x HW.                                           */
/*                                                                     */
/* All dimensions must be multiples of: OC%4==0, IC%4==0, HW%16==0.   */
/*                                                                     */
/* SCP layout:                                                         */
/*   scp_a .. scp_a + OC*IC/16 - 1  : weights (tenb=1 / tena side)    */
/*   scp_b .. scp_b + IC*HW/16 - 1  : activations (tenb=0 / tenb side)*/
/*                                                                     */
/* first_pass=true  -> RF cleared before FMA (first IC tile)           */
/* first_pass=false -> RF accumulated (subsequent IC tiles)            */
/* ------------------------------------------------------------------ */
static inline int tensor_matmul_tile(
    const float *A,          /* [OC, IC] weights in MRAM */
    const float *B,          /* [IC, HW] activations in MRAM */
    float       *C,          /* [OC, HW] output in MRAM */
    uint32_t OC, uint32_t IC, uint32_t HW,
    uint64_t scp_a,          /* SCP line for A */
    uint64_t scp_b,          /* SCP line for B */
    bool      first_pass)
{
    if (OC == 0 || IC == 0 || HW == 0) return 0;

    /* Hardware dimension encodings: number of chunks minus 1 */
    const uint64_t arows = (uint64_t)(OC / 4u) - 1u;
    const uint64_t acols = (uint64_t)(IC / 4u) - 1u;
    const uint64_t bcols = (uint64_t)(HW / 16u) - 1u;

    const uint64_t a_lines = (uint64_t)OC * (uint64_t)IC / 16u;
    const uint64_t b_lines = (uint64_t)IC * (uint64_t)HW / 16u;

    /* Load A (weights) → tenb=1 (tena side) */
    tensor_load(/*use_tmask=*/0, /*use_coop=*/0,
                scp_a, /*transformation=*/0,
                /*use_tenb=*/1,
                (uint64_t)A, /*offset=*/0,
                a_lines - 1u, /*stride=*/0x40, /*id=*/0);
    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* Load B (activations) → tenb=0 (tenb side) */
    tensor_load(/*use_tmask=*/0, /*use_coop=*/0,
                scp_b, /*transformation=*/0,
                /*use_tenb=*/0,
                (uint64_t)B, /*offset=*/0,
                b_lines - 1u, /*stride=*/0x40, /*id=*/1);
    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* FMA: C += A @ B */
    tensor_fma(/*use_tmask=*/0,
               bcols, arows, acols,
               /*offset=*/0,
               /*tenc_loc=*/0, /*tenb_unsigned=*/0, /*tena_unsigned=*/0,
               /*tenb_loc=*/1,     /* B is on tenb=0 side */
               scp_b,              /* SCP line of B */
               scp_a,              /* SCP line of A */
               /*opcode=*/0,
               first_pass);
    tensor_wait(TENSOR_FMA_WAIT);

    /* Store result from RF to MRAM */
    tensor_store(/*reg_stride=*/0, /*start_reg=*/0,
                 bcols, arows,
                 (uint64_t)C,
                 /*coop_store=*/0, /*stride=*/0x40);
    tensor_wait(TENSOR_STORE_WAIT);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Tensor-accelerated 1x1 conv                                         */
/*                                                                     */
/* Tiles the [OC, IC, HW] conv into TILE_OC x TILE_IC x TILE_HW       */
/* blocks that fit in SCP.  Dimensions must be multiples of 4/16.      */
/* For tiles that don't align, falls through to VPU dispatcher.        */
/* ------------------------------------------------------------------ */

/* SCP capacity shared between A and B sides: we reserve
 * 32 lines (512 floats) for weights, 224 lines for activations.
 * This allows tiles up to OC=32, IC=32, HW=64. */
#define TENSOR_SCP_A_LINES  32u   /* 512 floats for weights */
#define TENSOR_SCP_LINES    256u  /* total SCP lines available */

/* Tile sizes: balance register pressure vs SCP capacity.
 * Result registers needed = (OC/4) * (HW/16) * 4 <= ~64. */
#define TENSOR_TILE_OC  16u
#define TENSOR_TILE_IC  16u
#define TENSOR_TILE_HW  16u

/* Returns true if this conv has compatible dimensions. */
static inline bool tensor_can_handle(uint32_t IC, uint32_t OC,
                                     uint32_t H, uint32_t W_)
{
    if (!mh_is_t0(get_hart_id())) return false;
    /* Must be multiples of tile sizes */
    if (IC % TENSOR_TILE_IC != 0u) return false;
    if (OC % TENSOR_TILE_OC != 0u) return false;
    if ((H * W_) % TENSOR_TILE_HW != 0u) return false;
    /* Must fit in SCP */
    uint32_t HW = H * W_;
    uint32_t a_lines = TENSOR_TILE_OC * TENSOR_TILE_IC / 16u;
    uint32_t b_lines = TENSOR_TILE_IC * TENSOR_TILE_HW / 16u;
    if (a_lines > TENSOR_SCP_A_LINES) return false;
    if (b_lines > TENSOR_SCP_LINES - TENSOR_SCP_A_LINES) return false;
    return true;
}

static inline void conv2d_1x1_fp32_mh_tensor(uint32_t hid,
                                             const float *in, float *out,
                                             const float *W, const float *B,
                                             uint32_t IC, uint32_t H, uint32_t W_,
                                             uint32_t OC,
                                             uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    if (!tensor_can_handle(IC, OC, H, W_)) {
        /* Fall through to VPU */
        conv2d_1x1_disp(hid, in, out, W, B, IC, H, W_, OC, act);
        return;
    }

    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t HW = H * W_;
    const uint32_t OC_tiles = OC / TENSOR_TILE_OC;
    const uint32_t IC_tiles = IC / TENSOR_TILE_IC;
    const uint32_t HW_tiles = HW / TENSOR_TILE_HW;

    /* Partition OC tiles across harts */
    uint32_t t_lo, t_hi;
    mh_range(OC_tiles, cidx, &t_lo, &t_hi);

    for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
        const uint32_t oc0 = t_oc * TENSOR_TILE_OC;

        for (uint32_t t_ic = 0; t_ic < IC_tiles; t_ic++) {
            const uint32_t ic0 = t_ic * TENSOR_TILE_IC;
            const float *wtile = W + oc0 * IC + ic0;

            /* Load weights once per (oc,ic) tile — reused across HW */
            /* Weights stay at SCP_A lines 0..a_lines-1 */
            const uint64_t a_lines = TENSOR_TILE_OC * TENSOR_TILE_IC / 16u;
            tensor_load(0, 0, 0, 0, 1, (uint64_t)wtile, 0, a_lines - 1u, 0x40, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);

            for (uint32_t t_hw = 0; t_hw < HW_tiles; t_hw++) {
                const uint32_t hw0 = t_hw * TENSOR_TILE_HW;
                const float *atile = in + ic0 * HW + hw0;
                float *ctile = out + oc0 * HW + hw0;
                const bool first_pass = (t_ic == 0u);

                /* Load activation tile at SCP_B lines */
                const uint64_t b_lines = TENSOR_TILE_IC * TENSOR_TILE_HW / 16u;
                const uint64_t scp_b = TENSOR_SCP_A_LINES;
                tensor_load(0, 0, scp_b, 0, 0, (uint64_t)atile, 0, b_lines - 1u, 0x40, 1);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                /* FMA */
                const uint64_t arows = TENSOR_TILE_OC / 4u - 1u;
                const uint64_t acols = TENSOR_TILE_IC / 4u - 1u;
                const uint64_t bcols = TENSOR_TILE_HW / 16u - 1u;

                tensor_fma(0, bcols, arows, acols, 0,
                           0, 0, 0, 1, scp_b, 0, 0, first_pass);
                tensor_wait(TENSOR_FMA_WAIT);

                /* Store */
                tensor_store(0, 0, bcols, arows, (uint64_t)ctile, 0, 0x40);
                tensor_wait(TENSOR_STORE_WAIT);
            }
        }

        /* Apply bias + activation (scalar, once per OC tile) */
        if (B || act) {
            for (uint32_t oc = oc0; oc < oc0 + TENSOR_TILE_OC; oc++) {
                for (uint32_t hw = 0; hw < HW; hw++) {
                    float v = out[oc * HW + hw];
                    if (B) v += B[oc];
                    if (act == 1u) v = silu(v);
                    out[oc * HW + hw] = v;
                }
            }
        }
    }

    /* Evict our OC slice */
    uint32_t oc_lo = t_lo * TENSOR_TILE_OC;
    uint32_t oc_hi = t_hi * TENSOR_TILE_OC;
    if (oc_hi > oc_lo) {
        const uint32_t bytes = (oc_hi - oc_lo) * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

/* Override CONV_1x1 to route through tensor dispatcher */
#undef CONV_1x1
#define CONV_1x1(...) do { \
    conv2d_1x1_fp32_mh_tensor(hid, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)

#endif /* YOLO_USE_TENSOR */
#endif /* YOLO_TENSOR_H */
