/*
 * Tensor-unit accelerated helpers for YOLO inference on ET-SoC1.
 *
 * Provides tile-based matrix multiply (tensor_fma) for 1×1 convolutions,
 * hardware convolution acceleration for 3×3 convolutions, and cross-hart
 * reductions for softmax/NMS.
 *
 * This is an OPTIONAL overlay. When YOLO_USE_TENSOR is defined, the
 * conv1x1 and conv3x3 operations can be routed through the tensor unit
 * instead of the VPU. The tile sizes are chosen to fit within the SCP
 * scratchpad (part of L1D reconfigured in split mode).
 *
 * Reference: docs/et_soc1_hardware.md
 *            <erbium/isa/tensors.h>
 *            et-platform/test-compute-kernels/src/tl_tfma_tstore_fc/
 */
#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#include <stdint.h>
#include <stdbool.h>

#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

/* ------------------------------------------------------------------ */
/* SCP / L1D configuration                                            */
/* ------------------------------------------------------------------ */

/* Enable SCP mode: reconfigure L1D as split + scratchpad.
 * On erbium-soc1sim this requires an M-mode syscall.
 * Returns 0 on success, non-zero on failure. */
static inline int tensor_scp_enable(void)
{
    /* Check if already in SCP mode */
    if (get_l1d_mode() == l1d_scp)
        return 0;

    /* Request split + SCP via syscall */
    int64_t ret = set_l1_cache_control(/*d1_split=*/1, /*scp_en=*/1);
    if (ret != 0)
        return (int)ret;

    /* U-mode cache control */
    ucache_control(/*scp_en=*/1, /*cacheop_rate=*/0, /*cacheop_max=*/0);

    return (get_l1d_mode() == l1d_scp) ? 0 : -1;
}

/* Restore L1D to shared mode. */
static inline int tensor_scp_disable(void)
{
    return (int)set_l1_cache_control(/*d1_split=*/0, /*scp_en=*/0);
}

/* ------------------------------------------------------------------ */
/* SCP tile dimensions                                                */
/* ------------------------------------------------------------------ */

/*
 * The SCP scratchpad is a portion of L1D. On erbium, L1D is ~32-64 KB
 * per minion. In split mode, roughly half is available as SCP.
 * We use conservative tile sizes:
 *
 *   TILE_OC  = 16  — output channels per tile
 *   TILE_IC  = 32  — input channels per tile
 *   TILE_HW  = 64  — spatial positions per tile (for large feature maps)
 *
 * Each FP32 value is 4 bytes. A tile of weights: TILE_OC * TILE_IC * 4.
 * A tile of activations: TILE_IC * TILE_HW * 4.
 * Total SCP needed: ~ (16*32 + 32*64) * 4 = 10 KB — fits comfortably.
 */

#define TENSOR_TILE_OC  16u
#define TENSOR_TILE_IC  32u
#define TENSOR_TILE_HW  64u

/* ------------------------------------------------------------------ */
/* Tensor matmul: C[OC, HW] += W[OC, IC] @ X[IC, HW]                 */
/*                                                                     */
/* This processes one tile where:                                      */
/*   OC <= TENSOR_TILE_OC                                              */
/*   IC <= TENSOR_TILE_IC                                              */
/*   HW == HW_tile (full spatial extent or a subset)                   */
/*                                                                     */
/* Weights (A) are loaded into SCP lines 0..(OC*IC/16 - 1).            */
/* Activations (B) are loaded into SCP lines 32..(32 + IC*HW/16 - 1).  */
/* The FMA accumulates into RF registers, then stored to memory.       */
/* ------------------------------------------------------------------ */

static inline void tensor_matmul_tile(
    const float *W,          /* [OC, IC] weights */
    const float *X,          /* [IC, HW] activations */
    float       *C,          /* [OC, HW] output (accumulated) */
    uint32_t OC,
    uint32_t IC,
    uint32_t HW,
    bool      clear_rf)      /* true = first tile (zero RF), false = accumulate */
{
    if (OC == 0 || IC == 0 || HW == 0) return;

    /*
     * tensor_fma parameters:
     *   b_num_col = (HW / 16) - 1  — HW in units of 16, minus 1
     *   a_num_rows = OC / 4 - 1    — OC in units of 4, minus 1
     *   a_num_cols = IC / 4 - 1    — IC in units of 4, minus 1
     *
     * These are hardware tile dimensions. For simplicity we require
     * OC%4==0, IC%4==0, HW%16==0.
     */
    const uint64_t bcols  = (HW  / 16u) - 1u;
    const uint64_t arows  = (OC  / 4u)  - 1u;
    const uint64_t acols  = (IC  / 4u)  - 1u;

    /* Load weights into SCP lines 0..(OC*IC/64 - 1) (each line = 64 bytes = 16 floats) */
    /* First load: tenb=0 (Tensilica B side) */
    tensor_load(
        /*use_tmask=*/0, /*use_coop=*/0, /*dst_start=*/0,
        /*transformation=*/0, /*use_tenb=*/0,
        (uint64_t)W,
        /*offset=*/0,
        /*num_lines=*/(OC * IC) / 16u - 1u,
        /*stride=*/0x40,
        /*id=*/0);

    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* Load activations into SCP lines 32..(32 + IC*HW/16 - 1), tenb=1 (A side) */
    tensor_load(
        /*use_tmask=*/0, /*use_coop=*/0, /*dst_start=*/32,
        /*transformation=*/0, /*use_tenb=*/1,
        (uint64_t)X,
        /*offset=*/0,
        /*num_lines=*/(IC * HW) / 16u - 1u,
        /*stride=*/0x40,
        /*id=*/1);

    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* FMA: C += W @ X */
    tensor_fma(
        /*use_tmask=*/0,
        bcols,
        arows,
        acols,
        /*offset=*/0,
        /*tenc_loc=*/0,
        /*tenb_unsigned=*/0,
        /*tena_unsigned=*/0,
        /*tenb_loc=*/0,
        /*scp_loc_b=*/0,        /* B starts at SCP line 0 */
        /*scp_loc_a=*/32,        /* A starts at SCP line 32 */
        /*opcode=*/0,
        /*first_pass=*/clear_rf);

    tensor_wait(TENSOR_FMA_WAIT);

    /* Store RF to memory. Each result register holds 16 floats = 1 cache line.
     * Total lines = (OC * HW) / 16. We store from RF register 0. */
    tensor_store(
        /*reg_stride=*/0,
        /*start_reg=*/0,
        /*cols=*/(HW / 16u) - 1u,
        /*Arows=*/(OC / 4u) - 1u,
        (uint64_t)C,
        /*coop_store=*/0,
        /*stride=*/0x40);

    tensor_wait(TENSOR_STORE_WAIT);
}

/* ------------------------------------------------------------------ */
/* Full 1×1 convolution via tensor tiling                              */
/*                                                                     */
/* Splits a large 1×1 conv [OC, IC, H, W] into tiles that fit SCP.    */
/* ------------------------------------------------------------------ */

static inline void tensor_conv1x1(
    const float *in,  float *out,
    const float *W,   const float *B,
    uint32_t IC, uint32_t OC,
    uint32_t H,  uint32_t W_,
    bool      use_silu)
{
    const uint32_t HW = H * W_;

    for (uint32_t oc0 = 0; oc0 < OC; oc0 += TENSOR_TILE_OC) {
        const uint32_t oc_n = (OC - oc0 < TENSOR_TILE_OC) ? (OC - oc0) : TENSOR_TILE_OC;

        /* Clear output tile (or apply bias) */
        if (B) {
            for (uint32_t oc = 0; oc < oc_n; oc++) {
                const float bias = B[oc0 + oc];
                for (uint32_t s = 0; s < HW; s++)
                    out[(oc0 + oc) * HW + s] = bias;
            }
        }

        for (uint32_t ic0 = 0; ic0 < IC; ic0 += TENSOR_TILE_IC) {
            const uint32_t ic_n = (IC - ic0 < TENSOR_TILE_IC) ? (IC - ic0) : TENSOR_TILE_IC;

            for (uint32_t hw0 = 0; hw0 < HW; hw0 += TENSOR_TILE_HW) {
                const uint32_t hw_n = (HW - hw0 < TENSOR_TILE_HW) ? (HW - hw0) : TENSOR_TILE_HW;

                /* Pad to 4/16 alignment for tensor unit */
                uint32_t oc_pad = (oc_n + 3u) & ~3u;
                uint32_t ic_pad = (ic_n + 3u) & ~3u;
                uint32_t hw_pad = (hw_n + 15u) & ~15u;

                /* We need padded copies for the tensor unit alignment.
                 * In practice, YOLO dimensions are already multiples of 4/16:
                 *   OC ∈ {16, 32, 64, 80, 128, 256}
                 *   IC ∈ {16, 32, 64, 128, 256}
                 *   HW ∈ {144, 576, 2304}
                 * So oc_pad == oc_n, ic_pad == ic_n, hw_pad == hw_n.
                 * The tensor_matmul_tile function handles the rest. */
                if ((oc_n & 3u) == 0u && (ic_n & 3u) == 0u && (hw_n & 15u) == 0u) {
                    const float *wtile = W + (oc0 * IC + ic0);
                    const float *xtile = in + (ic0 * HW + hw0);
                    float       *ctile = out + (oc0 * HW + hw0);
                    tensor_matmul_tile(wtile, xtile, ctile,
                                       oc_n, ic_n, hw_n,
                                       /*clear_rf=*/(ic0 == 0u));
                } else {
                    /* Fallback to scalar for irregular tiles (unlikely in YOLO) */
                    for (uint32_t oc = 0; oc < oc_n; oc++) {
                        for (uint32_t s = 0; s < hw_n; s++) {
                            float acc = out[(oc0 + oc) * HW + hw0 + s];
                            for (uint32_t ic = 0; ic < ic_n; ic++) {
                                acc += W[(oc0 + oc) * IC + (ic0 + ic)]
                                     * in[(ic0 + ic) * HW + hw0 + s];
                            }
                            if (use_silu) acc = (acc > 8.0f) ? acc :
                                                (acc < -8.0f) ? 0.0f :
                                                acc * fast_recip(1.0f + my_expf(-acc));
                            out[(oc0 + oc) * HW + hw0 + s] = acc;
                        }
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* 3×3 convolution via hardware convolution accelerator                */
/* ------------------------------------------------------------------ */

static inline void tensor_conv3x3(
    const float *in,  float *out,
    const float *W,   const float *B,
    uint32_t IC, uint32_t OC,
    uint32_t IH, uint32_t IW,
    uint32_t OH, uint32_t OW,
    uint32_t stride,
    uint32_t pad,
    bool      use_silu)
{
    /* TODO: Use convolution_ctrl + convolution_size + tensor_load/FMA
     * to accelerate 3×3 convs. The convolution accelerator controls
     * the sliding-window addressing; data still flows through SCP.
     *
     * For now this is a placeholder — the VPU implementation is faster
     * than a naive tile-based matmul for 3×3.
     *
     * Reference: et-platform/test-compute-kernels/src/mlp/ for
     * convolution accelerator usage patterns.
     */
    (void)IH; (void)IW; (void)stride; (void)pad;
    /* Fall through to VPU implementation at the call site */
}

/* ------------------------------------------------------------------ */
/* Cross-hart softmax via tensor_reduce                                */
/* ------------------------------------------------------------------ */

/* Softmax over rows using cross-hart FMAX + FADD reduction.
 * Each hart owns a subset of rows. The tree depth is 3 (8 minions). */
static inline void tensor_softmax_rows(float *x, uint32_t M, uint32_t N,
                                       uint32_t hid)
{
    if (!mh_is_t0(hid)) return;

    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t row_lo, row_hi;
    mh_range(M, cidx, &row_lo, &row_hi);

    for (uint32_t i = row_lo; i < row_hi; i++) {
        float *row = x + i * N;
        /* Local max */
        float local_max = row[0];
        for (uint32_t j = 1; j < N; j++)
            if (row[j] > local_max) local_max = row[j];

        /* Cross-hart max via tensor_reduce_auto */
        float global_max = tensor_reduce_float(local_max, TENSOR_REDUCE_OP_FMAX,
                                               1, 0, 3);  /* auto tree, depth 3 */

        /* Local sum of exp(x - global_max) */
        float local_sum = 0.0f;
        for (uint32_t j = 0; j < N; j++) {
            row[j] = my_expf(row[j] - global_max);
            local_sum += row[j];
        }

        /* Cross-hart sum via tensor_reduce_auto */
        float global_sum = tensor_reduce_float(local_sum, TENSOR_REDUCE_OP_FADD,
                                               1, 0, 3);

        /* Normalize */
        const float inv = fast_recip(global_sum);
        for (uint32_t j = 0; j < N; j++)
            row[j] *= inv;
    }
}

#endif /* YOLO_TENSOR_H */
