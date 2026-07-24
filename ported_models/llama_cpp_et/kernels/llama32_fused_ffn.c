/*
 * Fused FFN kernel for Llama 3.2 1B on ET-SoC1.
 *
 * Combines gate_proj, up_proj, SiLU, elementwise multiply, and down_proj
 * into one kernel launch. Keeps intermediate results in L2 scratchpad
 * instead of writing/reading them from DRAM.
 *
 * For M=1 (token generation), saves ~5 DRAM round-trips per FFN layer.
 *
 * Multi-hart (all 2048 harts across 32 shires):
 *   Phase 1 — gate+up   : each hart computes 4 rows (8192/2048)
 *   Phase 2 — SiLU×mul   : each hart applies SiLU(gate) × up (own 4 rows)
 *   Phase 3 — down_proj  : each hart computes 1 row  (2048/2048)
 *
 * Memory model:
 *   Phase 1→2: intra-hart (each hart reads its own writes) — FENCE only
 *   Phase 2→3: cross-shire (all harts read all gated data) — evict past L2
 *              + ET_BARRIER_GLOBAL
 *
 * Entry: _start → entry_point(params, env)
 */

#include <stdint.h>
#include <string.h>

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

/* ----------------------------------------------------------------- */
/* Fused FFN parameters — host writes before launch                  */
/* ----------------------------------------------------------------- */
struct ffn_params {
    const float     * input;       /* [hidden] f32 */
    const block_q8_0 * W_gate;     /* [intermediate, hidden] Q8_0 */
    const block_q8_0 * W_up;       /* [intermediate, hidden] Q8_0 */
    const block_q8_0 * W_down;     /* [hidden, intermediate] Q8_0 */
    float           * output;      /* [hidden] f32 */
    float           * scratch;     /* [intermediate × 2] f32 — gate + up, then gated */
    int64_t           hidden;      /* H  — 2048 */
    int64_t           inter;       /* I  — 8192 */
    size_t            stride_gate; /* bytes per row of W_gate */
    size_t            stride_up;   /* bytes per row of W_up */
    size_t            stride_down; /* bytes per row of W_down */
};

#define TOTAL_HARTS 2048

/* ----------------------------------------------------------------- */
/* Scalar SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))         */
/* Uses ET VPU helpers from math_fp.h.                               */
/* ----------------------------------------------------------------- */
static inline float silu_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float exp_neg_x   = et_expf(-x);
    float denominator = 1.0f + exp_neg_x;
    return et_fdiv(x, denominator);
}

/* ----------------------------------------------------------------- */
/* Entry point                                                        */
/* ----------------------------------------------------------------- */
int entry_point(struct ffn_params * params, void * env) {
    (void)env;

    uint64_t hart_id = get_hart_id();
    int64_t  H = params->hidden;
    int64_t  I = params->inter;

    const float     * input   = params->input;
    const block_q8_0 * W_gate = params->W_gate;
    const block_q8_0 * W_up   = params->W_up;
    const block_q8_0 * W_down = params->W_down;
    float           * output  = params->output;
    float           * scratch = params->scratch;

    size_t stride_gate = params->stride_gate;
    size_t stride_up   = params->stride_up;
    size_t stride_down = params->stride_down;

    int64_t H_blocks = H >> 5;    /* 64 */
    int64_t I_blocks = I >> 5;    /* 256 */

    /* ----------------------------------------------------------------- */
    /* Phase 1: gate[i] and up[i] for our shard of [0..I)                */
    /*                                                                   */
    /* Each hart writes 4 consecutive elements to scratch[0..I) and      */
    /* scratch[I..2I). Own shard only — no cross-hart reads yet.         */
    /* ----------------------------------------------------------------- */
    int64_t rows_per = (I + TOTAL_HARTS - 1) / TOTAL_HARTS;  /* 4 */
    int64_t r0 = hart_id * rows_per;
    int64_t r1 = r0 + rows_per;
    if (r1 > I) r1 = I;

    if (r0 < I) {
        for (int64_t i = r0; i < r1; i++) {
            /* compute_row_dot_q8_0 handles mask save/restore internally */
            scratch[i]     = compute_row_dot_q8_0(
                (const block_q8_0 *)((const char *)W_gate + i * stride_gate),
                input, H_blocks);
            scratch[I + i] = compute_row_dot_q8_0(
                (const block_q8_0 *)((const char *)W_up   + i * stride_up),
                input, H_blocks);
        }
        /* Flush to L2 so other harts in same shire could read (if needed) */
        FENCE;
        flush_to_l2(&scratch[r0], 1, 64);
        flush_to_l2(&scratch[I + r0], 1, 64);
        WAIT_CACHEOPS;
    }

    /* Phase 1→2 barrier: each hart reads its own data, so FENCE suffices.
     * But keep a shire barrier to mirror existing patterns and provide
     * a clean sequencing point. */
    FENCE;
    et_barrier(ET_BARRIER_SHIRE);

    /* ----------------------------------------------------------------- */
    /* Phase 2: gated[i] = SiLU(gate[i]) × up[i]                        */
    /*                                                                   */
    /* Reads from scratch[i] (gate) and scratch[I+i] (up), both written  */
    /* by this hart in phase 1.  Writes result to scratch[i] (overwrite).*/
    /* No cross-hart dependency.                                         */
    /* ----------------------------------------------------------------- */
    if (r0 < I) {
        for (int64_t i = r0; i < r1; i++) {
            scratch[i] = silu_f32(scratch[i]) * scratch[I + i];
        }
        /* Evict past L2 to DRAM — phase 3 reads from ALL harts */
        FENCE;
        evict_region_past_l2(&scratch[r0], (size_t)(r1 - r0) * sizeof(float));
        WAIT_CACHEOPS;
    }

    /* Global barrier: all 2048 harts across all 32 shires.
     * Required because phase 3 reads gated data written by harts in other
     * shires.  Writes were evicted past L2 to DRAM above, so the barrier
     * ensures all evictions complete before anyone reads. */
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);

    /* ----------------------------------------------------------------- */
    /* Phase 3: output[j] = W_down[j] · gated                            */
    /*                                                                   */
    /* Each hart reads the FULL gated array scratch[0..I) to compute     */
    /* its assigned output row(s).  The gated array was fully assembled  */
    /* in phase 2 and made visible by the eviction + global barrier.     */
    /* ----------------------------------------------------------------- */
    rows_per = (H + TOTAL_HARTS - 1) / TOTAL_HARTS;  /* 1 */
    r0 = hart_id * rows_per;
    r1 = r0 + rows_per;
    if (r1 > H) r1 = H;

    if (r0 < H) {
        for (int64_t j = r0; j < r1; j++) {
            output[j] = compute_row_dot_q8_0(
                (const block_q8_0 *)((const char *)W_down + j * stride_down),
                scratch, I_blocks);
        }
        evict_region_past_l2(&output[r0], (size_t)(r1 - r0) * sizeof(float));
    }

    return 0;
}
