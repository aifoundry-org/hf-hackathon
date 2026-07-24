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
#include "math_fp.h"
#include "platform.h"

/* ----------------------------------------------------------------- */
/* Fused FFN parameters — host writes into shared buffer before launch */
/* ----------------------------------------------------------------- */
struct ffn_params {
    uint64_t input_off;      /* offset of input [hidden] f32 in shared buffer */
    uint64_t wgate_off;      /* offset of W_gate [intermediate, hidden] Q8_0 */
    uint64_t wup_off;        /* offset of W_up   [intermediate, hidden] Q8_0 */
    uint64_t wdown_off;      /* offset of W_down [hidden, intermediate] Q8_0 */
    uint64_t output_off;     /* offset of output [hidden] f32 */
    uint64_t scratch_off;    /* offset of scratch [intermediate * 3] f32 (gate + up + gated) */
    int64_t  hidden;         /* H — 2048 */
    int64_t  inter;          /* I — 8192 */
    int64_t  h_blocks;       /* hidden / 32 */
    int64_t  i_blocks;       /* intermediate / 32 */
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
int main(uintptr_t arg_area) {
    uint64_t hart_id = get_hart_id();

    /* T0-only: only even harts use VPU. Odd harts idle and skip to barrier. */
    if ((hart_id & 1u) != 0u) {
        /* Still need to participate in global barriers for phases 2->3 */
        goto phase2_barrier;
    }

    /* Resolve buffer base from arg_area */
    uint8_t *base;
    if (arg_area == 0 || arg_area == ~(uintptr_t)0) {
        extern char heap0_end[];
        base = (uint8_t *)heap0_end - 80u * 1024u * 1024u;
    } else {
        uintptr_t ptr = *(volatile uintptr_t *)arg_area;
        base = (uint8_t *)((ptr == 0 || ptr == ~(uintptr_t)0)
                           ? (uintptr_t)heap0_end - 80u * 1024u * 1024u : ptr);
    }

    /* Enable L1 scratchpad for tensor/VPU operations */
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    /* Read params from arg_area (host writes them before launch) */
    struct ffn_params params;
    __builtin_memcpy(&params, (const void *)arg_area, sizeof(params));

    int64_t  H = params.hidden;
    int64_t  I = params.inter;
    int64_t  H_blocks = params.h_blocks;  /* H / 32 */
    int64_t  I_blocks = params.i_blocks;  /* I / 32 */

    const float     * input   = (const float     *)(base + params.input_off);
    const block_q8_0 * W_gate = (const block_q8_0 *)(base + params.wgate_off);
    const block_q8_0 * W_up   = (const block_q8_0 *)(base + params.wup_off);
    const block_q8_0 * W_down = (const block_q8_0 *)(base + params.wdown_off);
    float           * output  = (float           *)(base + params.output_off);
    float           * scratch = (float           *)(base + params.scratch_off);

    /* ----------------------------------------------------------------- */
    /* Phase 1: gate[i] and up[i] for our shard of [0..I)                */
    /*                                                                   */
    /* Each T0 hart handles 8 consecutive rows of gate+up (I=8192,       */
    /* 1024 active T0 harts = 8 rows each). Writes to scratch[0..I)     */
    /* and scratch[I..2I). Own shard only — no cross-hart reads yet.     */
    /* ----------------------------------------------------------------- */
    /* 1024 T0 harts out of 2048 total */
    uint64_t tid = hart_id >> 1;
    uint64_t active_t0 = TOTAL_HARTS >> 1;  /* 1024 */

    int64_t rows_per = (I + (int64_t)active_t0 - 1) / (int64_t)active_t0;  /* 8 */
    int64_t r0 = (int64_t)tid * rows_per;
    int64_t r1 = r0 + rows_per;
    if (r1 > I) r1 = I;

    if (r0 < I) {
        q8_dot_state st;
        q8_dot_begin(&st);

        for (int64_t i = r0; i < r1; i++) {
            q8_dot_reset();
            q8_dot_tile(W_gate + i * (H_blocks), input, H_blocks);
            scratch[i]     = q8_dot_reduce();

            q8_dot_reset();
            q8_dot_tile(W_up   + i * (H_blocks), input, H_blocks);
            scratch[I + i] = q8_dot_reduce();
        }

        q8_dot_end(&st);

        /* Flush to L2 */
        FENCE;
        flush_to_l2(&scratch[r0], 1, 64);
        flush_to_l2(&scratch[I + r0], 1, 64);
        WAIT_CACHEOPS;
    }

    /* Phase 1→2 barrier: each hart reads its own data — FENCE suffices.
     * But keep a shire barrier as a clean sequencing point. */
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

phase2_barrier:
    /* Global barrier: all 2048 harts across all 32 shires.
     * Required because phase 3 reads gated data written by harts in other
     * shires.  Writes were evicted past L2 to DRAM above, so the barrier
     * ensures all evictions complete before anyone reads. */
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);

    /* Odd harts (non-T0) can now leave — they never used VPU */
    if ((hart_id & 1u) != 0u) return 0;

    /* ----------------------------------------------------------------- */
    /* Phase 3: output[j] = W_down[j] · gated                            */
    /*                                                                   */
    /* Each T0 hart reads the FULL gated array scratch[0..I) to compute  */
    /* its assigned output row(s).                                      */
    /* ----------------------------------------------------------------- */
    rows_per = (H + (int64_t)active_t0 - 1) / (int64_t)active_t0;  /* 2 */
    r0 = (int64_t)tid * rows_per;
    r1 = r0 + rows_per;
    if (r1 > H) r1 = H;

    if (r0 < H) {
        q8_dot_state st;
        q8_dot_begin(&st);

        for (int64_t j = r0; j < r1; j++) {
            q8_dot_reset();
            q8_dot_tile(W_down + j * (I_blocks), scratch, I_blocks);
            output[j] = q8_dot_reduce();
        }

        q8_dot_end(&st);

        FENCE;
        evict_region_past_l2(&output[r0], (size_t)(r1 - r0) * sizeof(float));
        WAIT_CACHEOPS;
    }

    return 0;
}
