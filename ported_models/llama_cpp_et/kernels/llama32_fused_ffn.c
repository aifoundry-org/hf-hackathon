/*
 * Fused FFN kernel for Llama 3.2 1B on ET-SoC1.
 *
 * Combines gate_proj, up_proj, SiLU, elementwise multiply, and down_proj
 * into one kernel launch. Keeps intermediate results in scratchpad instead
 * of writing/reading them from DRAM between separate kernel calls.
 *
 * Self-contained: all needed helpers (Q8_0 dot, barrier, evict) are inlined.
 * No dependency on repo-internal block_ops.h or platform.h additions.
 *
 * Entry: main(uintptr_t arg_area)
 *   arg_area → pointer to uint8_t *buffer_base (written by host launcher)
 *
 * Multi-hart: all 2048 harts, but only T0 (even) harts do VPU compute.
 *   Phase 1: gate+up — each T0 hart handles inter/T0_harts rows
 *   Phase 2: SiLU(gate)*up — same shard, no cross-hart reads
 *   Phase 3: down — each T0 hart handles hidden/T0_harts rows
 *
 * Global barrier (phase 2→3): implemented via shared atomic counter.
 */

#include <stdint.h>
#include <string.h>

/* ET SDK headers (from toolchain install at /opt/et) */
#include "etsoc/isa/hart.h"
#include "etsoc/isa/barriers.h"
#include "etsoc/isa/cacheops-umode.h"
#include "etsoc/isa/atomic.h"
#include "etsoc/isa/fcc.h"
#include "etsoc/isa/flb.h"
#include "etsoc/common/utils.h"

/* ----------------------------------------------------------------- */
/* Fused FFN parameters                                              */
/* ----------------------------------------------------------------- */
struct ffn_params {
    uint64_t input_off;      /* offset of input [hidden] f32 in shared buffer */
    uint64_t wgate_off;      /* offset of W_gate [intermediate, hidden] Q8_0 */
    uint64_t wup_off;        /* offset of W_up   [intermediate, hidden] Q8_0 */
    uint64_t wdown_off;      /* offset of W_down [hidden, intermediate] Q8_0 */
    uint64_t output_off;     /* offset of output [hidden] f32 */
    uint64_t scratch_off;    /* offset of scratch [intermediate * 3] f32 */
    int64_t  hidden;         /* H — 2048 */
    int64_t  inter;          /* I — 8192 */
    int64_t  h_blocks;       /* hidden / 32 */
    int64_t  i_blocks;       /* intermediate / 32 */
};

/* Q8_0 constants */
#define Q8_BLK   32
#define Q8_BYTES 34   /* sizeof(block_q8_0) = 2 (f16 scale) + 32 (int8) */

#define TOTAL_HARTS  2048
#define T0_HARTS     1024   /* TOTAL_HARTS / 2 */

/* ----------------------------------------------------------------- */
/* Inline helpers                                                     */
/* ----------------------------------------------------------------- */

/* f16 (uint16 LE) → f32. Flush subnormals to zero. */
static inline float f16_to_f32(uint16_t raw) {
    uint32_t s = (uint32_t)(raw & 0x8000u) << 16;
    uint32_t e = (raw >> 10) & 0x1fu;
    uint32_t m = raw & 0x3ffu;
    uint32_t f32u = (e == 0) ? s : s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &f32u, 4); return r;
}

/* Per-block Q8_0 × f32 dot product: a_block · b_col_start (32 elements).
 * Scalar, no VPU — correct and portable. The board's VPU-vectorized version
 * is in block_ops.h::compute_block_dot_product_q8_0; swap this inline
 * version for the vectorized one once VPU integration is verified. */
static inline float block_dot_q8_0(const uint8_t *blk, const float *b) {
    uint16_t scale_raw;
    memcpy(&scale_raw, blk, 2);
    float scale = f16_to_f32(scale_raw);
    const int8_t *q = (const int8_t *)(blk + 2);
    double acc = 0.0;  /* use double to avoid float accumulation error */
    for (int i = 0; i < Q8_BLK; i++) {
        acc += (double)q[i] * (double)b[i];
    }
    return (float)(acc * (double)scale);
}

/* Full-row Q8_0 × f32 dot product: sum over K_blocks blocks. */
static inline float row_dot_q8_0(const uint8_t *q_row, const float *b_col, int64_t K_blocks) {
    double acc = 0.0;
    for (int64_t kb = 0; kb < K_blocks; kb++) {
        acc += (double)block_dot_q8_0(q_row + kb * Q8_BYTES, b_col + kb * Q8_BLK);
    }
    return (float)acc;
}

/* SiLU: x * sigmoid(x) */
static inline float silu_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float e = 1.0f + __builtin_expf(-x);
    return x / e;
}

/* Global barrier across all active harts.
 * All harts call this; returns when all have arrived.
 * Uses a shared counter in DRAM (global_addr must be the same for all harts).
 * Counter is cache-line-sized to avoid false sharing. */
static uint32_t __attribute__((aligned(64))) g_barrier_count = 0;

static inline void global_barrier(uint32_t num_harts) {
    /* Local shire barrier first to ensure all harts in this shire arrive
     * together before the global increment */
    FENCE;
    uint32_t local_minion = ((uint32_t)get_hart_id() >> 1) & 0x1Fu;
    uint32_t mask = 1u << local_minion;
    shire_barrier(local_minion, 0, 2, mask, mask);
    FENCE;

    /* One hart per minion does the global atomic increment.
     * Hart 0 in each minion (even hart_id) does the atomic. */
    if ((get_hart_id() & 1u) == 0u) {
        uint32_t prev = atomic_add_global_32(&g_barrier_count, 1u);
        if (prev + 1u == num_harts / 2u) {
            /* Last T0 hart: reset counter, do a global fence-like sync */
            atomic_store_global_32(&g_barrier_count, 0u);
            FENCE;
            /* Release all harts by sending FCC credits.
             * Each shire has 32 minions × 2 harts = 64 harts.
             * We send credits to the FCC 0 of each minion's T0/T1. */
            for (uint64_t sid = 0; sid < 32; sid++) {
                fcc_send(sid, THREAD_0, FCC_0, 0xFFFFFFFFu);
                fcc_send(sid, THREAD_1, FCC_0, 0xFFFFFFFFu);
            }
        }
    }

    /* Wait for FCC credit from the releasing hart */
    fcc_consume(FCC_0);
    FENCE;
}

/* Evict a region from L1+L2 to DRAM so other shires can read it.
 * Splits into hardware-compatible chunks (max 16 cache lines per call). */
static inline void evict_region(const void *addr, size_t bytes) {
    if (!addr || bytes == 0) return;
    const uint64_t CL = 64;
    uint64_t base = (uint64_t)addr & ~(CL - 1);
    uint64_t end = ((uint64_t)addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t nlines = (end - base) / CL;
    for (uint64_t off = 0; off < nlines; off += 16) {
        uint64_t batch = nlines - off;
        if (batch > 16) batch = 16;
        /* EvictVA CSR 0x89F, dest=10 (L3/DRAM) in bits 59:58 */
        uint64_t csr_val = (0x2ULL << 58) | ((uint64_t)(base + off * CL) & 0xFFFFFFFFFFC0ULL) | ((batch - 1) & 0xF);
        uint64_t x31_val = CL & 0xFFFFFFFFFFC0ULL;
        __asm__ __volatile__("mv x31, %[x31]\n csrw 0x89F, %[val]\n" :: [x31] "r"(x31_val), [val] "r"(csr_val) : "x31", "memory");
    }
    WAIT_CACHEOPS;
    FENCE;
}

/* ----------------------------------------------------------------- */
/* Main entry point                                                   */
/* ----------------------------------------------------------------- */
int main(uintptr_t arg_area) {
    uint64_t hart_id = get_hart_id();

    /* arg_area points to a copy of the 8-byte device-buffer pointer.
     * The host writes param struct at offset 0 of the device buffer. */
    uint8_t *base = (uint8_t *)(arg_area ? *(volatile uint64_t *)arg_area : 0);
    if (!base) return 1;

    /* Read params from device buffer offset 0 */
    struct ffn_params params;
    __builtin_memcpy(&params, base, sizeof(params));

    int64_t H = params.hidden;
    int64_t I = params.inter;
    int64_t H_blocks = params.h_blocks;  /* H / 32 */
    int64_t I_blocks = params.i_blocks;  /* I / 32 */

    const float *input   = (const float *)(base + params.input_off);
    const uint8_t *Wgate = (const uint8_t *)(base + params.wgate_off);
    const uint8_t *Wup   = (const uint8_t *)(base + params.wup_off);
    const uint8_t *Wdown = (const uint8_t *)(base + params.wdown_off);
    float *output  = (float *)(base + params.output_off);
    float *scratch = (float *)(base + params.scratch_off);

    /* T0 (even) harts do compute. Odd harts skip phase 1/2 but must
     * still participate in the global barrier. */
    int is_t0 = ((hart_id & 1u) == 0u);
    uint64_t tid = hart_id >> 1;  /* T0 index 0..1023 */

    /* ----------------------------------------------------------------- */
    /* Phase 1: gate[i] and up[i] for our shard of [0..I)                */
    /* ----------------------------------------------------------------- */
    int64_t rows_per = (I + (int64_t)T0_HARTS - 1) / (int64_t)T0_HARTS;
    int64_t r0 = (int64_t)tid * rows_per;
    int64_t r1 = r0 + rows_per;
    if (r1 > I) r1 = I;

    if (is_t0 && r0 < I) {
        for (int64_t i = r0; i < r1; i++) {
            scratch[i]     = row_dot_q8_0(Wgate + i * (H_blocks * Q8_BYTES), input, H_blocks);
            scratch[I + i] = row_dot_q8_0(Wup   + i * (H_blocks * Q8_BYTES), input, H_blocks);
        }
        FENCE;
        evict_region(&scratch[r0], (size_t)(r1 - r0) * sizeof(float));
        evict_region(&scratch[I + r0], (size_t)(r1 - r0) * sizeof(float));
    }

    /* Local (minion) barrier: wait for both harts in this minion */
    FENCE;
    uint32_t local_minion = ((uint32_t)hart_id >> 1) & 0x1Fu;
    uint32_t mmask = 1u << local_minion;
    shire_barrier(local_minion, 0, 2, mmask, mmask);
    FENCE;

    /* ----------------------------------------------------------------- */
    /* Phase 2: gated[i] = SiLU(gate[i]) × up[i]                        */
    /* ----------------------------------------------------------------- */
    if (is_t0 && r0 < I) {
        for (int64_t i = r0; i < r1; i++) {
            scratch[i] = silu_f32(scratch[i]) * scratch[I + i];
        }
        FENCE;
        evict_region(&scratch[r0], (size_t)(r1 - r0) * sizeof(float));
    }

    /* Global barrier: all 2048 harts. Phase 3 reads the FULL gated array
     * written by all harts, so everyone must evict before anyone reads. */
    global_barrier(TOTAL_HARTS);

    /* ----------------------------------------------------------------- */
    /* Phase 3: output[j] = W_down[j] · gated                            */
    /* ----------------------------------------------------------------- */
    if (!is_t0) return 0;  /* Odd harts are done */

    rows_per = (H + (int64_t)T0_HARTS - 1) / (int64_t)T0_HARTS;
    r0 = (int64_t)tid * rows_per;
    r1 = r0 + rows_per;
    if (r1 > H) r1 = H;

    if (r0 < H) {
        for (int64_t j = r0; j < r1; j++) {
            output[j] = row_dot_q8_0(Wdown + j * (I_blocks * Q8_BYTES), scratch, I_blocks);
        }
        FENCE;
        evict_region(&output[r0], (size_t)(r1 - r0) * sizeof(float));
    }

    return 0;
}
