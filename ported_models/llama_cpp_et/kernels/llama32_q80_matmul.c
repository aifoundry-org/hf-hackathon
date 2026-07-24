/*
 * Q8_0 × FP32 fused matmul for Llama 3.2 1B on ET-SoC1.
 *
 * Loaded by the GGML ET backend at runtime (GGML_ET_KERNELS_PATH).
 * Replaces the default mul_mat for Q8_0 × f32 with a multi-hart
 * VPU-vectorized version: fuses dequant + matmul, 8-wide fmadd.ps.
 *
 * GGML Q8_0 layout (src0 = weights, shape [N, K]):
 *   Row j (output feature j): K_blocks = ceil(K/32) blocks
 *   Each block: [f16 scale LE][32 × int8 qs] = 34 bytes
 *   Row stride: K_blocks * 34 bytes
 *   Element (j, kb, lane): B[j * K_blocks * 34 + kb * 34 + 2 + lane]
 *
 * Multi-hart: each T0 hart handles a contiguous slice of output cols.
 * VPU: 8-wide using flq2/fsq2/fmadd.ps/fbcx.ps.
 *
 * Build:
 *   riscv64-unknown-elf-gcc -march=rv64imfc -mabi=lp64f \
 *     -O2 -ffast-math -fno-tree-loop-distribute-patterns \
 *     -c llama32_q80_matmul.c -o llama32_q80_matmul.elf
 */

#include <stdint.h>
#include <string.h>

#include "erbium/isa/hart.h"
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/fcc.h"
#include "erbium/isa/flb.h"

/* ----------------------------------------------------------------- */
/* Config                                                             */
/* ----------------------------------------------------------------- */
#ifndef UBERK_HARTS
#define UBERK_HARTS 8          /* T0 harts to use */
#endif

#define BLK_N       32          /* Q8_0 block size */
#define BLK_BYTES   34          /* 2 (f16 scale) + 32 (int8) */
#define VPU_W       8           /* VPU width */

/* Default shared-buffer layout */
#define PARAM_OFF   0x0000u
#define A_OFF       0x010000u   /* A: M x K x f32 */
#define B_OFF       0x100000u   /* B: N x K_blocks x 34 (Q8_0) */
#define C_OFF       0x500000u   /* C: M x N x f32 */

extern char heap0_end[];

/* ----------------------------------------------------------------- */
/* Params — host writes this before launch                            */
/* ----------------------------------------------------------------- */
struct params {
    uint32_t M, N, K;        /* dimensions */
    uint32_t a_off, b_off, c_off; /* buffer offsets */
    uint32_t harts;           /* active T0 count */
};

/* ----------------------------------------------------------------- */
/* VPU helpers                                                       */
/* ----------------------------------------------------------------- */
static inline float vbcx(float x) {
    float r;
    union { float f; uint32_t u; } u; u.f = x;
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(r) : "r"((uint64_t)u.u));
    return r;
}
static inline float vld8(const float *p) {
    float r;
    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(r) : "r"(p));
    return r;
}
static inline void vst8(float *p, float v) {
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(p), "f"(v) : "memory");
}
static inline float vfmadd(float a, float b, float c) {
    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a) : "f"(b), "f"(c));
    return a;
}
static inline float vfmul(float a, float b) {
    float r;
    __asm__ volatile("fmul.ps %0, %1, %2\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

/* f16 (uint16 LE) → f32.  Flush subnormals to zero. */
static inline float f16tof32(uint16_t raw) {
    uint32_t s = (uint32_t)(raw & 0x8000u) << 16;
    uint32_t e = (raw >> 10) & 0x1fu;
    uint32_t m = raw & 0x3ffu;
    uint32_t f32u = (e == 0) ? s : s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &f32u, 4); return r;
}

/* Load 8 int8, convert to f32, return packed VPU register. */
static inline float vld8_i8(const int8_t *q) {
    float tmp[8];
    for (int i = 0; i < 8; i++) tmp[i] = (float)q[i];
    return vld8(tmp);
}

/* ----------------------------------------------------------------- */
/* Entry                                                             */
/* ----------------------------------------------------------------- */
int main(uintptr_t arg_area) {
    uint32_t hid = get_hart_id();
    if ((hid & 1u) != 0u) return 0;       /* T0 only */
    uint32_t tid = hid >> 1;

    /* Resolve buffer base */
    uint8_t *base;
    if (arg_area == 0 || arg_area == ~(uintptr_t)0) {
        base = (uint8_t *)heap0_end - 80u * 1024u * 1024u;
    } else {
        uintptr_t p = *(volatile uintptr_t *)arg_area;
        base = (uint8_t *)((p == 0 || p == ~(uintptr_t)0)
                           ? (uintptr_t)heap0_end - 80u * 1024u * 1024u : p);
    }

    /* Read params */
    struct params p;
    memcpy(&p, base + PARAM_OFF, sizeof(p));
    uint32_t M     = p.M ? p.M : 1;
    uint32_t N     = p.N ? p.N : 2048;
    uint32_t K     = p.K ? p.K : 2048;
    uint32_t a_off = p.a_off ? p.a_off : A_OFF;
    uint32_t b_off = p.b_off ? p.b_off : B_OFF;
    uint32_t c_off = p.c_off ? p.c_off : C_OFF;
    uint32_t harts = p.harts ? p.harts : UBERK_HARTS;

    float   *A = (float   *)(base + a_off);
    uint8_t *B = (uint8_t *)(base + b_off);
    float   *C = (float   *)(base + c_off);

    /* Each T0 hart handles a slice of output columns (N) */
    uint32_t cols_h = (N + harts - 1u) / harts;
    uint32_t c0 = tid * cols_h;
    uint32_t c1 = c0 + cols_h;
    if (c1 > N) c1 = N;
    if (c0 >= N) return 0;

    uint32_t K_blk = (K + BLK_N - 1u) / BLK_N;   /* blocks per row */

    /* ----------------------------------------------------------------- */
    /* Main: C[m, j..j+7] = A[m, :] @ B[j..j+7, :]^T                    */
    /* ----------------------------------------------------------------- */
    for (uint32_t m = 0; m < M; m++) {
        const float *a_row = A + m * K;

        for (uint32_t j = c0; j < c1; j += VPU_W) {
            float acc;

            /* acc = 0 via broadcast */
            { union { float f; uint32_t u; } z; z.f = 0.0f; acc = vbcx(z.f); }

            for (uint32_t kb = 0; kb < K_blk; kb++) {
                uint32_t k0 = kb * BLK_N;
                uint32_t klen = K - k0;
                if (klen > BLK_N) klen = BLK_N;

                /* Pointer to Q8_0 block for output col j, block kb:
                 *   B_row_start = j * K_blk * 34
                 *   Block = B_row_start + kb * 34                          */
                uint8_t *blk = B + (uint64_t)j * K_blk * BLK_BYTES + (uint64_t)kb * BLK_BYTES;

                float s = f16tof32(*(uint16_t *)blk);
                float s8 = vbcx(s);
                const int8_t *q = (const int8_t *)(blk + 2);

                uint32_t l = 0;
                for (; l + VPU_W <= klen; l += VPU_W) {
                    float a8 = vld8(a_row + k0 + l);          /* 8 activations */
                    float q8 = vld8_i8(q + l);                 /* 8 quants → f32 */
                    float p  = vfmul(a8, q8);                  /* elementwise product */
                    acc      = vfmadd(acc, p, s8);             /* acc += product * scale */
                }
                /* Remainder (< 8) — scalar */
                for (; l < klen; l++) {
                    acc = (float)((double)acc + (double)a_row[k0 + l] * (double)q[l] * (double)s);
                }
            }

            vst8(C + m * N + j, acc);
        }
    }

    /* Evict our slice */
    if (c1 > c0) {
        uint64_t bytes = (uint64_t)M * (c1 - c0) * sizeof(float);
        evict((const void *)(C + c0), bytes);
    }
    WAIT_CACHEOPS;
    FENCE;

    return 0;
}
