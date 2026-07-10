/* Shared helpers for yolov10n kernels: math primitives, conv2d, SiLU,
 * concat/split/add, evict-and-fence. Single-hart FP32 scalar baseline.
 *
 * These are intentionally simple - readability beats speed.  Optimized
 * variants land in later milestones once correctness is established.
 */
#ifndef YOLO_COMMON_H
#define YOLO_COMMON_H

#include <stdint.h>
#include "erbium/isa/atomic.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/utils.h"

extern char heap0_end[];
#define BUFFER_SIZE         (80u * 1024u * 1024u)

static inline float fast_recip(float x) {
    union { float f; uint32_t u; } v; v.f = x;
    v.u = 0x7EF311C3u - v.u; float r = v.f;
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    return r;
}
static inline float my_expf(float x) {
    if (x >  88.0f) x =  88.0f;
    if (x < -88.0f) x = -88.0f;
    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;
    int k = (int)(x * inv_ln2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * ln2;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    union { float f; uint32_t u; } v = { p };
    int32_t exp_bias = (int32_t)((v.u >> 23) & 0xFF) + k;
    if (exp_bias > 254) v.u = 0x7F7FFFFFu;
    else if (exp_bias < 1) v.u = 0;
    else v.u = (v.u & 0x807FFFFFu) | ((uint32_t)exp_bias << 23);
    return v.f;
}
static inline float silu(float x) {
    if (x > 8.0f) return x;
    if (x < -8.0f) return 0.0f;
    return x * fast_recip(1.0f + my_expf(-x));
}

static inline uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - BUFFER_SIZE;
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - BUFFER_SIZE;
    return ptr;
}

/* Generic Conv2d: NCHW input/output. Activation:
 *   ACT=0  none
 *   ACT=1  SiLU
 */
static void conv2d_fp32(const float *in, float *out,
                        const float *W, const float *B,
                        uint32_t IC, uint32_t IH, uint32_t IW,
                        uint32_t OC, uint32_t OH, uint32_t OW,
                        uint32_t KH, uint32_t KW,
                        uint32_t SH, uint32_t SW,
                        uint32_t PH, uint32_t PW,
                        uint32_t act)
{
    for (uint32_t oc = 0; oc < OC; oc++) {
        const float bias = B[oc];
        for (uint32_t oh = 0; oh < OH; oh++) {
            int32_t ih_base = (int32_t)(oh * SH) - (int32_t)PH;
            uint32_t ky_start = (ih_base < 0) ? (uint32_t)(-ih_base) : 0u;
            uint32_t ky_end = KH;
            if (ih_base + (int32_t)KH > (int32_t)IH) ky_end = (uint32_t)((int32_t)IH - ih_base);
            
            for (uint32_t ow = 0; ow < OW; ow++) {
                int32_t iw_base = (int32_t)(ow * SW) - (int32_t)PW;
                uint32_t kx_start = (iw_base < 0) ? (uint32_t)(-iw_base) : 0u;
                uint32_t kx_end = KW;
                if (iw_base + (int32_t)KW > (int32_t)IW) kx_end = (uint32_t)((int32_t)IW - iw_base);
                
                float acc = bias;
                for (uint32_t ic = 0; ic < IC; ic++) {
                    for (uint32_t ky = ky_start; ky < ky_end; ky++) {
                        const uint32_t ih = (uint32_t)(ih_base + (int32_t)ky);
                        for (uint32_t kx = kx_start; kx < kx_end; kx++) {
                            const uint32_t iw = (uint32_t)(iw_base + (int32_t)kx);
                            const float v = in[(ic * IH + ih) * IW + iw];
                            const float w = W[((oc * IC + ic) * KH + ky) * KW + kx];
                            acc += w * v;
                        }
                    }
                }
                if (act == 1u) acc = silu(acc);
                out[(oc * OH + oh) * OW + ow] = acc;
            }
        }
    }
}

/* Depthwise conv: groups == IC == OC, filter shape [OC, 1, KH, KW]. */
static void conv2d_dw_fp32(const float *in, float *out,
                           const float *W, const float *B,
                           uint32_t C, uint32_t IH, uint32_t IW,
                           uint32_t OH, uint32_t OW,
                           uint32_t KH, uint32_t KW,
                           uint32_t SH, uint32_t SW,
                           uint32_t PH, uint32_t PW,
                           uint32_t act)
{
    for (uint32_t c = 0; c < C; c++) {
        const float bias = B[c];
        for (uint32_t oh = 0; oh < OH; oh++) {
            for (uint32_t ow = 0; ow < OW; ow++) {
                float acc = bias;
                for (uint32_t ky = 0; ky < KH; ky++) {
                    const int32_t ih = (int32_t)(oh * SH) - (int32_t)PH + (int32_t)ky;
                    if (ih < 0 || ih >= (int32_t)IH) continue;
                    for (uint32_t kx = 0; kx < KW; kx++) {
                        const int32_t iw = (int32_t)(ow * SW) - (int32_t)PW + (int32_t)kx;
                        if (iw < 0 || iw >= (int32_t)IW) continue;
                        const float v = in[(c * IH + (uint32_t)ih) * IW + (uint32_t)iw];
                        const float w = W[(c * KH + ky) * KW + kx];
                        acc += w * v;
                    }
                }
                if (act == 1u) acc = silu(acc);
                out[(c * OH + oh) * OW + ow] = acc;
            }
        }
    }
}

/* Add NCHW tensors elementwise: y = a + b */
static inline void add_chw(const float *a, const float *b, float *y, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] = a[i] + b[i];
}

/* Concat along channel axis: out[c=0..Ca,*,*] = a, out[c=Ca..Ca+Cb,*,*] = b. */
static inline void concat_c_chw(const float *a, uint32_t Ca,
                                const float *b, uint32_t Cb,
                                float *out, uint32_t H, uint32_t W) {
    const uint32_t bytes_a = Ca * H * W;
    const uint32_t bytes_b = Cb * H * W;
    for (uint32_t i = 0; i < bytes_a; i++) out[i] = a[i];
    for (uint32_t i = 0; i < bytes_b; i++) out[bytes_a + i] = b[i];
}

/* Split along channel axis: a = in[c=0..Ca,*,*], b = in[c=Ca..,*,*]. */
static inline void split_c_chw(const float *in, uint32_t Cin,
                               float *a, uint32_t Ca,
                               float *b, uint32_t Cb,
                               uint32_t H, uint32_t W) {
    (void)Cin;
    const uint32_t hw = H * W;
    for (uint32_t c = 0; c < Ca; c++)
        for (uint32_t i = 0; i < hw; i++) a[c*hw + i] = in[c*hw + i];
    for (uint32_t c = 0; c < Cb; c++)
        for (uint32_t i = 0; i < hw; i++) b[c*hw + i] = in[(Ca+c)*hw + i];
}

/* MaxPool 2D NCHW. */
static void maxpool_fp32(const float *in, float *out,
                         uint32_t C, uint32_t IH, uint32_t IW,
                         uint32_t OH, uint32_t OW,
                         uint32_t KH, uint32_t KW,
                         uint32_t SH, uint32_t SW,
                         uint32_t PH, uint32_t PW)
{
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < OH; oh++) {
            for (uint32_t ow = 0; ow < OW; ow++) {
                float m = -3.4e38f;
                for (uint32_t ky = 0; ky < KH; ky++) {
                    const int32_t ih = (int32_t)(oh * SH) - (int32_t)PH + (int32_t)ky;
                    if (ih < 0 || ih >= (int32_t)IH) continue;
                    for (uint32_t kx = 0; kx < KW; kx++) {
                        const int32_t iw = (int32_t)(ow * SW) - (int32_t)PW + (int32_t)kx;
                        if (iw < 0 || iw >= (int32_t)IW) continue;
                        const float v = in[(c * IH + (uint32_t)ih) * IW + (uint32_t)iw];
                        if (v > m) m = v;
                    }
                }
                out[(c * OH + oh) * OW + ow] = m;
            }
        }
    }
}

#define EVICT_AND_FENCE(addr, bytes) do { \
    evict((const void *)(addr), (uint64_t)(bytes)); \
    WAIT_CACHEOPS; \
    FENCE; \
} while (0)

/* -- multi-hart helpers -- */
#include "erbium/isa/barriers.h"
#include "erbium/isa/fcc.h"
#include "erbium/isa/flb.h"

#define MH_FLB        1u
#ifndef YOLO_RESERVE_MINION0
#define YOLO_RESERVE_MINION0 0
#endif

#ifdef BENCH_THREAD0_ONLY
#define MH_T0_MASK    ((1u << ACTIVE_HARTS) - 1u)
#define MH_T1_MASK    0u
#define MH_TOTAL      ACTIVE_HARTS
#define MH_NUM_T0     ACTIVE_HARTS
#define MH_BARRIER_OFFSET 0x8000u
struct mh_barrier_state {
    uint32_t count;
    uint32_t epoch;
    uint32_t reserved[14];
};
static volatile struct mh_barrier_state *g_mh_barrier;
static inline int      mh_is_active_hart(uint32_t hid) { (void)hid; return get_thread_id() == 0u && get_minion_id() < ACTIVE_HARTS; }
static inline uint32_t mh_t0_idx(uint32_t hid) { (void)hid; return get_minion_id(); }
static inline int      mh_is_t0(uint32_t hid) { return mh_is_active_hart(hid); }
static inline int      mh_is_leader(uint32_t hid) { (void)hid; return get_thread_id() == 0u && get_minion_id() == 0u; }
static inline void     mh_init_barrier(uint8_t *base) { g_mh_barrier = (volatile struct mh_barrier_state *)(base + MH_BARRIER_OFFSET); }
static inline void     mh_atomic_barrier(void)
{
    if (ACTIVE_HARTS <= 1u) return;
    volatile struct mh_barrier_state *const barrier = g_mh_barrier;
    const uint32_t epoch = atomic_load_local_32(&barrier->epoch);
    const uint32_t prior = atomic_add_local_32(&barrier->count, 1u);

    if (prior + 1u == ACTIVE_HARTS) {
        atomic_store_local_32(&barrier->count, 0u);
        FENCE;
        atomic_add_local_32(&barrier->epoch, 1u);
    } else {
        while (atomic_load_local_32(&barrier->epoch) == epoch) {
            FENCE;
        }
    }
    FENCE;
}
#elif YOLO_RESERVE_MINION0
#define MH_T0_MASK    0xFEu
#define MH_T1_MASK    0xFEu
#define MH_TOTAL      14u
#define MH_NUM_T0     7u
#define MH_LEADER_HART 2u
static inline int      mh_is_active_hart(uint32_t hid) { return hid >= 2u && hid < 16u; }
static inline uint32_t mh_t0_idx(uint32_t hid) { return (hid >> 1u) - 1u; }
static inline int      mh_is_t0 (uint32_t hid) { return mh_is_active_hart(hid) && ((hid & 1u) == 0u); }
#else
#define MH_T0_MASK    0xFFu
#define MH_T1_MASK    0xFFu
#define MH_TOTAL      16u
#define MH_NUM_T0     8u
#define MH_LEADER_HART 0u
static inline int      mh_is_active_hart(uint32_t hid) { return hid < 16u; }
static inline uint32_t mh_t0_idx(uint32_t hid) { return hid >> 1u; }
static inline int      mh_is_t0 (uint32_t hid) { return (hid & 1u) == 0u; }
#endif

#ifndef BENCH_THREAD0_ONLY
static inline int      mh_is_leader(uint32_t hid) { return hid == MH_LEADER_HART; }
static inline void     mh_init_barrier(uint8_t *base) { (void)base; }
#endif

#ifdef BENCH_THREAD0_ONLY
#define MH_BARRIER() do { \
    FENCE; WAIT_CACHEOPS; \
    mh_atomic_barrier(); \
} while (0)
#else
#define MH_BARRIER() do { \
    FENCE; WAIT_CACHEOPS; \
    (void)shire_barrier(MH_FLB, FCC_0, MH_TOTAL, MH_T0_MASK, MH_T1_MASK); \
} while (0)
#endif

/* Convenience macros: auto-barrier after each multi-hart conv. */
#define CONV_MH(...)    do { conv2d_fp32_mh(hid, __VA_ARGS__);    MH_BARRIER(); } while (0)
#define CONV_DW_MH(...) do { conv2d_dw_fp32_mh(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* Hart-0 only block: STMT runs on hart 0; eviction + barrier follow. */
#define H0_RUN(STMT, ADDR, BYTES) do { \
    if (is_h0) { \
        STMT; \
        evict((const void *)(ADDR), (uint64_t)(BYTES)); \
        WAIT_CACHEOPS; FENCE; \
    } \
    MH_BARRIER(); \
} while (0)

/* Range slice: hart `t0_idx` (in 0..MH_NUM_T0) gets [lo, hi) of [0..N).
 * Uniform partition; remainder distributed to lowest indices. */
static inline void mh_range(uint32_t N, uint32_t t0_idx,
                            uint32_t *lo, uint32_t *hi)
{
    *lo = (N * t0_idx) / MH_NUM_T0;
    *hi = (N * (t0_idx + 1u)) / MH_NUM_T0;
}

/* Multi-hart Conv2d: split by output channel across either 8 T0 harts
 * or all 16 harts (controlled by the YOLO_USE_16HART build flag).  Caller
 * is responsible for the barrier afterwards (we only evict our own slice). */
#ifdef YOLO_USE_16HART
#if YOLO_RESERVE_MINION0
#define YOLO_NHART 14u
static inline uint32_t yolo_compute_idx(uint32_t hid) { return hid - 2u; }
static inline int      yolo_is_compute(uint32_t hid) { return mh_is_active_hart(hid); }
#else
#define YOLO_NHART 16u
static inline uint32_t yolo_compute_idx(uint32_t hid) { return hid; }
static inline int      yolo_is_compute(uint32_t hid) { (void)hid; return 1; }
#endif
#else
#define YOLO_NHART  MH_NUM_T0
static inline uint32_t yolo_compute_idx(uint32_t hid) { return mh_t0_idx(hid); }
static inline int      yolo_is_compute(uint32_t hid) { return mh_is_t0(hid); }
#endif

static inline void yolo_range(uint32_t N, uint32_t idx,
                              uint32_t *lo, uint32_t *hi)
{
    *lo = (N * idx) / YOLO_NHART;
    *hi = (N * (idx + 1u)) / YOLO_NHART;
}

static void conv2d_fp32_mh(uint32_t hid,
                           const float *in, float *out,
                           const float *W, const float *B,
                           uint32_t IC, uint32_t IH, uint32_t IW,
                           uint32_t OC, uint32_t OH, uint32_t OW,
                           uint32_t KH, uint32_t KW,
                           uint32_t SH, uint32_t SW,
                           uint32_t PH, uint32_t PW,
                           uint32_t act)
{
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t oc_lo, oc_hi;
    yolo_range(OC, cidx, &oc_lo, &oc_hi);

    for (uint32_t oc = oc_lo; oc < oc_hi; oc++) {
        const float bias = B[oc];
        for (uint32_t oh = 0; oh < OH; oh++) {
            int32_t ih_base = (int32_t)(oh * SH) - (int32_t)PH;
            uint32_t ky_start = (ih_base < 0) ? (uint32_t)(-ih_base) : 0u;
            uint32_t ky_end = KH;
            if (ih_base + (int32_t)KH > (int32_t)IH) ky_end = (uint32_t)((int32_t)IH - ih_base);
            
            for (uint32_t ow = 0; ow < OW; ow++) {
                int32_t iw_base = (int32_t)(ow * SW) - (int32_t)PW;
                uint32_t kx_start = (iw_base < 0) ? (uint32_t)(-iw_base) : 0u;
                uint32_t kx_end = KW;
                if (iw_base + (int32_t)KW > (int32_t)IW) kx_end = (uint32_t)((int32_t)IW - iw_base);
                
                float acc = bias;
                for (uint32_t ic = 0; ic < IC; ic++) {
                    for (uint32_t ky = ky_start; ky < ky_end; ky++) {
                        const uint32_t ih = (uint32_t)(ih_base + (int32_t)ky);
                        for (uint32_t kx = kx_start; kx < kx_end; kx++) {
                            const uint32_t iw = (uint32_t)(iw_base + (int32_t)kx);
                            const float v = in[(ic * IH + ih) * IW + iw];
                            const float w = W[((oc * IC + ic) * KH + ky) * KW + kx];
                            acc += w * v;
                        }
                    }
                }
                if (act == 1u) acc = silu(acc);
                out[(oc * OH + oh) * OW + ow] = acc;
            }
        }
    }
    /* Evict our slice. */
    if (oc_hi > oc_lo) {
        const uint32_t bytes = (oc_hi - oc_lo) * OH * OW * sizeof(float);
        evict((const void *)(out + oc_lo * OH * OW), bytes);
    }
}

/* VPU-vectorized 1x1 Conv2d (stride=1, pad=0). Multi-hart by OC.
 *
 * Inner loop: each output pixel (oh, ow_block..ow_block+7) gets 8 lanes
 * accumulated in VPU register f0 via fmadd.ps with a broadcast scalar
 * weight and a flq2-loaded 8-lane input vector.
 *
 * Caller must guarantee KH=KW=1, stride=1, pad=0, OW % 8 == 0 (true for
 * all our YOLO 1x1 convs since OW in {16, 32, 64, 128, 256}).
 */
static void conv2d_1x1_fp32_mh_vpu(uint32_t hid,
                                   const float *in, float *out,
                                   const float *W, const float *B,
                                   uint32_t IC, uint32_t H, uint32_t W_,
                                   uint32_t OC,
                                   uint32_t act)
{
    /* VPU lives only on even (T0) harts.  Odd harts idle and just hit the
     * outer barrier afterwards. */
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t oc_lo, oc_hi;
    *(volatile uint32_t *)&oc_lo = (OC * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&oc_hi = (OC * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t oc = oc_lo; oc < oc_hi; oc++) {
        const float bias_v = B[oc];
        union { float f; uint32_t u; } bb; bb.f = bias_v;
        for (uint32_t oh = 0; oh < H; oh++) {
            for (uint32_t ow8 = 0; ow8 < W_; ow8 += 8u) {
                register float acc asm("f0");
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc) : "r"((uint64_t)bb.u));
                for (uint32_t ic = 0; ic < IC; ic++) {
                    const float w_scalar = W[oc * IC + ic];
                    union { float f; uint32_t u; } ww; ww.f = w_scalar;
                    float v_pkg;
                    float w_pkg;
                    const float *src = in + (ic * H + oh) * W_ + ow8;
                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                     : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                }
                /* Store 8 lanes; apply scalar SiLU/activation. */
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                /* throwaway fsq2 to flush store buffer (depth-anything lesson) */
                __asm__ volatile("fence rw, rw" ::: "memory");
                float *dst = out + (oc * H + oh) * W_ + ow8;
                if (act == 1u) {
                    for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]);
                } else {
                    for (int l = 0; l < 8; l++) dst[l] = acc_buf[l];
                }
            }
        }
    }
    if (oc_hi > oc_lo) {
        const uint32_t bytes = (oc_hi - oc_lo) * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

#define CONV_1x1_VPU(...)  do { conv2d_1x1_fp32_mh_vpu(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* OC-blocked VPU 1x1 conv: 8 output channels accumulated simultaneously per
 * (oh, ow8) tile.  Input is loaded ONCE per (ic, ow8) and reused across all
 * 8 oc lanes - 8x less memory bandwidth than the per-OC version above.
 *
 * Constraints: OC % 8 == 0, OW % 8 == 0. */
static void conv2d_1x1_fp32_mh_vpu_oc8(uint32_t hid,
                                       const float *in, float *out,
                                       const float *W, const float *B,
                                       uint32_t IC, uint32_t H, uint32_t W_,
                                       uint32_t OC,
                                       uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    /* Slice OC by groups of 8 (whole tile). */
    const uint32_t oc_tiles = OC / 8u;
    uint32_t tile_lo, tile_hi;
    *(volatile uint32_t *)&tile_lo = (oc_tiles * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&tile_hi = (oc_tiles * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t tile = tile_lo; tile < tile_hi; tile++) {
        const uint32_t oc0 = tile * 8u;
        for (uint32_t oh = 0; oh < H; oh++) {
            for (uint32_t ow8 = 0; ow8 < W_; ow8 += 8u) {
                /* 8 accumulators, one per OC lane in this tile. */
                register float a0 asm("f0"), a1 asm("f1"), a2 asm("f2"), a3 asm("f3");
                register float a4 asm("f4"), a5 asm("f5"), a6 asm("f6"), a7 asm("f7");
                /* Initialize each to bias broadcast. */
#define INIT_ACC(REG, OC_OFFSET) do { \
    union { float f; uint32_t u; } _bb; _bb.f = B[oc0 + OC_OFFSET]; \
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(REG) : "r"((uint64_t)_bb.u)); \
} while (0)
                INIT_ACC(a0, 0); INIT_ACC(a1, 1); INIT_ACC(a2, 2); INIT_ACC(a3, 3);
                INIT_ACC(a4, 4); INIT_ACC(a5, 5); INIT_ACC(a6, 6); INIT_ACC(a7, 7);
#undef INIT_ACC

                for (uint32_t ic = 0; ic < IC; ic++) {
                    float v_pkg;
                    register float w0 asm("f20"), w1 asm("f21"), w2 asm("f22"), w3 asm("f23");
                    register float w4 asm("f24"), w5 asm("f25"), w6 asm("f26"), w7 asm("f27");
                    const float *src = in + (ic * H + oh) * W_ + ow8;
                    
                    union { float f; uint32_t u; } w0_u; w0_u.f = W[(oc0 + 0) * IC + ic];
                    union { float f; uint32_t u; } w1_u; w1_u.f = W[(oc0 + 1) * IC + ic];
                    union { float f; uint32_t u; } w2_u; w2_u.f = W[(oc0 + 2) * IC + ic];
                    union { float f; uint32_t u; } w3_u; w3_u.f = W[(oc0 + 3) * IC + ic];
                    union { float f; uint32_t u; } w4_u; w4_u.f = W[(oc0 + 4) * IC + ic];
                    union { float f; uint32_t u; } w5_u; w5_u.f = W[(oc0 + 5) * IC + ic];
                    union { float f; uint32_t u; } w6_u; w6_u.f = W[(oc0 + 6) * IC + ic];
                    union { float f; uint32_t u; } w7_u; w7_u.f = W[(oc0 + 7) * IC + ic];
                    
                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                    
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w0) : "r"((uint64_t)w0_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w1) : "r"((uint64_t)w1_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w2) : "r"((uint64_t)w2_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w3) : "r"((uint64_t)w3_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w4) : "r"((uint64_t)w4_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w5) : "r"((uint64_t)w5_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w6) : "r"((uint64_t)w6_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w7) : "r"((uint64_t)w7_u.u));

                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a0) : "f"(v_pkg), "f"(w0));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a1) : "f"(v_pkg), "f"(w1));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a2) : "f"(v_pkg), "f"(w2));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a3) : "f"(v_pkg), "f"(w3));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a4) : "f"(v_pkg), "f"(w4));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a5) : "f"(v_pkg), "f"(w5));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a6) : "f"(v_pkg), "f"(w6));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a7) : "f"(v_pkg), "f"(w7));
                }

                /* Store each accumulator with optional SiLU. */
#define STORE_ACC(REG, OC_OFFSET) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    float *dst = out + ((oc0 + OC_OFFSET) * H + oh) * W_ + ow8; \
    if (act == 1u) { \
        for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]); \
    } else { \
        for (int l = 0; l < 8; l++) dst[l] = acc_buf[l]; \
    } \
} while (0)
                STORE_ACC(a0, 0); STORE_ACC(a1, 1); STORE_ACC(a2, 2); STORE_ACC(a3, 3);
                STORE_ACC(a4, 4); STORE_ACC(a5, 5); STORE_ACC(a6, 6); STORE_ACC(a7, 7);
#undef STORE_ACC
            }
        }
    }
    if (tile_hi > tile_lo) {
        const uint32_t oc_lo = tile_lo * 8u;
        const uint32_t bytes = (tile_hi - tile_lo) * 8u * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

#define CONV_1x1_VPU8(...) do { conv2d_1x1_fp32_mh_vpu_oc8(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* OC16-blocked 1x1: 16 accumulators in f0..f15.  Needs OC % 16 == 0
 * AND OC large enough for all 8 T0 harts to get at least one tile (OC>=128). */
static void conv2d_1x1_fp32_mh_vpu_oc16(uint32_t hid,
                                        const float *in, float *out,
                                        const float *W, const float *B,
                                        uint32_t IC, uint32_t H, uint32_t W_,
                                        uint32_t OC,
                                        uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t oc_tiles = OC / 16u;
    uint32_t tile_lo, tile_hi;
    *(volatile uint32_t *)&tile_lo = (oc_tiles * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&tile_hi = (oc_tiles * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t tile = tile_lo; tile < tile_hi; tile++) {
        const uint32_t oc0 = tile * 16u;
        for (uint32_t oh = 0; oh < H; oh++) {
            for (uint32_t ow8 = 0; ow8 < W_; ow8 += 8u) {
                register float a0 asm("f0"), a1 asm("f1"), a2 asm("f2"), a3 asm("f3");
                register float a4 asm("f4"), a5 asm("f5"), a6 asm("f6"), a7 asm("f7");
                register float a8 asm("f10"),a9 asm("f11"),aA asm("f12"),aB asm("f13");
                register float aC asm("f14"),aD asm("f15"),aE asm("f16"),aF asm("f17");
#define INIT_ACC(REG, OO) do { \
    union { float f; uint32_t u; } _bb; _bb.f = B[oc0 + OO]; \
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(REG) : "r"((uint64_t)_bb.u)); \
} while (0)
                INIT_ACC(a0, 0); INIT_ACC(a1, 1); INIT_ACC(a2, 2); INIT_ACC(a3, 3);
                INIT_ACC(a4, 4); INIT_ACC(a5, 5); INIT_ACC(a6, 6); INIT_ACC(a7, 7);
                INIT_ACC(a8, 8); INIT_ACC(a9, 9); INIT_ACC(aA,10); INIT_ACC(aB,11);
                INIT_ACC(aC,12); INIT_ACC(aD,13); INIT_ACC(aE,14); INIT_ACC(aF,15);
#undef INIT_ACC

                for (uint32_t ic = 0; ic < IC; ic++) {
                    float v_pkg;
                    register float w0 asm("f20"), w1 asm("f21"), w2 asm("f22"), w3 asm("f23");
                    register float w4 asm("f24"), w5 asm("f25"), w6 asm("f26"), w7 asm("f27");
                    const float *src = in + (ic * H + oh) * W_ + ow8;
                    
                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                    
                    // Batch 1 (0-7)
                    union { float f; uint32_t u; } w0_u; w0_u.f = W[(oc0 + 0) * IC + ic];
                    union { float f; uint32_t u; } w1_u; w1_u.f = W[(oc0 + 1) * IC + ic];
                    union { float f; uint32_t u; } w2_u; w2_u.f = W[(oc0 + 2) * IC + ic];
                    union { float f; uint32_t u; } w3_u; w3_u.f = W[(oc0 + 3) * IC + ic];
                    union { float f; uint32_t u; } w4_u; w4_u.f = W[(oc0 + 4) * IC + ic];
                    union { float f; uint32_t u; } w5_u; w5_u.f = W[(oc0 + 5) * IC + ic];
                    union { float f; uint32_t u; } w6_u; w6_u.f = W[(oc0 + 6) * IC + ic];
                    union { float f; uint32_t u; } w7_u; w7_u.f = W[(oc0 + 7) * IC + ic];
                    
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w0) : "r"((uint64_t)w0_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w1) : "r"((uint64_t)w1_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w2) : "r"((uint64_t)w2_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w3) : "r"((uint64_t)w3_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w4) : "r"((uint64_t)w4_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w5) : "r"((uint64_t)w5_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w6) : "r"((uint64_t)w6_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w7) : "r"((uint64_t)w7_u.u));

                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a0) : "f"(v_pkg), "f"(w0));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a1) : "f"(v_pkg), "f"(w1));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a2) : "f"(v_pkg), "f"(w2));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a3) : "f"(v_pkg), "f"(w3));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a4) : "f"(v_pkg), "f"(w4));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a5) : "f"(v_pkg), "f"(w5));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a6) : "f"(v_pkg), "f"(w6));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a7) : "f"(v_pkg), "f"(w7));

                    // Batch 2 (8-15)
                    w0_u.f = W[(oc0 + 8) * IC + ic];
                    w1_u.f = W[(oc0 + 9) * IC + ic];
                    w2_u.f = W[(oc0 + 10) * IC + ic];
                    w3_u.f = W[(oc0 + 11) * IC + ic];
                    w4_u.f = W[(oc0 + 12) * IC + ic];
                    w5_u.f = W[(oc0 + 13) * IC + ic];
                    w6_u.f = W[(oc0 + 14) * IC + ic];
                    w7_u.f = W[(oc0 + 15) * IC + ic];
                    
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w0) : "r"((uint64_t)w0_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w1) : "r"((uint64_t)w1_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w2) : "r"((uint64_t)w2_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w3) : "r"((uint64_t)w3_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w4) : "r"((uint64_t)w4_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w5) : "r"((uint64_t)w5_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w6) : "r"((uint64_t)w6_u.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w7) : "r"((uint64_t)w7_u.u));

                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a8) : "f"(v_pkg), "f"(w0));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a9) : "f"(v_pkg), "f"(w1));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aA) : "f"(v_pkg), "f"(w2));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aB) : "f"(v_pkg), "f"(w3));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aC) : "f"(v_pkg), "f"(w4));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aD) : "f"(v_pkg), "f"(w5));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aE) : "f"(v_pkg), "f"(w6));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(aF) : "f"(v_pkg), "f"(w7));
                }

#define STORE_ACC(REG, OO) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    float *dst = out + ((oc0 + OO) * H + oh) * W_ + ow8; \
    if (act == 1u) { \
        for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]); \
    } else { \
        for (int l = 0; l < 8; l++) dst[l] = acc_buf[l]; \
    } \
} while (0)
                STORE_ACC(a0, 0); STORE_ACC(a1, 1); STORE_ACC(a2, 2); STORE_ACC(a3, 3);
                STORE_ACC(a4, 4); STORE_ACC(a5, 5); STORE_ACC(a6, 6); STORE_ACC(a7, 7);
                STORE_ACC(a8, 8); STORE_ACC(a9, 9); STORE_ACC(aA,10); STORE_ACC(aB,11);
                STORE_ACC(aC,12); STORE_ACC(aD,13); STORE_ACC(aE,14); STORE_ACC(aF,15);
#undef STORE_ACC
            }
        }
    }
    if (tile_hi > tile_lo) {
        const uint32_t oc_lo = tile_lo * 16u;
        const uint32_t bytes = (tile_hi - tile_lo) * 16u * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

/* Dispatcher: OC>=128 -> OC16, OC>=64 -> OC8, else per-OC. */
static inline void conv2d_1x1_disp(uint32_t hid,
                                   const float *in, float *out,
                                   const float *W, const float *B,
                                   uint32_t IC, uint32_t H, uint32_t W_,
                                   uint32_t OC,
                                   uint32_t act)
{
    /* Fixed OC16 RAW pipeline hazard! Safe to use for OC>=128 */
    if   (OC >= 128u) conv2d_1x1_fp32_mh_vpu_oc16(hid, in, out, W, B, IC, H, W_, OC, act);
    else if(OC >= 64u)conv2d_1x1_fp32_mh_vpu_oc8 (hid, in, out, W, B, IC, H, W_, OC, act);
    else              conv2d_1x1_fp32_mh_vpu    (hid, in, out, W, B, IC, H, W_, OC, act);
}
#define CONV_1x1(...) do { conv2d_1x1_disp(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* VPU-vectorized 3x3 Conv2d (stride=1, pad=1, OW % 8 == 0).
 * Adapted from depth-anything M10 conv3x3_pad1_fp32_vpu, multi-hart by OC.
 */
static void conv2d_3x3_p1_fp32_mh_vpu(uint32_t hid,
                                      const float *in, float *out,
                                      const float *W, const float *B,
                                      uint32_t IC, uint32_t H, uint32_t W_,
                                      uint32_t OC,
                                      uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t oc_lo, oc_hi;
    *(volatile uint32_t *)&oc_lo = (OC * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&oc_hi = (OC * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t oc = oc_lo; oc < oc_hi; oc++) {
        const float bias_v = B[oc];
        union { float f; uint32_t u; } bb; bb.f = bias_v;
        for (int32_t oh = 0; oh < (int32_t)H; oh++) {
            int32_t is_h_edge = (oh == 0) || (oh == (int32_t)H - 1);
            for (int32_t ow8 = 0; ow8 < (int32_t)W_; ow8 += 8) {
                int32_t is_w_edge = (ow8 == 0) || (ow8 + 8 > (int32_t)W_ - 1);
                register float acc asm("f0");
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc) : "r"((uint64_t)bb.u));

                if (!is_h_edge && !is_w_edge) {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        for (uint32_t ky = 0; ky < 3u; ky++) {
                            const int32_t ih = oh + (int32_t)ky - 1;
                            for (uint32_t kx = 0; kx < 3u; kx++) {
                                const int32_t iw = ow8 + (int32_t)kx - 1;
                                const float w_scalar = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                                float v_pkg;
                                float w_pkg;
                                union { float f; uint32_t u; } ww; ww.f = w_scalar;
                                const float *src = in + (ic * H + (uint32_t)ih) * W_ + (uint32_t)iw;
                                __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                                 : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                            }
                        }
                    }
                } else {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        for (uint32_t ky = 0; ky < 3u; ky++) {
                            const int32_t ih = oh + (int32_t)ky - 1;
                            if (ih < 0 || ih >= (int32_t)H) continue;
                            for (uint32_t kx = 0; kx < 3u; kx++) {
                                const int32_t iw = ow8 + (int32_t)kx - 1;
                                const float w_scalar = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                                if (iw >= 0 && iw + 7 < (int32_t)W_) {
                                    float v_pkg;
                                    float w_pkg;
                                    union { float f; uint32_t u; } ww; ww.f = w_scalar;
                                    const float *src = in + (ic * H + (uint32_t)ih) * W_ + (uint32_t)iw;
                                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                                     : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                                } else {
                                    /* Edge: dump acc, scalar-update each of 8 lanes, reload */
                                    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                                    __asm__ volatile("fence rw, rw" ::: "memory");
                                    for (int lane = 0; lane < 8; lane++) {
                                        const int32_t iw_l = ow8 + lane + (int32_t)kx - 1;
                                        if (iw_l >= 0 && iw_l < (int32_t)W_) {
                                            acc_buf[lane] += in[(ic * H + (uint32_t)ih) * W_ + (uint32_t)iw_l] * w_scalar;
                                        }
                                    }
                                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(acc) : "r"(acc_buf));
                                }
                            }
                        }
                    }
                }

                /* Store and apply scalar SiLU per lane. */
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                __asm__ volatile("fence rw, rw" ::: "memory");
                float *dst = out + (oc * H + (uint32_t)oh) * W_ + (uint32_t)ow8;
                if (act == 1u) {
                    for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]);
                } else {
                    for (int l = 0; l < 8; l++) dst[l] = acc_buf[l];
                }
            }
        }
    }
    if (oc_hi > oc_lo) {
        const uint32_t bytes = (oc_hi - oc_lo) * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

#define CONV_3x3_P1_VPU(...) do { conv2d_3x3_p1_fp32_mh_vpu(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* OC-blocked VPU 3x3 stride=1 pad=1.  8 OC accumulated simultaneously per
 * (oh, ow8) tile - input v_pkg is loaded once per (ic, ky, kx, ow8) and
 * reused across all 8 oc lanes. */
static void conv2d_3x3_p1_fp32_mh_vpu_oc8(uint32_t hid,
                                          const float *in, float *out,
                                          const float *W, const float *B,
                                          uint32_t IC, uint32_t H, uint32_t W_,
                                          uint32_t OC,
                                          uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t oc_tiles = OC / 8u;
    uint32_t tile_lo, tile_hi;
    *(volatile uint32_t *)&tile_lo = (oc_tiles * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&tile_hi = (oc_tiles * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t tile = tile_lo; tile < tile_hi; tile++) {
        const uint32_t oc0 = tile * 8u;
        for (int32_t oh = 0; oh < (int32_t)H; oh++) {
            for (int32_t ow8 = 0; ow8 < (int32_t)W_; ow8 += 8) {
                register float a0 asm("f0"), a1 asm("f1"), a2 asm("f2"), a3 asm("f3");
                register float a4 asm("f4"), a5 asm("f5"), a6 asm("f6"), a7 asm("f7");
#define INIT_ACC(REG, OO) do { \
    union { float f; uint32_t u; } _bb; _bb.f = B[oc0 + OO]; \
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(REG) : "r"((uint64_t)_bb.u)); \
} while (0)
                INIT_ACC(a0, 0); INIT_ACC(a1, 1); INIT_ACC(a2, 2); INIT_ACC(a3, 3);
                INIT_ACC(a4, 4); INIT_ACC(a5, 5); INIT_ACC(a6, 6); INIT_ACC(a7, 7);
#undef INIT_ACC

                for (uint32_t ic = 0; ic < IC; ic++) {
                    for (uint32_t ky = 0; ky < 3u; ky++) {
                        const int32_t ih = oh + (int32_t)ky - 1;
                        if (ih < 0 || ih >= (int32_t)H) continue;
                        for (uint32_t kx = 0; kx < 3u; kx++) {
                            const int32_t iw = ow8 + (int32_t)kx - 1;
                            float v_pkg;
                            float w_pkg;
                            if (iw >= 0 && iw + 7 < (int32_t)W_) {
                                const float *src = in + (ic * H + (uint32_t)ih) * W_ + (uint32_t)iw;
                                register float w0 asm("f20"), w1 asm("f21"), w2 asm("f22"), w3 asm("f23");
                                register float w4 asm("f24"), w5 asm("f25"), w6 asm("f26"), w7 asm("f27");
                                
                                union { float f; uint32_t u; } w0_u; w0_u.f = W[((oc0 + 0) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w1_u; w1_u.f = W[((oc0 + 1) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w2_u; w2_u.f = W[((oc0 + 2) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w3_u; w3_u.f = W[((oc0 + 3) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w4_u; w4_u.f = W[((oc0 + 4) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w5_u; w5_u.f = W[((oc0 + 5) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w6_u; w6_u.f = W[((oc0 + 6) * IC + ic) * 9u + ky * 3u + kx];
                                union { float f; uint32_t u; } w7_u; w7_u.f = W[((oc0 + 7) * IC + ic) * 9u + ky * 3u + kx];
                                
                                __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                                
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w0) : "r"((uint64_t)w0_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w1) : "r"((uint64_t)w1_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w2) : "r"((uint64_t)w2_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w3) : "r"((uint64_t)w3_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w4) : "r"((uint64_t)w4_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w5) : "r"((uint64_t)w5_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w6) : "r"((uint64_t)w6_u.u));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w7) : "r"((uint64_t)w7_u.u));

                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a0) : "f"(v_pkg), "f"(w0));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a1) : "f"(v_pkg), "f"(w1));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a2) : "f"(v_pkg), "f"(w2));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a3) : "f"(v_pkg), "f"(w3));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a4) : "f"(v_pkg), "f"(w4));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a5) : "f"(v_pkg), "f"(w5));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a6) : "f"(v_pkg), "f"(w6));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(a7) : "f"(v_pkg), "f"(w7));
                            } else {
                                /* Edge case: scalar lane updates of all 8 accs. */
#define EDGE_ONE(REG, OO) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    const float w_scalar = W[((oc0 + OO) * IC + ic) * 9u + ky * 3u + kx]; \
    for (int lane = 0; lane < 8; lane++) { \
        const int32_t iw_l = ow8 + lane + (int32_t)kx - 1; \
        if (iw_l >= 0 && iw_l < (int32_t)W_) { \
            acc_buf[lane] += in[(ic * H + (uint32_t)ih) * W_ + (uint32_t)iw_l] * w_scalar; \
        } \
    } \
    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(REG) : "r"(acc_buf)); \
} while (0)
                                EDGE_ONE(a0, 0); EDGE_ONE(a1, 1); EDGE_ONE(a2, 2); EDGE_ONE(a3, 3);
                                EDGE_ONE(a4, 4); EDGE_ONE(a5, 5); EDGE_ONE(a6, 6); EDGE_ONE(a7, 7);
#undef EDGE_ONE
                            }
                        }
                    }
                }

#define STORE_ACC(REG, OO) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    float *dst = out + ((oc0 + OO) * H + (uint32_t)oh) * W_ + (uint32_t)ow8; \
    if (act == 1u) { \
        for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]); \
    } else { \
        for (int l = 0; l < 8; l++) dst[l] = acc_buf[l]; \
    } \
} while (0)
                STORE_ACC(a0, 0); STORE_ACC(a1, 1); STORE_ACC(a2, 2); STORE_ACC(a3, 3);
                STORE_ACC(a4, 4); STORE_ACC(a5, 5); STORE_ACC(a6, 6); STORE_ACC(a7, 7);
#undef STORE_ACC
            }
        }
    }
    if (tile_hi > tile_lo) {
        const uint32_t oc_lo = tile_lo * 8u;
        const uint32_t bytes = (tile_hi - tile_lo) * 8u * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

/* VPU 3x3 stride=2 pad=1.  Process 4 output cols per VPU iteration via an
 * 8-lane fmadd.ps where only lanes 0, 2, 4, 6 carry valid output-aligned
 * input data - lanes 1/3/5/7 compute garbage and we discard them on store.
 * Net VPU speedup: 4x (vs 8x for stride=1).  Constraint: OW % 4 == 0
 * (true for 16, 32, 64, 128, 256). */
static void conv2d_3x3_s2_p1_fp32_mh_vpu(uint32_t hid,
                                          const float *in, float *out,
                                          const float *W, const float *B,
                                          uint32_t IC, uint32_t IH, uint32_t IW,
                                          uint32_t OC, uint32_t OH, uint32_t OW,
                                          uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t oc_lo, oc_hi;
    *(volatile uint32_t *)&oc_lo = (OC * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&oc_hi = (OC * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t oc = oc_lo; oc < oc_hi; oc++) {
        const float bias_v = B[oc];
        union { float f; uint32_t u; } bb; bb.f = bias_v;
        for (int32_t oh = 0; oh < (int32_t)OH; oh++) {
            int32_t is_h_edge = (oh == 0); // No bottom edge because stride 2 padding 1 never overshoots by 2
            for (int32_t ow4 = 0; ow4 < (int32_t)OW; ow4 += 4) {
                int32_t is_w_edge = (ow4 == 0) || (ow4 * 2 + 8 > (int32_t)IW - 1);
                register float acc asm("f0");
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc) : "r"((uint64_t)bb.u));

                if (!is_h_edge && !is_w_edge) {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        for (uint32_t ky = 0; ky < 3u; ky++) {
                            const int32_t ih = oh * 2 + (int32_t)ky - 1;
                            for (uint32_t kx = 0; kx < 3u; kx++) {
                                const int32_t iw_base = ow4 * 2 + (int32_t)kx - 1;
                                const float w_scalar = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                                float v_pkg;
                                float w_pkg;
                                union { float f; uint32_t u; } ww; ww.f = w_scalar;
                                const float *src = in + (ic * IH + (uint32_t)ih) * IW + (uint32_t)iw_base;
                                __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                                 : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                            }
                        }
                    }
                } else {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        for (uint32_t ky = 0; ky < 3u; ky++) {
                            const int32_t ih = oh * 2 + (int32_t)ky - 1;
                            if (ih < 0 || ih >= (int32_t)IH) continue;
                            for (uint32_t kx = 0; kx < 3u; kx++) {
                                const int32_t iw_base = ow4 * 2 + (int32_t)kx - 1;  /* lane-0 input col */
                                const float w_scalar = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                                if (iw_base >= 0 && iw_base + 7 < (int32_t)IW) {
                                    /* Fast path: 8 contiguous input cols loaded; only
                                     * even lanes (0,2,4,6) contribute to valid output. */
                                    float v_pkg;
                                    float w_pkg;
                                    union { float f; uint32_t u; } ww; ww.f = w_scalar;
                                    const float *src = in + (ic * IH + (uint32_t)ih) * IW + (uint32_t)iw_base;
                                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                                     : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                                } else {
                                    /* Edge: scalar update of even lanes. */
                                    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                                    __asm__ volatile("fence rw, rw" ::: "memory");
                                    for (int lane = 0; lane < 4; lane++) {
                                        const int32_t iw_l = iw_base + 2 * lane;
                                        if (iw_l >= 0 && iw_l < (int32_t)IW) {
                                            acc_buf[2 * lane] += in[(ic * IH + (uint32_t)ih) * IW + (uint32_t)iw_l] * w_scalar;
                                        }
                                    }
                                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(acc) : "r"(acc_buf));
                                }
                            }
                        }
                    }
                }

                /* Store the 4 valid lanes to out[oc, oh, ow4..ow4+3]. */
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                __asm__ volatile("fence rw, rw" ::: "memory");
                float *dst = out + (oc * OH + (uint32_t)oh) * OW + (uint32_t)ow4;
                if (act == 1u) {
                    dst[0] = silu(acc_buf[0]); dst[1] = silu(acc_buf[2]);
                    dst[2] = silu(acc_buf[4]); dst[3] = silu(acc_buf[6]);
                } else {
                    dst[0] = acc_buf[0]; dst[1] = acc_buf[2];
                    dst[2] = acc_buf[4]; dst[3] = acc_buf[6];
                }
            }
        }
    }
    if (oc_hi > oc_lo) {
        const uint32_t bytes = (oc_hi - oc_lo) * OH * OW * sizeof(float);
        evict((const void *)(out + oc_lo * OH * OW), bytes);
    }
}
#define CONV_3x3_S2_P1_VPU(...) do { conv2d_3x3_s2_p1_fp32_mh_vpu(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* OC4-blocked VPU 3x3 stride=1 pad=1.  4 OC accumulated simultaneously per
 * (oh, ow8) tile.  Lower register pressure than OC8 to avoid the M18 hang. */
static void conv2d_3x3_p1_fp32_mh_vpu_oc4(uint32_t hid,
                                          const float *in, float *out,
                                          const float *W, const float *B,
                                          uint32_t IC, uint32_t H, uint32_t W_,
                                          uint32_t OC,
                                          uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t oc_tiles = OC / 4u;
    uint32_t tile_lo, tile_hi;
    *(volatile uint32_t *)&tile_lo = (oc_tiles * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&tile_hi = (oc_tiles * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t tile = tile_lo; tile < tile_hi; tile++) {
        const uint32_t oc0 = tile * 4u;
        for (int32_t oh = 0; oh < (int32_t)H; oh++) {
            for (int32_t ow8 = 0; ow8 < (int32_t)W_; ow8 += 8) {
                register float a0 asm("f0"), a1 asm("f1"), a2 asm("f2"), a3 asm("f3");
#define INIT_ACC(REG, OO) do { \
    union { float f; uint32_t u; } _bb; _bb.f = B[oc0 + OO]; \
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(REG) : "r"((uint64_t)_bb.u)); \
} while (0)
                INIT_ACC(a0, 0); INIT_ACC(a1, 1); INIT_ACC(a2, 2); INIT_ACC(a3, 3);
#undef INIT_ACC

                for (uint32_t ic = 0; ic < IC; ic++) {
                    for (uint32_t ky = 0; ky < 3u; ky++) {
                        const int32_t ih = oh + (int32_t)ky - 1;
                        if (ih < 0 || ih >= (int32_t)H) continue;
                        for (uint32_t kx = 0; kx < 3u; kx++) {
                            const int32_t iw = ow8 + (int32_t)kx - 1;
                            float v_pkg;
                            float w_pkg;
                            if (iw >= 0 && iw + 7 < (int32_t)W_) {
                                const float *src = in + (ic * H + (uint32_t)ih) * W_ + (uint32_t)iw;
                                __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
#define FMADD_ONE(REG, OO) do { \
    union { float f; uint32_t u; } _ww; _ww.f = W[((oc0 + OO) * IC + ic) * 9u + ky * 3u + kx]; \
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)_ww.u)); \
    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(REG) : "f"(v_pkg), "f"(w_pkg)); \
} while (0)
                                FMADD_ONE(a0, 0); FMADD_ONE(a1, 1); FMADD_ONE(a2, 2); FMADD_ONE(a3, 3);
#undef FMADD_ONE
                            } else {
#define EDGE_ONE(REG, OO) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    const float w_scalar = W[((oc0 + OO) * IC + ic) * 9u + ky * 3u + kx]; \
    for (int lane = 0; lane < 8; lane++) { \
        const int32_t iw_l = ow8 + lane + (int32_t)kx - 1; \
        if (iw_l >= 0 && iw_l < (int32_t)W_) { \
            acc_buf[lane] += in[(ic * H + (uint32_t)ih) * W_ + (uint32_t)iw_l] * w_scalar; \
        } \
    } \
    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(REG) : "r"(acc_buf)); \
} while (0)
                                EDGE_ONE(a0, 0); EDGE_ONE(a1, 1); EDGE_ONE(a2, 2); EDGE_ONE(a3, 3);
#undef EDGE_ONE
                            }
                        }
                    }
                }

#define STORE_ACC(REG, OO) do { \
    __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(REG) : "memory"); \
    __asm__ volatile("fence rw, rw" ::: "memory"); \
    float *dst = out + ((oc0 + OO) * H + (uint32_t)oh) * W_ + (uint32_t)ow8; \
    if (act == 1u) { \
        for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]); \
    } else { \
        for (int l = 0; l < 8; l++) dst[l] = acc_buf[l]; \
    } \
} while (0)
                STORE_ACC(a0, 0); STORE_ACC(a1, 1); STORE_ACC(a2, 2); STORE_ACC(a3, 3);
#undef STORE_ACC
            }
        }
    }
    if (tile_hi > tile_lo) {
        const uint32_t oc_lo = tile_lo * 4u;
        const uint32_t bytes = (tile_hi - tile_lo) * 4u * H * W_ * sizeof(float);
        evict((const void *)(out + oc_lo * H * W_), bytes);
    }
}

static inline void conv2d_3x3_p1_disp(uint32_t hid,
                                      const float *in, float *out,
                                      const float *W, const float *B,
                                      uint32_t IC, uint32_t H, uint32_t W_,
                                      uint32_t OC,
                                      uint32_t act)
{
    /* OC8 hangs the silicon (M18); use OC4 for OC>=32, per-OC for smaller. */
    if (OC >= 32u) conv2d_3x3_p1_fp32_mh_vpu_oc4(hid, in, out, W, B, IC, H, W_, OC, act);
    else           conv2d_3x3_p1_fp32_mh_vpu    (hid, in, out, W, B, IC, H, W_, OC, act);
}
#define CONV_3x3_P1(...) do { conv2d_3x3_p1_disp(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* VPU-vectorized depthwise 3x3 (stride=1 pad=1, OW % 8 == 0). */
static void conv2d_dw3x3_s1_p1_fp32_mh_vpu(uint32_t hid,
                                           const float *in, float *out,
                                           const float *W, const float *B,
                                           uint32_t C, uint32_t H, uint32_t W_,
                                           uint32_t act)
{
    if (!mh_is_t0(hid)) return;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t c_lo, c_hi;
    *(volatile uint32_t *)&c_lo = (C * cidx) / MH_NUM_T0;
    *(volatile uint32_t *)&c_hi = (C * (cidx + 1u)) / MH_NUM_T0;

    float acc_buf[8] __attribute__((aligned(32)));

    for (uint32_t c = c_lo; c < c_hi; c++) {
        const float bias_v = B[c];
        union { float f; uint32_t u; } bb; bb.f = bias_v;
        const float *wp = W + c * 9u;   /* 3x3 weights for channel c */
        for (int32_t oh = 0; oh < (int32_t)H; oh++) {
            for (int32_t ow8 = 0; ow8 < (int32_t)W_; ow8 += 8) {
                register float acc asm("f0");
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc) : "r"((uint64_t)bb.u));

                for (uint32_t ky = 0; ky < 3u; ky++) {
                    const int32_t ih = oh + (int32_t)ky - 1;
                    if (ih < 0 || ih >= (int32_t)H) continue;
                    for (uint32_t kx = 0; kx < 3u; kx++) {
                        const int32_t iw = ow8 + (int32_t)kx - 1;
                        const float w_scalar = wp[ky * 3u + kx];
                        if (iw >= 0 && iw + 7 < (int32_t)W_) {
                            float v_pkg;
                            float w_pkg;
                            union { float f; uint32_t u; } ww; ww.f = w_scalar;
                            const float *src = in + (c * H + (uint32_t)ih) * W_ + (uint32_t)iw;
                            __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(v_pkg) : "r"(src));
                            __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(w_pkg) : "r"((uint64_t)ww.u));
                            __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                             : "+f"(acc) : "f"(v_pkg), "f"(w_pkg));
                        } else {
                            __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                            __asm__ volatile("fence rw, rw" ::: "memory");
                            for (int lane = 0; lane < 8; lane++) {
                                const int32_t iw_l = ow8 + lane + (int32_t)kx - 1;
                                if (iw_l >= 0 && iw_l < (int32_t)W_) {
                                    acc_buf[lane] += in[(c * H + (uint32_t)ih) * W_ + (uint32_t)iw_l] * w_scalar;
                                }
                            }
                            __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(acc) : "r"(acc_buf));
                        }
                    }
                }

                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(acc_buf), "f"(acc) : "memory");
                __asm__ volatile("fence rw, rw" ::: "memory");
                float *dst = out + (c * H + (uint32_t)oh) * W_ + (uint32_t)ow8;
                if (act == 1u) {
                    for (int l = 0; l < 8; l++) dst[l] = silu(acc_buf[l]);
                } else {
                    for (int l = 0; l < 8; l++) dst[l] = acc_buf[l];
                }
            }
        }
    }
    if (c_hi > c_lo) {
        const uint32_t bytes = (c_hi - c_lo) * H * W_ * sizeof(float);
        evict((const void *)(out + c_lo * H * W_), bytes);
    }
}

#define CONV_DW3x3_S1_P1_VPU(...) do { conv2d_dw3x3_s1_p1_fp32_mh_vpu(hid, __VA_ARGS__); MH_BARRIER(); } while (0)

/* -- Multi-hart helpers for the residual / concat / activation tail -- */

/* mh_copy_floats: split N float copies across all compute harts. */
static inline void mh_copy_floats(uint32_t hid, float *dst, const float *src, uint32_t N) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t lo, hi;
    yolo_range(N, cidx, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) dst[i] = src[i];
    if (hi > lo) evict((const void *)(dst + lo), (hi - lo) * sizeof(float));
}

/* mh_add_floats: y = a + b, multi-hart. */
static inline void mh_add_floats(uint32_t hid, float *y, const float *a, const float *b, uint32_t N) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t lo, hi;
    yolo_range(N, cidx, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) y[i] = a[i] + b[i];
    if (hi > lo) evict((const void *)(y + lo), (hi - lo) * sizeof(float));
}

/* mh_iadd_floats: y += b, in place, multi-hart. */
static inline void mh_iadd_floats(uint32_t hid, float *y, const float *b, uint32_t N) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t lo, hi;
    yolo_range(N, cidx, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) y[i] += b[i];
    if (hi > lo) evict((const void *)(y + lo), (hi - lo) * sizeof(float));
}

/* Multi-hart concat of 3 same-shape blocks into 3*N floats. */
static inline void mh_concat3(uint32_t hid, float *dst,
                              const float *a, const float *b, const float *c,
                              uint32_t N) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t lo, hi;
    yolo_range(N, cidx, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) dst[          i] = a[i];
    for (uint32_t i = lo; i < hi; i++) dst[1u*N   + i] = b[i];
    for (uint32_t i = lo; i < hi; i++) dst[2u*N   + i] = c[i];
    if (hi > lo) {
        evict((const void *)(dst + lo),         (hi - lo) * sizeof(float));
        evict((const void *)(dst + 1u*N + lo),  (hi - lo) * sizeof(float));
        evict((const void *)(dst + 2u*N + lo),  (hi - lo) * sizeof(float));
    }
}

/* Multi-hart concat of 4 same-shape blocks (used by SPPF). */
static inline void mh_concat4(uint32_t hid, float *dst,
                              const float *a, const float *b,
                              const float *c, const float *d,
                              uint32_t N) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t lo, hi;
    yolo_range(N, cidx, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) dst[          i] = a[i];
    for (uint32_t i = lo; i < hi; i++) dst[1u*N   + i] = b[i];
    for (uint32_t i = lo; i < hi; i++) dst[2u*N   + i] = c[i];
    for (uint32_t i = lo; i < hi; i++) dst[3u*N   + i] = d[i];
    if (hi > lo) {
        evict((const void *)(dst + lo),         (hi - lo) * sizeof(float));
        evict((const void *)(dst + 1u*N + lo),  (hi - lo) * sizeof(float));
        evict((const void *)(dst + 2u*N + lo),  (hi - lo) * sizeof(float));
        evict((const void *)(dst + 3u*N + lo),  (hi - lo) * sizeof(float));
    }
}

#define MH_COPY(DST, SRC, N)             do { mh_copy_floats(hid, (DST), (SRC), (N)); MH_BARRIER(); } while (0)
#define MH_ADD(Y, A, B, N)               do { mh_add_floats(hid, (Y), (A), (B), (N)); MH_BARRIER(); } while (0)
#define MH_IADD(Y, B, N)                 do { mh_iadd_floats(hid, (Y), (B), (N)); MH_BARRIER(); } while (0)
#define MH_CONCAT3(DST, A, B, C, N)      do { mh_concat3(hid, (DST), (A), (B), (C), (N)); MH_BARRIER(); } while (0)
#define MH_CONCAT4(DST, A, B, C, D, N)   do { mh_concat4(hid, (DST), (A), (B), (C), (D), (N)); MH_BARRIER(); } while (0)

/* Multi-hart 5x5 maxpool stride=1 pad=2 (used in SPPF). */
static void mh_maxpool5_s1_p2(uint32_t hid, const float *in, float *out,
                              uint32_t C, uint32_t H, uint32_t W) {
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t c_lo, c_hi;
    yolo_range(C, cidx, &c_lo, &c_hi);
    for (uint32_t c = c_lo; c < c_hi; c++) {
        for (uint32_t oh = 0; oh < H; oh++) {
            for (uint32_t ow = 0; ow < W; ow++) {
                float m = -3.4e38f;
                for (uint32_t ky = 0; ky < 5u; ky++) {
                    const int32_t ih = (int32_t)oh - 2 + (int32_t)ky;
                    if (ih < 0 || ih >= (int32_t)H) continue;
                    for (uint32_t kx = 0; kx < 5u; kx++) {
                        const int32_t iw = (int32_t)ow - 2 + (int32_t)kx;
                        if (iw < 0 || iw >= (int32_t)W) continue;
                        const float v = in[(c * H + (uint32_t)ih) * W + (uint32_t)iw];
                        if (v > m) m = v;
                    }
                }
                out[(c * H + oh) * W + ow] = m;
            }
        }
    }
    if (c_hi > c_lo) evict((const void *)(out + c_lo * H * W), (c_hi - c_lo) * H * W * sizeof(float));
}
#define MH_MAXPOOL5(IN, OUT, C, H, W) do { mh_maxpool5_s1_p2(hid, (IN), (OUT), (C), (H), (W)); MH_BARRIER(); } while (0)

/* Multi-hart depthwise Conv2d (groups=C). */
static void conv2d_dw_fp32_mh(uint32_t hid,
                              const float *in, float *out,
                              const float *W, const float *B,
                              uint32_t C, uint32_t IH, uint32_t IW,
                              uint32_t OH, uint32_t OW,
                              uint32_t KH, uint32_t KW,
                              uint32_t SH, uint32_t SW,
                              uint32_t PH, uint32_t PW,
                              uint32_t act)
{
    if (!yolo_is_compute(hid)) return;
    const uint32_t cidx = yolo_compute_idx(hid);
    uint32_t c_lo, c_hi;
    yolo_range(C, cidx, &c_lo, &c_hi);

    for (uint32_t c = c_lo; c < c_hi; c++) {
        const float bias = B[c];
        for (uint32_t oh = 0; oh < OH; oh++) {
            for (uint32_t ow = 0; ow < OW; ow++) {
                float acc = bias;
                for (uint32_t ky = 0; ky < KH; ky++) {
                    const int32_t ih = (int32_t)(oh * SH) - (int32_t)PH + (int32_t)ky;
                    if (ih < 0 || ih >= (int32_t)IH) continue;
                    for (uint32_t kx = 0; kx < KW; kx++) {
                        const int32_t iw = (int32_t)(ow * SW) - (int32_t)PW + (int32_t)kx;
                        if (iw < 0 || iw >= (int32_t)IW) continue;
                        const float v = in[(c * IH + (uint32_t)ih) * IW + (uint32_t)iw];
                        const float w = W[(c * KH + ky) * KW + kx];
                        acc += w * v;
                    }
                }
                if (act == 1u) acc = silu(acc);
                out[(c * OH + oh) * OW + ow] = acc;
            }
        }
    }
    if (c_hi > c_lo) {
        const uint32_t bytes = (c_hi - c_lo) * OH * OW * sizeof(float);
        evict((const void *)(out + c_lo * OH * OW), bytes);
    }
}

/* Matmul helpers for batched-2D PSA attention.
 * matmul_2d: [M,K] @ [K,N] -> [M,N]
 */
static inline void matmul_2d_fp32(const float *A, const float *B, float *C,
                                  uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
    }
}

/* Softmax over rows (last axis): for each row of length N, compute
 * x = exp(x - max(x)) / sum(exp(x - max(x))) */
static inline void softmax_rows(float *x, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        float *row = x + i * N;
        float m = row[0];
        for (uint32_t j = 1; j < N; j++) if (row[j] > m) m = row[j];
        float s = 0.0f;
        for (uint32_t j = 0; j < N; j++) { row[j] = my_expf(row[j] - m); s += row[j]; }
        const float inv = fast_recip(s);
        for (uint32_t j = 0; j < N; j++) row[j] *= inv;
    }
}

/* Nearest-neighbor upsample 2x: [C, H, W] -> [C, 2H, 2W] */
static inline void upsample_nearest_2x(const float *in, float *out,
                                       uint32_t C, uint32_t H, uint32_t W)
{
    const uint32_t OH = H * 2u, OW = W * 2u;
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < OH; oh++) {
            const uint32_t ih = oh / 2u;
            for (uint32_t ow = 0; ow < OW; ow++) {
                const uint32_t iw = ow / 2u;
                out[(c * OH + oh) * OW + ow] = in[(c * H + ih) * W + iw];
            }
        }
    }
}

/* Transpose last two axes of a 2D tile: [M,N] -> [N,M] */
static inline void transpose_2d(const float *in, float *out, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++)
            out[j*M + i] = in[i*N + j];
}

#endif
