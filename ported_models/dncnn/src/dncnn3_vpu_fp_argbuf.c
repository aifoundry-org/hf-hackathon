/*
 * DnCNN-shaped VPU FP32 benchmark.
 *
 * This is intentionally separate from the int8 scalar DnCNN checksum path.  It
 * keeps the same 64x64x16/layer structure, but stores activations as FP32 and
 * uses Erbium VPU packed-single instructions for the dense hidden/final
 * channel dot products.
 */

#include <stdint.h>

#include "erbium/isa/atomic.h"
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 1u
#endif

#ifndef DNCNN_PASSES
#define DNCNN_PASSES 1u
#endif

#define DNCNN_MAGIC       0xD3C11003u
#define IMG_W             64u
#define IMG_H             64u
#define CH                16u
#define LAYERS            5u
#define K                 3u
#define IMG_BYTES         (IMG_W * IMG_H)
#define ACT_BYTES         (IMG_W * IMG_H * CH * sizeof(float))
#define W0_BYTES          (CH * K * K)
#define WH_BYTES          ((LAYERS - 2u) * CH * CH * K * K)
#define WF_BYTES          (CH * K * K)
#define WEIGHTS_BYTES     (W0_BYTES + WH_BYTES + WF_BYTES)
#define WPACK_FLOATS      (WH_BYTES + WF_BYTES)
#define FIRST_SCALE       (1.0f / 128.0f)
#define HIDDEN_SCALE      (1.0f / 256.0f)
#define FINAL_SCALE       (1.0f / 16.0f)

#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x0000u
#define SUMMARY_OFFSET    0x1000u
#define BARRIER_OFFSET    0x1800u
#define INPUT_OFFSET      0x2000u
#define WEIGHTS_OFFSET    0x4000u
#define OUTPUT_OFFSET     0x10000u
#define ACT0_OFFSET       0x20000u
#define ACT1_OFFSET       (ACT0_OFFSET + ACT_BYTES)
#define WPACK_OFFSET      (ACT1_OFFSET + ACT_BYTES)
#define WPACK_STRIDE      (WPACK_FLOATS * sizeof(float))
#define PRIVATE_OFFSET    0x120000u
#define PRIVATE_STRIDE    (2u * ACT_BYTES)

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct dncnn_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t row0;
	uint32_t row1;
	uint32_t active_harts;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[7];
};

struct dncnn_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t passes;
	uint32_t width;
	uint32_t height;
	uint32_t channels;
	uint32_t layers;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t slot_checksum_sum;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t reserved[3];
};

struct bench_barrier_state {
	uint32_t count;
	uint32_t epoch;
	uint32_t reserved[14];
};

static volatile struct bench_barrier_state *g_barrier;

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	return ptr;
}

static inline uint32_t active_mask_t0(void)
{
#ifdef BENCH_THREAD0_ONLY
	if (ACTIVE_HARTS >= 32u) {
		return 0xffffffffu;
	}
	return (1u << ACTIVE_HARTS) - 1u;
#else
	uint32_t mask = 0;

	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
#endif
}

static inline uint32_t active_mask_t1(void)
{
#ifdef BENCH_THREAD0_ONLY
	return 0u;
#else
	uint32_t mask = 0;

	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
#endif
}

static inline uint32_t bench_hart_id(void)
{
#ifdef BENCH_THREAD0_ONLY
	return get_minion_id();
#else
	return get_hart_id() & 0x3fu;
#endif
}

static inline int bench_hart_enabled(uint32_t hart_id)
{
#ifdef BENCH_THREAD0_ONLY
	return get_thread_id() == 0u && hart_id < ACTIVE_HARTS && hart_id < 32u;
#else
	return hart_id < ACTIVE_HARTS && hart_id < 16u;
#endif
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
#ifdef BENCH_THREAD0_ONLY
		volatile struct bench_barrier_state *const barrier = g_barrier;
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
#else
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
#endif
	}
}

static inline int clamp_coord(int v, unsigned limit)
{
	if (v < 0) {
		return 0;
	}
	if (v >= (int)limit) {
		return (int)limit - 1;
	}
	return v;
}

static inline float relu_f32(float v)
{
	return v < 0.0f ? 0.0f : v;
}

static inline uint8_t clamp_u8_from_f32(float v)
{
	if (v < 0.0f) {
		return 0u;
	}
	if (v > 255.0f) {
		return 255u;
	}
	return (uint8_t)(v + 0.5f);
}

static inline float vpu_dot16_f32(const float *a, const float *b)
{
	__attribute__((aligned(32))) float tmp[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"flq2    f1, 0(%[a0])\n"
		"flq2    f2, 0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[a0])\n"
		"flq2    f2, 32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2    f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [b0] "r"(b), [tmp] "r"(tmp)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum += tmp[i];
	}

	return sum;
}

static inline float vpu_accum3x3_dot16_f32(const float *p, const float *w_oc,
					   int row)
{
	__attribute__((aligned(32))) float tmp[8];
	const uint64_t zero = 0;
	const float *const p0 = p - row - (int)CH;
	const float *const p1 = p - row;
	const float *const p2 = p - row + (int)CH;
	const float *const p3 = p - (int)CH;
	const float *const p4 = p;
	const float *const p5 = p + (int)CH;
	const float *const p6 = p + row - (int)CH;
	const float *const p7 = p + row;
	const float *const p8 = p + row + (int)CH;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"flq2    f1, 0(%[p0])\n"
		"flq2    f2, 0(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p0])\n"
		"flq2    f2, 32(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p1])\n"
		"flq2    f2, 64(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p1])\n"
		"flq2    f2, 96(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p2])\n"
		"flq2    f2, 128(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p2])\n"
		"flq2    f2, 160(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p3])\n"
		"flq2    f2, 192(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p3])\n"
		"flq2    f2, 224(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p4])\n"
		"flq2    f2, 256(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p4])\n"
		"flq2    f2, 288(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p5])\n"
		"flq2    f2, 320(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p5])\n"
		"flq2    f2, 352(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p6])\n"
		"flq2    f2, 384(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p6])\n"
		"flq2    f2, 416(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p7])\n"
		"flq2    f2, 448(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p7])\n"
		"flq2    f2, 480(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 0(%[p8])\n"
		"flq2    f2, 512(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[p8])\n"
		"flq2    f2, 544(%[w])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2    f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [p0] "r"(p0), [p1] "r"(p1),
		  [p2] "r"(p2), [p3] "r"(p3), [p4] "r"(p4),
		  [p5] "r"(p5), [p6] "r"(p6), [p7] "r"(p7),
		  [p8] "r"(p8), [w] "r"(w_oc), [tmp] "r"(tmp)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum += tmp[i];
	}

	return sum;
}

static inline void vpu_accum3x3_dot16x2_f32(const float *p,
					    const float *w0,
					    const float *w1,
					    int row,
					    float *out0,
					    float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;
	const float *const p0 = p - row - (int)CH;
	const float *const p1 = p - row;
	const float *const p2 = p - row + (int)CH;
	const float *const p3 = p - (int)CH;
	const float *const p4 = p;
	const float *const p5 = p + (int)CH;
	const float *const p6 = p + row - (int)CH;
	const float *const p7 = p + row;
	const float *const p8 = p + row + (int)CH;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"flq2    f1, 0(%[p0])\n"
		"flq2    f2, 0(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 0(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p0])\n"
		"flq2    f2, 32(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 32(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p1])\n"
		"flq2    f2, 64(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 64(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p1])\n"
		"flq2    f2, 96(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 96(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p2])\n"
		"flq2    f2, 128(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 128(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p2])\n"
		"flq2    f2, 160(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 160(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p3])\n"
		"flq2    f2, 192(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 192(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p3])\n"
		"flq2    f2, 224(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 224(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p4])\n"
		"flq2    f2, 256(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 256(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p4])\n"
		"flq2    f2, 288(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 288(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p5])\n"
		"flq2    f2, 320(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 320(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p5])\n"
		"flq2    f2, 352(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 352(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p6])\n"
		"flq2    f2, 384(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 384(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p6])\n"
		"flq2    f2, 416(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 416(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p7])\n"
		"flq2    f2, 448(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 448(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p7])\n"
		"flq2    f2, 480(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 480(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 0(%[p8])\n"
		"flq2    f2, 512(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 512(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[p8])\n"
		"flq2    f2, 544(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 544(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		:
		: [zero] "r"(zero), [p0] "r"(p0), [p1] "r"(p1),
		  [p2] "r"(p2), [p3] "r"(p3), [p4] "r"(p4),
		  [p5] "r"(p5), [p6] "r"(p6), [p7] "r"(p7),
		  [p8] "r"(p8), [w0] "r"(w0), [w1] "r"(w1),
		  [tmp0] "r"(tmp0), [tmp1] "r"(tmp1)
		: "memory", "f0", "f1", "f2", "f3", "f4");

	float sum0 = 0.0f;
	float sum1 = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum0 += tmp0[i];
		sum1 += tmp1[i];
	}

	*out0 = sum0;
	*out1 = sum1;
}

static void pack_weights(const int8_t *weights, float *wpack)
{
	const int8_t *hidden = weights + W0_BYTES;

	for (uint32_t layer = 0; layer < (LAYERS - 2u); layer++) {
		float *dst_layer = wpack + layer * CH * K * K * CH;
		const int8_t *src_layer = hidden + layer * CH * CH * K * K;

		for (uint32_t oc = 0; oc < CH; oc++) {
			for (uint32_t k = 0; k < K * K; k++) {
				float *dst = dst_layer + (oc * K * K + k) * CH;

				for (uint32_t ic = 0; ic < CH; ic++) {
					dst[ic] = (float)src_layer[oc * CH * K * K +
								  ic * K * K + k];
				}
			}
		}
	}

	const int8_t *final = weights + W0_BYTES + WH_BYTES;
	float *final_dst = wpack + WH_BYTES;

	for (uint32_t k = 0; k < K * K; k++) {
		for (uint32_t ic = 0; ic < CH; ic++) {
			final_dst[k * CH + ic] = (float)final[ic * K * K + k];
		}
	}
}

static float hidden_acc_scalar(const float *input, const float *w_oc,
			       uint32_t y, uint32_t x)
{
	float acc = 0.0f;

	for (int ky = -1; ky <= 1; ky++) {
		const int yy = clamp_coord((int)y + ky, IMG_H);

		for (int kx = -1; kx <= 1; kx++) {
			const int xx = clamp_coord((int)x + kx, IMG_W);
			const float *pix = input + ((unsigned)yy * IMG_W +
						    (unsigned)xx) * CH;
			const float *w = w_oc + ((ky + 1) * 3 + (kx + 1)) * CH;

			for (uint32_t ic = 0; ic < CH; ic++) {
				acc += pix[ic] * w[ic];
			}
		}
	}

	return acc;
}

static float hidden_acc_vpu(const float *input, const float *w_oc,
			    uint32_t y, uint32_t x)
{
	const float *const p = input + (y * IMG_W + x) * CH;
	const int row = IMG_W * CH;

#ifdef DNCNN_VPU_ACCUM_3X3
	return vpu_accum3x3_dot16_f32(p, w_oc, row);
#else
	float acc = 0.0f;

	acc += vpu_dot16_f32(p - row - (int)CH, w_oc + 0u * CH);
	acc += vpu_dot16_f32(p - row, w_oc + 1u * CH);
	acc += vpu_dot16_f32(p - row + (int)CH, w_oc + 2u * CH);
	acc += vpu_dot16_f32(p - (int)CH, w_oc + 3u * CH);
	acc += vpu_dot16_f32(p, w_oc + 4u * CH);
	acc += vpu_dot16_f32(p + (int)CH, w_oc + 5u * CH);
	acc += vpu_dot16_f32(p + row - (int)CH, w_oc + 6u * CH);
	acc += vpu_dot16_f32(p + row, w_oc + 7u * CH);
	acc += vpu_dot16_f32(p + row + (int)CH, w_oc + 8u * CH);

	return acc;
#endif
}

static void conv_first_fp(const uint8_t *input, const int8_t *weights,
			  float *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const int8_t *const w = weights + oc * K * K;
		int32_t bias = 0;

		for (uint32_t i = 0; i < K * K; i++) {
			bias -= 128 * (int32_t)w[i];
		}

		for (uint32_t y = row0; y < row1; y++) {
			for (uint32_t x = 0; x < IMG_W; x++) {
				int32_t acc = bias;

				for (int ky = -1; ky <= 1; ky++) {
					const int yy = clamp_coord((int)y + ky, IMG_H);

					for (int kx = -1; kx <= 1; kx++) {
						const int xx = clamp_coord((int)x + kx, IMG_W);
						acc += (int32_t)input[(unsigned)yy * IMG_W +
								      (unsigned)xx] *
						       w[(ky + 1) * 3 + (kx + 1)];
					}
				}

				output[(y * IMG_W + x) * CH + oc] =
					relu_f32((float)acc * FIRST_SCALE);
			}
		}
	}
}

static void conv_hidden_fp(const float *input, const float *weights,
			   float *output, uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_OC2
	for (uint32_t oc = 0; oc < CH; oc += 2u) {
		const float *const w0 = weights + oc * K * K * CH;
		const float *const w1 = w0 + K * K * CH;

		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);

			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior = interior_y &&
					x > 0u && x < (IMG_W - 1u);
				float acc0;
				float acc1;

				if (interior) {
					const float *const p =
						input + (y * IMG_W + x) * CH;
					vpu_accum3x3_dot16x2_f32(p, w0, w1,
								  IMG_W * CH,
								  &acc0, &acc1);
				} else {
					acc0 = hidden_acc_scalar(input, w0, y, x);
					acc1 = hidden_acc_scalar(input, w1, y, x);
				}

				output[(y * IMG_W + x) * CH + oc] =
					relu_f32(acc0 * HIDDEN_SCALE);
				output[(y * IMG_W + x) * CH + oc + 1u] =
					relu_f32(acc1 * HIDDEN_SCALE);
			}
		}
	}
#else
	for (uint32_t oc = 0; oc < CH; oc++) {
		const float *const w_oc = weights + oc * K * K * CH;

		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);

			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior = interior_y &&
					x > 0u && x < (IMG_W - 1u);
				const float acc = interior ?
					hidden_acc_vpu(input, w_oc, y, x) :
					hidden_acc_scalar(input, w_oc, y, x);

				output[(y * IMG_W + x) * CH + oc] =
					relu_f32(acc * HIDDEN_SCALE);
			}
		}
	}
#endif
}

static void conv_final_fp(const float *input, const float *weights,
			  uint8_t *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++) {
		const int interior_y = y > 0u && y < (IMG_H - 1u);

		for (uint32_t x = 0; x < IMG_W; x++) {
			const int interior = interior_y &&
				x > 0u && x < (IMG_W - 1u);
			const float acc = interior ?
				hidden_acc_vpu(input, weights, y, x) :
				hidden_acc_scalar(input, weights, y, x);

			output[y * IMG_W + x] =
				clamp_u8_from_f32(128.0f + acc * FINAL_SCALE);
		}
	}
}

static uint32_t stripe_checksum(const uint8_t *output,
				uint32_t row0, uint32_t row1)
{
	uint32_t sum = 0;

	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			sum += output[y * IMG_W + x];
		}
	}

	return sum;
}

static void evict_activation_write_float(float *buffer,
					 uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_BOUNDARY_ONLY_EVICT
	if (row0 > 0u) {
		evict(buffer + row0 * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
	if (row1 < IMG_H && row1 > row0 + 1u) {
		evict(buffer + (row1 - 1u) * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
#else
	evict(buffer + row0 * IMG_W * CH,
	      (row1 - row0) * IMG_W * CH * sizeof(float));
#endif
}

static void evict_activation_read_float(const float *buffer,
					uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_SKIP_READ_EVICT
	(void)buffer;
	(void)row0;
	(void)row1;
#elif defined(DNCNN_VPU_BOUNDARY_ONLY_EVICT)
	if (row0 > 0u) {
		evict(buffer + (row0 - 1u) * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
	if (row1 < IMG_H) {
		evict(buffer + row1 * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
#else
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;

	evict(buffer + read_row0 * IMG_W * CH,
	      (read_row1 - read_row0) * IMG_W * CH * sizeof(float));
#endif
}

static void prefetch_activation_read_float(const float *buffer,
					   uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_PREFETCH_READ_WINDOW
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;
	uint64_t addr = (uint64_t)(buffer + read_row0 * IMG_W * CH);
	uint64_t lines = ((read_row1 - read_row0) * IMG_W * CH *
			  sizeof(float) + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
#else
	(void)buffer;
	(void)row0;
	(void)row1;
#endif
}

static void prefetch_wpack_float(const float *wpack)
{
#ifdef DNCNN_VPU_PREFETCH_WPACK
	uint64_t addr = (uint64_t)wpack;
	uint64_t lines = (WPACK_FLOATS * sizeof(float) + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
#else
	(void)wpack;
#endif
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = bench_hart_id();

	if (!bench_hart_enabled(hart_id)) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	const uint8_t *const input = base + INPUT_OFFSET;
	const int8_t *const weights = (const int8_t *)(base + WEIGHTS_OFFSET);
	uint8_t *const final_output = base + OUTPUT_OFFSET;
	float *const act0 = (float *)(base + ACT0_OFFSET);
	float *const act1 = (float *)(base + ACT1_OFFSET);
#ifdef DNCNN_VPU_SHARED_WPACK
	float *const wpack = (float *)(base + WPACK_OFFSET);
#else
	float *const wpack = (float *)(base + WPACK_OFFSET + hart_id * WPACK_STRIDE);
#endif
	float *const priv0 = (float *)(base + PRIVATE_OFFSET +
				       hart_id * PRIVATE_STRIDE);
	float *const priv1 = priv0 + (ACT_BYTES / sizeof(float));
	volatile struct dncnn_slot *const slots =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
	volatile struct dncnn_summary *const summary =
		(volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);
	g_barrier = (volatile struct bench_barrier_state *)(base + BARRIER_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

#ifdef DNCNN_VPU_SHARED_WPACK
	if (hart_id == 0u) {
		pack_weights(weights, wpack);
		FENCE;
		evict(wpack, WPACK_FLOATS * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
#ifndef DNCNN_VPU_SKIP_WPACK_ALLHART_EVICT
	evict(wpack, WPACK_FLOATS * sizeof(float));
	WAIT_CACHEOPS;
#endif
	prefetch_wpack_float(wpack);
#else
	pack_weights(weights, wpack);
	prefetch_wpack_float(wpack);
#endif
	FENCE;

	for (uint32_t pass = 0; pass < DNCNN_PASSES; pass++) {
#ifdef DNCNN_VPU_PRIVATE_FUSED
		const uint32_t first_row0 = row0 > 4u ? row0 - 4u : 0u;
		const uint32_t first_row1 = row1 + 4u < IMG_H ? row1 + 4u : IMG_H;
		const uint32_t hidden0_row0 = row0 > 3u ? row0 - 3u : 0u;
		const uint32_t hidden0_row1 = row1 + 3u < IMG_H ? row1 + 3u : IMG_H;
		const uint32_t hidden1_row0 = row0 > 2u ? row0 - 2u : 0u;
		const uint32_t hidden1_row1 = row1 + 2u < IMG_H ? row1 + 2u : IMG_H;
		const uint32_t hidden2_row0 = row0 > 1u ? row0 - 1u : 0u;
		const uint32_t hidden2_row1 = row1 + 1u < IMG_H ? row1 + 1u : IMG_H;
		const float *hidden_w = wpack;
		const float *const final_w = wpack + WH_BYTES;

		conv_first_fp(input, weights, priv0, first_row0, first_row1);
		FENCE;
		conv_hidden_fp(priv0, hidden_w, priv1, hidden0_row0, hidden0_row1);
		FENCE;
		conv_hidden_fp(priv1, hidden_w + CH * K * K * CH, priv0,
			       hidden1_row0, hidden1_row1);
		FENCE;
		conv_hidden_fp(priv0, hidden_w + 2u * CH * K * K * CH, priv1,
			       hidden2_row0, hidden2_row1);
		FENCE;
		conv_final_fp(priv1, final_w, final_output, row0, row1);
		FENCE;
#ifdef DNCNN_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DNCNN_PASSES)
#endif
		{
			evict(final_output + row0 * IMG_W,
			      (row1 - row0) * IMG_W);
			WAIT_CACHEOPS;
		}
#ifdef DNCNN_VPU_PRIVATE_FUSED_PASS_BARRIER
		bench_barrier();
#endif
#else
		conv_first_fp(input, weights, act0, row0, row1);
		FENCE;
		evict_activation_write_float(act0, row0, row1);
		WAIT_CACHEOPS;
		bench_barrier();

		const float *hidden_w = wpack;

		for (uint32_t layer = 0; layer < (LAYERS - 2u); layer++) {
			const float *const src = (layer & 1u) ? act1 : act0;
			float *const dst = (layer & 1u) ? act0 : act1;
			evict_activation_read_float(src, row0, row1);
			WAIT_CACHEOPS;
			prefetch_activation_read_float(src, row0, row1);

			conv_hidden_fp(src, hidden_w + layer * CH * K * K * CH,
				       dst, row0, row1);
			FENCE;
			evict_activation_write_float(dst, row0, row1);
			WAIT_CACHEOPS;
			bench_barrier();
		}

		const float *const last_hidden = ((LAYERS - 2u) & 1u) ? act1 : act0;
		const float *const final_w = wpack + WH_BYTES;

		evict_activation_read_float(last_hidden, row0, row1);
		WAIT_CACHEOPS;
		prefetch_activation_read_float(last_hidden, row0, row1);

		conv_final_fp(last_hidden, final_w, final_output, row0, row1);
		FENCE;
#ifdef DNCNN_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DNCNN_PASSES)
#endif
		{
		evict(final_output + row0 * IMG_W, (row1 - row0) * IMG_W);
		WAIT_CACHEOPS;
		}
#ifndef DNCNN_SKIP_FINAL_BARRIER
#ifndef DNCNN_SKIP_NONLAST_FINAL_BARRIER
		bench_barrier();
#else
		if (pass + 1u == DNCNN_PASSES) {
			bench_barrier();
		}
#endif
#endif
#endif
	}

	const uint32_t checksum = stripe_checksum(final_output, row0, row1);
	volatile struct dncnn_slot *const slot =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = DNCNN_MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->row0 = row0;
	slot->row1 = row1;
	slot->active_harts = ACTIVE_HARTS;
	slot->checksum = checksum;
	slot->done = 1u;

	FENCE;
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t slot_checksum_sum = 0;
		uint32_t output_sum = 0;

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == DNCNN_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}

		for (uint32_t i = 0; i < IMG_BYTES; i++) {
			output_sum += final_output[i];
		}

		const uint64_t macs_per_pass =
			(uint64_t)IMG_W * IMG_H *
			((uint64_t)CH * K * K +
			 (uint64_t)(LAYERS - 2u) * CH * CH * K * K +
			 (uint64_t)CH * K * K);
		const uint64_t ops = macs_per_pass * DNCNN_PASSES * 2u;

		summary->magic = DNCNN_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->passes = DNCNN_PASSES;
		summary->width = IMG_W;
		summary->height = IMG_H;
		summary->channels = CH;
		summary->layers = LAYERS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_sum = output_sum;
		summary->slot_checksum_sum = slot_checksum_sum;
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);

		FENCE;
		evict(summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
