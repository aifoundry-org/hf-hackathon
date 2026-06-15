/*
 * Real ONNX-tensor audit for Whisper Tiny En encoder block 0 MLP MatMul.
 *
 * Audited node:
 *   /encoder/blocks.0/mlp/0/MatMul
 *
 * The linked blobs are exported from ONNXRuntime:
 *   A:   /encoder/blocks.0/mlp_ln/LayerNormalization_output_0, rows 0..127
 *   B:   onnx::MatMul_780, transposed to N x K for the local VPU kernel
 *   REF: /encoder/blocks.0/mlp/0/MatMul_output_0, rows 0..127
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

extern const char _binary_mlp0_act_128x384_bin_start[];
extern const char _binary_mlp0_weight_t_1536x384_bin_start[];
extern const char _binary_mlp0_ref_128x1536_bin_start[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif

#ifndef MAGIC
#define MAGIC             0x574D4C50u
#endif
#ifndef M_ROWS
#define M_ROWS            128u
#endif
#ifndef K_DIM
#define K_DIM             384u
#endif
#ifndef N_COLS
#define N_COLS            1536u
#endif

#ifndef SLOT_BYTES
#define SLOT_BYTES        64u
#endif
#ifndef SLOTS_OFFSET
#define SLOTS_OFFSET      0x0000u
#endif
#ifndef SUMMARY_OFFSET
#define SUMMARY_OFFSET    0x1000u
#endif
#ifndef OUT_OFFSET
#define OUT_OFFSET        0x4000u
#endif
#ifndef ACT_OFFSET
#define ACT_OFFSET        0xD0000u
#endif
#ifndef WT_OFFSET
#define WT_OFFSET         0x100000u
#endif
#ifndef REF_OFFSET
#define REF_OFFSET        0x340000u
#endif
#ifndef VPU_DOT_COLS
#define VPU_DOT_COLS      2u
#endif
#ifndef VPU_ACCUM_FULL_K
#define VPU_ACCUM_FULL_K  0u
#endif

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct audit_slot {
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

struct audit_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t rows;
	uint32_t k_dim;
	uint32_t n_cols;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_hash;
	uint32_t reference_hash;
	uint32_t max_abs_scaled;
	uint32_t mean_abs_scaled;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t reserved[3];
};

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
	uint32_t mask = 0;

	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
}

static inline uint32_t active_mask_t1(void)
{
	uint32_t mask = 0;

	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

static void copy_bytes(void *dst, const void *src, uint32_t bytes)
{
	uint8_t *d = (uint8_t *)dst;
	const uint8_t *s = (const uint8_t *)src;

	for (uint32_t i = 0; i < bytes; i++) {
		d[i] = s[i];
	}
}

static inline float abs_f32(float v)
{
	return v < 0.0f ? -v : v;
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

static uint32_t hash_rows(const float *y, uint32_t cols, uint32_t row0,
			  uint32_t row1)
{
	uint32_t sum = 0;

	for (uint32_t r = row0; r < row1; r++) {
		for (uint32_t c = 0; c < cols; c++) {
			sum += clamp_u8_from_f32(128.0f + y[r * cols + c] * 0.25f);
		}
	}

	return sum;
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

static inline void vpu_dot16x2_f32(const float *a, const float *w0,
				   const float *w1, float *out0,
				   float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"flq2    f1, 0(%[a0])\n"
		"flq2    f2, 0(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 0(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[a0])\n"
		"flq2    f2, 32(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 32(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [w0] "r"(w0),
		  [w1] "r"(w1), [tmp0] "r"(tmp0), [tmp1] "r"(tmp1)
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

static inline void vpu_dotkx2_f32(const float *a, const float *w0,
				  const float *w1, float *out0,
				  float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;
	const float *ap = a;
	const float *bp0 = w0;
	const float *bp1 = w1;
	uint32_t chunks = K_DIM / 16u;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"beqz    %[chunks], 2f\n"
		"1:\n"
		"flq2    f1, 0(%[ap])\n"
		"flq2    f2, 0(%[bp0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 0(%[bp1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[ap])\n"
		"flq2    f2, 32(%[bp0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 32(%[bp1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"addi    %[ap], %[ap], 64\n"
		"addi    %[bp0], %[bp0], 64\n"
		"addi    %[bp1], %[bp1], 64\n"
		"addi    %[chunks], %[chunks], -1\n"
		"bnez    %[chunks], 1b\n"
		"2:\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		: [ap] "+r"(ap), [bp0] "+r"(bp0), [bp1] "+r"(bp1),
		  [chunks] "+r"(chunks)
		: [zero] "r"(zero), [tmp0] "r"(tmp0), [tmp1] "r"(tmp1)
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

static inline void vpu_dot16x4_f32(const float *a, const float *w0,
				   const float *w1, const float *w2,
				   const float *w3, float *out0,
				   float *out1, float *out2,
				   float *out3)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	__attribute__((aligned(32))) float tmp2[8];
	__attribute__((aligned(32))) float tmp3[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"fbcx.ps f5, %[zero]\n"
		"fbcx.ps f7, %[zero]\n"
		"flq2    f1, 0(%[a0])\n"
		"flq2    f2, 0(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 0(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f6, 0(%[w2])\n"
		"fmadd.ps f5, f1, f6, f5\n"
		"flq2    f8, 0(%[w3])\n"
		"fmadd.ps f7, f1, f8, f7\n"
		"flq2    f1, 32(%[a0])\n"
		"flq2    f2, 32(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 32(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f6, 32(%[w2])\n"
		"fmadd.ps f5, f1, f6, f5\n"
		"flq2    f8, 32(%[w3])\n"
		"fmadd.ps f7, f1, f8, f7\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		"fsq2    f5, 0(%[tmp2])\n"
		"fsq2    f7, 0(%[tmp3])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [w0] "r"(w0),
		  [w1] "r"(w1), [w2] "r"(w2), [w3] "r"(w3),
		  [tmp0] "r"(tmp0), [tmp1] "r"(tmp1),
		  [tmp2] "r"(tmp2), [tmp3] "r"(tmp3)
		: "memory", "f0", "f1", "f2", "f3", "f4", "f5",
		  "f6", "f7", "f8");

	float sum0 = 0.0f;
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float sum3 = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum0 += tmp0[i];
		sum1 += tmp1[i];
		sum2 += tmp2[i];
		sum3 += tmp3[i];
	}

	*out0 = sum0;
	*out1 = sum1;
	*out2 = sum2;
	*out3 = sum3;
}

static void matmul_tile(const float *a, const float *bt, float *out,
			uint32_t row0, uint32_t row1)
{
	for (uint32_t r = row0; r < row1; r++) {
		const float *const ar = a + r * K_DIM;

#if VPU_DOT_COLS >= 4
		for (uint32_t c = 0; c < N_COLS; c += 4u) {
			float acc0 = 0.0f;
			float acc1 = 0.0f;
			float acc2 = 0.0f;
			float acc3 = 0.0f;
			const float *const b0 = bt + c * K_DIM;
			const float *const b1 = bt + (c + 1u) * K_DIM;
			const float *const b2 = bt + (c + 2u) * K_DIM;
			const float *const b3 = bt + (c + 3u) * K_DIM;

			for (uint32_t k = 0; k < K_DIM; k += 16u) {
				float part0;
				float part1;
				float part2;
				float part3;

				vpu_dot16x4_f32(ar + k, b0 + k, b1 + k,
						b2 + k, b3 + k, &part0,
						&part1, &part2, &part3);
				acc0 += part0;
				acc1 += part1;
				acc2 += part2;
				acc3 += part3;
			}
			out[r * N_COLS + c] = acc0;
			out[r * N_COLS + c + 1u] = acc1;
			out[r * N_COLS + c + 2u] = acc2;
			out[r * N_COLS + c + 3u] = acc3;
		}
#else
		for (uint32_t c = 0; c < N_COLS; c += 2u) {
			float acc0 = 0.0f;
			float acc1 = 0.0f;
			const float *const b0 = bt + c * K_DIM;
			const float *const b1 = bt + (c + 1u) * K_DIM;

#if VPU_ACCUM_FULL_K
			vpu_dotkx2_f32(ar, b0, b1, &acc0, &acc1);
#else
			for (uint32_t k = 0; k < K_DIM; k += 16u) {
				float part0;
				float part1;

				vpu_dot16x2_f32(ar + k, b0 + k, b1 + k,
						&part0, &part1);
				acc0 += part0;
				acc1 += part1;
			}
#endif
			out[r * N_COLS + c] = acc0;
			out[r * N_COLS + c + 1u] = acc1;
		}
#endif
	}
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const out = (float *)(base + OUT_OFFSET);
	float *const act = (float *)(base + ACT_OFFSET);
	float *const wt = (float *)(base + WT_OFFSET);
	float *const ref = (float *)(base + REF_OFFSET);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	const uint32_t row0 = (M_ROWS * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (M_ROWS * (hart_id + 1u)) / ACTIVE_HARTS;

	if (hart_id == 0u) {
		copy_bytes(act, _binary_mlp0_act_128x384_bin_start,
			   M_ROWS * K_DIM * sizeof(float));
		copy_bytes(wt, _binary_mlp0_weight_t_1536x384_bin_start,
			   N_COLS * K_DIM * sizeof(float));
		copy_bytes(ref, _binary_mlp0_ref_128x1536_bin_start,
			   M_ROWS * N_COLS * sizeof(float));
		FENCE;
		evict(act, M_ROWS * K_DIM * sizeof(float));
		evict(wt, N_COLS * K_DIM * sizeof(float));
		evict(ref, M_ROWS * N_COLS * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	matmul_tile(act, wt, out, row0, row1);
	FENCE;
	evict(out + row0 * N_COLS, (row1 - row0) * N_COLS * sizeof(float));
	WAIT_CACHEOPS;
	bench_barrier();

	float max_abs = 0.0f;
	float sum_abs = 0.0f;

	for (uint32_t r = row0; r < row1; r++) {
		for (uint32_t c = 0; c < N_COLS; c++) {
			const float d = abs_f32(out[r * N_COLS + c] -
					       ref[r * N_COLS + c]);
			if (d > max_abs) {
				max_abs = d;
			}
			sum_abs += d;
		}
	}

	volatile struct audit_slot *const slot =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->row0 = row0;
	slot->row1 = row1;
	slot->active_harts = ACTIVE_HARTS;
	slot->checksum = hash_rows(out, N_COLS, row0, row1);
	slot->done = 1u;

	FENCE;
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t output_hash = 0;
		uint32_t reference_hash = 0;
		float full_max_abs = 0.0f;
		float full_sum_abs = 0.0f;

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
			}
		}

		output_hash = hash_rows(out, N_COLS, 0u, M_ROWS);
		reference_hash = hash_rows(ref, N_COLS, 0u, M_ROWS);
		for (uint32_t i = 0; i < M_ROWS * N_COLS; i++) {
			const float d = abs_f32(out[i] - ref[i]);
			if (d > full_max_abs) {
				full_max_abs = d;
			}
			full_sum_abs += d;
		}

		const uint64_t ops = (uint64_t)M_ROWS * N_COLS * K_DIM * 2u;
		summary->magic = MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->rows = M_ROWS;
		summary->k_dim = K_DIM;
		summary->n_cols = N_COLS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
		summary->reference_hash = reference_hash;
		summary->max_abs_scaled = (uint32_t)(full_max_abs * 1000000.0f);
		summary->mean_abs_scaled =
			(uint32_t)((full_sum_abs / (float)(M_ROWS * N_COLS)) *
				   1000000.0f);
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);

		FENCE;
		evict(summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	(void)max_abs;
	(void)sum_abs;
	return 0;
}
