/*
 * Real ONNX-tensor audit for one Whisper decoder single-row MatMul.
 *
 * This kernel targets decoder token projections where the left input is one
 * token row: [1, K] x [K, N] -> [1, N]. Work is split by output columns so
 * all selected harts do useful work even when M=1.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef INPUTS_PRELOADED
extern const char _binary_decoder_act_bin_start[];
extern const char _binary_decoder_weight_t_bin_start[];
extern const char _binary_decoder_ref_bin_start[];
#endif

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef AUDIT_REF
#define AUDIT_REF 1u
#endif
#ifndef ARGMAX_ONLY
#define ARGMAX_ONLY 0u
#endif

#ifndef MAGIC
#define MAGIC             0x57444C47u
#endif
#ifndef K_DIM
#define K_DIM             384u
#endif
#ifndef N_COLS
#define N_COLS            4096u
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
#ifndef TMP_OFFSET
#define TMP_OFFSET        0x8000u
#endif
#ifndef ACT_OFFSET
#define ACT_OFFSET        0x10000u
#endif
#ifndef WT_OFFSET
#define WT_OFFSET         0x20000u
#endif
#ifndef REF_OFFSET
#define REF_OFFSET        0xD00000u
#endif

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0
#define TMP_STRIDE_COLS   ((N_COLS + 15u) & ~15u)

struct audit_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t col0;
	uint32_t col1;
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

static uint32_t hash_cols(const float *y, uint32_t col0, uint32_t col1)
{
	uint32_t sum = 0;

	for (uint32_t c = col0; c < col1; c++) {
		sum += clamp_u8_from_f32(128.0f + y[c] * 0.25f);
	}

	return sum;
}

static inline uint32_t f32_bits(float v)
{
	union {
		float f;
		uint32_t u;
	} bits;

	bits.f = v;
	return bits.u;
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

static void matmul_cols(const float *a, const float *bt, float *out,
			uint32_t col0, uint32_t col1)
{
	for (uint32_t c = col0; c < col1; c += 2u) {
		float acc0 = 0.0f;
		float acc1 = 0.0f;
		const float *const w0 = bt + c * K_DIM;
		const float *const w1 = bt + (c + 1u) * K_DIM;

		vpu_dotkx2_f32(a, w0, w1, &acc0, &acc1);
		out[c] = acc0;
		out[c + 1u] = acc1;
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
	float *const tmp = (float *)(base + TMP_OFFSET);
	float *const act = (float *)(base + ACT_OFFSET);
	float *const wt = (float *)(base + WT_OFFSET);
	float *const ref = (float *)(base + REF_OFFSET);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	const uint32_t pairs = N_COLS / 2u;
	const uint32_t pair0 = (pairs * hart_id) / ACTIVE_HARTS;
	const uint32_t pair1 = (pairs * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t col0 = pair0 * 2u;
	const uint32_t col1 = pair1 * 2u;

#ifndef INPUTS_PRELOADED
	if (hart_id == 0u) {
		copy_bytes(act, _binary_decoder_act_bin_start,
			   K_DIM * sizeof(float));
		copy_bytes(wt, _binary_decoder_weight_t_bin_start,
			   N_COLS * K_DIM * sizeof(float));
#if AUDIT_REF
		copy_bytes(ref, _binary_decoder_ref_bin_start,
			   N_COLS * sizeof(float));
#endif
		FENCE;
		evict(act, K_DIM * sizeof(float));
		evict(wt, N_COLS * K_DIM * sizeof(float));
#if AUDIT_REF
		evict(ref, N_COLS * sizeof(float));
#endif
		WAIT_CACHEOPS;
	}
#else
	if (hart_id == 0u) {
		FENCE;
		evict(act, K_DIM * sizeof(float));
		evict(wt, N_COLS * K_DIM * sizeof(float));
#if AUDIT_REF
		evict(ref, N_COLS * sizeof(float));
#endif
		WAIT_CACHEOPS;
	}
#endif
	bench_barrier();

	float *const hart_tmp = tmp + hart_id * TMP_STRIDE_COLS;

	matmul_cols(act, wt, hart_tmp, col0, col1);
	FENCE;
	if (col1 > col0) {
		evict(hart_tmp + col0, (col1 - col0) * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

#if !ARGMAX_ONLY || AUDIT_REF
	if (hart_id == 0u) {
		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			const uint32_t p0 = (pairs * h) / ACTIVE_HARTS;
			const uint32_t p1 = (pairs * (h + 1u)) / ACTIVE_HARTS;
			const uint32_t c0 = p0 * 2u;
			const uint32_t c1 = p1 * 2u;
			const float *const src = tmp + h * TMP_STRIDE_COLS;

			for (uint32_t c = c0; c < c1; c++) {
				out[c] = src[c];
			}
		}
		FENCE;
		evict(out, N_COLS * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
#endif

	float max_abs = 0.0f;
	float sum_abs = 0.0f;

#if AUDIT_REF
	for (uint32_t c = col0; c < col1; c++) {
		const float d = abs_f32(out[c] - ref[c]);
		if (d > max_abs) {
			max_abs = d;
		}
		sum_abs += d;
	}
#endif

	volatile struct audit_slot *const slot =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->col0 = col0;
	slot->col1 = col1;
	slot->active_harts = ACTIVE_HARTS;
#if ARGMAX_ONLY && !AUDIT_REF
	slot->checksum = hash_cols(hart_tmp, col0, col1);
#else
	slot->checksum = hash_cols(out, col0, col1);
#endif
	slot->done = 1u;

	FENCE;
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		float full_max_abs = 0.0f;
		float full_sum_abs = 0.0f;
		uint32_t output_hash = 0;
		uint32_t argmax_col = 0;
		float argmax_value = -3.4028234663852886e38f;

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
			}
		}

#if ARGMAX_ONLY && !AUDIT_REF
		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			const uint32_t p0 = (pairs * h) / ACTIVE_HARTS;
			const uint32_t p1 = (pairs * (h + 1u)) / ACTIVE_HARTS;
			const uint32_t c0 = p0 * 2u;
			const uint32_t c1 = p1 * 2u;
			const float *const src = tmp + h * TMP_STRIDE_COLS;

			output_hash += hash_cols(src, c0, c1);
			for (uint32_t c = c0; c < c1; c++) {
				const float v = src[c];
				if (v > argmax_value ||
				    (v == argmax_value && c < argmax_col)) {
					argmax_value = v;
					argmax_col = c;
				}
			}
		}
#else
		output_hash = hash_cols(out, 0u, N_COLS);
		for (uint32_t c = 0; c < N_COLS; c++) {
			const float v = out[c];
			if (v > argmax_value ||
			    (v == argmax_value && c < argmax_col)) {
				argmax_value = v;
				argmax_col = c;
			}
		}
#endif

#if AUDIT_REF
		for (uint32_t c = 0; c < N_COLS; c++) {
			const float d = abs_f32(out[c] - ref[c]);
			if (d > full_max_abs) {
				full_max_abs = d;
			}
			full_sum_abs += d;
		}
#endif

		const uint64_t ops = (uint64_t)N_COLS * K_DIM * 2u;
		summary->magic = MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->rows = 1u;
		summary->k_dim = K_DIM;
		summary->n_cols = N_COLS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
#if AUDIT_REF
		summary->reference_hash = hash_cols(ref, 0u, N_COLS);
#else
		summary->reference_hash = 0u;
#endif
		summary->max_abs_scaled = (uint32_t)(full_max_abs * 1000000.0f);
		const float inv_cols = 1.0f / (float)N_COLS;
		summary->mean_abs_scaled =
			(uint32_t)((full_sum_abs * inv_cols) * 1000000.0f);
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);
		summary->reserved[0] = argmax_col;
		summary->reserved[1] = f32_bits(argmax_value);
		summary->reserved[2] = ARGMAX_ONLY;

		FENCE;
		evict(summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	(void)max_abs;
	(void)sum_abs;
	return 0;
}
