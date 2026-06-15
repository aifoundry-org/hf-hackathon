/*
 * Real Whisper decoder tail audit:
 *   final LayerNorm input -> final LayerNorm -> vocab logits tile argmax.
 *
 * This moves one more graph boundary onto ET-SoC1 than the logits-only audit.
 * The host still supplies ONNXRuntime tensors as the audit oracle, but the
 * token-choice value is produced from silicon-computed LayerNorm output.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef MAGIC
#define MAGIC 0x57444C54u
#endif
#ifndef K_DIM
#define K_DIM 384u
#endif
#ifndef LN_DIM
#define LN_DIM 384u
#endif
#ifndef N_COLS
#define N_COLS 4096u
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
#ifndef LN_OUT_OFFSET
#define LN_OUT_OFFSET     0x4000u
#endif
#ifndef PARTIAL_OFFSET
#define PARTIAL_OFFSET    0x8000u
#endif
#ifndef LN_PARAM_OFFSET
#define LN_PARAM_OFFSET   0x9000u
#endif
#ifndef LN_IN_OFFSET
#define LN_IN_OFFSET      0x10000u
#endif
#ifndef LN_WEIGHT_OFFSET
#define LN_WEIGHT_OFFSET  0x20000u
#endif
#ifndef LN_BIAS_OFFSET
#define LN_BIAS_OFFSET    0x30000u
#endif
#ifndef LN_REF_OFFSET
#define LN_REF_OFFSET     0x40000u
#endif
#ifndef WT_OFFSET
#define WT_OFFSET         0x50000u
#endif

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

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
	uint32_t argmax_col;
	uint32_t argmax_value_bits;
	uint32_t reserved[5];
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

struct partial {
	float sum;
	float sumsq;
	uint32_t count;
	uint32_t pad;
	uint32_t reserved[12];
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
	uint32_t mask = 0u;

	for (uint32_t h = 0u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
	}

	return mask;
}

static inline uint32_t active_mask_t1(void)
{
	uint32_t mask = 0u;

	for (uint32_t h = 1u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
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

static uint32_t hash_f32(const float *x, uint32_t i0, uint32_t i1)
{
	uint32_t sum = 0u;

	for (uint32_t i = i0; i < i1; i++) {
		sum += clamp_u8_from_f32(128.0f + x[i] * 0.25f);
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

static float fast_inv_sqrtf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	v.f = x;
	v.u = 0x5f3759dfu - (v.u >> 1u);
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	return v.f;
}

static inline void vpu_dotkx2_f32(const float *a, const float *w0,
				  const float *w1, float *out0,
				  float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0u;
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

	for (uint32_t i = 0u; i < 8u; i++) {
		sum0 += tmp0[i];
		sum1 += tmp1[i];
	}

	*out0 = sum0;
	*out1 = sum1;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const ln_out = (float *)(base + LN_OUT_OFFSET);
	volatile struct partial *const partials =
		(volatile struct partial *)(base + PARTIAL_OFFSET);
	volatile float *const ln_param = (volatile float *)(base + LN_PARAM_OFFSET);
	float *const ln_in = (float *)(base + LN_IN_OFFSET);
	float *const ln_weight = (float *)(base + LN_WEIGHT_OFFSET);
	float *const ln_bias = (float *)(base + LN_BIAS_OFFSET);
	float *const ln_ref = (float *)(base + LN_REF_OFFSET);
	float *const wt = (float *)(base + WT_OFFSET);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	if (hart_id == 0u) {
		FENCE;
		evict(ln_in, LN_DIM * sizeof(float));
		evict(ln_weight, LN_DIM * sizeof(float));
		evict(ln_bias, LN_DIM * sizeof(float));
		evict(ln_ref, LN_DIM * sizeof(float));
		evict(wt, N_COLS * K_DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint32_t ln_lines = LN_DIM / 16u;
	const uint32_t line0 = (ln_lines * hart_id) / ACTIVE_HARTS;
	const uint32_t line1 = (ln_lines * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t i0 = line0 * 16u;
	const uint32_t i1 = line1 * 16u;
	float sum = 0.0f;
	float sumsq = 0.0f;

	for (uint32_t i = i0; i < i1; i++) {
		const float v = ln_in[i];
		sum += v;
		sumsq += v * v;
	}
	partials[hart_id].sum = sum;
	partials[hart_id].sumsq = sumsq;
	partials[hart_id].count = i1 - i0;
	FENCE;
	evict((void *)&partials[hart_id], sizeof(partials[hart_id]));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		float total = 0.0f;
		float total_sq = 0.0f;
		uint32_t count = 0u;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			total += partials[h].sum;
			total_sq += partials[h].sumsq;
			count += partials[h].count;
		}

		(void)count;
		const float inv_count = 1.0f / (float)LN_DIM;
		const float mean = total * inv_count;
		const float var = (total_sq * inv_count) - (mean * mean);
		ln_param[0] = mean;
		ln_param[1] = fast_inv_sqrtf(var + 0.00001f);
		FENCE;
		evict((void *)ln_param, 2u * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const float mean = ln_param[0];
	const float inv = ln_param[1];
	for (uint32_t i = i0; i < i1; i++) {
		ln_out[i] = ((ln_in[i] - mean) * inv) * ln_weight[i] + ln_bias[i];
	}
	FENCE;
	if (i1 > i0) {
		evict(ln_out + i0, (i1 - i0) * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint32_t pairs = N_COLS / 2u;
	const uint32_t pair0 = (pairs * hart_id) / ACTIVE_HARTS;
	const uint32_t pair1 = (pairs * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t col0 = pair0 * 2u;
	const uint32_t col1 = pair1 * 2u;
	float best = -3.4028234663852886e38f;
	uint32_t argmax_col = col0;

	for (uint32_t c = col0; c < col1; c += 2u) {
		float acc0 = 0.0f;
		float acc1 = 0.0f;
		vpu_dotkx2_f32(ln_out, wt + c * K_DIM,
			       wt + (c + 1u) * K_DIM, &acc0, &acc1);
		if (acc0 > best || (acc0 == best && c < argmax_col)) {
			best = acc0;
			argmax_col = c;
		}
		if (acc1 > best || (acc1 == best && (c + 1u) < argmax_col)) {
			best = acc1;
			argmax_col = c + 1u;
		}
	}

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
	slot->checksum = hash_f32(ln_out, i0, i1);
	slot->done = 1u;
	slot->argmax_col = argmax_col;
	slot->argmax_value_bits = f32_bits(best);
	FENCE;
	evict((void *)slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t done_count = 0u;
		uint32_t active_mask = 0u;
		uint32_t tile_argmax = 0u;
		float tile_best = -3.4028234663852886e38f;
		float ln_max_abs = 0.0f;
		float ln_sum_abs = 0.0f;
		uint32_t output_hash = 0u;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				union {
					float f;
					uint32_t u;
				} bits;

				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				bits.u = slots[h].argmax_value_bits;
				if (bits.f > tile_best ||
				    (bits.f == tile_best &&
				     slots[h].argmax_col < tile_argmax)) {
					tile_best = bits.f;
					tile_argmax = slots[h].argmax_col;
				}
				output_hash += slots[h].checksum;
			}
		}

		for (uint32_t i = 0u; i < LN_DIM; i++) {
			const float d = abs_f32(ln_out[i] - ln_ref[i]);
			if (d > ln_max_abs) {
				ln_max_abs = d;
			}
			ln_sum_abs += d;
		}

		const uint64_t ops = (uint64_t)N_COLS * K_DIM * 2u;
		summary->magic = MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->rows = 1u;
		summary->k_dim = K_DIM;
		summary->n_cols = N_COLS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
		summary->reference_hash = hash_f32(ln_ref, 0u, LN_DIM);
		summary->max_abs_scaled = (uint32_t)(ln_max_abs * 1000000.0f);
		const float inv_cols = 1.0f / (float)LN_DIM;
		summary->mean_abs_scaled =
			(uint32_t)((ln_sum_abs * inv_cols) * 1000000.0f);
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);
		summary->reserved[0] = tile_argmax;
		summary->reserved[1] = f32_bits(tile_best);
		summary->reserved[2] = 2u;
		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
