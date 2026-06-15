/*
 * Resident raw-INT8 Whisper decoder tail probe.
 *
 * Device memory layout:
 *   0x0000000..0x0ffffff  16 MiB runtime arena
 *   0x1000000..0x4ffffff  64 MiB packed resident INT8 weight region
 *
 * The host supplies the final decoder LayerNorm input plus dequantized LN
 * scale/bias in the runtime arena.  This kernel computes LayerNorm and the
 * complete vocab projection/argmax using the raw INT8 final-projection weight
 * directly out of the resident weight region.
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
#define MAGIC 0x57524938u
#endif
#ifndef K_DIM
#define K_DIM 384u
#endif
#ifndef N_COLS
#define N_COLS 51864u
#endif
#ifndef RUNTIME_REGION_BYTES
#define RUNTIME_REGION_BYTES (16u * 1024u * 1024u)
#endif
#ifndef WEIGHT_REGION_OFFSET
#define WEIGHT_REGION_OFFSET (16u * 1024u * 1024u)
#endif
#ifndef RESIDENT_WT_OFFSET
#define RESIDENT_WT_OFFSET 0u
#endif
#ifndef WT_SCALE
#define WT_SCALE 1.0f
#endif

#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x0000u
#define SUMMARY_OFFSET    0x1000u
#define PARAM_OFFSET      0x2000u
#define LN_OUT_OFFSET     0x4000u
#define PARTIAL_OFFSET    0x8000u
#define LN_PARAM_OFFSET   0x9000u
#define LN_IN_OFFSET      0x10000u
#define LN_WEIGHT_OFFSET  0x20000u
#define LN_BIAS_OFFSET    0x30000u
#define LN_REF_OFFSET     0x40000u
#define ACCUM_OFFSET      0x50000u

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
	uint32_t k_dim;
	uint32_t n_cols;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_hash;
	uint32_t reference_hash;
	uint32_t ln_max_abs_scaled;
	uint32_t ln_mean_abs_scaled;
	uint32_t argmax_col;
	uint32_t argmax_value_bits;
	uint32_t expected_argmax_col;
	uint32_t argmax_match;
	uint32_t hpm3_lo;
	uint32_t hpm3_hi;
	uint32_t hpm4_lo;
	uint32_t hpm4_hi;
	uint32_t hpm5_lo;
	uint32_t hpm5_hi;
	uint32_t hpm6_lo;
	uint32_t hpm6_hi;
	uint32_t hpm7_lo;
	uint32_t hpm7_hi;
	uint32_t hpm8_lo;
	uint32_t hpm8_hi;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t reserved[4];
};

struct run_params {
	uint32_t magic;
	uint32_t expected_argmax_col;
	uint32_t expected_argmax_value_bits;
	uint32_t resident_wt_offset;
	uint32_t n_cols;
	uint32_t k_dim;
	uint32_t wt_scale_bits;
	uint32_t reserved[9];
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
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
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

static inline uint64_t read_hpm3(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter3" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm4(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter4" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm5(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter5" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm6(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter6" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm7(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter7" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm8(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter8" : "=r"(v));
	return v;
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

static inline float f32_from_bits(uint32_t u)
{
	union {
		float f;
		uint32_t u;
	} bits;

	bits.u = u;
	return bits.f;
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

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	volatile struct run_params *const params =
		(volatile struct run_params *)(base + PARAM_OFFSET);
	float *const ln_out = (float *)(base + LN_OUT_OFFSET);
	volatile struct partial *const partials =
		(volatile struct partial *)(base + PARTIAL_OFFSET);
	volatile float *const ln_param = (volatile float *)(base + LN_PARAM_OFFSET);
	float *const ln_in = (float *)(base + LN_IN_OFFSET);
	float *const ln_weight = (float *)(base + LN_WEIGHT_OFFSET);
	float *const ln_bias = (float *)(base + LN_BIAS_OFFSET);
	float *const ln_ref = (float *)(base + LN_REF_OFFSET);
	float *const accum_all = (float *)(base + ACCUM_OFFSET);
	const int8_t *const wt =
		(const int8_t *)(base + WEIGHT_REGION_OFFSET + RESIDENT_WT_OFFSET);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	if (hart_id == 0u) {
		FENCE;
		evict(ln_in, K_DIM * sizeof(float));
		evict(ln_weight, K_DIM * sizeof(float));
		evict(ln_bias, K_DIM * sizeof(float));
		evict(ln_ref, K_DIM * sizeof(float));
		evict((void *)params, sizeof(*params));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint64_t hpm3_0 = read_hpm3();
	const uint64_t hpm4_0 = read_hpm4();
	const uint64_t hpm5_0 = read_hpm5();
	const uint64_t hpm6_0 = read_hpm6();
	const uint64_t hpm7_0 = read_hpm7();
	const uint64_t hpm8_0 = read_hpm8();

	const uint32_t ln_lines = K_DIM / 16u;
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
		const float inv_count = 1.0f / (float)K_DIM;
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

	const uint32_t max_cols_per_hart =
		(N_COLS + ACTIVE_HARTS - 1u) / ACTIVE_HARTS;
	const uint32_t col0 = (N_COLS * hart_id) / ACTIVE_HARTS;
	const uint32_t col1 = (N_COLS * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t local_cols = col1 - col0;
	float *const accum = accum_all + hart_id * max_cols_per_hart;

	for (uint32_t j = 0u; j < local_cols; j++) {
		accum[j] = 0.0f;
	}

	for (uint32_t k = 0u; k < K_DIM; k++) {
		const float a = ln_out[k] * WT_SCALE;
		const int8_t *const row = wt + k * N_COLS + col0;

		for (uint32_t j = 0u; j < local_cols; j++) {
			accum[j] += a * (float)row[j];
		}
	}

	float best = -3.4028234663852886e38f;
	uint32_t argmax_col = col0;
	for (uint32_t j = 0u; j < local_cols; j++) {
		const float v = accum[j];
		const uint32_t c = col0 + j;

		if (v > best || (v == best && c < argmax_col)) {
			best = v;
			argmax_col = c;
		}
	}

	const uint64_t hpm3_1 = read_hpm3();
	const uint64_t hpm4_1 = read_hpm4();
	const uint64_t hpm5_1 = read_hpm5();
	const uint64_t hpm6_1 = read_hpm6();
	const uint64_t hpm7_1 = read_hpm7();
	const uint64_t hpm8_1 = read_hpm8();

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
		uint32_t out_hash = 0u;
		uint32_t argmax = 0u;
		float global_best = -3.4028234663852886e38f;
		float ln_max_abs = 0.0f;
		float ln_sum_abs = 0.0f;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				const float v = f32_from_bits(slots[h].argmax_value_bits);

				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				out_hash += slots[h].checksum;
				if (v > global_best ||
				    (v == global_best && slots[h].argmax_col < argmax)) {
					global_best = v;
					argmax = slots[h].argmax_col;
				}
			}
		}

		for (uint32_t i = 0u; i < K_DIM; i++) {
			const float d = abs_f32(ln_out[i] - ln_ref[i]);

			if (d > ln_max_abs) {
				ln_max_abs = d;
			}
			ln_sum_abs += d;
		}

		const uint64_t ops = (uint64_t)N_COLS * K_DIM * 2u;
		const uint64_t dhpm3 = hpm3_1 - hpm3_0;
		const uint64_t dhpm4 = hpm4_1 - hpm4_0;
		const uint64_t dhpm5 = hpm5_1 - hpm5_0;
		const uint64_t dhpm6 = hpm6_1 - hpm6_0;
		const uint64_t dhpm7 = hpm7_1 - hpm7_0;
		const uint64_t dhpm8 = hpm8_1 - hpm8_0;

		summary->magic = MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->k_dim = K_DIM;
		summary->n_cols = N_COLS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = out_hash;
		summary->reference_hash = hash_f32(ln_ref, 0u, K_DIM);
		summary->ln_max_abs_scaled = (uint32_t)(ln_max_abs * 1000000.0f);
		summary->ln_mean_abs_scaled =
			(uint32_t)((ln_sum_abs * (1.0f / (float)K_DIM)) * 1000000.0f);
		summary->argmax_col = argmax;
		summary->argmax_value_bits = f32_bits(global_best);
		summary->expected_argmax_col = params->expected_argmax_col;
		summary->argmax_match =
			(argmax == params->expected_argmax_col) ? 1u : 0u;
		summary->hpm3_lo = (uint32_t)dhpm3;
		summary->hpm3_hi = (uint32_t)(dhpm3 >> 32);
		summary->hpm4_lo = (uint32_t)dhpm4;
		summary->hpm4_hi = (uint32_t)(dhpm4 >> 32);
		summary->hpm5_lo = (uint32_t)dhpm5;
		summary->hpm5_hi = (uint32_t)(dhpm5 >> 32);
		summary->hpm6_lo = (uint32_t)dhpm6;
		summary->hpm6_hi = (uint32_t)(dhpm6 >> 32);
		summary->hpm7_lo = (uint32_t)dhpm7;
		summary->hpm7_hi = (uint32_t)(dhpm7 >> 32);
		summary->hpm8_lo = (uint32_t)dhpm8;
		summary->hpm8_hi = (uint32_t)(dhpm8 >> 32);
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);
		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
