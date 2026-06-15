/*
 * Whisper native-on-chip scheduling smoke test.
 *
 * This is a small auditable kernel for the intended full-native Whisper split:
 * even harts run the VPU-heavy decoder logits MatMul, while odd harts run
 * scalar graph work shaped like LayerNorm.  It proves the role split,
 * barriers, and non-coherent cache handoff on silicon before moving more ONNX
 * nodes from the host into the device kernel.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef SMOKE_STOP_PHASE
#define SMOKE_STOP_PHASE 0u
#endif
#ifndef SMOKE_MODE
#define SMOKE_MODE 0u
#endif
#ifndef MAGIC
#define MAGIC 0x57485350u
#endif
#ifndef K_DIM
#define K_DIM 384u
#endif
#ifndef N_COLS
#define N_COLS 1024u
#endif
#ifndef LN_DIM
#define LN_DIM 384u
#endif

#ifndef SLOT_BYTES
#define SLOT_BYTES 64u
#endif
#ifndef SLOTS_OFFSET
#define SLOTS_OFFSET 0x0000u
#endif
#ifndef SUMMARY_OFFSET
#define SUMMARY_OFFSET 0x1000u
#endif
#ifndef LOGITS_OUT_OFFSET
#define LOGITS_OUT_OFFSET 0x4000u
#endif
#ifndef ACT_OFFSET
#define ACT_OFFSET 0x20000u
#endif
#ifndef WT_OFFSET
#define WT_OFFSET 0x30000u
#endif
#ifndef LOGITS_REF_OFFSET
#define LOGITS_REF_OFFSET 0x1B0000u
#endif
#ifndef LN_IN_OFFSET
#define LN_IN_OFFSET 0x1C0000u
#endif
#ifndef LN_REF_OFFSET
#define LN_REF_OFFSET 0x1D0000u
#endif
#ifndef LN_OUT_OFFSET
#define LN_OUT_OFFSET 0x1E0000u
#endif
#ifndef LN_PARTIAL_OFFSET
#define LN_PARTIAL_OFFSET 0x1F0000u
#endif
#ifndef LN_PARAM_OFFSET
#define LN_PARAM_OFFSET 0x1F1000u
#endif

#ifndef BENCH_FLB
#define BENCH_FLB 2u
#endif
#ifndef BENCH_FCC
#define BENCH_FCC FCC_0
#endif
#define VPU_HARTS (ACTIVE_HARTS / 2u)
#define SCALAR_HARTS (ACTIVE_HARTS / 2u)
#define SMOKE_MODE_BOTH 0u
#define SMOKE_MODE_VPU_ONLY 1u
#define SMOKE_MODE_SCALAR_ONLY 2u
#define SMOKE_MODE_BARRIER_ONLY 3u

struct hart_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t role;
	uint32_t work0;
	uint32_t work1;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[7];
};

struct summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t vpu_harts;
	uint32_t scalar_harts;
	uint32_t vpu_done;
	uint32_t scalar_done;
	uint32_t vpu_active_mask;
	uint32_t scalar_active_mask;
	uint32_t logits_argmax;
	uint32_t logits_max_abs_scaled;
	uint32_t ln_max_abs_scaled;
	uint32_t ln_mean_abs_scaled;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t phase;
	uint32_t mode;
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

static inline void write_u64_words(volatile uint32_t *lo,
				   volatile uint32_t *hi, uint64_t v)
{
	*lo = (uint32_t)v;
	*hi = (uint32_t)(v >> 32);
}

static void publish_checkpoint(volatile struct summary *report,
			       uint32_t phase)
{
	report->magic = MAGIC;
	report->active_harts = ACTIVE_HARTS;
	report->vpu_harts = VPU_HARTS;
	report->scalar_harts = SCALAR_HARTS;
	report->vpu_done = 0u;
	report->scalar_done = 0u;
	report->vpu_active_mask = 0u;
	report->scalar_active_mask = 0u;
	report->logits_argmax = 0u;
	report->logits_max_abs_scaled = 0u;
	report->ln_max_abs_scaled = 0u;
	report->ln_mean_abs_scaled = 0u;
	report->ops_lo = 0u;
	report->ops_hi = 0u;
	report->phase = phase;
	report->mode = SMOKE_MODE;
	FENCE;
	evict((void *)report, sizeof(*report));
	WAIT_CACHEOPS;
}

static void mark_checkpoint_slot(volatile struct hart_slot *slot,
				 uint32_t hart_id, uint32_t phase)
{
	slot->magic = MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->role = phase;
	slot->work0 = 0u;
	slot->work1 = 0u;
	slot->checksum = MAGIC ^ hart_id ^ (phase << 16u);
	slot->done = 1u;
	FENCE;
	evict((void *)slot, sizeof(*slot));
	WAIT_CACHEOPS;
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
	float *const logits_out = (float *)(base + LOGITS_OUT_OFFSET);
	float *const act = (float *)(base + ACT_OFFSET);
	float *const wt = (float *)(base + WT_OFFSET);
	float *const logits_ref = (float *)(base + LOGITS_REF_OFFSET);
	float *const ln_in = (float *)(base + LN_IN_OFFSET);
	float *const ln_ref = (float *)(base + LN_REF_OFFSET);
	float *const ln_out = (float *)(base + LN_OUT_OFFSET);
	volatile struct partial *const partials =
		(volatile struct partial *)(base + LN_PARTIAL_OFFSET);
	volatile float *const ln_param = (volatile float *)(base + LN_PARAM_OFFSET);
	volatile struct hart_slot *const slots =
		(volatile struct hart_slot *)(base + SLOTS_OFFSET);
	volatile struct summary *const report =
		(volatile struct summary *)(base + SUMMARY_OFFSET);
	uint64_t hpm3_start = 0u;
	uint64_t hpm4_start = 0u;
	uint64_t hpm5_start = 0u;
	uint64_t hpm6_start = 0u;
	uint64_t hpm7_start = 0u;
	uint64_t hpm8_start = 0u;

	if (hart_id == 0u) {
		hpm3_start = read_hpm3();
		hpm4_start = read_hpm4();
		hpm5_start = read_hpm5();
		hpm6_start = read_hpm6();
		hpm7_start = read_hpm7();
		hpm8_start = read_hpm8();
	}

	if (SMOKE_MODE == SMOKE_MODE_BARRIER_ONLY && SMOKE_STOP_PHASE == 3u) {
		bench_barrier();
		mark_checkpoint_slot(&slots[hart_id], hart_id, 3u);
		if (hart_id == 0u) {
			publish_checkpoint(report, 3u);
		}
		return 0;
	}

	mark_checkpoint_slot(&slots[hart_id], hart_id, 1u);
	if (SMOKE_STOP_PHASE == 1u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 1u);
		}
		return 0;
	}

	if (hart_id == 0u && SMOKE_MODE != SMOKE_MODE_BARRIER_ONLY) {
		FENCE;
		evict(act, K_DIM * sizeof(float));
		evict(wt, N_COLS * K_DIM * sizeof(float));
		evict(logits_ref, N_COLS * sizeof(float));
		evict(ln_in, LN_DIM * sizeof(float));
		evict(ln_ref, LN_DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	if (SMOKE_STOP_PHASE == 2u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 2u);
		}
		return 0;
	}

	bench_barrier();
	mark_checkpoint_slot(&slots[hart_id], hart_id, 3u);
	if (SMOKE_STOP_PHASE == 3u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 3u);
		}
		return 0;
	}

	const uint32_t thread_id = get_thread_id();
	const uint32_t local_id = hart_id >> 1u;

	if (thread_id == 0u && SMOKE_MODE != SMOKE_MODE_SCALAR_ONLY &&
	    SMOKE_MODE != SMOKE_MODE_BARRIER_ONLY) {
		const uint32_t pairs = N_COLS / 2u;
		const uint32_t pair0 = (pairs * local_id) / VPU_HARTS;
		const uint32_t pair1 = (pairs * (local_id + 1u)) / VPU_HARTS;
		const uint32_t col0 = pair0 * 2u;
		const uint32_t col1 = pair1 * 2u;

		for (uint32_t c = col0; c < col1; c += 2u) {
			float acc0;
			float acc1;
			vpu_dotkx2_f32(act, wt + c * K_DIM,
				       wt + (c + 1u) * K_DIM,
				       &acc0, &acc1);
			logits_out[c] = acc0;
			logits_out[c + 1u] = acc1;
		}
		FENCE;
		if (col1 > col0) {
			evict(logits_out + col0, (col1 - col0) * sizeof(float));
			WAIT_CACHEOPS;
		}
		slots[hart_id].magic = MAGIC;
		slots[hart_id].hart_id = hart_id;
		slots[hart_id].minion_id = get_minion_id();
		slots[hart_id].thread_id = thread_id;
		slots[hart_id].role = 1u;
		slots[hart_id].work0 = col0;
		slots[hart_id].work1 = col1;
		slots[hart_id].checksum = hash_f32(logits_out, col0, col1);
		slots[hart_id].done = 1u;
		FENCE;
		evict((void *)&slots[hart_id], sizeof(slots[hart_id]));
		WAIT_CACHEOPS;
	} else if (thread_id == 1u && SMOKE_MODE != SMOKE_MODE_VPU_ONLY &&
		   SMOKE_MODE != SMOKE_MODE_BARRIER_ONLY) {
		const uint32_t i0 = (LN_DIM * local_id) / SCALAR_HARTS;
		const uint32_t i1 = (LN_DIM * (local_id + 1u)) / SCALAR_HARTS;
		float sum = 0.0f;
		float sumsq = 0.0f;

		for (uint32_t i = i0; i < i1; i++) {
			const float v = ln_in[i];
			sum += v;
			sumsq += v * v;
		}
		partials[local_id].sum = sum;
		partials[local_id].sumsq = sumsq;
		partials[local_id].count = i1 - i0;
		FENCE;
		evict((void *)&partials[local_id], sizeof(partials[local_id]));
		WAIT_CACHEOPS;
	} else {
		mark_checkpoint_slot(&slots[hart_id], hart_id, 4u);
	}
	if (SMOKE_STOP_PHASE == 4u) {
		mark_checkpoint_slot(&slots[hart_id], hart_id, 4u);
		if (hart_id == 0u) {
			publish_checkpoint(report, 4u);
		}
		return 0;
	}

	bench_barrier();
	if (SMOKE_STOP_PHASE == 5u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 5u);
		}
		return 0;
	}

	if (hart_id == 1u && SMOKE_MODE != SMOKE_MODE_VPU_ONLY &&
	    SMOKE_MODE != SMOKE_MODE_BARRIER_ONLY) {
		float sum = 0.0f;
		float sumsq = 0.0f;
		uint32_t count = 0u;

		for (uint32_t i = 0u; i < SCALAR_HARTS; i++) {
			sum += partials[i].sum;
			sumsq += partials[i].sumsq;
			count += partials[i].count;
		}

		(void)count;
		const float inv_count = 1.0f / (float)LN_DIM;
		const float mean = sum * inv_count;
		const float var = (sumsq * inv_count) - (mean * mean);
		ln_param[0] = mean;
		ln_param[1] = fast_inv_sqrtf(var + 0.00001f);
		FENCE;
		evict((void *)ln_param, 2u * sizeof(float));
		WAIT_CACHEOPS;
	}

	if (SMOKE_STOP_PHASE == 8u) {
		mark_checkpoint_slot(&slots[hart_id], hart_id, 8u);
		if (hart_id == 0u) {
			publish_checkpoint(report, 8u);
		}
		return 0;
	}

	bench_barrier();
	if (SMOKE_STOP_PHASE == 6u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 6u);
		}
		return 0;
	}

	if (thread_id == 1u && SMOKE_MODE != SMOKE_MODE_VPU_ONLY &&
	    SMOKE_MODE != SMOKE_MODE_BARRIER_ONLY) {
		const uint32_t i0 = (LN_DIM * local_id) / SCALAR_HARTS;
		const uint32_t i1 = (LN_DIM * (local_id + 1u)) / SCALAR_HARTS;
		const float mean = ln_param[0];
		const float inv = ln_param[1];

		for (uint32_t i = i0; i < i1; i++) {
			ln_out[i] = (ln_in[i] - mean) * inv;
		}
		FENCE;
		if (i1 > i0) {
			evict(ln_out + i0, (i1 - i0) * sizeof(float));
			WAIT_CACHEOPS;
		}
		slots[hart_id].magic = MAGIC;
		slots[hart_id].hart_id = hart_id;
		slots[hart_id].minion_id = get_minion_id();
		slots[hart_id].thread_id = thread_id;
		slots[hart_id].role = 2u;
		slots[hart_id].work0 = i0;
		slots[hart_id].work1 = i1;
		slots[hart_id].checksum = hash_f32(ln_out, i0, i1);
		slots[hart_id].done = 1u;
		FENCE;
		evict((void *)&slots[hart_id], sizeof(slots[hart_id]));
		WAIT_CACHEOPS;
	}

	bench_barrier();
	if (SMOKE_STOP_PHASE == 7u) {
		if (hart_id == 0u) {
			publish_checkpoint(report, 7u);
		}
		return 0;
	}

	if (hart_id == 0u) {
		uint32_t vpu_done = 0u;
		uint32_t scalar_done = 0u;
		uint32_t vpu_mask = 0u;
		uint32_t scalar_mask = 0u;
		float logits_max_abs = 0.0f;
		float ln_max_abs = 0.0f;
		float ln_sum_abs = 0.0f;
		float best = logits_out[0];
		uint32_t argmax = 0u;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic != MAGIC || slots[h].done != 1u) {
				continue;
			}
			if (slots[h].role == 1u) {
				vpu_done++;
				vpu_mask |= 1u << slots[h].hart_id;
			} else if (slots[h].role == 2u) {
				scalar_done++;
				scalar_mask |= 1u << slots[h].hart_id;
			}
		}

		for (uint32_t i = 0u; i < N_COLS; i++) {
			const float y = logits_out[i];
			const float d = abs_f32(y - logits_ref[i]);

			if (d > logits_max_abs) {
				logits_max_abs = d;
			}
			if (y > best) {
				best = y;
				argmax = i;
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
		report->magic = MAGIC;
		report->active_harts = ACTIVE_HARTS;
		report->vpu_harts = VPU_HARTS;
		report->scalar_harts = SCALAR_HARTS;
		report->vpu_done = vpu_done;
		report->scalar_done = scalar_done;
		report->vpu_active_mask = vpu_mask;
		report->scalar_active_mask = scalar_mask;
		report->logits_argmax = argmax;
		report->logits_max_abs_scaled =
			(uint32_t)(logits_max_abs * 1000000.0f);
		report->ln_max_abs_scaled = (uint32_t)(ln_max_abs * 1000000.0f);
		const float ln_inv_count = 1.0f / (float)LN_DIM;
		report->ln_mean_abs_scaled =
			(uint32_t)((ln_sum_abs * ln_inv_count) * 1000000.0f);
		report->ops_lo = (uint32_t)ops;
		report->ops_hi = (uint32_t)(ops >> 32);
		report->phase = 8u;
		report->mode = SMOKE_MODE;
		write_u64_words(&report->hpm3_lo, &report->hpm3_hi,
				read_hpm3() - hpm3_start);
		write_u64_words(&report->hpm4_lo, &report->hpm4_hi,
				read_hpm4() - hpm4_start);
		write_u64_words(&report->hpm5_lo, &report->hpm5_hi,
				read_hpm5() - hpm5_start);
		write_u64_words(&report->hpm6_lo, &report->hpm6_hi,
				read_hpm6() - hpm6_start);
		write_u64_words(&report->hpm7_lo, &report->hpm7_hi,
				read_hpm7() - hpm7_start);
		write_u64_words(&report->hpm8_lo, &report->hpm8_hi,
				read_hpm8() - hpm8_start);
		FENCE;
		evict((void *)report, sizeof(*report));
		WAIT_CACHEOPS;
	}

	return 0;
}
