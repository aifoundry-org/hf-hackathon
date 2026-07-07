/*
 * Depth-Anything/ViT-shaped Erbium VPU FP32 benchmark.
 *
 * This is a kernel-level benchmark, not a full model runner.  It captures the
 * main transformer work expected from Depth Anything V2-style backbones:
 * token dot products, attention-value matmul, and MLP projections.
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

#ifndef DEPTH_PASSES
#define DEPTH_PASSES 1u
#endif

#ifndef DEPTH_MAGIC
#define DEPTH_MAGIC       0xD3717001u
#endif
#ifndef TOK
#define TOK               192u
#endif
#ifndef DIM
#define DIM               64u
#endif
#ifndef HIDDEN
#define HIDDEN            128u
#endif
#define X_FLOATS          (TOK * DIM)
#define XT_FLOATS         (DIM * TOK)
#define S_FLOATS          (TOK * TOK)
#define O_FLOATS          (TOK * DIM)
#define H_FLOATS          (TOK * HIDDEN)
#define Y_FLOATS          (TOK * DIM)
#define W1_FLOATS         (HIDDEN * DIM)
#define W2_FLOATS         (DIM * HIDDEN)

#ifndef SLOT_BYTES
#define SLOT_BYTES        64u
#endif
#ifndef SLOTS_OFFSET
#define SLOTS_OFFSET      0x0000u
#endif
#ifndef SUMMARY_OFFSET
#define SUMMARY_OFFSET    0x1000u
#endif
#ifndef BARRIER_OFFSET
#define BARRIER_OFFSET    0x1800u
#endif
#ifndef X_OFFSET
#define X_OFFSET          0x4000u
#endif
#ifndef XT_OFFSET
#define XT_OFFSET         0x10000u
#endif
#ifndef W1_OFFSET
#define W1_OFFSET         0x20000u
#endif
#ifndef W2_OFFSET
#define W2_OFFSET         0x30000u
#endif
#ifndef S_OFFSET
#define S_OFFSET          0x40000u
#endif
#ifndef O_OFFSET
#define O_OFFSET          0x70000u
#endif
#ifndef H_OFFSET
#define H_OFFSET          0x80000u
#endif
#ifndef Y_OFFSET
#define Y_OFFSET          0xA0000u
#endif

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct depth_slot {
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

struct depth_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t passes;
	uint32_t tokens;
	uint32_t dim;
	uint32_t hidden;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t slot_checksum_sum;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t reserved[4];
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

static float vpu_dot_f32(const float *a, const float *b, uint32_t k)
{
	float acc = 0.0f;

	for (uint32_t i = 0; i < k; i += 16u) {
		acc += vpu_dot16_f32(a + i, b + i);
	}

	return acc;
}

static void vpu_dotx2_f32(const float *a, const float *b0,
			  const float *b1, uint32_t k,
			  float *out0, float *out1)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;

	for (uint32_t i = 0; i < k; i += 16u) {
		float part0;
		float part1;

		vpu_dot16x2_f32(a + i, b0 + i, b1 + i, &part0, &part1);
		acc0 += part0;
		acc1 += part1;
	}

	*out0 = acc0;
	*out1 = acc1;
}

static inline void vpu_dot16x4_f32(const float *a, const float *w0,
				   const float *w1, const float *w2, const float *w3,
				   float *out0, float *out1, float *out2, float *out3)
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
		  [tmp0] "r"(tmp0), [tmp1] "r"(tmp1), [tmp2] "r"(tmp2), [tmp3] "r"(tmp3)
		: "memory", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8");

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

static void vpu_dotx4_f32(const float *a, const float *b0,
			  const float *b1, const float *b2, const float *b3,
			  uint32_t k,
			  float *out0, float *out1, float *out2, float *out3)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;

	for (uint32_t i = 0; i < k; i += 16u) {
		float part0;
		float part1;
		float part2;
		float part3;

		vpu_dot16x4_f32(a + i, b0 + i, b1 + i, b2 + i, b3 + i, &part0, &part1, &part2, &part3);
		acc0 += part0;
		acc1 += part1;
		acc2 += part2;
		acc3 += part3;
	}

	*out0 = acc0;
	*out1 = acc1;
	*out2 = acc2;
	*out3 = acc3;
}

static void prefetch_block(const void *ptr, uint32_t bytes)
{
	uint64_t addr = (uint64_t)ptr;
	uint64_t lines = (bytes + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
}

static void init_model(float *x, float *xt, float *w1, float *w2)
{
	for (uint32_t t = 0; t < TOK; t++) {
		for (uint32_t d = 0; d < DIM; d++) {
			const uint32_t idx = t * DIM + d;
			const float v = (float)((idx * 13u + 5u) & 0xffu) *
				(1.0f / 255.0f);
			x[idx] = v;
			xt[d * TOK + t] = v;
		}
	}

	for (uint32_t i = 0; i < W1_FLOATS; i++) {
		const int32_t v = (int32_t)((i * 19u + 11u) % 29u) - 14;
		w1[i] = (float)v * (1.0f / 8.0f);
	}

	for (uint32_t i = 0; i < W2_FLOATS; i++) {
		const int32_t v = (int32_t)((i * 23u + 3u) % 31u) - 15;
		w2[i] = (float)v * (1.0f / 8.0f);
	}
}

static void maybe_evict_read_rows(const float *a, uint32_t cols,
				  uint32_t row0, uint32_t row1)
{
#ifndef DEPTH_SKIP_READ_EVICT
	evict(a + row0 * cols, (row1 - row0) * cols * sizeof(float));
	WAIT_CACHEOPS;
#else
	(void)a;
	(void)cols;
	(void)row0;
	(void)row1;
#endif
}

static void maybe_prefetch_inputs(const float *a, uint32_t a_cols,
				  const float *b, uint32_t b_rows,
				  uint32_t k, uint32_t row0,
				  uint32_t row1)
{
#ifdef DEPTH_PREFETCH_A_ROWS
	prefetch_block(a + row0 * a_cols,
		       (row1 - row0) * a_cols * sizeof(float));
#else
	(void)a;
	(void)a_cols;
	(void)row0;
	(void)row1;
#endif

#ifdef DEPTH_PREFETCH_B
	prefetch_block(b, b_rows * k * sizeof(float));
#else
	(void)b;
	(void)b_rows;
	(void)k;
#endif
}

static void matmul_t(const float *a, const float *b, float *out,
		     uint32_t m, uint32_t n, uint32_t k,
		     uint32_t row0, uint32_t row1,
		     float scale, uint32_t relu)
{
	(void)m;
	maybe_evict_read_rows(a, k, row0, row1);
	maybe_prefetch_inputs(a, k, b, n, k, row0, row1);

#ifdef DEPTH_VPU_OC4
	for (uint32_t r = row0; r < row1; r++) {
		const float *const ar = a + r * k;

		for (uint32_t c = 0; c < n; c += 4u) {
			float acc0, acc1, acc2, acc3;

			vpu_dotx4_f32(ar, b + c * k, b + (c + 1u) * k, b + (c + 2u) * k, b + (c + 3u) * k,
				      k, &acc0, &acc1, &acc2, &acc3);
			acc0 *= scale;
			acc1 *= scale;
			acc2 *= scale;
			acc3 *= scale;
			if (relu) {
				acc0 = relu_f32(acc0);
				acc1 = relu_f32(acc1);
				acc2 = relu_f32(acc2);
				acc3 = relu_f32(acc3);
			}
			out[r * n + c] = acc0;
			out[r * n + c + 1u] = acc1;
			out[r * n + c + 2u] = acc2;
			out[r * n + c + 3u] = acc3;
		}
	}
#elif defined(DEPTH_VPU_OC2)
	for (uint32_t r = row0; r < row1; r++) {
		const float *const ar = a + r * k;

		for (uint32_t c = 0; c < n; c += 2u) {
			float acc0;
			float acc1;

			vpu_dotx2_f32(ar, b + c * k, b + (c + 1u) * k,
				      k, &acc0, &acc1);
			acc0 *= scale;
			acc1 *= scale;
			if (relu) {
				acc0 = relu_f32(acc0);
				acc1 = relu_f32(acc1);
			}
			out[r * n + c] = acc0;
			out[r * n + c + 1u] = acc1;
		}
	}
#else
	for (uint32_t r = row0; r < row1; r++) {
		const float *const ar = a + r * k;

		for (uint32_t c = 0; c < n; c++) {
			float acc = vpu_dot_f32(ar, b + c * k, k) * scale;

			if (relu) {
				acc = relu_f32(acc);
			}
			out[r * n + c] = acc;
		}
	}
#endif
}

static void evict_rows(float *out, uint32_t cols, uint32_t row0,
		       uint32_t row1)
{
	evict(out + row0 * cols, (row1 - row0) * cols * sizeof(float));
	WAIT_CACHEOPS;
}

static uint32_t stripe_checksum(const float *y, uint32_t row0, uint32_t row1)
{
	uint32_t sum = 0;

	for (uint32_t r = row0; r < row1; r++) {
		for (uint32_t c = 0; c < DIM; c++) {
			sum += clamp_u8_from_f32(128.0f + y[r * DIM + c] * 16.0f);
		}
	}

	return sum;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = bench_hart_id();

	if (!bench_hart_enabled(hart_id)) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const x = (float *)(base + X_OFFSET);
	float *const xt = (float *)(base + XT_OFFSET);
	float *const w1 = (float *)(base + W1_OFFSET);
	float *const w2 = (float *)(base + W2_OFFSET);
	float *const scores = (float *)(base + S_OFFSET);
	float *const attn_out = (float *)(base + O_OFFSET);
	float *const hidden = (float *)(base + H_OFFSET);
	float *const yout = (float *)(base + Y_OFFSET);
	volatile struct depth_slot *const slots =
		(volatile struct depth_slot *)(base + SLOTS_OFFSET);
	volatile struct depth_summary *const summary =
		(volatile struct depth_summary *)(base + SUMMARY_OFFSET);
	g_barrier = (volatile struct bench_barrier_state *)(base + BARRIER_OFFSET);

	const uint32_t row0 = (TOK * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (TOK * (hart_id + 1u)) / ACTIVE_HARTS;

	if (hart_id == 0u) {
		init_model(x, xt, w1, w2);
		FENCE;
		evict(x, X_FLOATS * sizeof(float));
		evict(xt, XT_FLOATS * sizeof(float));
		evict(w1, W1_FLOATS * sizeof(float));
		evict(w2, W2_FLOATS * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

#ifdef DEPTH_PREFETCH_STATIC
	prefetch_block(x, X_FLOATS * sizeof(float));
	prefetch_block(xt, XT_FLOATS * sizeof(float));
	prefetch_block(w1, W1_FLOATS * sizeof(float));
	prefetch_block(w2, W2_FLOATS * sizeof(float));
#endif

	for (uint32_t pass = 0; pass < DEPTH_PASSES; pass++) {
		matmul_t(x, x, scores, TOK, TOK, DIM, row0, row1,
			 1.0f / (float)DIM, 0u);
		FENCE;
		evict_rows(scores, TOK, row0, row1);
		bench_barrier();

		matmul_t(scores, xt, attn_out, TOK, DIM, TOK, row0, row1,
			 1.0f / (float)TOK, 0u);
		FENCE;
		evict_rows(attn_out, DIM, row0, row1);
		bench_barrier();

		matmul_t(attn_out, w1, hidden, TOK, HIDDEN, DIM, row0, row1,
			 1.0f / (float)DIM, 1u);
		FENCE;
		evict_rows(hidden, HIDDEN, row0, row1);
		bench_barrier();

		matmul_t(hidden, w2, yout, TOK, DIM, HIDDEN, row0, row1,
			 1.0f / (float)HIDDEN, 0u);
		FENCE;
#ifdef DEPTH_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DEPTH_PASSES)
#endif
		{
			evict_rows(yout, DIM, row0, row1);
		}
		bench_barrier();
	}

	const uint32_t checksum = stripe_checksum(yout, row0, row1);
	volatile struct depth_slot *const slot =
		(volatile struct depth_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = DEPTH_MAGIC;
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
			if (slots[h].magic == DEPTH_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}

		output_sum = stripe_checksum(yout, 0u, TOK);

		const uint64_t macs_per_pass =
			(uint64_t)TOK * TOK * DIM +
			(uint64_t)TOK * DIM * TOK +
			(uint64_t)TOK * HIDDEN * DIM +
			(uint64_t)TOK * DIM * HIDDEN;
		const uint64_t ops = macs_per_pass * DEPTH_PASSES * 2u;

		summary->magic = DEPTH_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->passes = DEPTH_PASSES;
		summary->tokens = TOK;
		summary->dim = DIM;
		summary->hidden = HIDDEN;
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
