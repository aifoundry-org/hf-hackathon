/*
 * Real Whisper data-movement audit kernel.
 *
 * This validates graph nodes whose math is a deterministic rearrangement or
 * selection of FP32 values: Gather, Slice, Concat, Reshape, and Unsqueeze.
 * The host provides a packed source vector, a uint32 index map, and an ORT
 * reference output.  Each active hart writes out[i] = src[index[i]].
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
#define MAGIC 0x57494350u
#endif
#ifndef ELEM_COUNT
#define ELEM_COUNT 384u
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
#ifndef INDEX_OFFSET
#define INDEX_OFFSET      0x200000u
#endif
#ifndef SRC_OFFSET
#define SRC_OFFSET        0x600000u
#endif
#ifndef REF_OFFSET
#define REF_OFFSET        0xB00000u
#endif

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct audit_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t i0;
	uint32_t i1;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[8];
};

struct audit_summary {
	uint32_t magic;
	uint32_t elem_count;
	uint32_t active_harts;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_hash;
	uint32_t reference_hash;
	uint32_t max_abs_scaled;
	uint32_t mean_abs_scaled;
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
	uint32_t reserved[11];
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

static inline uint32_t line_count_for(uint32_t elems)
{
	return (elems + 15u) / 16u;
}

static inline uint32_t part_i0(uint32_t elems, uint32_t hart_id)
{
	const uint32_t lines = line_count_for(elems);
	const uint32_t line0 = (lines * hart_id) / ACTIVE_HARTS;

	return line0 * 16u;
}

static inline uint32_t part_i1(uint32_t elems, uint32_t hart_id)
{
	const uint32_t lines = line_count_for(elems);
	const uint32_t line1 = (lines * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t i1 = line1 * 16u;

	return i1 > elems ? elems : i1;
}

static float fast_recipf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	if (x == 0.0f) {
		return 3.4028234663852886e38f;
	}

	v.f = x;
	v.u = 0x7EF311C3u - v.u;
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	return v.f;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const out = (float *)(base + OUT_OFFSET);
	const uint32_t *const index = (const uint32_t *)(base + INDEX_OFFSET);
	const float *const src = (const float *)(base + SRC_OFFSET);
	const float *const ref = (const float *)(base + REF_OFFSET);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	if (hart_id == 0u) {
		FENCE;
		evict((void *)index, ELEM_COUNT * sizeof(uint32_t));
		evict((void *)ref, ELEM_COUNT * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint64_t hpm3_start = read_hpm3();
	const uint64_t hpm4_start = read_hpm4();
	const uint64_t hpm5_start = read_hpm5();
	const uint64_t hpm6_start = read_hpm6();
	const uint64_t hpm7_start = read_hpm7();
	const uint64_t hpm8_start = read_hpm8();

	const uint32_t i0 = part_i0(ELEM_COUNT, hart_id);
	const uint32_t i1 = part_i1(ELEM_COUNT, hart_id);

	for (uint32_t i = i0; i < i1; i++) {
		out[i] = src[index[i]];
	}

	FENCE;
	if (i1 > i0) {
		evict(out + i0, (i1 - i0) * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint64_t hpm3_end = read_hpm3();
	const uint64_t hpm4_end = read_hpm4();
	const uint64_t hpm5_end = read_hpm5();
	const uint64_t hpm6_end = read_hpm6();
	const uint64_t hpm7_end = read_hpm7();
	const uint64_t hpm8_end = read_hpm8();

	volatile struct audit_slot *const slot =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->i0 = i0;
	slot->i1 = i1;
	slot->checksum = hash_f32(out, i0, i1);
	slot->done = 1u;
	FENCE;
	evict((void *)slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t done_count = 0u;
		uint32_t active_mask = 0u;
		uint32_t output_hash = 0u;
		float max_abs = 0.0f;
		float sum_abs = 0.0f;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				output_hash += slots[h].checksum;
			}
		}

		for (uint32_t i = 0u; i < ELEM_COUNT; i++) {
			const float d = abs_f32(out[i] - ref[i]);

			if (d > max_abs) {
				max_abs = d;
			}
			sum_abs += d;
		}

		const uint64_t dhpm3 = hpm3_end - hpm3_start;
		const uint64_t dhpm4 = hpm4_end - hpm4_start;
		const uint64_t dhpm5 = hpm5_end - hpm5_start;
		const uint64_t dhpm6 = hpm6_end - hpm6_start;
		const uint64_t dhpm7 = hpm7_end - hpm7_start;
		const uint64_t dhpm8 = hpm8_end - hpm8_start;

		summary->magic = MAGIC;
		summary->elem_count = ELEM_COUNT;
		summary->active_harts = ACTIVE_HARTS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
		summary->reference_hash = hash_f32(ref, 0u, ELEM_COUNT);
		summary->max_abs_scaled = (uint32_t)(max_abs * 1000000.0f);
		summary->mean_abs_scaled =
			(uint32_t)((sum_abs * fast_recipf((float)ELEM_COUNT)) *
				   1000000.0f);
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
		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
