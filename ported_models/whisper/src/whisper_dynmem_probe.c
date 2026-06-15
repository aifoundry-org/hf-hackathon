/*
 * Dynamic-memory launcher probe for ET-SoC1 Erbium.
 *
 * The host allocates more than the historical 16 MiB argbuf and loads a blob at
 * HIGH_OFFSET.  The kernel reads that high address through the launch-provided
 * device pointer and writes a compact summary at offset 0 so the host can dump
 * only the low page.
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
#define MAGIC 0x5744594Eu
#endif
#ifndef HIGH_OFFSET
#define HIGH_OFFSET (20u * 1024u * 1024u)
#endif
#ifndef BYTE_COUNT
#define BYTE_COUNT 4096u
#endif

#define BENCH_FLB 2u
#define BENCH_FCC FCC_0

struct slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t i0;
	uint32_t i1;
	uint32_t sum;
	uint32_t xorv;
	uint32_t done;
	uint32_t reserved[7];
};

struct summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t high_offset;
	uint32_t byte_count;
	uint32_t done_count;
	uint32_t active_mask;
	uint32_t sum;
	uint32_t xorv;
	uint32_t first_word;
	uint32_t last_word;
	uint32_t reserved[22];
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

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	volatile uint8_t *const high = (volatile uint8_t *)(base + HIGH_OFFSET);
	volatile struct slot *const slots =
		(volatile struct slot *)(base + 0x0000u);
	volatile struct summary *const s =
		(volatile struct summary *)(base + 0x1000u);
	volatile struct slot *const mine = &slots[hart_id];

	const uint32_t i0 = (BYTE_COUNT * hart_id) / ACTIVE_HARTS;
	const uint32_t i1 = (BYTE_COUNT * (hart_id + 1u)) / ACTIVE_HARTS;
	uint32_t sum = 0u;
	uint32_t xorv = 0u;

	for (uint32_t i = i0; i < i1; i++) {
		const uint32_t v = high[i];
		sum += v;
		xorv ^= v << ((i & 3u) * 8u);
	}

	mine->magic = MAGIC;
	mine->hart_id = hart_id;
	mine->minion_id = get_minion_id();
	mine->thread_id = get_thread_id();
	mine->i0 = i0;
	mine->i1 = i1;
	mine->sum = sum;
	mine->xorv = xorv;
	mine->done = 1u;
	FENCE;
	evict((void *)mine, sizeof(*mine));
	WAIT_CACHEOPS;

	bench_barrier();

	if (hart_id == 0u) {
		uint32_t done = 0u;
		uint32_t mask = 0u;
		uint32_t total_sum = 0u;
		uint32_t total_xor = 0u;

		for (uint32_t h = 0u; h < ACTIVE_HARTS && h < 16u; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				done++;
				mask |= 1u << h;
				total_sum += slots[h].sum;
				total_xor ^= slots[h].xorv;
			}
		}

		s->magic = MAGIC;
		s->active_harts = ACTIVE_HARTS;
		s->high_offset = HIGH_OFFSET;
		s->byte_count = BYTE_COUNT;
		s->done_count = done;
		s->active_mask = mask;
		s->sum = total_sum;
		s->xorv = total_xor;
		s->first_word = ((uint32_t)high[0]) |
				((uint32_t)high[1] << 8) |
				((uint32_t)high[2] << 16) |
				((uint32_t)high[3] << 24);
		s->last_word = ((uint32_t)high[BYTE_COUNT - 4u]) |
			       ((uint32_t)high[BYTE_COUNT - 3u] << 8) |
			       ((uint32_t)high[BYTE_COUNT - 2u] << 16) |
			       ((uint32_t)high[BYTE_COUNT - 1u] << 24);
		FENCE;
		evict((void *)s, sizeof(*s));
		WAIT_CACHEOPS;
	}

	bench_barrier();
	return 0;
}
