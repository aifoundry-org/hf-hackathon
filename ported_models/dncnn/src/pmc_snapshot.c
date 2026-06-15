/*
 * Lightweight U-mode PMU snapshot for exact DnCNN ELF profiling.
 *
 * Run this before and after an unmodified benchmark ELF.  It snapshots
 * hpmcounter3..8 on all local shire harts plus the public SC/MS PMC syscalls.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/syscall.h"

extern char heap0_end[];

#define SNAP_MAGIC       0x504D5350u
#define DIRECT_MAGIC     0x504D5344u
#define SYSCALL_MAGIC    0x504D5359u

#define DIRECT_OFFSET    0x0000u
#define SYSCALL_OFFSET   0x1000u
#define SUMMARY_OFFSET   0x2000u

#define ACTIVE_HARTS     16u
#define HPM_COUNT        6u
#define SC_BANKS         4u
#define MS_COUNT         8u
#define PMC_PER_BLOCK    3u

#define SNAP_FLB         2u
#define SNAP_FCC         FCC_0

struct snapshot_direct {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t reserved[4];
	uint64_t hpm[HPM_COUNT];
};

struct snapshot_syscall {
	uint32_t magic;
	uint32_t kind;       /* 0 = SC, 1 = MS */
	uint32_t block_id;   /* SC bank or MS id */
	uint32_t pmc_id;
	uint32_t shire_id;
	uint32_t hart_id;
	uint32_t reserved[2];
	uint64_t value;
};

struct snapshot_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t syscall_records;
	uint32_t shire_id;
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

static inline void snapshot_barrier(void)
{
	shire_barrier(SNAP_FLB, SNAP_FCC, ACTIVE_HARTS,
		      active_mask_t0(), active_mask_t1());
}

#define HPM_SAFE_READ(counter, value)                \
	do {                                         \
		__asm__ __volatile__(".p2align 4\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     : "=r"(value));       \
	} while (0)

static inline uint64_t read_hpm(uint32_t idx)
{
	uint64_t value = 0;

	switch (idx) {
	case 0:
		HPM_SAFE_READ("hpmcounter3", value);
		break;
	case 1:
		HPM_SAFE_READ("hpmcounter4", value);
		break;
	case 2:
		HPM_SAFE_READ("hpmcounter5", value);
		break;
	case 3:
		HPM_SAFE_READ("hpmcounter6", value);
		break;
	case 4:
		HPM_SAFE_READ("hpmcounter7", value);
		break;
	case 5:
		HPM_SAFE_READ("hpmcounter8", value);
		break;
	default:
		break;
	}

	return value;
}

static void sample_syscalls(uintptr_t base)
{
	volatile struct snapshot_syscall *records =
		(volatile struct snapshot_syscall *)(base + SYSCALL_OFFSET);
	const uint32_t shire = get_shire_id();
	const uint32_t hart = get_hart_id();
	uint32_t idx = 0;

	for (uint32_t bank = 0; bank < SC_BANKS; bank++) {
		for (uint32_t pmc = 0; pmc < PMC_PER_BLOCK; pmc++) {
			volatile struct snapshot_syscall *r = &records[idx++];

			r->magic = SYSCALL_MAGIC;
			r->kind = 0;
			r->block_id = bank;
			r->pmc_id = pmc;
			r->shire_id = shire;
			r->hart_id = hart;
			r->reserved[0] = 0;
			r->reserved[1] = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_SC_SAMPLE,
						     shire, bank, pmc);
		}
	}

	for (uint32_t ms = 0; ms < MS_COUNT; ms++) {
		for (uint32_t pmc = 0; pmc < PMC_PER_BLOCK; pmc++) {
			volatile struct snapshot_syscall *r = &records[idx++];

			r->magic = SYSCALL_MAGIC;
			r->kind = 1;
			r->block_id = ms;
			r->pmc_id = pmc;
			r->shire_id = shire;
			r->hart_id = hart;
			r->reserved[0] = 0;
			r->reserved[1] = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_MS_SAMPLE,
						     ms, pmc, 0);
		}
	}

	evict((const void *)(base + SYSCALL_OFFSET),
	      (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK *
		      sizeof(struct snapshot_syscall));
}

int main(uintptr_t arg_area)
{
	const uintptr_t base = buffer_base_from_args(arg_area);
	const uint32_t hart = get_hart_id();

	if (hart >= ACTIVE_HARTS) {
		return 0;
	}

	volatile struct snapshot_direct *direct =
		(volatile struct snapshot_direct *)(base + DIRECT_OFFSET);
	volatile struct snapshot_direct *rec = &direct[hart];

	rec->magic = DIRECT_MAGIC;
	rec->hart_id = hart;
	rec->minion_id = get_minion_id();
	rec->thread_id = get_thread_id();
	rec->reserved[0] = SNAP_MAGIC;
	rec->reserved[1] = 0;
	rec->reserved[2] = 0;
	rec->reserved[3] = 0;

	if (get_thread_id() == 0u) {
		for (uint32_t i = 0; i < HPM_COUNT; i++) {
			rec->hpm[i] = read_hpm(i);
		}
		evict((const void *)rec, sizeof(*rec));
	}
	snapshot_barrier();

	if (get_thread_id() == 1u) {
		for (uint32_t i = 0; i < HPM_COUNT; i++) {
			rec->hpm[i] = read_hpm(i);
		}
		evict((const void *)rec, sizeof(*rec));
	}
	snapshot_barrier();

	if (hart == 0u) {
		volatile struct snapshot_summary *summary =
			(volatile struct snapshot_summary *)(base + SUMMARY_OFFSET);

		sample_syscalls(base);
		summary->magic = SNAP_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->syscall_records = (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK;
		summary->shire_id = get_shire_id();
		for (uint32_t i = 0; i < 12u; i++) {
			summary->reserved[i] = 0;
		}
		evict((const void *)summary, sizeof(*summary));
	}

	snapshot_barrier();
	return 0;
}
