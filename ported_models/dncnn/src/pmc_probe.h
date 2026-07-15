/*
 * pmc_probe.h - drop-in ET-SoC1 hardware performance-counter probe.
 *
 * Brackets a compute region and records, per active hart, the six minion HPM
 * (hardware performance monitor) counters the firmware programs by default, plus
 * (on hart 0) the shire-cache
 * (L2) and memshire (DDR) PMCs sampled through the U-mode syscall the firmware
 * services in M-mode. The deltas land in a fixed DRAM region so the launcher's
 * --dump_after captures them and decode_pmc.py turns them into a report.
 *
 * Entirely behind -DDNCNN_PMC: with the flag off this header compiles to
 * nothing, so the leaderboard build is byte-for-byte unchanged.
 *
 * Firmware default event map (device-minion-runtime MachineMinion/main.c,
 * mm_setup_default_pmcs) - the counters are free-running, U-mode readable.
 * Authoritative copy is docs/perf_counters.md; keep this mirror in sync:
 *   hpm3 = minion cycles          (only enabled on minion 0 of a neighborhood)
 *   hpm4 = retired instructions, thread 0   (every hart)
 *   hpm5 = retired instructions, thread 1   (every hart)
 *   hpm6 = L2 miss requests                 (every hart)
 *   hpm7 = minion icache requests   (neigh event; only the neigh-lead hart)
 *   hpm8 = icache etlink requests   (neigh event; only the neigh-lead hart)
 * Shire-cache PMCs: P0 = all L2 reads, P1 = all L2 writes (per neigh bank).
 * Memshire  PMCs:  P0 = all mesh reads, P1 = all mesh writes (per DDR shire).
 */

#ifndef DNCNN_PMC_PROBE_H
#define DNCNN_PMC_PROBE_H

#ifdef DNCNN_PMC

#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium-soc1sim/isa/syscall.h"   /* syscall(), SYSCALL_PMC_SC_SAMPLE/MS_SAMPLE */

#define PMC_MAGIC       0x504D4331u   /* "PMC1" region header magic          */
#define PMC_HART_MAGIC  0x504D4348u   /* "PMCH" per-hart record magic        */
#define PMC_AGG_MAGIC   0x504D4341u   /* "PMCA" aggregate (hart 0) magic     */
#define PMC_VERSION     1u

#define PMC_HPM_COUNT   6u            /* hpmcounter3..8                       */
#define PMC_SC_BANKS    4u            /* neighborhood cache banks per shire   */
#define PMC_MS_COUNT    8u            /* memory shires (DDR controllers)      */
#define PMC_PMC_PER     3u            /* {cycle, p0, p1} per SC/MS block      */
#define PMC_MAX_HARTS   32u           /* fixed slot count -> layout is        */
                                      /* independent of ACTIVE_HARTS          */

/* Work-around for hardware bug RTLMIN-6496: 4 back-to-back reads inside a half
 * cacheline so the two minion threads never sample a counter mid-update. */
#define PMC_HPM_SAFE_READ(counter, value)                \
	do {                                             \
		__asm__ __volatile__(".p2align 4\n"      \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     : "=r"(value));         \
	} while (0)

static inline uint64_t pmc_read_hpm(uint32_t idx)
{
	uint64_t v = 0;
	switch (idx) {
	case 0: PMC_HPM_SAFE_READ("hpmcounter3", v); break;
	case 1: PMC_HPM_SAFE_READ("hpmcounter4", v); break;
	case 2: PMC_HPM_SAFE_READ("hpmcounter5", v); break;
	case 3: PMC_HPM_SAFE_READ("hpmcounter6", v); break;
	case 4: PMC_HPM_SAFE_READ("hpmcounter7", v); break;
	case 5: PMC_HPM_SAFE_READ("hpmcounter8", v); break;
	default: break;
	}
	return v;
}

/* Per-hart record, padded to 128 B (two cache lines) so each hart evicts only
 * its own lines - same false-sharing discipline as the kernel's band seams. */
struct pmc_hart_rec {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint64_t hpm_start[PMC_HPM_COUNT];   /* 48 B */
	uint64_t hpm_end[PMC_HPM_COUNT];     /* 48 B */
	uint64_t _pad[2];                    /* -> 128 B */
};
_Static_assert(sizeof(struct pmc_hart_rec) == 128u, "pmc_hart_rec must be 128 B");

/* Aggregate record, written by hart 0 only: shire-cache + memshire samples. */
struct pmc_agg_rec {
	uint32_t magic;
	uint32_t shire_id;
	uint32_t sc_supported;   /* 1 if the SC-sample syscall returned real data */
	uint32_t ms_supported;
	uint64_t sc_start[PMC_SC_BANKS][PMC_PMC_PER];
	uint64_t sc_end[PMC_SC_BANKS][PMC_PMC_PER];
	uint64_t ms_start[PMC_MS_COUNT][PMC_PMC_PER];
	uint64_t ms_end[PMC_MS_COUNT][PMC_PMC_PER];
};

/* Whole region: 128 B header, then the hart slots (128 B each, so slot i starts
 * at offset 128 + 128*i), then the aggregate. */
struct pmc_region {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t hpm_count;
	uint32_t version;
	uint64_t _pad[14];                        /* -> 128 B header             */
	struct pmc_hart_rec harts[PMC_MAX_HARTS]; /* offset 128, 32*128 = 4096 B */
	struct pmc_agg_rec  agg;                  /* offset 4224                 */
};
_Static_assert(sizeof(((struct pmc_region *)0)->harts) == 4096u, "hart table size");

/* -1 sentinel the firmware/driver returns for an unimplemented or bad counter. */
#define PMC_ERR ((uint64_t)~0ull)

static inline void pmc_sample_sc(struct pmc_agg_rec *agg, uint64_t (*dst)[PMC_PMC_PER])
{
	const uint64_t shire = get_shire_id();
	uint32_t ok = 0;
	for (uint32_t bank = 0; bank < PMC_SC_BANKS; bank++) {
		for (uint32_t pmc = 0; pmc < PMC_PMC_PER; pmc++) {
			uint64_t v = (uint64_t)syscall(SYSCALL_PMC_SC_SAMPLE, shire, bank, pmc);
			dst[bank][pmc] = v;
			if (v != PMC_ERR)
				ok = 1;
		}
	}
	agg->sc_supported = ok;
	agg->shire_id = (uint32_t)shire;
}

static inline void pmc_sample_ms(struct pmc_agg_rec *agg, uint64_t (*dst)[PMC_PMC_PER])
{
	uint32_t ok = 0;
	for (uint32_t ms = 0; ms < PMC_MS_COUNT; ms++) {
		for (uint32_t pmc = 0; pmc < PMC_PMC_PER; pmc++) {
			uint64_t v = (uint64_t)syscall(SYSCALL_PMC_MS_SAMPLE, ms, pmc, 0);
			dst[ms][pmc] = v;
			if (v != PMC_ERR)
				ok = 1;
		}
	}
	agg->ms_supported = ok;
}

/* Snapshot the start of the measured region. Every active hart records its own
 * minion HPM start; hart 0 additionally samples the shared SC/MS PMCs. */
static inline void pmc_probe_begin(void *region_base, uint32_t hart_id, uint32_t active_harts)
{
	struct pmc_region *const r = (struct pmc_region *)region_base;
	if (hart_id >= PMC_MAX_HARTS)
		return;
	struct pmc_hart_rec *const h = &r->harts[hart_id];
	h->magic     = PMC_HART_MAGIC;
	h->hart_id   = hart_id;
	h->minion_id = get_minion_id();
	h->thread_id = get_thread_id();

	if (hart_id == 0u) {
		r->magic        = PMC_MAGIC;
		r->active_harts = active_harts;
		r->hpm_count    = PMC_HPM_COUNT;
		r->version      = PMC_VERSION;
		r->agg.magic    = PMC_AGG_MAGIC;
		pmc_sample_sc(&r->agg, r->agg.sc_start);
		pmc_sample_ms(&r->agg, r->agg.ms_start);
	}

	/* Read the minion HPMs last so the SC/MS syscalls above are outside the
	 * per-hart cycle/retired-instruction window on hart 0. */
	for (uint32_t i = 0; i < PMC_HPM_COUNT; i++)
		h->hpm_start[i] = pmc_read_hpm(i);
}

/* Snapshot the end of the region and flush every touched line to DRAM so the
 * launcher's --dump_after captures it. Each hart evicts only its own record. */
static inline void pmc_probe_end(void *region_base, uint32_t hart_id)
{
	struct pmc_region *const r = (struct pmc_region *)region_base;
	if (hart_id >= PMC_MAX_HARTS)
		return;
	struct pmc_hart_rec *const h = &r->harts[hart_id];
	for (uint32_t i = 0; i < PMC_HPM_COUNT; i++)
		h->hpm_end[i] = pmc_read_hpm(i);

	if (hart_id == 0u) {
		pmc_sample_sc(&r->agg, r->agg.sc_end);
		pmc_sample_ms(&r->agg, r->agg.ms_end);
	}

	FENCE;
	evict((const void *)h, sizeof(*h));
	if (hart_id == 0u) {
		evict((const void *)r, 64u);            /* header line */
		evict((const void *)&r->agg, sizeof(r->agg));
	}
	WAIT_CACHEOPS;
}

#endif /* DNCNN_PMC */
#endif /* DNCNN_PMC_PROBE_H */
