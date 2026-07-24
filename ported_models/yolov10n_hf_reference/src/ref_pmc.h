/*
 * ref_pmc.h - isolated ET-SoC1 performance-counter support for the
 * correctness-first YOLOv10n Hugging Face ONNX reference port.
 *
 * Define YR_PMC to enable target-side sampling.  Calls remain valid no-ops
 * when YR_PMC is not defined, so a selected slice can be bracketed directly:
 *
 *     yr_pmc_begin(base + YR_PMC_DEVICE_OFFSET, hart_id, active_harts);
 *     run_selected_slice(...);
 *     yr_pmc_end(base + YR_PMC_DEVICE_OFFSET, hart_id);
 *
 * The calls directly bracket the selected slice: hart 0 takes the optional
 * shared shire-cache (SC) and memshire (MS) snapshots before reading the HPM
 * starts and after reading the HPM ends.  Thus shared syscalls are outside the
 * per-hart interval; as with any sequential snapshot, the small CSR-read
 * boundary cost remains.  Define either of these as 0 to omit its syscalls:
 *
 *     YR_PMC_SAMPLE_SC (default 1)
 *     YR_PMC_SAMPLE_MS (default 1)
 *
 * The fixed, little-endian binary format is decoded by tools/decode_pmc.py.
 * Its magic, version, sizes, offsets, and endian marker are deliberately part
 * of the public format below rather than inferred from a compiler dump.
 *
 * Firmware default event map (see docs/perf_counters.md):
 *   hpmcounter3 = minion cycles
 *   hpmcounter4 = retired instructions, thread 0
 *   hpmcounter5 = retired instructions, thread 1
 *   hpmcounter6 = L2 miss requests
 *   hpmcounter7 = minion icache requests
 *   hpmcounter8 = icache etlink requests
 */

#ifndef YR_REF_PMC_H
#define YR_REF_PMC_H

#include <stddef.h>
#include <stdint.h>

#define YR_PMC_REGION_MAGIC       0x4d505259u /* little-endian bytes "YRPM" */
#define YR_PMC_HART_MAGIC         0x48505259u /* little-endian bytes "YRPH" */
#define YR_PMC_AGGREGATE_MAGIC    0x41505259u /* little-endian bytes "YRPA" */
#define YR_PMC_FORMAT_VERSION     1u
#define YR_PMC_ENDIAN_MARKER      0x01020304u

#define YR_PMC_HPM_COUNT          6u /* hpmcounter3..hpmcounter8 */
#define YR_PMC_SC_BANKS           4u
#define YR_PMC_MS_COUNT           8u
#define YR_PMC_COUNTERS_PER_BLOCK 3u /* {cycles, event0, event1} */
#define YR_PMC_MAX_HARTS          32u

#define YR_PMC_HEADER_BYTES       128u
#define YR_PMC_HART_RECORD_BYTES  128u
#define YR_PMC_AGGREGATE_OFFSET   4224u
#define YR_PMC_REGION_BYTES       4816u

#define YR_PMC_FLAG_SC_REQUESTED  (1u << 0)
#define YR_PMC_FLAG_MS_REQUESTED  (1u << 1)
#define YR_PMC_KNOWN_FLAGS        \
	(YR_PMC_FLAG_SC_REQUESTED | YR_PMC_FLAG_MS_REQUESTED)

#define YR_PMC_ERROR_VALUE        ((uint64_t)~0ull)

#ifndef YR_PMC_SAMPLE_SC
#define YR_PMC_SAMPLE_SC 1
#endif

#ifndef YR_PMC_SAMPLE_MS
#define YR_PMC_SAMPLE_MS 1
#endif

/*
 * Each hart owns two cache lines.  This avoids false sharing when a future
 * multi-hart reference slice uses the same format; active_harts=1 is the
 * normal correctness-first configuration.
 */
struct yr_pmc_hart_record {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint64_t hpm_start[YR_PMC_HPM_COUNT];
	uint64_t hpm_end[YR_PMC_HPM_COUNT];
	uint64_t reserved[2];
};

/*
 * sc_supported_mask and ms_supported_mask have one bit per flattened
 * {bank/shire, counter} entry.  A set bit means both the start and end syscall
 * returned a value other than YR_PMC_ERROR_VALUE.  The header flags say which
 * families were requested at compile time.
 */
struct yr_pmc_aggregate_record {
	uint32_t magic;
	uint32_t shire_id;
	uint32_t sc_supported_mask;
	uint32_t ms_supported_mask;
	uint64_t sc_start[YR_PMC_SC_BANKS][YR_PMC_COUNTERS_PER_BLOCK];
	uint64_t sc_end[YR_PMC_SC_BANKS][YR_PMC_COUNTERS_PER_BLOCK];
	uint64_t ms_start[YR_PMC_MS_COUNT][YR_PMC_COUNTERS_PER_BLOCK];
	uint64_t ms_end[YR_PMC_MS_COUNT][YR_PMC_COUNTERS_PER_BLOCK];
};

/*
 * The first 128 bytes are a versioned header.  Keep the reserved bytes zero
 * only as a future-format convention; the v1 decoder does not require them.
 */
struct yr_pmc_region {
	uint32_t magic;
	uint32_t version;
	uint32_t region_bytes;
	uint32_t active_harts;
	uint32_t hpm_count;
	uint32_t max_harts;
	uint32_t flags;
	uint32_t endian_marker;
	uint64_t reserved[12];
	struct yr_pmc_hart_record harts[YR_PMC_MAX_HARTS];
	struct yr_pmc_aggregate_record aggregate;
};

_Static_assert(sizeof(struct yr_pmc_hart_record) == YR_PMC_HART_RECORD_BYTES,
	       "yr_pmc_hart_record binary size changed");
_Static_assert(offsetof(struct yr_pmc_region, harts) == YR_PMC_HEADER_BYTES,
	       "yr_pmc_region header binary size changed");
_Static_assert(offsetof(struct yr_pmc_region, aggregate) ==
		       YR_PMC_AGGREGATE_OFFSET,
	       "yr_pmc_region aggregate offset changed");
_Static_assert(sizeof(struct yr_pmc_region) == YR_PMC_REGION_BYTES,
	       "yr_pmc_region binary size changed");

#ifdef YR_PMC

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

#if YR_PMC_SAMPLE_SC || YR_PMC_SAMPLE_MS
#include "erbium-soc1sim/isa/syscall.h"
#endif

/*
 * RTLMIN-6496 workaround: four back-to-back reads within one half-cacheline
 * prevent the two minion threads from sampling a counter during its update.
 */
#define YR_PMC_SAFE_HPM_READ(counter, value)             \
	do {                                              \
		__asm__ __volatile__(".p2align 4\n"       \
				     "csrr %0," counter "\n"  \
				     "csrr %0," counter "\n"  \
				     "csrr %0," counter "\n"  \
				     "csrr %0," counter "\n"  \
				     : "=r"(value));          \
	} while (0)

static inline uint64_t yr_pmc_read_hpm(uint32_t index)
{
	uint64_t value = 0;

	switch (index) {
	case 0: YR_PMC_SAFE_HPM_READ("hpmcounter3", value); break;
	case 1: YR_PMC_SAFE_HPM_READ("hpmcounter4", value); break;
	case 2: YR_PMC_SAFE_HPM_READ("hpmcounter5", value); break;
	case 3: YR_PMC_SAFE_HPM_READ("hpmcounter6", value); break;
	case 4: YR_PMC_SAFE_HPM_READ("hpmcounter7", value); break;
	case 5: YR_PMC_SAFE_HPM_READ("hpmcounter8", value); break;
	default: break;
	}
	return value;
}

#if YR_PMC_SAMPLE_SC
static inline uint32_t yr_pmc_sample_sc(
	uint64_t (*destination)[YR_PMC_COUNTERS_PER_BLOCK])
{
	const uint64_t shire_id = get_shire_id();
	uint32_t supported_mask = 0;

	for (uint32_t bank = 0; bank < YR_PMC_SC_BANKS; ++bank) {
		for (uint32_t counter = 0;
		     counter < YR_PMC_COUNTERS_PER_BLOCK; ++counter) {
			const uint32_t bit =
				bank * YR_PMC_COUNTERS_PER_BLOCK + counter;
			const uint64_t value = (uint64_t)syscall(
				SYSCALL_PMC_SC_SAMPLE, shire_id, bank, counter);
			destination[bank][counter] = value;
			if (value != YR_PMC_ERROR_VALUE)
				supported_mask |= 1u << bit;
		}
	}
	return supported_mask;
}
#endif

#if YR_PMC_SAMPLE_MS
static inline uint32_t yr_pmc_sample_ms(
	uint64_t (*destination)[YR_PMC_COUNTERS_PER_BLOCK])
{
	uint32_t supported_mask = 0;

	for (uint32_t memory_shire = 0;
	     memory_shire < YR_PMC_MS_COUNT; ++memory_shire) {
		for (uint32_t counter = 0;
		     counter < YR_PMC_COUNTERS_PER_BLOCK; ++counter) {
			const uint32_t bit =
				memory_shire * YR_PMC_COUNTERS_PER_BLOCK + counter;
			const uint64_t value = (uint64_t)syscall(
				SYSCALL_PMC_MS_SAMPLE, memory_shire, counter, 0);
			destination[memory_shire][counter] = value;
			if (value != YR_PMC_ERROR_VALUE)
				supported_mask |= 1u << bit;
		}
	}
	return supported_mask;
}
#endif

/*
 * Take the start snapshot immediately before the selected slice.  Shared
 * counter syscalls happen before the HPM reads, so they are outside the
 * per-hart HPM interval.
 */
static inline void yr_pmc_begin(void *region_base, uint32_t hart_id,
				uint32_t active_harts)
{
	struct yr_pmc_region *const region =
		(struct yr_pmc_region *)region_base;
	struct yr_pmc_hart_record *hart;

	if (hart_id >= YR_PMC_MAX_HARTS)
		return;

	hart = &region->harts[hart_id];
	hart->magic = YR_PMC_HART_MAGIC;
	hart->hart_id = hart_id;
	hart->minion_id = get_minion_id();
	hart->thread_id = get_thread_id();
	hart->reserved[0] = 0;
	hart->reserved[1] = 0;

	if (hart_id == 0u) {
		uint32_t flags = 0;

#if YR_PMC_SAMPLE_SC
		flags |= YR_PMC_FLAG_SC_REQUESTED;
#endif
#if YR_PMC_SAMPLE_MS
		flags |= YR_PMC_FLAG_MS_REQUESTED;
#endif
		region->magic = YR_PMC_REGION_MAGIC;
		region->version = YR_PMC_FORMAT_VERSION;
		region->region_bytes = (uint32_t)sizeof(*region);
		region->active_harts = active_harts;
		region->hpm_count = YR_PMC_HPM_COUNT;
		region->max_harts = YR_PMC_MAX_HARTS;
		region->flags = flags;
		region->endian_marker = YR_PMC_ENDIAN_MARKER;
		for (uint32_t index = 0; index < 12u; ++index)
			region->reserved[index] = 0;
		region->aggregate.magic = YR_PMC_AGGREGATE_MAGIC;
		region->aggregate.shire_id = (uint32_t)get_shire_id();
		region->aggregate.sc_supported_mask = 0;
		region->aggregate.ms_supported_mask = 0;

#if YR_PMC_SAMPLE_SC
		region->aggregate.sc_supported_mask =
			yr_pmc_sample_sc(region->aggregate.sc_start);
#else
		for (uint32_t bank = 0; bank < YR_PMC_SC_BANKS; ++bank) {
			for (uint32_t counter = 0;
			     counter < YR_PMC_COUNTERS_PER_BLOCK; ++counter) {
				region->aggregate.sc_start[bank][counter] =
					YR_PMC_ERROR_VALUE;
				region->aggregate.sc_end[bank][counter] =
					YR_PMC_ERROR_VALUE;
			}
		}
#endif
#if YR_PMC_SAMPLE_MS
		region->aggregate.ms_supported_mask =
			yr_pmc_sample_ms(region->aggregate.ms_start);
#else
		for (uint32_t memory_shire = 0;
		     memory_shire < YR_PMC_MS_COUNT; ++memory_shire) {
			for (uint32_t counter = 0;
			     counter < YR_PMC_COUNTERS_PER_BLOCK; ++counter) {
				region->aggregate.ms_start[memory_shire][counter] =
					YR_PMC_ERROR_VALUE;
				region->aggregate.ms_end[memory_shire][counter] =
					YR_PMC_ERROR_VALUE;
			}
		}
#endif
	}

	for (uint32_t index = 0; index < YR_PMC_HPM_COUNT; ++index)
		hart->hpm_start[index] = yr_pmc_read_hpm(index);

	/* Do not let selected-slice memory operations move before the snapshot. */
	__asm__ __volatile__("" ::: "memory");
}

/*
 * Take the end snapshot immediately after the selected slice.  The compiler
 * barrier keeps selected-slice memory operations before the HPM reads.  Shared
 * samples follow those reads and therefore do not inflate the HPM deltas.
 */
static inline void yr_pmc_end(void *region_base, uint32_t hart_id)
{
	struct yr_pmc_region *const region =
		(struct yr_pmc_region *)region_base;
	struct yr_pmc_hart_record *hart;

	if (hart_id >= YR_PMC_MAX_HARTS)
		return;

	hart = &region->harts[hart_id];
	__asm__ __volatile__("" ::: "memory");
	for (uint32_t index = 0; index < YR_PMC_HPM_COUNT; ++index)
		hart->hpm_end[index] = yr_pmc_read_hpm(index);

	if (hart_id == 0u) {
#if YR_PMC_SAMPLE_SC
		region->aggregate.sc_supported_mask &=
			yr_pmc_sample_sc(region->aggregate.sc_end);
#endif
#if YR_PMC_SAMPLE_MS
		region->aggregate.ms_supported_mask &=
			yr_pmc_sample_ms(region->aggregate.ms_end);
#endif
	}

	/* Publish all counter stores before issuing cache maintenance. */
	__asm__ __volatile__("" ::: "memory");
	FENCE;
	evict((const void *)hart, sizeof(*hart));
	if (hart_id == 0u) {
		evict((const void *)region, YR_PMC_HEADER_BYTES);
		evict((const void *)&region->aggregate,
		      sizeof(region->aggregate));
	}
	WAIT_CACHEOPS;
	__asm__ __volatile__("" ::: "memory");
}

#else /* YR_PMC */

static inline void yr_pmc_begin(void *region_base, uint32_t hart_id,
				uint32_t active_harts)
{
	(void)region_base;
	(void)hart_id;
	(void)active_harts;
}

static inline void yr_pmc_end(void *region_base, uint32_t hart_id)
{
	(void)region_base;
	(void)hart_id;
}

#endif /* YR_PMC */

#endif /* YR_REF_PMC_H */
