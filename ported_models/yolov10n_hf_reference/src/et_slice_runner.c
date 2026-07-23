#include <stdint.h>

#include "erbium/isa/atomic.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/utils.h"

#include "ref_pmc.h"
#include "ref_runtime.h"
#include "slice_manifest.h"

#ifndef YR_MANIFEST_VERSION
#define YR_MANIFEST_VERSION 1u
#endif

/*
 * Hart count used by yr_hart_id()/yr_hart_count(), the barrier width, and
 * the PMC active-hart field below. yr_conv() uses it to split output
 * channels. Override with -DYR_NHART=8 to test thread-0-only occupancy.
 */
#ifndef YR_NHART
#define YR_NHART 16u
#endif


uint32_t yr_hart_id(void)
{
    return get_hart_id() & 0x3fu;
}


uint32_t yr_hart_count(void)
{
    return YR_NHART;
}


void yr_publish(const void *address, uint32_t bytes)
{
    evict(address, (uint64_t)bytes);
    WAIT_CACHEOPS;
}


/*
 * Software barrier state. Lives in the ELF image, not in the device buffer,
 * because the launcher allocates that buffer without zeroing it and a
 * counting barrier that starts from garbage never releases. The build uses
 * -fno-zero-initialized-in-bss so these zeros are loaded from the image.
 * Padded to a full cache line so no other variable shares it, which matters
 * because L1 is minion local and not coherent.
 */
struct yr_barrier_state {
    volatile uint32_t count;
    volatile uint32_t epoch;
    uint32_t padding[14];
};

static struct yr_barrier_state g_yr_barrier __attribute__((aligned(64))) = {
    0u, 0u, { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }
};


void yr_hart_barrier(void)
{
    uint32_t epoch;
    uint32_t prior;

    FENCE;
    WAIT_CACHEOPS;
    if (YR_NHART <= 1u) {
        return;
    }
    epoch = atomic_load_local_32(&g_yr_barrier.epoch);
    prior = atomic_add_local_32(&g_yr_barrier.count, 1u);
    if (prior + 1u == YR_NHART) {
        atomic_store_local_32(&g_yr_barrier.count, 0u);
        FENCE;
        (void)atomic_add_local_32(&g_yr_barrier.epoch, 1u);
    } else {
        while (atomic_load_local_32(&g_yr_barrier.epoch) == epoch) {
            FENCE;
        }
    }
    FENCE;
}


static uintptr_t yr_buffer_base_from_args(uintptr_t argument_area)
{
    uintptr_t pointer;
    if (argument_area == 0u || argument_area == ~(uintptr_t)0u) {
        return 0u;
    }
    pointer = *(volatile uintptr_t *)argument_area;
    if (pointer == 0u || pointer == ~(uintptr_t)0u) {
        return 0u;
    }
    return pointer;
}


#if YR_MANIFEST_VERSION >= 2
/*
 * Keep this check outside every PMC interval.  The schema-v2 stage table must
 * be an exact, gap-free partition of the generated node list, with local and
 * pinned-ONNX ordinals agreeing.  A malformed measurement table must never
 * silently skip or execute a graph node.
 */
static uint32_t yr_pmc_stage_manifest_valid(void)
{
    uint32_t stage_index;
    uint32_t next_local_node = 0u;
    uint32_t next_onnx_node = YR_FIRST_NODE;

    if (YR_PMC_STAGE_COUNT == 0u || YR_PMC_STAGE_STRIDE < YR_PMC_REGION_BYTES) {
        return 0u;
    }
    for (stage_index = 0u; stage_index < YR_PMC_STAGE_COUNT; ++stage_index) {
        const struct yr_pmc_stage_desc *stage = &yr_pmc_stages[stage_index];
        if (stage->first_local_node != next_local_node
            || stage->last_local_node < stage->first_local_node
            || stage->last_local_node >= YR_NODE_COUNT
            || stage->first_onnx_node != next_onnx_node
            || stage->last_onnx_node < stage->first_onnx_node
            || stage->last_onnx_node - stage->first_onnx_node
                != stage->last_local_node - stage->first_local_node) {
            return 0u;
        }
        next_local_node = stage->last_local_node + 1u;
        next_onnx_node = stage->last_onnx_node + 1u;
    }
    return next_local_node == YR_NODE_COUNT
        && next_onnx_node == YR_LAST_NODE + 1u;
}
#endif


int main(uintptr_t argument_area)
{
    uint8_t *base;
    struct yr_result_header *result;
    uint32_t status;
    const uint32_t hart_id = get_hart_id() & 0x3fu;

    /*
     * NUM_HARTS (the CRT gate deciding which physical harts even reach this
     * point) stays 16 in every configuration, including single-logical-hart
     * builds, because only NUM_HARTS=16 has been proven to boot on this
     * sys_emu. When YR_NHART is compiled down to 1 for a barrier-free
     * validation run, every hart still calls main(); only hart 0 may
     * proceed, since yr_hart_range() assumes yr_hart_id() < yr_hart_count()
     * and would compute out-of-range splits for any other physical hart.
     */
    if (YR_NHART <= 1u && hart_id != 0u) {
        return 0;
    }

    base = (uint8_t *)yr_buffer_base_from_args(argument_area);
    if (base == (uint8_t *)0) {
        return YR_STATUS_BAD_MANIFEST;
    }
    result = (struct yr_result_header *)(base + YR_RESULT_DEVICE_OFFSET);

    /*
     * yr_prepare_result() zeroes the whole workspace, so every hart must wait
     * for it to finish before any hart writes a Conv slice into that memory.
     * Only hart 0 owns the result header; the rest read result->status after
     * the barrier to decide whether to proceed.
     */
    FENCE;
    if (hart_id == 0u) {
        yr_prepare_result(base, result);
        yr_publish(result, YR_RESULT_HEADER_BYTES + YR_WORKSPACE_BYTES);
    }
    yr_hart_barrier();
    status = result->status;
    if (status != YR_STATUS_OK) {
        return status;
    }
#if YR_MANIFEST_VERSION >= 2
    if (hart_id == 0u) {
        if (!yr_pmc_stage_manifest_valid()) {
            result->status = YR_STATUS_BAD_MANIFEST;
        }
        yr_publish(result, YR_RESULT_HEADER_BYTES);
    }
    yr_hart_barrier();
    status = result->status;
    if (status != YR_STATUS_OK) {
        return status;
    }
    for (uint32_t stage_index = 0u;
         stage_index < YR_PMC_STAGE_COUNT;
         ++stage_index) {
        const struct yr_pmc_stage_desc *stage = &yr_pmc_stages[stage_index];
        uint8_t *pmc_base = base + YR_PMC_DEVICE_OFFSET
            + stage_index * YR_PMC_STAGE_STRIDE;
        yr_pmc_begin(pmc_base, hart_id, YR_NHART);
        status = yr_run_node_span(
            base, result, stage->first_local_node, stage->last_local_node);
        yr_pmc_end(pmc_base, hart_id);
        if (status != YR_STATUS_OK) {
            break;
        }
    }
#else
    yr_pmc_begin(base + YR_PMC_DEVICE_OFFSET, hart_id, YR_NHART);
    status = yr_run_selected(base, result);
    yr_pmc_end(base + YR_PMC_DEVICE_OFFSET, hart_id);
#endif

    /*
     * yr_finalize_result() hashes the whole workspace and only hart 0 calls
     * it, so every Conv slice from every hart must already be published by
     * the per-node barrier inside yr_run_node_span() before this point.
     */
    if (hart_id == 0u) {
        yr_finalize_result(base, result);
    }
    yr_hart_barrier();

    if (hart_id == 0u) {
        __asm__ __volatile__("" ::: "memory");
        FENCE;

        evict((const void *)(base + YR_RESULT_DEVICE_OFFSET),
              YR_RESULT_HEADER_BYTES + YR_WORKSPACE_BYTES);
#if YR_MANIFEST_VERSION >= 2
        for (uint32_t stage_index = 0u;
             stage_index < YR_PMC_STAGE_COUNT;
             ++stage_index) {
            evict(
                (const void *)(base + YR_PMC_DEVICE_OFFSET
                    + stage_index * YR_PMC_STAGE_STRIDE),
                YR_PMC_REGION_BYTES);
        }
#else
        evict((const void *)(base + YR_PMC_DEVICE_OFFSET), YR_PMC_REGION_BYTES);
#endif
        WAIT_CACHEOPS;
        __asm__ __volatile__("" ::: "memory");
        FENCE;
    }
    return status;
}
