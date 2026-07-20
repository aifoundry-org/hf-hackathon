#include <stdint.h>

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/utils.h"

#include "ref_pmc.h"
#include "ref_runtime.h"
#include "slice_manifest.h"

#ifndef YR_MANIFEST_VERSION
#define YR_MANIFEST_VERSION 1u
#endif


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

    /* Correctness reference path: exactly one hart, no implicit threading. */
    if (hart_id != 0u) {
        return 0;
    }
    base = (uint8_t *)yr_buffer_base_from_args(argument_area);
    if (base == (uint8_t *)0) {
        return YR_STATUS_BAD_MANIFEST;
    }
    result = (struct yr_result_header *)(base + YR_RESULT_DEVICE_OFFSET);

    FENCE;
    status = yr_prepare_result(base, result);
    if (status != YR_STATUS_OK) {
        return status;
    }
#if YR_MANIFEST_VERSION >= 2
    if (!yr_pmc_stage_manifest_valid()) {
        result->status = YR_STATUS_BAD_MANIFEST;
        return YR_STATUS_BAD_MANIFEST;
    }
    for (uint32_t stage_index = 0u;
         stage_index < YR_PMC_STAGE_COUNT;
         ++stage_index) {
        const struct yr_pmc_stage_desc *stage = &yr_pmc_stages[stage_index];
        uint8_t *pmc_base = base + YR_PMC_DEVICE_OFFSET
            + stage_index * YR_PMC_STAGE_STRIDE;
        yr_pmc_begin(pmc_base, hart_id, 1u);
        status = yr_run_node_span(
            base, result, stage->first_local_node, stage->last_local_node);
        yr_pmc_end(pmc_base, hart_id);
        if (status != YR_STATUS_OK) {
            break;
        }
    }
#else
    yr_pmc_begin(base + YR_PMC_DEVICE_OFFSET, hart_id, 1u);
    status = yr_run_selected(base, result);
    yr_pmc_end(base + YR_PMC_DEVICE_OFFSET, hart_id);
#endif
    yr_finalize_result(base, result);
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
    return status;
}
