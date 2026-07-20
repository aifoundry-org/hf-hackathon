#ifndef YOLOV10N_HF_REF_RUNTIME_H
#define YOLOV10N_HF_REF_RUNTIME_H

#include <stdint.h>

#define YR_RESULT_MAGIC 0x31465259u /* "YRF1", little endian */
#define YR_RESULT_VERSION 1u
#define YR_MATH_VERSION 1u

enum yr_status {
    YR_STATUS_OK = 0,
    YR_STATUS_BAD_MANIFEST = 1,
    YR_STATUS_UNSUPPORTED_OP = 2,
    YR_STATUS_UNSUPPORTED_SHAPE = 3
};

/*
 * Fixed 128-byte result header at offset zero of the launcher dump. The
 * selected tensors follow at YR_RESULT_HEADER_BYTES using offsets from the
 * generated slice manifest. PMC data has its own versioned region.
 */
struct yr_result_header {
    uint32_t magic;
    uint32_t version;
    uint32_t status;
    uint32_t failed_node;
    uint32_t failed_op;
    uint32_t first_node;
    uint32_t last_node;
    uint32_t node_count;
    uint32_t tensor_count;
    uint32_t workspace_bytes;
    uint32_t input_blob_bytes;
    uint32_t weight_blob_bytes;
    uint32_t math_version;
    uint32_t reserved32[3];
    uint64_t workspace_fnv1a;
    uint64_t reserved64[7];
};

_Static_assert(sizeof(struct yr_result_header) == 128u,
               "yr_result_header must remain 128 bytes");

/* Initialize and validate result/manifest bookkeeping outside PMC scope. */
uint32_t yr_prepare_result(
    uint8_t *device_base, struct yr_result_header *result);

/*
 * Execute only the generated inclusive ONNX node range. Call prepare first;
 * the caller owns PMC bracketing and cache publication.
 */
uint32_t yr_run_selected(uint8_t *device_base, struct yr_result_header *result);

/*
 * Execute an inclusive span of manifest-local node ordinals.  Full-graph ET
 * runners use this to put separate PMC begin/end pairs around measured
 * architecture stages.  Legacy slice callers continue to use
 * yr_run_selected(), which covers the entire generated manifest.
 */
uint32_t yr_run_node_span(
    uint8_t *device_base,
    struct yr_result_header *result,
    uint32_t first_local_node,
    uint32_t last_local_node);

/*
 * Compute the result integrity hash after the measured operator interval.
 * Keeping this O(workspace) bookkeeping separate ensures PMC covers only the
 * selected ONNX nodes.
 */
void yr_finalize_result(uint8_t *device_base, struct yr_result_header *result);

#endif
