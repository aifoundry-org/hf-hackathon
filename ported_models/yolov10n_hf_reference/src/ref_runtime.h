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

/*
 * Hart topology and cross-hart synchronization hooks. Each runner supplies
 * its own definitions so ref_runtime.c stays free of platform headers and
 * behaves identically on host and device once yr_hart_count() returns 1.
 */
uint32_t yr_hart_id(void);
uint32_t yr_hart_count(void);
void yr_publish(const void *address, uint32_t bytes);
void yr_hart_barrier(void);

/*
 * Optional fast path for a single Conv node, tried before the portable
 * scalar yr_conv(). Returns 1 if it computed the node and wrote output,
 * 0 if it declined (caller falls back to the scalar path unchanged).
 * ref_runtime.c supplies a weak stub that always declines, so the host
 * build and any ET build without a tensor-path source stay exactly as
 * they are today. An ET-only source file can override this symbol with
 * a real tensor-unit implementation without ref_runtime.c ever
 * including a platform or tensor header.
 *
 * On multi-hart builds each hart calls this independently and, when it
 * returns 1, may have only computed its own slice of output channels
 * (tile-aligned to the tensor unit's native width, not the raw per-hart
 * channel split yr_conv() itself uses). hart_oc_lo and hart_oc_hi report
 * that slice in output-channel units so the caller publishes exactly what
 * this hart wrote instead of assuming a different split. Both are set to
 * 0 before anything else runs, so a decline (return 0) or a hart that owns
 * no tiles this call always leaves them as an empty [0,0) range.
 */
struct yr_node_desc;
struct yr_tensor_desc;
uint32_t yr_conv_tensor(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *weight_desc,
    const struct yr_tensor_desc *bias_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    uint32_t *hart_oc_lo,
    uint32_t *hart_oc_hi);

#endif
