/*
 * Tensor-unit scaffolding for YOLO inference on ET-SoC1.
 *
 * When YOLO_USE_TENSOR is defined, this header overrides CONV_1x1
 * with a tensor-aware dispatcher.  Currently the tensor path is a
 * stub that falls through to the VPU — real tensor_fma routing
 * will be added layer-by-layer after validation on real silicon.
 *
 * Reference:
 *   et-platform/et-common-libs/include/erbium/isa/tensors.h
 *   et-platform/test-compute-kernels/src/tl_tfma_tstore_fc/
 *   docs/et_soc1_hardware.md
 */
#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#ifdef YOLO_USE_TENSOR

#include "erbium/isa/tensors.h"
#include "erbium/isa/cacheops-umode.h"

/* ------------------------------------------------------------------ */
/* SCP scratchpad init (call once at startup)                          */
/* Returns 0 on success.                                              */
/* ------------------------------------------------------------------ */
static inline int tensor_scp_enable(void)
{
    if (get_l1d_mode() == l1d_scp)
        return 0;
    int64_t ret = set_l1_cache_control(/*d1_split=*/1, /*scp_en=*/1);
    if (ret != 0) return (int)ret;
    ucache_control(/*scp_en=*/1, /*cacheop_rate=*/0, /*cacheop_max=*/0);
    return (get_l1d_mode() == l1d_scp) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Placeholder: tensor-accelerated 1x1 conv                            */
/* Currently falls through to VPU dispatcher.                          */
/* ------------------------------------------------------------------ */
static inline void conv2d_1x1_fp32_mh_tensor(uint32_t hid,
                                             const float *in, float *out,
                                             const float *W, const float *B,
                                             uint32_t IC, uint32_t H, uint32_t W_,
                                             uint32_t OC,
                                             uint32_t act)
{
    /* TODO: tensor-tiled 1x1 conv via tensor_load + tensor_fma + tensor_store.
     * See docs/et_soc1_hardware.md for the tensor unit API.
     * For now, delegate to the VPU. */
    conv2d_1x1_disp(hid, in, out, W, B, IC, H, W_, OC, act);
}

/* Override CONV_1x1 to route through tensor dispatcher */
#undef CONV_1x1
#define CONV_1x1(...) do { \
    conv2d_1x1_fp32_mh_tensor(hid, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)

#endif /* YOLO_USE_TENSOR */
#endif /* YOLO_TENSOR_H */
