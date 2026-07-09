/*
 * Microbenchmark for scalar vs VPU 3x3 stride-2 conv.
 * Runs ONE conv type in a tight loop, measures wall time via launcher.
 * Build two variants: define SCALAR or VPU.
 */
#include "yolo_common.h"

/* Tiny dimensions (IC=4, IH=8, IW=8, OC=8, OH=4, OW=4) */
#define T_IC 4u
#define T_IH 8u
#define T_IW 8u
#define T_OC 8u
#define T_OH 4u
#define T_OW 4u

#define ITERATIONS 1000

int main(uintptr_t arg_area) {
    uint32_t hid = get_hart_id();
    if (hid != 0) return 0;

    uint8_t *base = (uint8_t *)buffer_base_from_args(arg_area);

    float *input   = (float *)base;
    float *weights = (float *)(base + 0x10000);
    float *bias    = (float *)(base + 0x20000);
    float *out     = (float *)(base + 0x30000);

    /* Fill input */
    for (uint32_t i = 0; i < T_IC * T_IH * T_IW; i++)
        input[i] = (float)(i & 0xFF) * 0.01f;

    /* Fill weights */
    for (uint32_t i = 0; i < T_OC * T_IC * 9; i++)
        weights[i] = (float)(i & 0x7) * 0.1f - 0.3f;

    /* Fill bias */
    for (uint32_t i = 0; i < T_OC; i++)
        bias[i] = 0.1f;

    asm volatile("fence rw, rw" ::: "memory");

#ifdef SCALAR
    for (int i = 0; i < ITERATIONS; i++) {
        conv2d_fp32_mh(hid, input, out, (const float *)weights, (const float *)bias,
                       T_IC, T_IH, T_IW, T_OC, T_OH, T_OW,
                       3u, 3u, 2u, 2u, 1u, 1u, 0u);
    }
#else
    for (int i = 0; i < ITERATIONS; i++) {
        conv2d_3x3_s2_p1_fp32_mh_vpu(hid, input, out, (const float *)weights, (const float *)bias,
                                      T_IC, T_IH, T_IW, T_OC, T_OH, T_OW, 0u);
    }
#endif

    asm volatile("fence rw, rw" ::: "memory");
    return 0;
}