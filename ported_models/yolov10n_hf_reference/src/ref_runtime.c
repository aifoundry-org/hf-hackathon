/*
 * Portable scalar FP32 ONNX slice runtime.
 *
 * This file intentionally contains no VPU/TFMA paths, fusions, tiling,
 * threading, allocation, or libc calls. Each ONNX node remains a separately
 * materialized tensor so intermediate comparisons stay visible.
 */

#include <stdint.h>

#include "ref_runtime.h"
#include "slice_manifest.h"

#ifndef YR_MANIFEST_VERSION
#define YR_MANIFEST_VERSION 1u
#endif

static uint8_t *yr_tensor_raw(
    uint8_t *base, const struct yr_tensor_desc *tensor)
{
    uintptr_t address;
    if (tensor->storage == YR_STORAGE_INPUT) {
        address = (uintptr_t)base + YR_INPUT_DEVICE_OFFSET + tensor->offset;
    } else if (tensor->storage == YR_STORAGE_WEIGHTS) {
        address = (uintptr_t)base + YR_WEIGHT_DEVICE_OFFSET + tensor->offset;
    } else if (tensor->storage == YR_STORAGE_WORKSPACE) {
        address = (uintptr_t)base + YR_RESULT_DEVICE_OFFSET
                + YR_RESULT_HEADER_BYTES + tensor->offset;
    } else {
        return (uint8_t *)0;
    }
    return (uint8_t *)address;
}


static float *yr_tensor_ptr(uint8_t *base, const struct yr_tensor_desc *tensor)
{
    return (float *)yr_tensor_raw(base, tensor);
}


static uint32_t yr_tensor_dtype(
    const struct yr_tensor_desc *tensor)
{
#if YR_MANIFEST_VERSION >= 2
    return tensor->dtype;
#else
    (void)tensor;
    return 1u; /* YR_DTYPE_FLOAT in manifest v2. */
#endif
}


static uint32_t yr_dtype_bytes(uint32_t dtype)
{
    if (dtype == 1u) {
        return 4u;
    }
#if YR_MANIFEST_VERSION >= 2
    if (dtype == YR_DTYPE_INT64) {
        return 8u;
    }
#endif
    return 0u;
}


static uint32_t yr_range_valid(
    uint64_t offset, uint64_t bytes, uint64_t limit)
{
    return bytes <= limit && offset <= limit - bytes;
}


static uint32_t yr_memory_map_valid(void)
{
    return YR_RESULT_HEADER_BYTES >= sizeof(struct yr_result_header)
        && yr_range_valid(
            YR_RESULT_DEVICE_OFFSET,
            (uint64_t)YR_RESULT_HEADER_BYTES
                + (uint64_t)YR_WORKSPACE_BYTES,
            YR_MEM_SIZE)
        && yr_range_valid(
            YR_INPUT_DEVICE_OFFSET, YR_INPUT_BLOB_BYTES, YR_MEM_SIZE)
        && yr_range_valid(
            YR_WEIGHT_DEVICE_OFFSET, YR_WEIGHT_BLOB_BYTES, YR_MEM_SIZE);
}


static uint32_t yr_manifest_valid(void)
{
    uint32_t tensor_index;
    if (!yr_memory_map_valid()) {
        return 0u;
    }
    for (tensor_index = 0u; tensor_index < YR_TENSOR_COUNT; ++tensor_index) {
        const struct yr_tensor_desc *tensor = &yr_tensors[tensor_index];
        uint64_t elements = 1u;
        uint32_t dimension;
        uint32_t limit;
        const uint32_t element_bytes =
            yr_dtype_bytes(yr_tensor_dtype(tensor));
        if (tensor->rank > 6u
            || element_bytes == 0u
            || tensor->elements > UINT32_MAX / element_bytes
            || tensor->nbytes != tensor->elements * element_bytes
            || tensor->offset % element_bytes != 0u) {
            return 0u;
        }
        for (dimension = 0u; dimension < tensor->rank; ++dimension) {
            elements *= tensor->dims[dimension];
            if (elements > UINT32_MAX) {
                return 0u;
            }
        }
        if (elements != tensor->elements) {
            return 0u;
        }
        if (tensor->storage == YR_STORAGE_INPUT) {
            limit = YR_INPUT_BLOB_BYTES;
        } else if (tensor->storage == YR_STORAGE_WEIGHTS) {
            limit = YR_WEIGHT_BLOB_BYTES;
        } else if (tensor->storage == YR_STORAGE_WORKSPACE) {
            limit = YR_WORKSPACE_BYTES;
        } else {
            return 0u;
        }
        if (!yr_range_valid(tensor->offset, tensor->nbytes, limit)) {
            return 0u;
        }
    }
    return 1u;
}


static uint32_t yr_same_shape(
    const struct yr_tensor_desc *left,
    const struct yr_tensor_desc *right)
{
    uint32_t dimension;
    if (left->rank != right->rank) {
        return 0u;
    }
    for (dimension = 0u; dimension < left->rank; ++dimension) {
        if (left->dims[dimension] != right->dims[dimension]) {
            return 0u;
        }
    }
    return 1u;
}


#if YR_MANIFEST_VERSION >= 2
static int32_t yr_normalize_axis(int32_t axis, uint32_t rank)
{
    if (axis < 0) {
        axis += (int32_t)rank;
    }
    if (axis < 0 || axis >= (int32_t)rank) {
        return -1;
    }
    return axis;
}


static uint32_t yr_shape_product(
    const struct yr_tensor_desc *tensor,
    uint32_t first,
    uint32_t last,
    uint32_t *product)
{
    uint64_t value = 1u;
    uint32_t dimension;
    if (first > last || last > tensor->rank) {
        return 0u;
    }
    for (dimension = first; dimension < last; ++dimension) {
        value *= tensor->dims[dimension];
        if (value > UINT32_MAX) {
            return 0u;
        }
    }
    *product = (uint32_t)value;
    return 1u;
}


static uint32_t yr_broadcast_index(
    const struct yr_tensor_desc *input,
    const struct yr_tensor_desc *output,
    uint32_t output_index,
    uint32_t *input_index)
{
    uint32_t coordinate[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t dimension;
    uint32_t remaining = output_index;
    uint32_t result = 0u;
    uint32_t stride = 1u;

    if (input->rank > output->rank) {
        return 0u;
    }
    for (dimension = output->rank; dimension > 0u; --dimension) {
        const uint32_t dim = output->dims[dimension - 1u];
        if (dim == 0u) {
            return 0u;
        }
        coordinate[dimension - 1u] = remaining % dim;
        remaining /= dim;
    }
    for (dimension = input->rank; dimension > 0u; --dimension) {
        const uint32_t input_dim = input->dims[dimension - 1u];
        const uint32_t output_dimension =
            output->rank - input->rank + dimension - 1u;
        const uint32_t output_dim = output->dims[output_dimension];
        uint32_t selected;
        if (input_dim != 1u && input_dim != output_dim) {
            return 0u;
        }
        selected = input_dim == 1u ? 0u : coordinate[output_dimension];
        result += selected * stride;
        stride *= input_dim;
    }
    *input_index = result;
    return 1u;
}
#endif


static float yr_recip_positive(float value)
{
    /*
     * Positive-normal reciprocal seed followed by four Newton steps. This
     * avoids the ET U-mode fdiv hardware erratum while converging to FP32
     * precision for the sigmoid denominator (which is always in [1, 2]).
     */
    union {
        float f;
        uint32_t u;
    } seed;
    float result;
    seed.f = value;
    seed.u = 0x7ef311c3u - seed.u;
    result = seed.f;
    result = result * (2.0f - value * result);
    result = result * (2.0f - value * result);
    result = result * (2.0f - value * result);
    result = result * (2.0f - value * result);
    return result;
}


static float yr_expf(float value)
{
    /*
     * Range reduce to r in approximately [-ln(2)/2, ln(2)/2], evaluate an
     * eighth-order Taylor polynomial there, then form 2^k exactly. Sigmoid
     * calls this only with a non-positive argument, so overflow is impossible.
     */
    const float ln2 = 0.69314718055994530942f;
    const float inv_ln2 = 1.4426950408889634074f;
    int32_t exponent;
    float reduced;
    float polynomial;
    union {
        uint32_t u;
        float f;
    } scale;

    if (value <= -87.0f) {
        return 0.0f;
    }
    if (value >= 0.0f) {
        /* This path is included for direct testing; sigmoid passes <= 0. */
        if (value >= 88.0f) {
            value = 88.0f;
        }
    }
    exponent = (int32_t)(value * inv_ln2
                       + (value >= 0.0f ? 0.5f : -0.5f));
    reduced = value - (float)exponent * ln2;

    polynomial = 1.0f / 40320.0f;
    polynomial = polynomial * reduced + 1.0f / 5040.0f;
    polynomial = polynomial * reduced + 1.0f / 720.0f;
    polynomial = polynomial * reduced + 1.0f / 120.0f;
    polynomial = polynomial * reduced + 1.0f / 24.0f;
    polynomial = polynomial * reduced + 1.0f / 6.0f;
    polynomial = polynomial * reduced + 0.5f;
    polynomial = polynomial * reduced + 1.0f;
    polynomial = polynomial * reduced + 1.0f;

    if (exponent < -126) {
        return 0.0f;
    }
    if (exponent > 127) {
        scale.u = 0x7f7fffffu;
        return scale.f;
    }
    scale.u = (uint32_t)(exponent + 127) << 23;
    return polynomial * scale.f;
}


static float yr_sigmoid_scalar(float value)
{
    float exponential;
    if (value != value) {
        return value;
    }
    if (value >= 0.0f) {
        exponential = yr_expf(-value);
        return yr_recip_positive(1.0f + exponential);
    }
    exponential = yr_expf(value);
    return exponential * yr_recip_positive(1.0f + exponential);
}


/*
 * Integer division helpers with the rounding direction stated explicitly.
 * C truncates toward zero, which is the wrong direction for the negative
 * numerators produced by padded convolution windows.
 */
static int64_t yr_ceil_div(int64_t numerator, int64_t denominator)
{
    if (numerator >= 0) {
        return (numerator + denominator - 1) / denominator;
    }
    return -((-numerator) / denominator);
}


static int64_t yr_floor_div(int64_t numerator, int64_t denominator)
{
    if (numerator >= 0) {
        return numerator / denominator;
    }
    return -(((-numerator) + denominator - 1) / denominator);
}


/*
 * Clamp a kernel axis to the taps whose input coordinate lands inside the
 * tensor, where the coordinate is base plus tap times dilation. Returns the
 * number of valid taps and writes the first one to *first_tap. Because
 * dilation is positive the valid set is contiguous, so the bounds test leaves
 * the accumulation loop entirely.
 */
static uint32_t yr_tap_range(
    int64_t base, int64_t dilation, int64_t extent, int64_t taps,
    uint32_t *first_tap)
{
    int64_t lo = yr_ceil_div(-base, dilation);
    int64_t hi = yr_floor_div(extent - 1 - base, dilation);
    if (lo < 0) {
        lo = 0;
    }
    if (hi > taps - 1) {
        hi = taps - 1;
    }
    if (hi < lo) {
        *first_tap = 0u;
        return 0u;
    }
    *first_tap = (uint32_t)lo;
    return (uint32_t)(hi - lo + 1);
}


/*
 * Split [0, count) into yr_hart_count() near-equal pieces, remainder going to
 * the lowest-indexed harts, and return the piece owned by yr_hart_id(). With
 * a single hart this is [0, count), so single-hart callers are unaffected.
 */
static void yr_hart_range(uint32_t count, uint32_t *lo, uint32_t *hi)
{
    const uint32_t harts = yr_hart_count();
    const uint32_t id = yr_hart_id();
    *lo = (count * id) / harts;
    *hi = (count * (id + 1u)) / harts;
}


static uint32_t yr_conv(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *weight_desc,
    const struct yr_tensor_desc *bias_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    const float *weight,
    const float *bias,
    float *output)
{
    uint32_t batches, input_channels, input_h, input_w;
    uint32_t output_channels, output_h, output_w;
    uint32_t channels_per_group, outputs_per_group;
    uint32_t channel_stride, kernel_stride, dilated_row_stride, plane_stride;
    uint32_t interior_first, interior_end;
    int32_t column_step, column_step2, column_step3;
    int64_t interior_lo, interior_hi;
    uint32_t n, oc, oh, ow, icg, ky, kx;
    uint32_t oc_lo, oc_hi;
    int64_t effective_h, effective_w, padded_h, padded_w;
    int64_t expected_h, expected_w;

    if (yr_tensor_dtype(input_desc) != 1u
        || yr_tensor_dtype(weight_desc) != 1u
        || yr_tensor_dtype(output_desc) != 1u
        || (bias_desc != (const struct yr_tensor_desc *)0
            && yr_tensor_dtype(bias_desc) != 1u)
        || input_desc->rank != 4u || weight_desc->rank != 4u
        || output_desc->rank != 4u || node->group <= 0
        || node->kernel_h <= 0 || node->kernel_w <= 0
        || node->stride_h <= 0 || node->stride_w <= 0
        || node->dilation_h <= 0 || node->dilation_w <= 0
        || node->pad_top < 0 || node->pad_left < 0
        || node->pad_bottom < 0 || node->pad_right < 0) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    batches = input_desc->dims[0];
    input_channels = input_desc->dims[1];
    input_h = input_desc->dims[2];
    input_w = input_desc->dims[3];
    output_channels = output_desc->dims[1];
    output_h = output_desc->dims[2];
    output_w = output_desc->dims[3];
    effective_h =
        (int64_t)node->dilation_h * (node->kernel_h - 1) + 1;
    effective_w =
        (int64_t)node->dilation_w * (node->kernel_w - 1) + 1;
    padded_h =
        (int64_t)input_h + node->pad_top + node->pad_bottom;
    padded_w =
        (int64_t)input_w + node->pad_left + node->pad_right;
    if (padded_h < effective_h || padded_w < effective_w) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    expected_h = (padded_h - effective_h) / node->stride_h + 1;
    expected_w = (padded_w - effective_w) / node->stride_w + 1;
    if (output_desc->dims[0] != batches
        || expected_h != (int64_t)output_h
        || expected_w != (int64_t)output_w
        || input_channels % (uint32_t)node->group != 0u
        || output_channels % (uint32_t)node->group != 0u
        || weight_desc->dims[0] != output_channels
        || weight_desc->dims[1] != input_channels / (uint32_t)node->group
        || weight_desc->dims[2] != (uint32_t)node->kernel_h
        || weight_desc->dims[3] != (uint32_t)node->kernel_w) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    if (bias != (const float *)0
        && (bias_desc == (const struct yr_tensor_desc *)0
            || bias_desc->rank != 1u
            || bias_desc->elements != output_channels)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }

    channels_per_group = input_channels / (uint32_t)node->group;
    outputs_per_group = output_channels / (uint32_t)node->group;
    channel_stride = input_h * input_w;
    kernel_stride = (uint32_t)node->kernel_h * (uint32_t)node->kernel_w;
    dilated_row_stride = (uint32_t)node->dilation_h * input_w;
    plane_stride = output_h * output_w;
    column_step = node->stride_w;
    column_step2 = column_step * 2;
    column_step3 = column_step * 3;

    /*
     * Output columns whose whole kernel width lands inside the input need no
     * per-column bounds work, so four of them can share one weight load.
     * Columns outside this span keep the general single-output path.
     */
    interior_lo = yr_ceil_div(node->pad_left, node->stride_w);
    interior_hi = yr_floor_div(
        (int64_t)input_w - 1
            - (int64_t)(node->kernel_w - 1) * node->dilation_w
            + node->pad_left,
        node->stride_w);
    if (interior_hi > (int64_t)output_w - 1) {
        interior_hi = (int64_t)output_w - 1;
    }
    if (interior_hi < interior_lo) {
        interior_first = output_w;
        interior_end = output_w;
    } else {
        interior_first = (uint32_t)interior_lo;
        interior_end = (uint32_t)interior_hi + 1u;
    }

    /*
     * The tap loops below walk running pointers instead of recomputing the
     * flat input and weight indices per multiply-accumulate, and the padding
     * bounds are resolved once per output position by yr_tap_range(). Skipped
     * taps contributed nothing before, so the accumulation order over
     * (icg, ky, kx) is unchanged and results stay bit-identical.
     */
    yr_hart_range(output_channels, &oc_lo, &oc_hi);
    for (n = 0; n < batches; ++n) {
        const float *const batch_input =
            input + n * input_channels * channel_stride;
        float *const batch_output =
            output + n * output_channels * plane_stride;
        oc = oc_lo;
        /*
         * Two output channels at a time wherever the pair exists and shares a
         * group. Both read the same input values, so pairing halves the input
         * loads per multiply-accumulate. On the unrolled interior path that
         * takes one pass from four loads plus one weight for four products to
         * four loads plus two weights for eight, and it doubles the number of
         * independent accumulator chains. Each accumulator still sums over
         * (icg, ky, kx) in the original order, so results stay bit-identical.
         */
        while (oc + 1u < oc_hi
               && (oc + 1u) / outputs_per_group == oc / outputs_per_group) {
            const uint32_t group = oc / outputs_per_group;
            const float *const group_input = batch_input
                + group * channels_per_group * channel_stride;
            const float *const filter_a =
                weight + oc * channels_per_group * kernel_stride;
            const float *const filter_b =
                filter_a + channels_per_group * kernel_stride;
            const float initial_a = bias == (const float *)0 ? 0.0f : bias[oc];
            const float initial_b =
                bias == (const float *)0 ? 0.0f : bias[oc + 1u];
            float *out_a = batch_output + oc * plane_stride;
            float *out_b = out_a + plane_stride;
            for (oh = 0; oh < output_h; ++oh) {
                const int64_t base_h =
                    (int64_t)oh * node->stride_h - node->pad_top;
                uint32_t first_ky;
                const uint32_t ky_count = yr_tap_range(
                    base_h, node->dilation_h, input_h, node->kernel_h,
                    &first_ky);
                const float *row_origin;
                const float *row_filter_a;
                const float *row_filter_b;
                if (ky_count == 0u) {
                    for (ow = 0; ow < output_w; ++ow) {
                        *out_a++ = initial_a;
                        *out_b++ = initial_b;
                    }
                    continue;
                }
                row_origin = group_input
                    + (base_h + (int64_t)first_ky * node->dilation_h)
                      * (int64_t)input_w;
                row_filter_a = filter_a + first_ky * (uint32_t)node->kernel_w;
                row_filter_b = filter_b + first_ky * (uint32_t)node->kernel_w;
                ow = 0u;
                while (ow < output_w) {
                    const int64_t base_w =
                        (int64_t)ow * node->stride_w - node->pad_left;
                    if (ow >= interior_first && ow + 4u <= interior_end) {
                        const float *channel = row_origin + base_w;
                        const float *tap_weight_a = row_filter_a;
                        const float *tap_weight_b = row_filter_b;
                        float first_a = initial_a;
                        float second_a = initial_a;
                        float third_a = initial_a;
                        float fourth_a = initial_a;
                        float first_b = initial_b;
                        float second_b = initial_b;
                        float third_b = initial_b;
                        float fourth_b = initial_b;
                        for (icg = 0; icg < channels_per_group; ++icg) {
                            const float *row = channel;
                            const float *row_weight_a = tap_weight_a;
                            const float *row_weight_b = tap_weight_b;
                            for (ky = 0; ky < ky_count; ++ky) {
                                const float *value = row;
                                const float *coefficient_a = row_weight_a;
                                const float *coefficient_b = row_weight_b;
                                for (kx = 0; kx < (uint32_t)node->kernel_w;
                                     ++kx) {
                                    const float scale_a = *coefficient_a++;
                                    const float scale_b = *coefficient_b++;
                                    const float value0 = value[0];
                                    const float value1 = value[column_step];
                                    const float value2 = value[column_step2];
                                    const float value3 = value[column_step3];
                                    first_a += value0 * scale_a;
                                    second_a += value1 * scale_a;
                                    third_a += value2 * scale_a;
                                    fourth_a += value3 * scale_a;
                                    first_b += value0 * scale_b;
                                    second_b += value1 * scale_b;
                                    third_b += value2 * scale_b;
                                    fourth_b += value3 * scale_b;
                                    value += node->dilation_w;
                                }
                                row += dilated_row_stride;
                                row_weight_a += (uint32_t)node->kernel_w;
                                row_weight_b += (uint32_t)node->kernel_w;
                            }
                            channel += channel_stride;
                            tap_weight_a += kernel_stride;
                            tap_weight_b += kernel_stride;
                        }
                        out_a[0] = first_a;
                        out_a[1] = second_a;
                        out_a[2] = third_a;
                        out_a[3] = fourth_a;
                        out_b[0] = first_b;
                        out_b[1] = second_b;
                        out_b[2] = third_b;
                        out_b[3] = fourth_b;
                        out_a += 4;
                        out_b += 4;
                        ow += 4u;
                    } else {
                        uint32_t first_kx;
                        const uint32_t kx_count = yr_tap_range(
                            base_w, node->dilation_w, input_w, node->kernel_w,
                            &first_kx);
                        float accumulator_a = initial_a;
                        float accumulator_b = initial_b;
                        if (kx_count != 0u) {
                            const float *channel = row_origin + base_w
                                + (int64_t)first_kx * node->dilation_w;
                            const float *tap_weight_a = row_filter_a + first_kx;
                            const float *tap_weight_b = row_filter_b + first_kx;
                            for (icg = 0; icg < channels_per_group; ++icg) {
                                const float *row = channel;
                                const float *row_weight_a = tap_weight_a;
                                const float *row_weight_b = tap_weight_b;
                                for (ky = 0; ky < ky_count; ++ky) {
                                    const float *value = row;
                                    const float *coefficient_a = row_weight_a;
                                    const float *coefficient_b = row_weight_b;
                                    for (kx = 0; kx < kx_count; ++kx) {
                                        const float sample = *value;
                                        accumulator_a += sample
                                            * *coefficient_a++;
                                        accumulator_b += sample
                                            * *coefficient_b++;
                                        value += node->dilation_w;
                                    }
                                    row += dilated_row_stride;
                                    row_weight_a += (uint32_t)node->kernel_w;
                                    row_weight_b += (uint32_t)node->kernel_w;
                                }
                                channel += channel_stride;
                                tap_weight_a += kernel_stride;
                                tap_weight_b += kernel_stride;
                            }
                        }
                        *out_a++ = accumulator_a;
                        *out_b++ = accumulator_b;
                        ow += 1u;
                    }
                }
            }
            oc += 2u;
        }
        for (; oc < oc_hi; ++oc) {
            const uint32_t group = oc / outputs_per_group;
            const float *const group_input = batch_input
                + group * channels_per_group * channel_stride;
            const float *const filter =
                weight + oc * channels_per_group * kernel_stride;
            const float initial = bias == (const float *)0 ? 0.0f : bias[oc];
            float *out = batch_output + oc * plane_stride;
            for (oh = 0; oh < output_h; ++oh) {
                const int64_t base_h =
                    (int64_t)oh * node->stride_h - node->pad_top;
                uint32_t first_ky;
                const uint32_t ky_count = yr_tap_range(
                    base_h, node->dilation_h, input_h, node->kernel_h,
                    &first_ky);
                const float *row_origin;
                const float *row_filter;
                if (ky_count == 0u) {
                    for (ow = 0; ow < output_w; ++ow) {
                        *out++ = initial;
                    }
                    continue;
                }
                row_origin = group_input
                    + (base_h + (int64_t)first_ky * node->dilation_h)
                      * (int64_t)input_w;
                row_filter = filter + first_ky * (uint32_t)node->kernel_w;
                ow = 0u;
                while (ow < output_w) {
                    const int64_t base_w =
                        (int64_t)ow * node->stride_w - node->pad_left;
                    if (ow >= interior_first && ow + 4u <= interior_end) {
                        /*
                         * Four neighbouring output columns, each keeping its
                         * own accumulator and its own tap order, so the four
                         * dependency chains interleave and the weight is
                         * fetched once for the group.
                         */
                        const float *channel = row_origin + base_w;
                        const float *tap_weight = row_filter;
                        float first = initial;
                        float second = initial;
                        float third = initial;
                        float fourth = initial;
                        for (icg = 0; icg < channels_per_group; ++icg) {
                            const float *row = channel;
                            const float *row_weight = tap_weight;
                            for (ky = 0; ky < ky_count; ++ky) {
                                const float *value = row;
                                const float *coefficient = row_weight;
                                for (kx = 0; kx < (uint32_t)node->kernel_w;
                                     ++kx) {
                                    const float scale = *coefficient++;
                                    first += value[0] * scale;
                                    second += value[column_step] * scale;
                                    third += value[column_step2] * scale;
                                    fourth += value[column_step3] * scale;
                                    value += node->dilation_w;
                                }
                                row += dilated_row_stride;
                                row_weight += (uint32_t)node->kernel_w;
                            }
                            channel += channel_stride;
                            tap_weight += kernel_stride;
                        }
                        out[0] = first;
                        out[1] = second;
                        out[2] = third;
                        out[3] = fourth;
                        out += 4;
                        ow += 4u;
                    } else {
                        uint32_t first_kx;
                        const uint32_t kx_count = yr_tap_range(
                            base_w, node->dilation_w, input_w, node->kernel_w,
                            &first_kx);
                        float accumulator = initial;
                        if (kx_count != 0u) {
                            const float *channel = row_origin + base_w
                                + (int64_t)first_kx * node->dilation_w;
                            const float *tap_weight = row_filter + first_kx;
                            for (icg = 0; icg < channels_per_group; ++icg) {
                                const float *row = channel;
                                const float *row_weight = tap_weight;
                                for (ky = 0; ky < ky_count; ++ky) {
                                    const float *value = row;
                                    const float *coefficient = row_weight;
                                    for (kx = 0; kx < kx_count; ++kx) {
                                        accumulator += *value * *coefficient;
                                        value += node->dilation_w;
                                        ++coefficient;
                                    }
                                    row += dilated_row_stride;
                                    row_weight += (uint32_t)node->kernel_w;
                                }
                                channel += channel_stride;
                                tap_weight += kernel_stride;
                            }
                        }
                        *out++ = accumulator;
                        ow += 1u;
                    }
                }
            }
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_sigmoid(
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    float *output,
    uint32_t elem_lo,
    uint32_t elem_hi)
{
    uint32_t index;
    if (yr_tensor_dtype(input_desc) != 1u
        || yr_tensor_dtype(output_desc) != 1u
        || !yr_same_shape(input_desc, output_desc)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = elem_lo; index < elem_hi; ++index) {
        output[index] = yr_sigmoid_scalar(input[index]);
    }
    return YR_STATUS_OK;
}


static uint32_t yr_mul(
    const struct yr_tensor_desc *left_desc,
    const struct yr_tensor_desc *right_desc,
    const struct yr_tensor_desc *output_desc,
    const float *left,
    const float *right,
    float *output,
    uint32_t elem_lo,
    uint32_t elem_hi)
{
    uint32_t index;
    if (yr_tensor_dtype(left_desc) != 1u
        || yr_tensor_dtype(right_desc) != 1u
        || yr_tensor_dtype(output_desc) != 1u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    if (yr_same_shape(left_desc, output_desc)
        && yr_same_shape(right_desc, output_desc)) {
        for (index = elem_lo; index < elem_hi; ++index) {
            output[index] = left[index] * right[index];
        }
        return YR_STATUS_OK;
    }
    if (yr_same_shape(left_desc, output_desc)
        && right_desc->elements == 1u
        && right_desc->rank <= output_desc->rank) {
        for (index = elem_lo; index < elem_hi; ++index) {
            output[index] = left[index] * right[0];
        }
        return YR_STATUS_OK;
    }
    if (yr_same_shape(right_desc, output_desc)
        && left_desc->elements == 1u
        && left_desc->rank <= output_desc->rank) {
        for (index = elem_lo; index < elem_hi; ++index) {
            output[index] = left[0] * right[index];
        }
        return YR_STATUS_OK;
    }
#if YR_MANIFEST_VERSION >= 2
    /*
     * General static multidirectional broadcasting.  The pinned graph uses
     * this for N285: [1,4,8400] * [1,8400].
     */
    for (index = elem_lo; index < elem_hi; ++index) {
        uint32_t left_index;
        uint32_t right_index;
        if (!yr_broadcast_index(
                left_desc, output_desc, index, &left_index)
            || !yr_broadcast_index(
                right_desc, output_desc, index, &right_index)) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        output[index] = left[left_index] * right[right_index];
    }
    return YR_STATUS_OK;
#else
    return YR_STATUS_UNSUPPORTED_SHAPE;
#endif
}


static uint32_t yr_concat(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *const *input_descs,
    const struct yr_tensor_desc *output_desc,
    const float *const *inputs,
    float *output)
{
    int32_t axis = node->axis;
    uint32_t rank = output_desc->rank;
    uint32_t outer = 1u;
    uint32_t inner = 1u;
    uint32_t expected_axis = 0u;
    uint32_t input_index;
    uint32_t dimension;
    uint32_t outer_index;
    uint32_t output_axis_offset = 0u;

    if (rank == 0u || rank > 6u || node->input_count == 0u
        || node->input_count > 4u
        || yr_tensor_dtype(output_desc) != 1u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    if (axis < 0) {
        axis += (int32_t)rank;
    }
    if (axis < 0 || axis >= (int32_t)rank) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (dimension = 0u; dimension < (uint32_t)axis; ++dimension) {
        outer *= output_desc->dims[dimension];
    }
    for (dimension = (uint32_t)axis + 1u; dimension < rank; ++dimension) {
        inner *= output_desc->dims[dimension];
    }
    for (input_index = 0u; input_index < node->input_count; ++input_index) {
        const struct yr_tensor_desc *input_desc = input_descs[input_index];
        if (input_desc == (const struct yr_tensor_desc *)0
            || inputs[input_index] == (const float *)0
            || yr_tensor_dtype(input_desc) != 1u
            || input_desc->rank != rank) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        for (dimension = 0u; dimension < rank; ++dimension) {
            if (dimension != (uint32_t)axis
                && input_desc->dims[dimension] != output_desc->dims[dimension]) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
        }
        expected_axis += input_desc->dims[axis];
    }
    if (expected_axis != output_desc->dims[axis]) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }

    for (input_index = 0u; input_index < node->input_count; ++input_index) {
        const uint32_t input_axis = input_descs[input_index]->dims[axis];
        const uint32_t input_block = input_axis * inner;
        const uint32_t output_block = output_desc->dims[axis] * inner;
        for (outer_index = 0u; outer_index < outer; ++outer_index) {
            uint32_t element;
            const uint32_t source_base = outer_index * input_block;
            const uint32_t destination_base =
                outer_index * output_block + output_axis_offset * inner;
            for (element = 0u; element < input_block; ++element) {
                output[destination_base + element] =
                    inputs[input_index][source_base + element];
            }
        }
        output_axis_offset += input_axis;
    }
    return YR_STATUS_OK;
}


#if YR_MANIFEST_VERSION >= 2

static void yr_copy_bytes(
    uint8_t *destination, const uint8_t *source, uint32_t bytes)
{
    uint32_t index;
    for (index = 0u; index < bytes; ++index) {
        destination[index] = source[index];
    }
}


static uint32_t yr_copy_tensor(
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const uint8_t *input,
    uint8_t *output)
{
    if (yr_tensor_dtype(input_desc) != yr_tensor_dtype(output_desc)
        || input_desc->elements != output_desc->elements
        || input_desc->nbytes != output_desc->nbytes) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    yr_copy_bytes(output, input, output_desc->nbytes);
    return YR_STATUS_OK;
}


static uint32_t yr_reshape(
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *shape_desc,
    const struct yr_tensor_desc *output_desc,
    const int64_t *shape,
    const uint8_t *input,
    uint8_t *output)
{
    uint64_t known_elements = 1u;
    int32_t inferred_axis = -1;
    uint32_t dimension;

    /*
     * ONNX Reshape-13 with allowzero=0.  The package generator rejects any
     * other allowzero contract, while this check independently validates the
     * actual INT64 shape initializer and inferred static output descriptor.
     */
    if (yr_tensor_dtype(shape_desc) != YR_DTYPE_INT64
        || shape_desc->rank != 1u
        || shape_desc->elements != output_desc->rank
        || output_desc->rank > 6u
        || yr_tensor_dtype(input_desc) != yr_tensor_dtype(output_desc)
        || input_desc->elements != output_desc->elements) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (dimension = 0u; dimension < output_desc->rank; ++dimension) {
        const int64_t requested = shape[dimension];
        uint32_t expected;
        if (requested == 0) {
            if (dimension >= input_desc->rank) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
            expected = input_desc->dims[dimension];
        } else if (requested == -1) {
            if (inferred_axis >= 0) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
            inferred_axis = (int32_t)dimension;
            continue;
        } else if (requested > 0
                   && requested <= (int64_t)UINT32_MAX) {
            expected = (uint32_t)requested;
        } else {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        if (output_desc->dims[dimension] != expected) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        known_elements *= expected;
        if (known_elements > UINT32_MAX) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
    }
    if (inferred_axis >= 0) {
        if (known_elements == 0u
            || input_desc->elements % (uint32_t)known_elements != 0u
            || output_desc->dims[(uint32_t)inferred_axis]
                != input_desc->elements / (uint32_t)known_elements) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
    } else if (known_elements != input_desc->elements) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    return yr_copy_tensor(input_desc, output_desc, input, output);
}


static uint32_t yr_flatten(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const uint8_t *input,
    uint8_t *output)
{
    int32_t axis = node->axis;
    uint32_t outer;
    uint32_t inner;
    if (axis < 0) {
        axis += (int32_t)input_desc->rank;
    }
    if (axis < 0 || axis > (int32_t)input_desc->rank
        || output_desc->rank != 2u
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis, input_desc->rank, &inner)
        || output_desc->dims[0] != outer
        || output_desc->dims[1] != inner) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    return yr_copy_tensor(input_desc, output_desc, input, output);
}


static uint32_t yr_add_sub(
    const struct yr_tensor_desc *left_desc,
    const struct yr_tensor_desc *right_desc,
    const struct yr_tensor_desc *output_desc,
    const float *left,
    const float *right,
    float *output,
    uint32_t subtract,
    uint32_t elem_lo,
    uint32_t elem_hi)
{
    uint32_t index;
    if (yr_tensor_dtype(left_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(right_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || !yr_same_shape(left_desc, right_desc)
        || !yr_same_shape(left_desc, output_desc)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = elem_lo; index < elem_hi; ++index) {
        output[index] =
            subtract ? left[index] - right[index] : left[index] + right[index];
    }
    return YR_STATUS_OK;
}


static uint32_t yr_split(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *sizes_desc,
    const struct yr_tensor_desc *const *output_descs,
    const uint8_t *input,
    const int64_t *sizes,
    uint8_t *const *outputs)
{
    int32_t axis = yr_normalize_axis(node->axis, input_desc->rank);
    uint32_t outer;
    uint32_t inner;
    uint32_t element_bytes = yr_dtype_bytes(yr_tensor_dtype(input_desc));
    uint32_t output_index;
    uint32_t source_axis_offset = 0u;
    uint64_t axis_sum = 0u;

    if (axis < 0 || node->input_count != 2u
        || node->output_count == 0u || node->output_count > 3u
        || yr_tensor_dtype(sizes_desc) != YR_DTYPE_INT64
        || sizes_desc->elements != node->output_count
        || element_bytes == 0u
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis + 1u, input_desc->rank, &inner)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (output_index = 0u; output_index < node->output_count; ++output_index) {
        const struct yr_tensor_desc *output_desc = output_descs[output_index];
        uint32_t dimension;
        if (sizes[output_index] < 0
            || output_desc == (const struct yr_tensor_desc *)0
            || outputs[output_index] == (uint8_t *)0
            || yr_tensor_dtype(output_desc) != yr_tensor_dtype(input_desc)
            || output_desc->rank != input_desc->rank
            || output_desc->dims[axis] != (uint32_t)sizes[output_index]) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        for (dimension = 0u; dimension < input_desc->rank; ++dimension) {
            if (dimension != (uint32_t)axis
                && output_desc->dims[dimension]
                    != input_desc->dims[dimension]) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
        }
        axis_sum += (uint64_t)sizes[output_index];
    }
    if (axis_sum != input_desc->dims[axis]) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (output_index = 0u; output_index < node->output_count; ++output_index) {
        const uint32_t output_axis =
            output_descs[output_index]->dims[axis];
        const uint32_t copy_elements = output_axis * inner;
        const uint32_t source_block = input_desc->dims[axis] * inner;
        uint32_t outer_index;
        for (outer_index = 0u; outer_index < outer; ++outer_index) {
            const uint32_t source_element =
                outer_index * source_block + source_axis_offset * inner;
            const uint32_t destination_element =
                outer_index * copy_elements;
            yr_copy_bytes(
                outputs[output_index]
                    + destination_element * element_bytes,
                input + source_element * element_bytes,
                copy_elements * element_bytes);
        }
        source_axis_offset += output_axis;
    }
    return YR_STATUS_OK;
}


/*
 * channel_lo/channel_hi restrict the sweep to a sub-range of channels, the
 * same per-hart split Conv already uses, so multiple harts can share one
 * node's channels instead of every hart redundantly computing every
 * channel. Safe at the channel granularity used by the caller because
 * this graph's every NCHW plane (output_h * output_w * 4 bytes) is a
 * multiple of the 64-byte cache line, the same invariant the Conv publish
 * split already relies on.
 */
static uint32_t yr_maxpool(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    float *output,
    uint32_t channel_lo,
    uint32_t channel_hi)
{
    uint32_t n, channel, oh, ow, ky, kx;
    const uint32_t batches = input_desc->dims[0];
    const uint32_t channels = input_desc->dims[1];
    const uint32_t input_h = input_desc->dims[2];
    const uint32_t input_w = input_desc->dims[3];
    const uint32_t output_h = output_desc->dims[2];
    const uint32_t output_w = output_desc->dims[3];

    if (yr_tensor_dtype(input_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || input_desc->rank != 4u || output_desc->rank != 4u
        || output_desc->dims[0] != batches
        || output_desc->dims[1] != channels
        || node->kernel_h <= 0 || node->kernel_w <= 0
        || node->stride_h <= 0 || node->stride_w <= 0
        || node->dilation_h <= 0 || node->dilation_w <= 0
        || node->pad_top < 0 || node->pad_left < 0
        || node->pad_bottom < 0 || node->pad_right < 0
        || node->ceil_mode != 0
        || (uint64_t)output_h
            != ((uint64_t)input_h + (uint32_t)node->pad_top
                + (uint32_t)node->pad_bottom
                - ((uint64_t)(uint32_t)node->dilation_h
                   * ((uint32_t)node->kernel_h - 1u) + 1u))
               / (uint32_t)node->stride_h + 1u
        || (uint64_t)output_w
            != ((uint64_t)input_w + (uint32_t)node->pad_left
                + (uint32_t)node->pad_right
                - ((uint64_t)(uint32_t)node->dilation_w
                   * ((uint32_t)node->kernel_w - 1u) + 1u))
               / (uint32_t)node->stride_w + 1u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (n = 0u; n < batches; ++n) {
        for (channel = channel_lo; channel < channel_hi; ++channel) {
            for (oh = 0u; oh < output_h; ++oh) {
                for (ow = 0u; ow < output_w; ++ow) {
                    float best = 0.0f;
                    uint32_t have_value = 0u;
                    for (ky = 0u; ky < (uint32_t)node->kernel_h; ++ky) {
                        const int32_t ih =
                            (int32_t)(oh * (uint32_t)node->stride_h)
                            - node->pad_top
                            + (int32_t)(ky * (uint32_t)node->dilation_h);
                        if (ih < 0 || ih >= (int32_t)input_h) {
                            continue;
                        }
                        for (kx = 0u; kx < (uint32_t)node->kernel_w; ++kx) {
                            const int32_t iw =
                                (int32_t)(ow * (uint32_t)node->stride_w)
                                - node->pad_left
                                + (int32_t)(kx * (uint32_t)node->dilation_w);
                            float value;
                            if (iw < 0 || iw >= (int32_t)input_w) {
                                continue;
                            }
                            value = input[
                                ((n * channels + channel) * input_h
                                 + (uint32_t)ih) * input_w + (uint32_t)iw];
                            if (!have_value || value > best) {
                                best = value;
                                have_value = 1u;
                            }
                        }
                    }
                    if (!have_value) {
                        return YR_STATUS_UNSUPPORTED_SHAPE;
                    }
                    output[
                        ((n * channels + channel) * output_h + oh)
                        * output_w + ow] = best;
                }
            }
        }
    }
    return YR_STATUS_OK;
}


/*
 * channel_lo/channel_hi split channels across harts the same way yr_maxpool()
 * now does, resting on the same NCHW-plane-is-a-multiple-of-64-bytes
 * invariant.
 */
static uint32_t yr_resize(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *scales_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    const float *scales,
    float *output,
    uint32_t channel_lo,
    uint32_t channel_hi)
{
    uint32_t n, channel, oh, ow;
    const uint32_t input_h = input_desc->dims[2];
    const uint32_t input_w = input_desc->dims[3];
    const uint32_t output_h = output_desc->dims[2];
    const uint32_t output_w = output_desc->dims[3];
    if (node->input_count != 3u
        || node->resize_nearest_asymmetric_floor != 1
        || yr_tensor_dtype(input_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(scales_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || input_desc->rank != 4u || output_desc->rank != 4u
        || scales_desc->elements != 4u
        || scales[0] != 1.0f || scales[1] != 1.0f
        || scales[2] != 2.0f || scales[3] != 2.0f
        || output_desc->dims[0] != input_desc->dims[0]
        || output_desc->dims[1] != input_desc->dims[1]
        || output_h != input_h * 2u || output_w != input_w * 2u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (n = 0u; n < input_desc->dims[0]; ++n) {
        for (channel = channel_lo; channel < channel_hi; ++channel) {
            for (oh = 0u; oh < output_h; ++oh) {
                const uint32_t ih = oh / 2u;
                for (ow = 0u; ow < output_w; ++ow) {
                    const uint32_t iw = ow / 2u;
                    output[
                        ((n * output_desc->dims[1] + channel) * output_h + oh)
                        * output_w + ow] =
                        input[
                            ((n * input_desc->dims[1] + channel) * input_h + ih)
                            * input_w + iw];
                }
            }
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_matmul(
    const struct yr_tensor_desc *left_desc,
    const struct yr_tensor_desc *right_desc,
    const struct yr_tensor_desc *output_desc,
    const float *left,
    const float *right,
    float *output)
{
    uint32_t batch_count;
    uint32_t batch;
    uint32_t row;
    uint32_t column;
    uint32_t reduction;
    uint32_t dimension;
    uint32_t m, k, n;
    if (yr_tensor_dtype(left_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(right_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || left_desc->rank < 2u || left_desc->rank > 6u
        || right_desc->rank != left_desc->rank
        || output_desc->rank != left_desc->rank) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (dimension = 0u; dimension + 2u < left_desc->rank; ++dimension) {
        if (left_desc->dims[dimension] != right_desc->dims[dimension]
            || left_desc->dims[dimension] != output_desc->dims[dimension]) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
    }
    m = left_desc->dims[left_desc->rank - 2u];
    k = left_desc->dims[left_desc->rank - 1u];
    if (right_desc->dims[right_desc->rank - 2u] != k) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    n = right_desc->dims[right_desc->rank - 1u];
    if (output_desc->dims[output_desc->rank - 2u] != m
        || output_desc->dims[output_desc->rank - 1u] != n
        || !yr_shape_product(
            left_desc, 0u, left_desc->rank - 2u, &batch_count)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (batch = 0u; batch < batch_count; ++batch) {
        const uint32_t left_base = batch * m * k;
        const uint32_t right_base = batch * k * n;
        const uint32_t output_base = batch * m * n;
        for (row = 0u; row < m; ++row) {
            for (column = 0u; column < n; ++column) {
                float accumulator = 0.0f;
                for (reduction = 0u; reduction < k; ++reduction) {
                    accumulator +=
                        left[left_base + row * k + reduction]
                        * right[right_base + reduction * n + column];
                }
                output[output_base + row * n + column] = accumulator;
            }
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_softmax(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    float *output)
{
    const int32_t axis = yr_normalize_axis(node->axis, input_desc->rank);
    uint32_t outer;
    uint32_t inner;
    uint32_t outer_index;
    uint32_t inner_index;
    uint32_t axis_index;
    uint32_t axis_size;
    if (axis < 0
        || yr_tensor_dtype(input_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || !yr_same_shape(input_desc, output_desc)
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis + 1u, input_desc->rank, &inner)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    axis_size = input_desc->dims[axis];
    for (outer_index = 0u; outer_index < outer; ++outer_index) {
        for (inner_index = 0u; inner_index < inner; ++inner_index) {
            const uint32_t base =
                outer_index * axis_size * inner + inner_index;
            float maximum = input[base];
            float sum = 0.0f;
            float reciprocal;
            for (axis_index = 1u; axis_index < axis_size; ++axis_index) {
                const float value = input[base + axis_index * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }
            for (axis_index = 0u; axis_index < axis_size; ++axis_index) {
                const uint32_t offset = base + axis_index * inner;
                const float value = yr_expf(input[offset] - maximum);
                output[offset] = value;
                sum += value;
            }
            if (!(sum > 0.0f)) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
            reciprocal = yr_recip_positive(sum);
            for (axis_index = 0u; axis_index < axis_size; ++axis_index) {
                const uint32_t offset = base + axis_index * inner;
                output[offset] *= reciprocal;
            }
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_transpose(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const uint8_t *input,
    uint8_t *output)
{
    uint32_t input_strides[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t coordinates[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t seen = 0u;
    uint32_t element_bytes = yr_dtype_bytes(yr_tensor_dtype(input_desc));
    uint32_t dimension;
    uint32_t output_index;
    if (input_desc->rank == 0u || input_desc->rank > 6u
        || output_desc->rank != input_desc->rank
        || node->perm_count != input_desc->rank
        || yr_tensor_dtype(output_desc) != yr_tensor_dtype(input_desc)
        || output_desc->elements != input_desc->elements
        || element_bytes == 0u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    input_strides[input_desc->rank - 1u] = 1u;
    for (dimension = input_desc->rank - 1u; dimension > 0u; --dimension) {
        input_strides[dimension - 1u] =
            input_strides[dimension] * input_desc->dims[dimension];
    }
    for (dimension = 0u; dimension < input_desc->rank; ++dimension) {
        const int32_t source_dimension = node->perm[dimension];
        if (source_dimension < 0
            || source_dimension >= (int32_t)input_desc->rank
            || (seen & (1u << (uint32_t)source_dimension)) != 0u
            || output_desc->dims[dimension]
                != input_desc->dims[source_dimension]) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        seen |= 1u << (uint32_t)source_dimension;
    }
    for (output_index = 0u; output_index < output_desc->elements;
         ++output_index) {
        uint32_t remaining = output_index;
        uint32_t input_index = 0u;
        for (dimension = output_desc->rank; dimension > 0u; --dimension) {
            const uint32_t dim = output_desc->dims[dimension - 1u];
            coordinates[dimension - 1u] = remaining % dim;
            remaining /= dim;
        }
        for (dimension = 0u; dimension < output_desc->rank; ++dimension) {
            input_index += coordinates[dimension]
                * input_strides[(uint32_t)node->perm[dimension]];
        }
        yr_copy_bytes(
            output + output_index * element_bytes,
            input + input_index * element_bytes,
            element_bytes);
    }
    return YR_STATUS_OK;
}


static uint32_t yr_reduce_max(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    float *output)
{
    int32_t axis;
    uint32_t outer;
    uint32_t inner;
    uint32_t axis_size;
    uint32_t outer_index;
    uint32_t inner_index;
    uint32_t axis_index;
    if (node->axes_count != 1u
        || yr_tensor_dtype(input_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    axis = yr_normalize_axis(node->axes[0], input_desc->rank);
    if (axis < 0
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis + 1u, input_desc->rank, &inner)
        || output_desc->elements != outer * inner
        || (node->keepdims == 0
            && output_desc->rank + 1u != input_desc->rank)
        || (node->keepdims != 0
            && output_desc->rank != input_desc->rank)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    {
        uint32_t input_dimension;
        uint32_t output_dimension = 0u;
        for (input_dimension = 0u; input_dimension < input_desc->rank;
             ++input_dimension) {
            if (input_dimension == (uint32_t)axis) {
                if (node->keepdims != 0
                    && output_desc->dims[output_dimension++] != 1u) {
                    return YR_STATUS_UNSUPPORTED_SHAPE;
                }
            } else if (output_desc->dims[output_dimension++]
                       != input_desc->dims[input_dimension]) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
        }
    }
    axis_size = input_desc->dims[axis];
    for (outer_index = 0u; outer_index < outer; ++outer_index) {
        for (inner_index = 0u; inner_index < inner; ++inner_index) {
            const uint32_t base =
                outer_index * axis_size * inner + inner_index;
            float maximum = input[base];
            for (axis_index = 1u; axis_index < axis_size; ++axis_index) {
                const float value = input[base + axis_index * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }
            output[outer_index * inner + inner_index] = maximum;
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_topk(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *k_desc,
    const struct yr_tensor_desc *values_desc,
    const struct yr_tensor_desc *indices_desc,
    const float *input,
    const int64_t *k_data,
    float *values,
    int64_t *indices)
{
    const int32_t axis = yr_normalize_axis(node->axis, input_desc->rank);
    uint32_t outer;
    uint32_t inner;
    uint32_t axis_size;
    uint32_t k;
    uint32_t outer_index;
    uint32_t inner_index;
    if (axis < 0 || node->largest != 1 || node->sorted != 1
        || yr_tensor_dtype(input_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(k_desc) != YR_DTYPE_INT64
        || yr_tensor_dtype(values_desc) != YR_DTYPE_FLOAT
        || yr_tensor_dtype(indices_desc) != YR_DTYPE_INT64
        || k_desc->elements != 1u || k_data[0] <= 0
        || k_data[0] > (int64_t)UINT32_MAX
        || values_desc->rank != input_desc->rank
        || indices_desc->rank != input_desc->rank
        || !yr_same_shape(values_desc, indices_desc)
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis + 1u, input_desc->rank, &inner)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    k = (uint32_t)k_data[0];
    axis_size = input_desc->dims[axis];
    if (k > axis_size || values_desc->dims[axis] != k) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (outer_index = 0u; outer_index < outer; ++outer_index) {
        for (inner_index = 0u; inner_index < inner; ++inner_index) {
            uint32_t populated = 0u;
            uint32_t candidate_index;
            for (candidate_index = 0u; candidate_index < axis_size;
                 ++candidate_index) {
                const uint32_t input_offset =
                    (outer_index * axis_size + candidate_index) * inner
                    + inner_index;
                const float candidate = input[input_offset];
                uint32_t position = 0u;
                while (position < populated) {
                    const uint32_t output_offset =
                        (outer_index * k + position) * inner + inner_index;
                    const float existing = values[output_offset];
                    const int64_t existing_index = indices[output_offset];
                    if (candidate > existing
                        || (candidate == existing
                            && candidate_index < (uint32_t)existing_index)) {
                        break;
                    }
                    ++position;
                }
                if (position >= k) {
                    continue;
                }
                {
                    uint32_t shift =
                        populated < k ? populated : k - 1u;
                    while (shift > position) {
                        const uint32_t destination =
                            (outer_index * k + shift) * inner + inner_index;
                        const uint32_t source =
                            (outer_index * k + shift - 1u) * inner
                            + inner_index;
                        values[destination] = values[source];
                        indices[destination] = indices[source];
                        --shift;
                    }
                }
                values[(outer_index * k + position) * inner + inner_index] =
                    candidate;
                indices[(outer_index * k + position) * inner + inner_index] =
                    (int64_t)candidate_index;
                if (populated < k) {
                    ++populated;
                }
            }
            if (populated != k) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_unsqueeze(
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *axes_desc,
    const struct yr_tensor_desc *output_desc,
    const int64_t *axes,
    const uint8_t *input,
    uint8_t *output)
{
    uint32_t seen = 0u;
    uint32_t axis_index;
    if (yr_tensor_dtype(axes_desc) != YR_DTYPE_INT64
        || output_desc->rank != input_desc->rank + axes_desc->elements
        || output_desc->rank > 6u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (axis_index = 0u; axis_index < axes_desc->elements; ++axis_index) {
        int64_t axis = axes[axis_index];
        if (axis < 0) {
            axis += (int64_t)output_desc->rank;
        }
        if (axis < 0 || axis >= (int64_t)output_desc->rank
            || (seen & (1u << (uint32_t)axis)) != 0u
            || output_desc->dims[axis] != 1u) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        seen |= 1u << (uint32_t)axis;
    }
    {
        uint32_t input_dimension = 0u;
        uint32_t output_dimension;
        for (output_dimension = 0u;
             output_dimension < output_desc->rank;
             ++output_dimension) {
            if ((seen & (1u << output_dimension)) == 0u) {
                if (input_dimension >= input_desc->rank
                    || output_desc->dims[output_dimension]
                        != input_desc->dims[input_dimension++]) {
                    return YR_STATUS_UNSUPPORTED_SHAPE;
                }
            }
        }
        if (input_dimension != input_desc->rank) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
    }
    return yr_copy_tensor(input_desc, output_desc, input, output);
}


static uint32_t yr_tile(
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *repeats_desc,
    const struct yr_tensor_desc *output_desc,
    const int64_t *repeats,
    const uint8_t *input,
    uint8_t *output)
{
    uint32_t input_strides[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t coordinates[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t element_bytes = yr_dtype_bytes(yr_tensor_dtype(input_desc));
    uint32_t dimension;
    uint32_t output_index;
    if (yr_tensor_dtype(repeats_desc) != YR_DTYPE_INT64
        || repeats_desc->elements != input_desc->rank
        || output_desc->rank != input_desc->rank
        || yr_tensor_dtype(output_desc) != yr_tensor_dtype(input_desc)
        || input_desc->rank == 0u || input_desc->rank > 6u
        || element_bytes == 0u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    input_strides[input_desc->rank - 1u] = 1u;
    for (dimension = input_desc->rank - 1u; dimension > 0u; --dimension) {
        input_strides[dimension - 1u] =
            input_strides[dimension] * input_desc->dims[dimension];
    }
    for (dimension = 0u; dimension < input_desc->rank; ++dimension) {
        if (repeats[dimension] <= 0
            || repeats[dimension] > (int64_t)UINT32_MAX
            || output_desc->dims[dimension]
                != input_desc->dims[dimension]
                    * (uint32_t)repeats[dimension]) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
    }
    for (output_index = 0u; output_index < output_desc->elements;
         ++output_index) {
        uint32_t remaining = output_index;
        uint32_t input_index = 0u;
        for (dimension = output_desc->rank; dimension > 0u; --dimension) {
            const uint32_t dim = output_desc->dims[dimension - 1u];
            coordinates[dimension - 1u] = remaining % dim;
            remaining /= dim;
        }
        for (dimension = 0u; dimension < input_desc->rank; ++dimension) {
            input_index +=
                (coordinates[dimension] % input_desc->dims[dimension])
                * input_strides[dimension];
        }
        yr_copy_bytes(
            output + output_index * element_bytes,
            input + input_index * element_bytes,
            element_bytes);
    }
    return YR_STATUS_OK;
}


static uint32_t yr_gather_elements(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *data_desc,
    const struct yr_tensor_desc *indices_desc,
    const struct yr_tensor_desc *output_desc,
    const uint8_t *data,
    const int64_t *indices,
    uint8_t *output)
{
    const int32_t axis = yr_normalize_axis(node->axis, data_desc->rank);
    uint32_t data_strides[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t coordinates[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t element_bytes = yr_dtype_bytes(yr_tensor_dtype(data_desc));
    uint32_t dimension;
    uint32_t output_index;
    if (axis < 0 || data_desc->rank == 0u || data_desc->rank > 6u
        || indices_desc->rank != data_desc->rank
        || output_desc->rank != data_desc->rank
        || !yr_same_shape(indices_desc, output_desc)
        || yr_tensor_dtype(indices_desc) != YR_DTYPE_INT64
        || yr_tensor_dtype(output_desc) != yr_tensor_dtype(data_desc)
        || element_bytes == 0u) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    data_strides[data_desc->rank - 1u] = 1u;
    for (dimension = data_desc->rank - 1u; dimension > 0u; --dimension) {
        data_strides[dimension - 1u] =
            data_strides[dimension] * data_desc->dims[dimension];
    }
    for (output_index = 0u; output_index < output_desc->elements;
         ++output_index) {
        uint32_t remaining = output_index;
        uint32_t data_index = 0u;
        int64_t selected = indices[output_index];
        if (selected < 0) {
            selected += data_desc->dims[axis];
        }
        if (selected < 0 || selected >= (int64_t)data_desc->dims[axis]) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        for (dimension = output_desc->rank; dimension > 0u; --dimension) {
            const uint32_t dim = output_desc->dims[dimension - 1u];
            coordinates[dimension - 1u] = remaining % dim;
            remaining /= dim;
        }
        for (dimension = 0u; dimension < data_desc->rank; ++dimension) {
            const uint32_t coordinate =
                dimension == (uint32_t)axis
                ? (uint32_t)selected : coordinates[dimension];
            if (coordinate >= data_desc->dims[dimension]) {
                return YR_STATUS_UNSUPPORTED_SHAPE;
            }
            data_index += coordinate * data_strides[dimension];
        }
        yr_copy_bytes(
            output + output_index * element_bytes,
            data + data_index * element_bytes,
            element_bytes);
    }
    return YR_STATUS_OK;
}


static uint32_t yr_integer_binary(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *left_desc,
    const struct yr_tensor_desc *right_desc,
    const struct yr_tensor_desc *output_desc,
    const int64_t *left,
    const int64_t *right,
    int64_t *output,
    uint32_t modulo,
    uint32_t elem_lo,
    uint32_t elem_hi)
{
    uint32_t index;
    if (yr_tensor_dtype(left_desc) != YR_DTYPE_INT64
        || yr_tensor_dtype(right_desc) != YR_DTYPE_INT64
        || yr_tensor_dtype(output_desc) != YR_DTYPE_INT64
        || !yr_same_shape(left_desc, output_desc)
        || right_desc->elements != 1u
        || right[0] == 0
        || (modulo && node->fmod != 0)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = elem_lo; index < elem_hi; ++index) {
        output[index] =
            modulo ? left[index] % right[0] : left[index] / right[0];
    }
    return YR_STATUS_OK;
}


static uint32_t yr_cast(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const int64_t *input,
    float *output,
    uint32_t elem_lo,
    uint32_t elem_hi)
{
    uint32_t index;
    if (node->to != 1 /* ONNX TensorProto.FLOAT */
        || yr_tensor_dtype(input_desc) != YR_DTYPE_INT64
        || yr_tensor_dtype(output_desc) != YR_DTYPE_FLOAT
        || !yr_same_shape(input_desc, output_desc)) {
        return YR_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = elem_lo; index < elem_hi; ++index) {
        /*
         * The pinned Cast consumes class indices produced by Mod 80, hence
         * every value is exactly representable as signed INT32 and FP32.
         * Narrow only after checking the value.  This deliberately makes the
         * ET compiler emit the supported fcvt.s.w instruction instead of
         * fcvt.s.l, which traps in the repository system emulator.
         */
        if (input[index] < INT32_MIN || input[index] > INT32_MAX) {
            return YR_STATUS_UNSUPPORTED_SHAPE;
        }
        output[index] = (float)(int32_t)input[index];
    }
    return YR_STATUS_OK;
}

#endif /* YR_MANIFEST_VERSION >= 2 */


static uint64_t yr_fnv1a(const uint8_t *data, uint32_t bytes)
{
    uint64_t hash = 14695981039346656037ull;
    uint32_t index;
    for (index = 0; index < bytes; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ull;
    }
    return hash;
}


uint32_t yr_prepare_result(
    uint8_t *device_base, struct yr_result_header *result)
{
    uint32_t reserved_index;
    uint32_t workspace_index;
    uint8_t *workspace;

    if (device_base == (uint8_t *)0
        || result != (struct yr_result_header *)(
            device_base + YR_RESULT_DEVICE_OFFSET)
        || !yr_memory_map_valid()) {
        return YR_STATUS_BAD_MANIFEST;
    }

    result->magic = YR_RESULT_MAGIC;
    result->version = YR_RESULT_VERSION;
    result->status = YR_STATUS_OK;
    result->failed_node = 0xffffffffu;
    result->failed_op = 0u;
    result->first_node = YR_FIRST_NODE;
    result->last_node = YR_LAST_NODE;
    result->node_count = YR_NODE_COUNT;
    result->tensor_count = YR_TENSOR_COUNT;
    result->workspace_bytes = YR_WORKSPACE_BYTES;
    result->input_blob_bytes = YR_INPUT_BLOB_BYTES;
    result->weight_blob_bytes = YR_WEIGHT_BLOB_BYTES;
    result->math_version = YR_MATH_VERSION;
    for (reserved_index = 0u; reserved_index < 3u; ++reserved_index) {
        result->reserved32[reserved_index] = 0u;
    }
    result->workspace_fnv1a = 0u;
    for (reserved_index = 0u; reserved_index < 7u; ++reserved_index) {
        result->reserved64[reserved_index] = 0u;
    }
    if (!yr_manifest_valid()) {
        result->status = YR_STATUS_BAD_MANIFEST;
        return YR_STATUS_BAD_MANIFEST;
    }
    workspace =
        device_base + YR_RESULT_DEVICE_OFFSET + YR_RESULT_HEADER_BYTES;
    for (workspace_index = 0u; workspace_index < YR_WORKSPACE_BYTES;
         ++workspace_index) {
        workspace[workspace_index] = 0u;
    }
    return YR_STATUS_OK;
}


uint32_t yr_run_node_span(
    uint8_t *device_base,
    struct yr_result_header *result,
    uint32_t first_local_node,
    uint32_t last_local_node)
{
    uint32_t node_index;
    uint32_t status = YR_STATUS_OK;

    if (device_base == (uint8_t *)0
        || result == (struct yr_result_header *)0
        || first_local_node > last_local_node
        || last_local_node >= YR_NODE_COUNT) {
        if (result != (struct yr_result_header *)0) {
            result->status = YR_STATUS_BAD_MANIFEST;
        }
        return YR_STATUS_BAD_MANIFEST;
    }
    for (node_index = first_local_node;
         node_index <= last_local_node;
         ++node_index) {
        const struct yr_node_desc *node = &yr_nodes[node_index];
        const struct yr_tensor_desc *in0;
        const struct yr_tensor_desc *out0;
        uint8_t *in0_raw;
        uint8_t *out0_raw;
        uint32_t output_index;
        /*
         * Every non-Conv op previously ran redundantly on all 16 harts
         * (each hart computing the full output, then publishing only a
         * cache-line slice of it). Measurement this session showed that
         * redundant path corrupts data on real hardware from partway
         * through the graph onward, well beyond the narrower GatherElements
         * failures it was already known for; this reproduces even on the
         * board's previously-"proven" build, so it predates today's
         * changes and is not one specific op's bug. Conv is unaffected,
         * since each hart computes and publishes only its own output-channel
         * slice, never redundantly, so there is no multi-hart interaction
         * to race. Rather than chase the exact hardware mechanism further,
         * every non-Conv op now runs on hart 0 alone; every other hart
         * skips the work but still calls the per-node barrier once, the
         * same as hart 0 does after publishing below, so harts 1-15 wait
         * for hart 0 right here instead of racing ahead into the next
         * Conv node's input before hart 0 has finished producing it. Conv
         * still gets full 16-hart channel-partitioned throughput, which is
         * where nearly all of a CNN's compute cost lives, so this keeps
         * most of the real speedup while removing the entire class of bug
         * by construction instead of by a fix this session could not
         * fully verify.
         */
        if (node->op != YR_OP_CONV && yr_hart_id() != 0u) {
            yr_hart_barrier();
            continue;
        }
        if (node->input_count == 0u || node->output_count == 0u
            || node->inputs[0] >= YR_TENSOR_COUNT
            || node->outputs[0] >= YR_TENSOR_COUNT) {
            status = YR_STATUS_BAD_MANIFEST;
            goto fail;
        }
        for (output_index = 0u;
             output_index < node->output_count;
             ++output_index) {
            if (output_index >= 3u
                || node->outputs[output_index] >= YR_TENSOR_COUNT
                || yr_tensors[node->outputs[output_index]].storage
                    != YR_STORAGE_WORKSPACE) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
        }
        in0 = &yr_tensors[node->inputs[0]];
        out0 = &yr_tensors[node->outputs[0]];
        in0_raw = yr_tensor_raw(device_base, in0);
        out0_raw = yr_tensor_raw(device_base, out0);
        if (in0_raw == (uint8_t *)0 || out0_raw == (uint8_t *)0) {
            status = YR_STATUS_BAD_MANIFEST;
            goto fail;
        }

        if (node->op == YR_OP_CONV) {
            const struct yr_tensor_desc *weights;
            const struct yr_tensor_desc *bias_desc = 0;
            const float *bias_data = 0;
            uint32_t tensor_oc_lo = 0u, tensor_oc_hi = 0u;
            if ((node->input_count != 2u && node->input_count != 3u)
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            weights = &yr_tensors[node->inputs[1]];
            if (weights->storage != YR_STORAGE_WEIGHTS
                || yr_tensor_ptr(device_base, weights) == (float *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            if (node->input_count == 3u) {
                if (node->inputs[2] >= YR_TENSOR_COUNT) {
                    status = YR_STATUS_BAD_MANIFEST;
                    goto fail;
                }
                bias_desc = &yr_tensors[node->inputs[2]];
                bias_data = yr_tensor_ptr(device_base, bias_desc);
                if (bias_desc->storage != YR_STORAGE_WEIGHTS
                    || bias_data == (float *)0) {
                    status = YR_STATUS_BAD_MANIFEST;
                    goto fail;
                }
            }
            /*
             * yr_conv_tensor() partitions its own work per hart, computing
             * only its tile-aligned slice of output channels and reporting
             * that slice back through tensor_oc_lo/tensor_oc_hi so the
             * publish below covers exactly what this hart wrote. Harts that
             * own no tile for this node get an empty range and publish
             * nothing, which is why the range comes back from the callee
             * instead of being assumed here.
             *
             * The mode switch this path depends on now happens once per hart
             * in yr_conv_tensor_init() at runner entry, so nothing about it
             * is per-node or shared between harts any more.
             */
            if (YR_CONV_TENSOR_ENABLED
                && yr_conv_tensor(
                    node, in0, weights, bias_desc, out0, (float *)in0_raw,
                    yr_tensor_ptr(device_base, weights), bias_data,
                    (float *)out0_raw, &tensor_oc_lo, &tensor_oc_hi) != 0u) {
                status = YR_STATUS_OK;
            } else {
                status = yr_conv(
                    node, in0, weights, bias_desc, out0, (float *)in0_raw,
                    yr_tensor_ptr(device_base, weights), bias_data,
                    (float *)out0_raw);
                yr_hart_range(out0->dims[1], &tensor_oc_lo, &tensor_oc_hi);
            }
            if (status == YR_STATUS_OK) {
                const uint32_t oc_lo = tensor_oc_lo;
                const uint32_t oc_hi = tensor_oc_hi;
                if (oc_hi > oc_lo) {
                    const uint32_t plane_bytes =
                        out0->dims[2] * out0->dims[3] * 4u;
                    const uint32_t batch_bytes = out0->dims[1] * plane_bytes;
                    uint32_t batch_index;
                    for (batch_index = 0u; batch_index < out0->dims[0];
                         ++batch_index) {
                        yr_publish(
                            out0_raw + (uint64_t)batch_index * batch_bytes
                                + (uint64_t)oc_lo * plane_bytes,
                            (oc_hi - oc_lo) * plane_bytes);
                    }
                }
            }
            yr_hart_barrier();
        } else if (node->op == YR_OP_SIGMOID) {
            uint32_t elem_lo, elem_hi;
            if (node->input_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            elem_lo = 0u; elem_hi = out0->elements;
            status = yr_sigmoid(
                in0, out0, (float *)in0_raw, (float *)out0_raw,
                elem_lo, elem_hi);
        } else if (node->op == YR_OP_MUL) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
            uint32_t elem_lo, elem_hi;
            if (node->input_count != 2u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            in1 = &yr_tensors[node->inputs[1]];
            in1_raw = yr_tensor_raw(device_base, in1);
            if (in1_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            elem_lo = 0u; elem_hi = out0->elements;
            status = yr_mul(
                in0, in1, out0, (float *)in0_raw,
                (float *)in1_raw, (float *)out0_raw, elem_lo, elem_hi);
        } else if (node->op == YR_OP_CONCAT) {
            const struct yr_tensor_desc *input_descs[4] = {0, 0, 0, 0};
            const float *input_data[4] = {0, 0, 0, 0};
            uint32_t input_index;
            if (node->input_count == 0u || node->input_count > 4u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            for (input_index = 0u; input_index < node->input_count;
                 ++input_index) {
                if (node->inputs[input_index] >= YR_TENSOR_COUNT) {
                    status = YR_STATUS_BAD_MANIFEST;
                    goto fail;
                }
                input_descs[input_index] =
                    &yr_tensors[node->inputs[input_index]];
                input_data[input_index] =
                    yr_tensor_ptr(device_base, input_descs[input_index]);
            }
            status = yr_concat(
                node, input_descs, out0, input_data, (float *)out0_raw);
#if YR_MANIFEST_VERSION >= 2
        } else if (node->op == YR_OP_ADD || node->op == YR_OP_SUB) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
            uint32_t elem_lo, elem_hi;
            if (node->input_count != 2u
                || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            in1 = &yr_tensors[node->inputs[1]];
            in1_raw = yr_tensor_raw(device_base, in1);
            if (in1_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            elem_lo = 0u; elem_hi = out0->elements;
            status = yr_add_sub(
                in0, in1, out0, (float *)in0_raw, (float *)in1_raw,
                (float *)out0_raw, node->op == YR_OP_SUB, elem_lo, elem_hi);
        } else if (node->op == YR_OP_SPLIT) {
            const struct yr_tensor_desc *sizes_desc;
            const struct yr_tensor_desc *output_descs[3] = {0, 0, 0};
            uint8_t *output_data[3] = {0, 0, 0};
            uint8_t *sizes_raw;
            if (node->input_count != 2u
                || node->inputs[1] >= YR_TENSOR_COUNT
                || node->output_count > 3u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            sizes_desc = &yr_tensors[node->inputs[1]];
            sizes_raw = yr_tensor_raw(device_base, sizes_desc);
            for (output_index = 0u;
                 output_index < node->output_count;
                 ++output_index) {
                output_descs[output_index] =
                    &yr_tensors[node->outputs[output_index]];
                output_data[output_index] =
                    yr_tensor_raw(device_base, output_descs[output_index]);
            }
            if (sizes_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_split(
                node, in0, sizes_desc, output_descs, in0_raw,
                (const int64_t *)sizes_raw, output_data);
        } else if (node->op == YR_OP_MAXPOOL) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_maxpool(
                node, in0, out0, (float *)in0_raw, (float *)out0_raw,
                0u, out0->dims[1]);
        } else if (node->op == YR_OP_RESIZE) {
            const struct yr_tensor_desc *scales_desc;
            uint8_t *scales_raw;
            if (node->input_count != 3u || node->output_count != 1u
                || node->inputs[1] != UINT32_MAX
                || node->inputs[2] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            scales_desc = &yr_tensors[node->inputs[2]];
            scales_raw = yr_tensor_raw(device_base, scales_desc);
            if (scales_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_resize(
                node, in0, scales_desc, out0, (float *)in0_raw,
                (const float *)scales_raw, (float *)out0_raw,
                0u, out0->dims[1]);
        } else if (node->op == YR_OP_MATMUL) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            in1 = &yr_tensors[node->inputs[1]];
            in1_raw = yr_tensor_raw(device_base, in1);
            if (in1_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_matmul(
                in0, in1, out0, (float *)in0_raw, (float *)in1_raw,
                (float *)out0_raw);
        } else if (node->op == YR_OP_SOFTMAX) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_softmax(
                node, in0, out0, (float *)in0_raw, (float *)out0_raw);
        } else if (node->op == YR_OP_RESHAPE) {
            const struct yr_tensor_desc *shape_desc;
            uint8_t *shape_raw;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            shape_desc = &yr_tensors[node->inputs[1]];
            shape_raw = yr_tensor_raw(device_base, shape_desc);
            if (shape_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_reshape(
                in0, shape_desc, out0, (const int64_t *)shape_raw,
                in0_raw, out0_raw);
        } else if (node->op == YR_OP_FLATTEN) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_flatten(node, in0, out0, in0_raw, out0_raw);
        } else if (node->op == YR_OP_TRANSPOSE) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_transpose(
                node, in0, out0, in0_raw, out0_raw);
        } else if (node->op == YR_OP_REDUCEMAX) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_reduce_max(
                node, in0, out0, (float *)in0_raw, (float *)out0_raw);
        } else if (node->op == YR_OP_TOPK) {
            const struct yr_tensor_desc *k_desc;
            const struct yr_tensor_desc *indices_desc;
            uint8_t *k_raw;
            uint8_t *indices_raw;
            if (node->input_count != 2u || node->output_count != 2u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            k_desc = &yr_tensors[node->inputs[1]];
            indices_desc = &yr_tensors[node->outputs[1]];
            k_raw = yr_tensor_raw(device_base, k_desc);
            indices_raw = yr_tensor_raw(device_base, indices_desc);
            if (k_raw == (uint8_t *)0
                || indices_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_topk(
                node, in0, k_desc, out0, indices_desc,
                (float *)in0_raw, (const int64_t *)k_raw,
                (float *)out0_raw, (int64_t *)indices_raw);
        } else if (node->op == YR_OP_UNSQUEEZE) {
            const struct yr_tensor_desc *axes_desc;
            uint8_t *axes_raw;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            axes_desc = &yr_tensors[node->inputs[1]];
            axes_raw = yr_tensor_raw(device_base, axes_desc);
            if (axes_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_unsqueeze(
                in0, axes_desc, out0, (const int64_t *)axes_raw,
                in0_raw, out0_raw);
        } else if (node->op == YR_OP_TILE) {
            const struct yr_tensor_desc *repeats_desc;
            uint8_t *repeats_raw;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            repeats_desc = &yr_tensors[node->inputs[1]];
            repeats_raw = yr_tensor_raw(device_base, repeats_desc);
            if (repeats_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_tile(
                in0, repeats_desc, out0, (const int64_t *)repeats_raw,
                in0_raw, out0_raw);
        } else if (node->op == YR_OP_GATHERELEMENTS) {
            const struct yr_tensor_desc *indices_desc;
            uint8_t *indices_raw;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            indices_desc = &yr_tensors[node->inputs[1]];
            indices_raw = yr_tensor_raw(device_base, indices_desc);
            if (indices_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            status = yr_gather_elements(
                node, in0, indices_desc, out0, in0_raw,
                (const int64_t *)indices_raw, out0_raw);
        } else if (node->op == YR_OP_MOD || node->op == YR_OP_DIV) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
            uint32_t elem_lo, elem_hi;
            if (node->input_count != 2u || node->output_count != 1u
                || node->inputs[1] >= YR_TENSOR_COUNT) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            in1 = &yr_tensors[node->inputs[1]];
            in1_raw = yr_tensor_raw(device_base, in1);
            if (in1_raw == (uint8_t *)0) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            elem_lo = 0u; elem_hi = out0->elements;
            status = yr_integer_binary(
                node, in0, in1, out0, (const int64_t *)in0_raw,
                (const int64_t *)in1_raw, (int64_t *)out0_raw,
                node->op == YR_OP_MOD, elem_lo, elem_hi);
        } else if (node->op == YR_OP_CAST) {
            uint32_t elem_lo, elem_hi;
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            elem_lo = 0u; elem_hi = out0->elements;
            status = yr_cast(
                node, in0, out0, (const int64_t *)in0_raw,
                (float *)out0_raw, elem_lo, elem_hi);
#endif
        } else {
            status = YR_STATUS_UNSUPPORTED_OP;
        }

        /*
         * Only hart 0 ever reaches this code for a non-Conv op (every
         * other hart already skipped the node above), so hart 0 publishes
         * each output tensor's full range itself; there is no other
         * hart's slice to split off, and no other hart to race with. The
         * barrier call must run whether or not this node's status is OK,
         * matching the unconditional barrier the other 15 harts already
         * called above; skipping it on failure here would leave those 15
         * harts waiting forever on a 16th call that never comes.
         */
        if (node->op != YR_OP_CONV) {
            if (status == YR_STATUS_OK) {
                uint32_t output_publish_index;
                for (output_publish_index = 0u;
                     output_publish_index < node->output_count;
                     ++output_publish_index) {
                    const struct yr_tensor_desc *published_desc =
                        &yr_tensors[node->outputs[output_publish_index]];
                    uint8_t *published_raw =
                        yr_tensor_raw(device_base, published_desc);
                    if (published_raw != (uint8_t *)0
                        && published_desc->nbytes > 0u) {
                        yr_publish(published_raw, published_desc->nbytes);
                    }
                }
            }
            yr_hart_barrier();
        }
        if (status != YR_STATUS_OK) {
            goto fail;
        }
    }
    /*
     * Non-Conv nodes above no longer have a per-node barrier (only hart 0
     * ever runs them, so there is nothing to synchronize per node); this
     * one barrier re-syncs every hart before the caller starts the next
     * stage, whether or not this span's last node happened to be Conv.
     */
    yr_hart_barrier();

    return YR_STATUS_OK;

fail:
    result->status = status;
    result->failed_node = yr_nodes[node_index].onnx_index;
    result->failed_op = yr_nodes[node_index].op;
    return status;
}


uint32_t yr_run_selected(uint8_t *device_base, struct yr_result_header *result)
{
    return yr_run_node_span(
        device_base, result, 0u, YR_NODE_COUNT - 1u);
}


void yr_finalize_result(uint8_t *device_base, struct yr_result_header *result)
{
    const uint8_t *workspace;
    if (device_base == (uint8_t *)0
        || result != (struct yr_result_header *)(
            device_base + YR_RESULT_DEVICE_OFFSET)
        || !yr_manifest_valid()) {
        return;
    }
    workspace =
        device_base + YR_RESULT_DEVICE_OFFSET + YR_RESULT_HEADER_BYTES;
    result->workspace_fnv1a = yr_fnv1a(workspace, YR_WORKSPACE_BYTES);
}


/*
 * Default tensor-mode setup, does nothing. Overridden by the same ET-only
 * source that overrides yr_conv_tensor() below.
 */
__attribute__((weak))
void yr_conv_tensor_init(void)
{
}


/*
 * Default fast path, always declines. An ET-only source that defines this
 * symbol without the weak attribute overrides it at link time; the host
 * build and any ET build that does not list that source keep this stub, so
 * behavior stays exactly the portable scalar path.
 */
__attribute__((weak))
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
    uint32_t *hart_oc_hi)
{
    (void)node;
    (void)input_desc;
    (void)weight_desc;
    (void)bias_desc;
    (void)output_desc;
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    *hart_oc_lo = 0u;
    *hart_oc_hi = 0u;
    return 0u;
}
