/*
 * Portable scalar FP32 ONNX slice runtime.
 *
 * This file intentionally contains no VPU/TFMA paths, fusions, tiling,
 * threading, allocation, or libc calls. Each ONNX node remains a separately
 * materialized tensor so intermediate comparisons stay visible.
 */

#include <stdint.h>

#include "ref_runtime.h"
/*
 * The script builds put the generated manifest on the include path with -I, one
 * directory per slice. The single-translation-unit CI build cannot rely on the
 * working directory, so it includes the pinned full-graph manifest by relative
 * path first and defines this macro; the header it picks is the same file.
 */
#ifndef YR_SLICE_MANIFEST_PREINCLUDED
#include "slice_manifest.h"
#endif

#ifndef YR_MANIFEST_VERSION
#define YR_MANIFEST_VERSION 1u
#endif

/*
 * Build switch for the dedicated 1x1 stride-1 Conv path in yr_conv, on by
 * default. Set to 0 to route 1x1 convolutions through the general path
 * instead, which is how its board speedup is isolated from other changes.
 * The two paths produce bit-identical output, so this only affects speed.
 */
#ifndef YR_CONV_1X1_FAST
#define YR_CONV_1X1_FAST 1
#endif

/*
 * Packed vector version of the 1x1 path, ET target only. Runs the same
 * per-pixel matrix-vector product eight flat positions at a time on the vector
 * unit instead of one at a time in scalar, across all sixteen harts. The fused
 * multiply-add rounds differently from the scalar mul then add, so a build with
 * this validates by tolerance rather than the workspace hash.
 */
#ifndef YR_CONV_1X1_VPU
#define YR_CONV_1X1_VPU 1
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


/*
 * Same split as yr_hart_range but over a flat element count, with every
 * boundary snapped to a 64-byte cache line (16 floats). Elementwise ops use
 * this so each hart writes and evicts a range that shares no cache line with
 * any other hart's range; L1 is minion local and not coherent, so two harts
 * touching one line is exactly the corruption this avoids. The final range
 * clamps to count so a tail shorter than a line is written by one hart only.
 */
#define YR_CACHE_LINE_FLOATS 16u
static void yr_hart_elem_range(uint32_t count, uint32_t *lo, uint32_t *hi)
{
    const uint32_t harts = yr_hart_count();
    const uint32_t id = yr_hart_id();
    const uint32_t lines =
        (count + YR_CACHE_LINE_FLOATS - 1u) / YR_CACHE_LINE_FLOATS;
    uint32_t lo_line = (lines * id) / harts;
    uint32_t hi_line = (lines * (id + 1u)) / harts;
    *lo = lo_line * YR_CACHE_LINE_FLOATS;
    *hi = hi_line * YR_CACHE_LINE_FLOATS;
    if (*lo > count) {
        *lo = count;
    }
    if (*hi > count) {
        *hi = count;
    }
}


/*
 * Build switch for the dedicated 3x3 stride-1 same-padding Conv path, off by
 * default. On the board this path measured about 1.5s slower than the general
 * path over the full graph, so it stays off; set to 1 to route those
 * convolutions through it. The two produce bit-identical output, so this only
 * affects speed.
 */
#ifndef YR_CONV_3X3_FAST
#define YR_CONV_3X3_FAST 0
#endif

/*
 * Packed vector version of the 3x3 stride-1 same-pad path, ET target only. The
 * interior columns, whose whole 3x3 neighbourhood is in bounds, run eight
 * output positions at a time on the vector unit across all sixteen harts; the
 * border columns and rows stay on the scalar per-pixel helper. Fused
 * multiply-add rounds differently from the scalar path, so a build with this
 * validates by tolerance.
 */
#ifndef YR_CONV_3X3_VPU
#define YR_CONV_3X3_VPU 0
#endif


/*
 * Packed vector version of the 3x3 stride-1 same-pad DEPTHWISE path (group ==
 * channels, one input channel per output channel), ET target only. Depthwise is
 * the only conv class the graph never specialised, the general grouped path
 * cannot pair two output channels because outputs_per_group is 1, so every
 * depthwise channel runs pure scalar single-oc. These convs are only 0.7 percent
 * of the graph's multiply-accumulates but about 11 percent of its conv memory
 * traffic, and this graph is memory bound. Each channel is independent with no
 * input-channel reduction, so the interior eight-wide vector loop is the same as
 * yr_conv3x3_vpu with the icg loop removed and the input plane indexed by the
 * output channel. Fused multiply-add rounds differently from the scalar path, so
 * a build with this validates by tolerance.
 */
#ifndef YR_CONV_DW3X3_VPU
#define YR_CONV_DW3X3_VPU 1
#endif


/*
 * One output pixel of a 3x3 stride-1 pad-1 dilation-1 ungrouped Conv, for a
 * single output channel whose 3x3-per-input-channel weights start at w_oc.
 * Used only for the border pixels of yr_conv3x3_s1, where some taps fall
 * outside the input. It resolves the valid tap window with the same
 * yr_tap_range the general path uses and accumulates in the same
 * (input channel, ky, kx) order starting from the bias, so its result is
 * bit-identical to what the general Conv loop would produce for that pixel.
 */
static float yr_conv3x3_pixel(
    const float *input, const float *w_oc, float bias_value,
    uint32_t input_channels, uint32_t input_h, uint32_t input_w,
    int32_t oh, int32_t ow)
{
    const uint32_t channel_stride = input_h * input_w;
    const int64_t base_h = (int64_t)oh - 1;
    const int64_t base_w = (int64_t)ow - 1;
    uint32_t first_ky, first_kx;
    const uint32_t ky_count =
        yr_tap_range(base_h, 1, (int64_t)input_h, 3, &first_ky);
    const uint32_t kx_count =
        yr_tap_range(base_w, 1, (int64_t)input_w, 3, &first_kx);
    float accumulator = bias_value;
    uint32_t icg, ky, kx;
    if (ky_count == 0u || kx_count == 0u) {
        return accumulator;
    }
    for (icg = 0; icg < input_channels; ++icg) {
        const float *row = input + (uint64_t)icg * channel_stride
            + (uint64_t)(base_h + (int64_t)first_ky) * input_w
            + (base_w + (int64_t)first_kx);
        const float *coefficient_row =
            w_oc + (uint64_t)icg * 9u + (uint64_t)first_ky * 3u + first_kx;
        for (ky = 0; ky < ky_count; ++ky) {
            const float *value = row;
            const float *coefficient = coefficient_row;
            for (kx = 0; kx < kx_count; ++kx) {
                accumulator += *value * *coefficient;
                value += 1;
                coefficient += 1;
            }
            row += input_w;
            coefficient_row += 3u;
        }
    }
    return accumulator;
}


/*
 * Dedicated 3x3 stride-1 same-padding ungrouped Conv, 53.3 percent of this
 * graph's multiply-accumulates and its single largest arithmetic cost. The
 * general path handles it correctly but pays for a runtime-count kernel loop
 * on every output pixel; here the kernel is fixed at 3x3 so the nine taps
 * unroll to straight-line code the compiler can pipeline. Interior pixels
 * (every tap in bounds, the overwhelming majority) take the unrolled path,
 * two output channels and four output columns at a time to match the general
 * path's register use and reuse each input load. The one-pixel border strip
 * defers to yr_conv3x3_pixel. Every accumulator sums over input channel then
 * ky then kx starting from the bias, the exact order the general loop uses,
 * so the output is bit-identical.
 */
static uint32_t yr_conv3x3_s1(
    uint32_t batches, uint32_t input_channels, uint32_t output_channels,
    uint32_t input_h, uint32_t input_w,
    const float *input, const float *weight, const float *bias, float *output)
{
    const uint32_t hw = input_h * input_w;
    const uint32_t weight_oc_stride = input_channels * 9u;
    uint32_t oc_lo, oc_hi, n, oc, icg;
    int32_t oh, ow;
    yr_hart_range(output_channels, &oc_lo, &oc_hi);
    for (n = 0; n < batches; ++n) {
        const float *const batch_input =
            input + (uint64_t)n * input_channels * hw;
        float *const batch_output =
            output + (uint64_t)n * output_channels * hw;
        oc = oc_lo;
        while (oc + 1u < oc_hi) {
            const float *const weight_a = weight + (uint64_t)oc * weight_oc_stride;
            const float *const weight_b = weight_a + weight_oc_stride;
            const float bias_a = bias == (const float *)0 ? 0.0f : bias[oc];
            const float bias_b =
                bias == (const float *)0 ? 0.0f : bias[oc + 1u];
            float *const out_a = batch_output + (uint64_t)oc * hw;
            float *const out_b = out_a + hw;
            for (oh = 0; oh < (int32_t)input_h; ++oh) {
                if (oh >= 1 && oh <= (int32_t)input_h - 2) {
                    float *const row_a = out_a + (uint64_t)oh * input_w;
                    float *const row_b = out_b + (uint64_t)oh * input_w;
                    row_a[0] = yr_conv3x3_pixel(
                        batch_input, weight_a, bias_a, input_channels,
                        input_h, input_w, oh, 0);
                    row_b[0] = yr_conv3x3_pixel(
                        batch_input, weight_b, bias_b, input_channels,
                        input_h, input_w, oh, 0);
                    ow = 1;
                    while (ow + 3 <= (int32_t)input_w - 2) {
                        float a0 = bias_a, a1 = bias_a, a2 = bias_a, a3 = bias_a;
                        float b0 = bias_b, b1 = bias_b, b2 = bias_b, b3 = bias_b;
                        for (icg = 0; icg < input_channels; ++icg) {
                            const float *const wa = weight_a + (uint64_t)icg * 9u;
                            const float *const wb = weight_b + (uint64_t)icg * 9u;
                            const float *top_left = batch_input
                                + (uint64_t)icg * hw
                                + (uint64_t)(oh - 1) * input_w + (ow - 1);
                            uint32_t ky, kx;
                            for (ky = 0u; ky < 3u; ++ky) {
                                const float *const pr = top_left
                                    + (uint64_t)ky * input_w;
                                for (kx = 0u; kx < 3u; ++kx) {
                                    const float wav = wa[ky * 3u + kx];
                                    const float wbv = wb[ky * 3u + kx];
                                    const float v0 = pr[kx];
                                    const float v1 = pr[kx + 1u];
                                    const float v2 = pr[kx + 2u];
                                    const float v3 = pr[kx + 3u];
                                    a0 += wav * v0; a1 += wav * v1;
                                    a2 += wav * v2; a3 += wav * v3;
                                    b0 += wbv * v0; b1 += wbv * v1;
                                    b2 += wbv * v2; b3 += wbv * v3;
                                }
                            }
                        }
                        row_a[ow] = a0; row_a[ow + 1] = a1;
                        row_a[ow + 2] = a2; row_a[ow + 3] = a3;
                        row_b[ow] = b0; row_b[ow + 1] = b1;
                        row_b[ow + 2] = b2; row_b[ow + 3] = b3;
                        ow += 4;
                    }
                    for (; ow <= (int32_t)input_w - 2; ++ow) {
                        row_a[ow] = yr_conv3x3_pixel(
                            batch_input, weight_a, bias_a, input_channels,
                            input_h, input_w, oh, ow);
                        row_b[ow] = yr_conv3x3_pixel(
                            batch_input, weight_b, bias_b, input_channels,
                            input_h, input_w, oh, ow);
                    }
                    row_a[input_w - 1u] = yr_conv3x3_pixel(
                        batch_input, weight_a, bias_a, input_channels,
                        input_h, input_w, oh, (int32_t)input_w - 1);
                    row_b[input_w - 1u] = yr_conv3x3_pixel(
                        batch_input, weight_b, bias_b, input_channels,
                        input_h, input_w, oh, (int32_t)input_w - 1);
                } else {
                    for (ow = 0; ow < (int32_t)input_w; ++ow) {
                        out_a[(uint64_t)oh * input_w + ow] = yr_conv3x3_pixel(
                            batch_input, weight_a, bias_a, input_channels,
                            input_h, input_w, oh, ow);
                        out_b[(uint64_t)oh * input_w + ow] = yr_conv3x3_pixel(
                            batch_input, weight_b, bias_b, input_channels,
                            input_h, input_w, oh, ow);
                    }
                }
            }
            oc += 2u;
        }
        for (; oc < oc_hi; ++oc) {
            const float *const weight_oc =
                weight + (uint64_t)oc * weight_oc_stride;
            const float bias_value = bias == (const float *)0 ? 0.0f : bias[oc];
            float *const out_oc = batch_output + (uint64_t)oc * hw;
            for (oh = 0; oh < (int32_t)input_h; ++oh) {
                for (ow = 0; ow < (int32_t)input_w; ++ow) {
                    out_oc[(uint64_t)oh * input_w + ow] = yr_conv3x3_pixel(
                        batch_input, weight_oc, bias_value, input_channels,
                        input_h, input_w, oh, ow);
                }
            }
        }
    }
    return YR_STATUS_OK;
}


#if YR_CONV_3X3_VPU && defined(__riscv)
/*
 * Packed eight-wide 3x3 stride-1 same-pad conv. One output channel at a time so
 * every channel count maps cleanly across harts, the interior columns whose
 * whole neighbourhood is in bounds run eight positions at a time on the vector
 * unit, and the first and last column of each interior row plus the top and
 * bottom rows fall back to the scalar per-pixel helper. Accumulates over input
 * channel then ky then kx from the bias, the same order the scalar path uses.
 */
static uint32_t yr_conv3x3_vpu(
    uint32_t batches, uint32_t input_channels, uint32_t output_channels,
    uint32_t input_h, uint32_t input_w,
    const float *input, const float *weight, const float *bias, float *output)
{
    const uint32_t hw = input_h * input_w;
    const uint32_t weight_oc_stride = input_channels * 9u;
    uint32_t oc_lo, oc_hi, n, oc, icg;
    int32_t oh, ow;
    yr_hart_range(output_channels, &oc_lo, &oc_hi);
    for (n = 0; n < batches; ++n) {
        const float *const batch_input =
            input + (uint64_t)n * input_channels * hw;
        float *const batch_output =
            output + (uint64_t)n * output_channels * hw;
        for (oc = oc_lo; oc < oc_hi; ++oc) {
            const float *const weight_oc =
                weight + (uint64_t)oc * weight_oc_stride;
            const float bias_value = bias == (const float *)0 ? 0.0f : bias[oc];
            union { float f; uint32_t u; } bo = { bias_value };
            float *const out_oc = batch_output + (uint64_t)oc * hw;
            for (oh = 0; oh < (int32_t)input_h; ++oh) {
                if (oh >= 1 && oh <= (int32_t)input_h - 2) {
                    float *const row = out_oc + (uint64_t)oh * input_w;
                    row[0] = yr_conv3x3_pixel(batch_input, weight_oc, bias_value,
                        input_channels, input_h, input_w, oh, 0);
                    ow = 1;
                    while (ow + 8 <= (int32_t)input_w - 1) {
                        float acc;
                        __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc)
                            : "r"((uint64_t)bo.u));
                        for (icg = 0; icg < input_channels; ++icg) {
                            const float *const wc = weight_oc + (uint64_t)icg * 9u;
                            const float *const top_left = batch_input
                                + (uint64_t)icg * hw
                                + (uint64_t)(oh - 1) * input_w + (ow - 1);
                            uint32_t ky, kx;
                            for (ky = 0u; ky < 3u; ++ky) {
                                const float *const pr = top_left
                                    + (uint64_t)ky * input_w;
                                for (kx = 0u; kx < 3u; ++kx) {
                                    union { float f; uint32_t u; } cw =
                                        { wc[ky * 3u + kx] };
                                    float iv, wv;
                                    __asm__ volatile("flq2 %0, 0(%1)\n"
                                        : "=f"(iv) : "r"(pr + kx));
                                    __asm__ volatile("fbcx.ps %0, %1\n"
                                        : "=f"(wv) : "r"((uint64_t)cw.u));
                                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                        : "+f"(acc) : "f"(iv), "f"(wv));
                                }
                            }
                        }
                        __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(row + ow),
                            "f"(acc) : "memory");
                        ow += 8;
                    }
                    for (; ow <= (int32_t)input_w - 2; ++ow) {
                        row[ow] = yr_conv3x3_pixel(batch_input, weight_oc,
                            bias_value, input_channels, input_h, input_w, oh, ow);
                    }
                    row[input_w - 1u] = yr_conv3x3_pixel(batch_input, weight_oc,
                        bias_value, input_channels, input_h, input_w, oh,
                        (int32_t)input_w - 1);
                } else {
                    for (ow = 0; ow < (int32_t)input_w; ++ow) {
                        out_oc[(uint64_t)oh * input_w + ow] = yr_conv3x3_pixel(
                            batch_input, weight_oc, bias_value, input_channels,
                            input_h, input_w, oh, ow);
                    }
                }
            }
        }
    }
    return YR_STATUS_OK;
}
#endif


#if YR_CONV_DW3X3_VPU && defined(__riscv)
/*
 * One output pixel of a 3x3 stride-1 pad-1 dilation-1 depthwise Conv, for a
 * single channel whose nine weights start at w9 and whose input plane starts at
 * chan. Used only for the border pixels of yr_conv_dw3x3_s1_vpu, where some taps
 * fall outside the input. Skips out-of-range taps and accumulates in (ky, kx)
 * order from the bias, matching the general depthwise loop for that pixel.
 */
static float yr_dw3x3_pixel(
    const float *chan, const float *w9, float bias_value,
    uint32_t input_h, uint32_t input_w, int32_t oh, int32_t ow)
{
    float acc = bias_value;
    int32_t ky;
    for (ky = 0; ky < 3; ++ky) {
        const int32_t ih = oh + ky - 1;
        int32_t kx;
        if (ih < 0 || ih >= (int32_t)input_h) {
            continue;
        }
        for (kx = 0; kx < 3; ++kx) {
            const int32_t iw = ow + kx - 1;
            if (iw < 0 || iw >= (int32_t)input_w) {
                continue;
            }
            acc += w9[ky * 3 + kx]
                * chan[(uint64_t)ih * input_w + iw];
        }
    }
    return acc;
}

/*
 * Packed eight-wide 3x3 stride-1 same-pad depthwise conv. One channel at a time
 * so the channel count maps cleanly across harts; the interior columns whose
 * whole 3x3 neighbourhood is in bounds run eight positions at a time on the
 * vector unit, and the first and last column of each interior row plus the top
 * and bottom rows fall back to the scalar per-pixel helper. Each channel reads
 * only its own input plane (no input-channel reduction) and accumulates over ky
 * then kx from the bias, the same order the scalar path uses.
 */
static uint32_t yr_conv_dw3x3_s1_vpu(
    uint32_t batches, uint32_t channels, uint32_t input_h, uint32_t input_w,
    const float *input, const float *weight, const float *bias, float *output)
{
    const uint32_t hw = input_h * input_w;
    uint32_t c_lo, c_hi, n, c;
    int32_t oh, ow;
    yr_hart_range(channels, &c_lo, &c_hi);
    for (n = 0; n < batches; ++n) {
        const float *const batch_input =
            input + (uint64_t)n * channels * hw;
        float *const batch_output =
            output + (uint64_t)n * channels * hw;
        for (c = c_lo; c < c_hi; ++c) {
            const float *const chan = batch_input + (uint64_t)c * hw;
            const float *const w9 = weight + (uint64_t)c * 9u;
            const float bias_value = bias == (const float *)0 ? 0.0f : bias[c];
            union { float f; uint32_t u; } bo = { bias_value };
            float *const out_c = batch_output + (uint64_t)c * hw;
            for (oh = 0; oh < (int32_t)input_h; ++oh) {
                if (oh >= 1 && oh <= (int32_t)input_h - 2) {
                    float *const row = out_c + (uint64_t)oh * input_w;
                    row[0] = yr_dw3x3_pixel(chan, w9, bias_value,
                        input_h, input_w, oh, 0);
                    ow = 1;
                    while (ow + 8 <= (int32_t)input_w - 1) {
                        float acc;
                        const float *const top_left = chan
                            + (uint64_t)(oh - 1) * input_w + (ow - 1);
                        uint32_t ky, kx;
                        __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(acc)
                            : "r"((uint64_t)bo.u));
                        for (ky = 0u; ky < 3u; ++ky) {
                            const float *const pr = top_left
                                + (uint64_t)ky * input_w;
                            for (kx = 0u; kx < 3u; ++kx) {
                                union { float f; uint32_t u; } cw =
                                    { w9[ky * 3u + kx] };
                                float iv, wv;
                                __asm__ volatile("flq2 %0, 0(%1)\n"
                                    : "=f"(iv) : "r"(pr + kx));
                                __asm__ volatile("fbcx.ps %0, %1\n"
                                    : "=f"(wv) : "r"((uint64_t)cw.u));
                                __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                    : "+f"(acc) : "f"(iv), "f"(wv));
                            }
                        }
                        __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(row + ow),
                            "f"(acc) : "memory");
                        ow += 8;
                    }
                    for (; ow <= (int32_t)input_w - 2; ++ow) {
                        row[ow] = yr_dw3x3_pixel(chan, w9, bias_value,
                            input_h, input_w, oh, ow);
                    }
                    row[input_w - 1u] = yr_dw3x3_pixel(chan, w9, bias_value,
                        input_h, input_w, oh, (int32_t)input_w - 1);
                } else {
                    for (ow = 0; ow < (int32_t)input_w; ++ow) {
                        out_c[(uint64_t)oh * input_w + ow] = yr_dw3x3_pixel(
                            chan, w9, bias_value, input_h, input_w, oh, ow);
                    }
                }
            }
        }
    }
    return YR_STATUS_OK;
}
#endif


/*
 * Build switch for the dedicated 3x3 stride-2 Conv path, off by default. On the
 * board this path measured slightly slower than the general path over the full
 * graph (about 0.8s), so it stays off; set to 1 to route those convolutions
 * through it. The two produce bit-identical output, so this only affects speed.
 */
#ifndef YR_CONV_3X3_S2_FAST
#define YR_CONV_3X3_S2_FAST 0
#endif


/*
 * One output pixel of a 3x3 stride-s pad dilation-1 ungrouped Conv, for a
 * single output channel whose 3x3-per-input-channel weights start at w_oc.
 * Used only for the border columns of yr_conv3x3_s2, where some taps fall
 * outside the input. It resolves the valid tap window with the same
 * yr_tap_range the general path uses and accumulates in the same
 * (input channel, ky, kx) order starting from the bias, so its result is
 * bit-identical to what the general Conv loop would produce for that pixel.
 */
static float yr_conv3x3_s2_pixel(
    const float *input, const float *w_oc, float bias_value,
    uint32_t input_channels, uint32_t input_h, uint32_t input_w,
    int32_t oh, int32_t ow, uint32_t stride,
    uint32_t pad_top, uint32_t pad_left)
{
    const uint32_t channel_stride = input_h * input_w;
    const int64_t base_h = (int64_t)oh * (int64_t)stride - (int64_t)pad_top;
    const int64_t base_w = (int64_t)ow * (int64_t)stride - (int64_t)pad_left;
    uint32_t first_ky, first_kx;
    const uint32_t ky_count =
        yr_tap_range(base_h, 1, (int64_t)input_h, 3, &first_ky);
    const uint32_t kx_count =
        yr_tap_range(base_w, 1, (int64_t)input_w, 3, &first_kx);
    float accumulator = bias_value;
    uint32_t icg, ky, kx;
    if (ky_count == 0u || kx_count == 0u) {
        return accumulator;
    }
    for (icg = 0; icg < input_channels; ++icg) {
        const float *row = input + (uint64_t)icg * channel_stride
            + (uint64_t)(base_h + (int64_t)first_ky) * input_w
            + (base_w + (int64_t)first_kx);
        const float *coefficient_row =
            w_oc + (uint64_t)icg * 9u + (uint64_t)first_ky * 3u + first_kx;
        for (ky = 0; ky < ky_count; ++ky) {
            const float *value = row;
            const float *coefficient = coefficient_row;
            for (kx = 0; kx < kx_count; ++kx) {
                accumulator += *value * *coefficient;
                value += 1;
                coefficient += 1;
            }
            row += input_w;
            coefficient_row += 3u;
        }
    }
    return accumulator;
}


/*
 * Dedicated 3x3 stride-2 dilation-1 ungrouped Conv, 10.1 percent of this
 * graph's multiply-accumulates (the downsampling convolutions). The general
 * path already blocks this shape four output columns and two output channels
 * at a time; this path keeps that blocking but fixes the kernel at 3x3 with
 * unit dilation, so the tap loops fold to constants the compiler can pipeline
 * instead of the general path's runtime kernel width and dilation. Column
 * borders defer to yr_conv3x3_s2_pixel; top and bottom border rows fall out of
 * yr_tap_range's row count inline, exactly as the general path does. Every
 * accumulator sums over input channel then ky then kx starting from the bias,
 * the same order the general loop uses, so the output is bit-identical.
 */
static uint32_t yr_conv3x3_s2(
    uint32_t batches, uint32_t input_channels, uint32_t output_channels,
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w,
    uint32_t pad_top, uint32_t pad_left,
    uint32_t interior_first, uint32_t interior_end,
    const float *input, const float *weight, const float *bias, float *output)
{
    const uint32_t stride = 2u;
    const uint32_t channel_stride = input_h * input_w;
    const uint32_t plane_stride = output_h * output_w;
    const uint32_t weight_oc_stride = input_channels * 9u;
    const int32_t column_step = (int32_t)stride;
    const int32_t column_step2 = column_step * 2;
    const int32_t column_step3 = column_step * 3;
    uint32_t oc_lo, oc_hi, n, oc, icg, ky;
    int32_t oh, ow;
    yr_hart_range(output_channels, &oc_lo, &oc_hi);
    for (n = 0; n < batches; ++n) {
        const float *const batch_input =
            input + (uint64_t)n * input_channels * channel_stride;
        float *const batch_output =
            output + (uint64_t)n * output_channels * plane_stride;
        oc = oc_lo;
        while (oc + 1u < oc_hi) {
            const float *const weight_a =
                weight + (uint64_t)oc * weight_oc_stride;
            const float *const weight_b = weight_a + weight_oc_stride;
            const float bias_a = bias == (const float *)0 ? 0.0f : bias[oc];
            const float bias_b =
                bias == (const float *)0 ? 0.0f : bias[oc + 1u];
            float *const out_a = batch_output + (uint64_t)oc * plane_stride;
            float *const out_b = out_a + plane_stride;
            for (oh = 0; oh < (int32_t)output_h; ++oh) {
                const int64_t base_h =
                    (int64_t)oh * (int64_t)stride - (int64_t)pad_top;
                uint32_t first_ky;
                const uint32_t ky_count =
                    yr_tap_range(base_h, 1, (int64_t)input_h, 3, &first_ky);
                float *const row_a = out_a + (uint64_t)oh * output_w;
                float *const row_b = out_b + (uint64_t)oh * output_w;
                const float *row_origin;
                const float *wrow_a;
                const float *wrow_b;
                if (ky_count == 0u) {
                    for (ow = 0; ow < (int32_t)output_w; ++ow) {
                        row_a[ow] = bias_a;
                        row_b[ow] = bias_b;
                    }
                    continue;
                }
                row_origin = batch_input
                    + (uint64_t)(base_h + (int64_t)first_ky) * input_w;
                wrow_a = weight_a + (uint64_t)first_ky * 3u;
                wrow_b = weight_b + (uint64_t)first_ky * 3u;
                ow = 0;
                while (ow < (int32_t)output_w) {
                    const int64_t base_w =
                        (int64_t)ow * (int64_t)stride - (int64_t)pad_left;
                    if ((uint32_t)ow >= interior_first
                        && (uint32_t)ow + 4u <= interior_end) {
                        const float *channel = row_origin + base_w;
                        const float *tap_a = wrow_a;
                        const float *tap_b = wrow_b;
                        float a0 = bias_a, a1 = bias_a, a2 = bias_a, a3 = bias_a;
                        float b0 = bias_b, b1 = bias_b, b2 = bias_b, b3 = bias_b;
                        for (icg = 0; icg < input_channels; ++icg) {
                            const float *row = channel;
                            const float *ca = tap_a;
                            const float *cb = tap_b;
                            for (ky = 0; ky < ky_count; ++ky) {
                                const float *value = row;
                                uint32_t kxi;
                                for (kxi = 0; kxi < 3u; ++kxi) {
                                    const float sa = ca[kxi];
                                    const float sb = cb[kxi];
                                    const float v0 = value[0];
                                    const float v1 = value[column_step];
                                    const float v2 = value[column_step2];
                                    const float v3 = value[column_step3];
                                    a0 += v0 * sa; a1 += v1 * sa;
                                    a2 += v2 * sa; a3 += v3 * sa;
                                    b0 += v0 * sb; b1 += v1 * sb;
                                    b2 += v2 * sb; b3 += v3 * sb;
                                    value += 1;
                                }
                                row += input_w;
                                ca += 3u;
                                cb += 3u;
                            }
                            channel += channel_stride;
                            tap_a += 9u;
                            tap_b += 9u;
                        }
                        row_a[ow] = a0; row_a[ow + 1] = a1;
                        row_a[ow + 2] = a2; row_a[ow + 3] = a3;
                        row_b[ow] = b0; row_b[ow + 1] = b1;
                        row_b[ow + 2] = b2; row_b[ow + 3] = b3;
                        ow += 4;
                    } else {
                        row_a[ow] = yr_conv3x3_s2_pixel(
                            batch_input, weight_a, bias_a, input_channels,
                            input_h, input_w, oh, ow, stride,
                            pad_top, pad_left);
                        row_b[ow] = yr_conv3x3_s2_pixel(
                            batch_input, weight_b, bias_b, input_channels,
                            input_h, input_w, oh, ow, stride,
                            pad_top, pad_left);
                        ow += 1;
                    }
                }
            }
            oc += 2u;
        }
        for (; oc < oc_hi; ++oc) {
            const float *const weight_oc =
                weight + (uint64_t)oc * weight_oc_stride;
            const float bias_value =
                bias == (const float *)0 ? 0.0f : bias[oc];
            float *const out_oc = batch_output + (uint64_t)oc * plane_stride;
            for (oh = 0; oh < (int32_t)output_h; ++oh) {
                const int64_t base_h =
                    (int64_t)oh * (int64_t)stride - (int64_t)pad_top;
                uint32_t first_ky;
                const uint32_t ky_count =
                    yr_tap_range(base_h, 1, (int64_t)input_h, 3, &first_ky);
                float *const row = out_oc + (uint64_t)oh * output_w;
                const float *row_origin;
                const float *wrow;
                if (ky_count == 0u) {
                    for (ow = 0; ow < (int32_t)output_w; ++ow) {
                        row[ow] = bias_value;
                    }
                    continue;
                }
                row_origin = batch_input
                    + (uint64_t)(base_h + (int64_t)first_ky) * input_w;
                wrow = weight_oc + (uint64_t)first_ky * 3u;
                ow = 0;
                while (ow < (int32_t)output_w) {
                    const int64_t base_w =
                        (int64_t)ow * (int64_t)stride - (int64_t)pad_left;
                    if ((uint32_t)ow >= interior_first
                        && (uint32_t)ow + 4u <= interior_end) {
                        const float *channel = row_origin + base_w;
                        const float *tap = wrow;
                        float a0 = bias_value, a1 = bias_value;
                        float a2 = bias_value, a3 = bias_value;
                        for (icg = 0; icg < input_channels; ++icg) {
                            const float *rp = channel;
                            const float *cc = tap;
                            for (ky = 0; ky < ky_count; ++ky) {
                                const float *value = rp;
                                uint32_t kxi;
                                for (kxi = 0; kxi < 3u; ++kxi) {
                                    const float s = cc[kxi];
                                    a0 += value[0] * s;
                                    a1 += value[column_step] * s;
                                    a2 += value[column_step2] * s;
                                    a3 += value[column_step3] * s;
                                    value += 1;
                                }
                                rp += input_w;
                                cc += 3u;
                            }
                            channel += channel_stride;
                            tap += 9u;
                        }
                        row[ow] = a0; row[ow + 1] = a1;
                        row[ow + 2] = a2; row[ow + 3] = a3;
                        ow += 4;
                    } else {
                        row[ow] = yr_conv3x3_s2_pixel(
                            batch_input, weight_oc, bias_value, input_channels,
                            input_h, input_w, oh, ow, stride,
                            pad_top, pad_left);
                        ow += 1;
                    }
                }
            }
        }
    }
    return YR_STATUS_OK;
}


#if YR_CONV_1X1_VPU && defined(__riscv)
/*
 * Packed eight-wide 1x1 stride-1 conv, out = bias + sum over ic of
 * w[oc][ic] * in[ic][hw]. Same per-pixel matrix-vector product as the scalar
 * fast path but the accumulate runs eight flat positions at a time on the
 * vector unit, two output channels sharing each activation load. Every hart
 * owns an output-channel range. hw is a multiple of eight for every 1x1 node
 * in this graph; a scalar tail covers any remainder.
 */
static void yr_conv_1x1_vpu(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batches, uint32_t input_channels, uint32_t output_channels,
    uint32_t hw, uint32_t oc_lo, uint32_t oc_hi)
{
    uint32_t n, oc, ic, p;
    for (n = 0u; n < batches; ++n) {
        const float *batch_input = input + (uint64_t)n * input_channels * hw;
        float *batch_output = output + (uint64_t)n * output_channels * hw;
        oc = oc_lo;
        for (; oc + 1u < oc_hi; oc += 2u) {
            const float *wa = weight + (uint64_t)oc * input_channels;
            const float *wb = wa + input_channels;
            float *out_a = batch_output + (uint64_t)oc * hw;
            float *out_b = out_a + hw;
            union { float f; uint32_t u; } ba = { bias ? bias[oc] : 0.0f };
            union { float f; uint32_t u; } bb = { bias ? bias[oc + 1u] : 0.0f };
            for (p = 0u; p + 8u <= hw; p += 8u) {
                const float *in_plane = batch_input + p;
                float va, vb;
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(va)
                    : "r"((uint64_t)ba.u));
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vb)
                    : "r"((uint64_t)bb.u));
                for (ic = 0u; ic < input_channels; ++ic) {
                    union { float f; uint32_t u; } cwa = { wa[ic] };
                    union { float f; uint32_t u; } cwb = { wb[ic] };
                    float iv, wav, wbv;
                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(iv)
                        : "r"(in_plane));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(wav)
                        : "r"((uint64_t)cwa.u));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(wbv)
                        : "r"((uint64_t)cwb.u));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(va)
                        : "f"(iv), "f"(wav));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(vb)
                        : "f"(iv), "f"(wbv));
                    in_plane += hw;
                }
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(out_a + p), "f"(va)
                    : "memory");
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(out_b + p), "f"(vb)
                    : "memory");
            }
            for (; p < hw; ++p) {
                float a = ba.f, b = bb.f;
                const float *in_plane = batch_input + p;
                for (ic = 0u; ic < input_channels; ++ic) {
                    const float iv = *in_plane;
                    a += wa[ic] * iv;
                    b += wb[ic] * iv;
                    in_plane += hw;
                }
                out_a[p] = a;
                out_b[p] = b;
            }
        }
        for (; oc < oc_hi; ++oc) {
            const float *wr = weight + (uint64_t)oc * input_channels;
            float *out_row = batch_output + (uint64_t)oc * hw;
            union { float f; uint32_t u; } bo = { bias ? bias[oc] : 0.0f };
            for (p = 0u; p + 8u <= hw; p += 8u) {
                const float *in_plane = batch_input + p;
                float va;
                __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(va)
                    : "r"((uint64_t)bo.u));
                for (ic = 0u; ic < input_channels; ++ic) {
                    union { float f; uint32_t u; } cw = { wr[ic] };
                    float iv, wv;
                    __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(iv)
                        : "r"(in_plane));
                    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(wv)
                        : "r"((uint64_t)cw.u));
                    __asm__ volatile("fmadd.ps %0, %1, %2, %0\n" : "+f"(va)
                        : "f"(iv), "f"(wv));
                    in_plane += hw;
                }
                __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(out_row + p), "f"(va)
                    : "memory");
            }
            for (; p < hw; ++p) {
                float a = bo.f;
                const float *in_plane = batch_input + p;
                for (ic = 0u; ic < input_channels; ++ic) {
                    a += wr[ic] * *in_plane;
                    in_plane += hw;
                }
                out_row[p] = a;
            }
        }
    }
}
#endif


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
     * Dedicated 1x1 stride-1 ungrouped path, 35.9 percent of this graph's
     * multiply-accumulates. Such a Conv is a per-pixel [OC x IC] by [IC]
     * matrix-vector product with no neighbourhood, so the spatial dimension
     * collapses to a flat length-HW vector and none of the general path's
     * tap-range, kernel, or padding machinery applies. Dropping all of it lets
     * the inner loop be a plain accumulate the compiler can pipeline. Two
     * output channels share each input read (the same pairing the general path
     * uses), four flat positions run in parallel, and each accumulator sums
     * over input channels in ascending order starting from the bias, exactly
     * the order the general loop below uses, so the result is bit-identical.
     */
#if YR_CONV_1X1_VPU && defined(__riscv)
    if (node->kernel_h == 1 && node->kernel_w == 1
        && node->stride_h == 1 && node->stride_w == 1
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->group == 1
        && (node->pad_top | node->pad_left | node->pad_bottom
            | node->pad_right) == 0) {
        yr_hart_range(output_channels, &oc_lo, &oc_hi);
        yr_conv_1x1_vpu(input, weight, bias, output, batches, input_channels,
                        output_channels, channel_stride, oc_lo, oc_hi);
        return YR_STATUS_OK;
    }
#endif
    if (YR_CONV_1X1_FAST
        && node->kernel_h == 1 && node->kernel_w == 1
        && node->stride_h == 1 && node->stride_w == 1
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->group == 1
        && (node->pad_top | node->pad_left | node->pad_bottom
            | node->pad_right) == 0) {
        const uint32_t hw = channel_stride;
        yr_hart_range(output_channels, &oc_lo, &oc_hi);
        for (n = 0; n < batches; ++n) {
            const float *const batch_input =
                input + (uint64_t)n * input_channels * hw;
            float *const batch_output =
                output + (uint64_t)n * output_channels * hw;
            oc = oc_lo;
            while (oc + 1u < oc_hi) {
                const float *const weight_a = weight + (uint64_t)oc * input_channels;
                const float *const weight_b = weight_a + input_channels;
                const float initial_a =
                    bias == (const float *)0 ? 0.0f : bias[oc];
                const float initial_b =
                    bias == (const float *)0 ? 0.0f : bias[oc + 1u];
                float *const out_a = batch_output + (uint64_t)oc * hw;
                float *const out_b = out_a + hw;
                uint32_t p = 0u;
                while (p + 4u <= hw) {
                    float a0 = initial_a, a1 = initial_a;
                    float a2 = initial_a, a3 = initial_a;
                    float b0 = initial_b, b1 = initial_b;
                    float b2 = initial_b, b3 = initial_b;
                    const float *in_plane = batch_input + p;
                    for (icg = 0; icg < input_channels; ++icg) {
                        const float wa = weight_a[icg];
                        const float wb = weight_b[icg];
                        const float i0 = in_plane[0];
                        const float i1 = in_plane[1];
                        const float i2 = in_plane[2];
                        const float i3 = in_plane[3];
                        a0 += wa * i0; a1 += wa * i1;
                        a2 += wa * i2; a3 += wa * i3;
                        b0 += wb * i0; b1 += wb * i1;
                        b2 += wb * i2; b3 += wb * i3;
                        in_plane += hw;
                    }
                    out_a[p] = a0; out_a[p + 1u] = a1;
                    out_a[p + 2u] = a2; out_a[p + 3u] = a3;
                    out_b[p] = b0; out_b[p + 1u] = b1;
                    out_b[p + 2u] = b2; out_b[p + 3u] = b3;
                    p += 4u;
                }
                for (; p < hw; ++p) {
                    float a = initial_a;
                    float b = initial_b;
                    const float *in_plane = batch_input + p;
                    for (icg = 0; icg < input_channels; ++icg) {
                        const float iv = *in_plane;
                        a += weight_a[icg] * iv;
                        b += weight_b[icg] * iv;
                        in_plane += hw;
                    }
                    out_a[p] = a;
                    out_b[p] = b;
                }
                oc += 2u;
            }
            for (; oc < oc_hi; ++oc) {
                const float *const weight_row =
                    weight + (uint64_t)oc * input_channels;
                const float initial =
                    bias == (const float *)0 ? 0.0f : bias[oc];
                float *const out_row = batch_output + (uint64_t)oc * hw;
                uint32_t p = 0u;
                while (p + 4u <= hw) {
                    float a0 = initial, a1 = initial;
                    float a2 = initial, a3 = initial;
                    const float *in_plane = batch_input + p;
                    for (icg = 0; icg < input_channels; ++icg) {
                        const float w = weight_row[icg];
                        a0 += w * in_plane[0];
                        a1 += w * in_plane[1];
                        a2 += w * in_plane[2];
                        a3 += w * in_plane[3];
                        in_plane += hw;
                    }
                    out_row[p] = a0; out_row[p + 1u] = a1;
                    out_row[p + 2u] = a2; out_row[p + 3u] = a3;
                    p += 4u;
                }
                for (; p < hw; ++p) {
                    float a = initial;
                    const float *in_plane = batch_input + p;
                    for (icg = 0; icg < input_channels; ++icg) {
                        a += weight_row[icg] * *in_plane;
                        in_plane += hw;
                    }
                    out_row[p] = a;
                }
            }
        }
        return YR_STATUS_OK;
    }

    /*
     * Dedicated 3x3 stride-1 same-padding ungrouped path, the graph's largest
     * arithmetic cost. Same-padding here means one pixel on every side, which
     * with a 3x3 kernel keeps output spatial dims equal to input.
     */
#if YR_CONV_3X3_VPU && defined(__riscv)
    if (node->kernel_h == 3 && node->kernel_w == 3
        && node->stride_h == 1 && node->stride_w == 1
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->group == 1
        && node->pad_top == 1 && node->pad_left == 1
        && node->pad_bottom == 1 && node->pad_right == 1) {
        return yr_conv3x3_vpu(
            batches, input_channels, output_channels, input_h, input_w,
            input, weight, bias, output);
    }
#endif
    if (YR_CONV_3X3_FAST
        && node->kernel_h == 3 && node->kernel_w == 3
        && node->stride_h == 1 && node->stride_w == 1
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->group == 1
        && node->pad_top == 1 && node->pad_left == 1
        && node->pad_bottom == 1 && node->pad_right == 1) {
        return yr_conv3x3_s1(
            batches, input_channels, output_channels, input_h, input_w,
            input, weight, bias, output);
    }

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
     * Dedicated 3x3 stride-2 dilation-1 ungrouped path, the downsampling
     * convolutions. Reuses the interior column span computed just above.
     */
    if (YR_CONV_3X3_S2_FAST
        && node->kernel_h == 3 && node->kernel_w == 3
        && node->stride_h == 2 && node->stride_w == 2
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->group == 1) {
        return yr_conv3x3_s2(
            batches, input_channels, output_channels,
            input_h, input_w, output_h, output_w,
            (uint32_t)node->pad_top, (uint32_t)node->pad_left,
            interior_first, interior_end,
            input, weight, bias, output);
    }

    /*
     * Dedicated 3x3 stride-1 same-pad depthwise path (group == channels, one
     * input channel per output channel), the only conv class with no vector or
     * paired path in the general loop below. Guarded on the exact depthwise
     * shape so any other grouped conv still falls through to the general path.
     */
#if YR_CONV_DW3X3_VPU && defined(__riscv)
    if ((uint32_t)node->group == input_channels
        && output_channels == input_channels
        && channels_per_group == 1u
        && node->kernel_h == 3 && node->kernel_w == 3
        && node->stride_h == 1 && node->stride_w == 1
        && node->dilation_h == 1 && node->dilation_w == 1
        && node->pad_top == 1 && node->pad_left == 1
        && node->pad_bottom == 1 && node->pad_right == 1
        && output_h == input_h && output_w == input_w) {
        return yr_conv_dw3x3_s1_vpu(
            batches, input_channels, input_h, input_w,
            input, weight, bias, output);
    }
#endif

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


#ifndef YR_SILU_VPU
#define YR_SILU_VPU 1
#endif

#if YR_SILU_VPU && defined(__riscv)
/*
 * Packed eight-wide SiLU on the vector unit, out = x * sigmoid(x). Builds the
 * sigmoid straight from the hardware exponential and reciprocal instead of the
 * scalar polynomial, eight lanes at a time. sigmoid(x) is 1 / (1 + e^-x), and
 * e^-x is exp2(-x * log2(e)) since the hardware exponential is base two. Runs
 * only on the ET target where these packed ops exist, and never on the host
 * reference, so it sits behind its own build flag. The exponential differs in
 * the last bits from the scalar path, so a build using this validates by
 * tolerance, not by the workspace hash. The parallel elementwise slices are
 * cache-line aligned, so the length is a multiple of sixteen and the eight-wide
 * loop needs no scalar tail.
 */
static void yr_silu_vpu_slice(
    const float *input, float *output, uint32_t elem_lo, uint32_t elem_hi)
{
    union { float f; uint32_t u; } z = { 0.0f };
    union { float f; uint32_t u; } o = { 1.0f };
    union { float f; uint32_t u; } l = { 1.4426950408889634f };
    float vz, vo, vl2e;
    uint32_t i;
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vz) : "r"((uint64_t)z.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vo) : "r"((uint64_t)o.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vl2e) : "r"((uint64_t)l.u));
    for (i = elem_lo; i + 8u <= elem_hi; i += 8u) {
        float x, t;
        __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(x) : "r"(input + i));
        __asm__ volatile(
            "fsub.ps %[t], %[z], %[x]\n"
            "fmul.ps %[t], %[t], %[l2e]\n"
            "fexp.ps %[t], %[t]\n"
            "fadd.ps %[t], %[t], %[o]\n"
            "frcp.ps %[t], %[t]\n"
            "fmul.ps %[t], %[t], %[x]\n"
            : [t] "=&f"(t)
            : [x] "f"(x), [z] "f"(vz), [o] "f"(vo), [l2e] "f"(vl2e));
        __asm__ volatile("fsq2 %1, 0(%0)\n" :: "r"(output + i), "f"(t)
                         : "memory");
    }
}
#endif


/*
 * Fused SiLU over one elementwise slice, out = x * sigmoid(x). Folds a
 * Sigmoid node and the Mul node that follows it into a single pass, so the
 * intermediate sigmoid tensor is never written or read back. The scalar op is
 * the same yr_sigmoid_scalar and a multiply the standalone Sigmoid then Mul
 * would run, in the same order, so the fused output is bit identical to the
 * two-node path and validates against the same workspace hash. With the packed
 * vector build the same slice runs through yr_silu_vpu_slice instead, faster
 * but only tolerance close to the scalar hash.
 */
static uint32_t yr_silu(
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
#if YR_SILU_VPU && defined(__riscv)
    yr_silu_vpu_slice(input, output, elem_lo, elem_hi);
    return YR_STATUS_OK;
#else
    for (index = elem_lo; index < elem_hi; ++index) {
        const float x = input[index];
        output[index] = x * yr_sigmoid_scalar(x);
    }
    return YR_STATUS_OK;
#endif
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
    float *output,
    uint32_t out_lo,
    uint32_t out_hi)
{
    int32_t axis = node->axis;
    uint32_t rank = output_desc->rank;
    uint32_t inner = 1u;
    uint32_t expected_axis = 0u;
    uint32_t input_index;
    uint32_t dimension;
    uint32_t output_block;
    uint32_t axis_offsets[4] = {0u, 0u, 0u, 0u};
    uint32_t flat_index;

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

    output_block = output_desc->dims[axis] * inner;
    {
        uint32_t cumulative = 0u;
        for (input_index = 0u; input_index < node->input_count;
             ++input_index) {
            axis_offsets[input_index] = cumulative;
            cumulative += input_descs[input_index]->dims[axis];
        }
    }
    if (out_hi > output_desc->elements) {
        out_hi = output_desc->elements;
    }
    for (flat_index = out_lo; flat_index < out_hi; ++flat_index) {
        const uint32_t outer_index = flat_index / output_block;
        const uint32_t rem = flat_index % output_block;
        const uint32_t axis_pos = rem / inner;
        const uint32_t inner_index = rem % inner;
        uint32_t which = 0u;
        while (which + 1u < node->input_count
               && axis_pos >= axis_offsets[which + 1u]) {
            ++which;
        }
        {
            const uint32_t local_axis = axis_pos - axis_offsets[which];
            const uint32_t input_block =
                input_descs[which]->dims[axis] * inner;
            output[flat_index] =
                inputs[which][outer_index * input_block
                              + local_axis * inner + inner_index];
        }
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
 * Split across harts, on by default. Split materialises each output as a real
 * copy (yr_split calls yr_copy_bytes) and otherwise runs on hart 0 alone while
 * the other fifteen harts idle; it is about 6 percent of the graph's output
 * traffic and this graph is memory bound. For the outer==1 shape (the C2f
 * channel splits, where the dims before the split axis multiply to one) each
 * output is a single contiguous block, so every hart copies a disjoint
 * cache-line-aligned sub-range of each output and publishes what it wrote. Any
 * other shape falls back to the serial yr_split, so output stays bit-identical.
 */
#ifndef YR_SPLIT_MH
#define YR_SPLIT_MH 1
#endif

#if YR_SPLIT_MH
/*
 * Multi-hart Split for the outer==1 shape, the dims before the split axis
 * multiply to one, so each output is a single contiguous block of copy_elements
 * floats taken from a disjoint slice of the input. Every hart copies a
 * cache-line-aligned sub-range of each output (yr_hart_elem_range) and publishes
 * exactly that slice, so the sixteen harts share the copy that yr_split ran
 * alone. Repeats yr_split's shape validation and returns UNSUPPORTED_SHAPE for
 * anything it cannot split this way; the caller only routes here after the
 * dispatch guard confirmed the shape, and every hart sees the same static
 * manifest so they agree on the outcome. Accumulation-free copy, so the result
 * is bit-identical to the serial path.
 */
static uint32_t yr_split_mh(
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
        || element_bytes != (uint32_t)sizeof(float)
        || !yr_shape_product(input_desc, 0u, (uint32_t)axis, &outer)
        || !yr_shape_product(
            input_desc, (uint32_t)axis + 1u, input_desc->rank, &inner)
        || outer != 1u) {
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
    /*
     * outer == 1, so output j's destination is the contiguous block
     * [0, copy_elements) and its source starts at source_axis_offset * inner.
     * Each hart owns a disjoint line-aligned [lo, hi) of that block.
     */
    for (output_index = 0u; output_index < node->output_count; ++output_index) {
        const uint32_t output_axis = output_descs[output_index]->dims[axis];
        const uint32_t copy_elements = output_axis * inner;
        uint32_t lo, hi;
        yr_hart_elem_range(copy_elements, &lo, &hi);
        if (hi > lo) {
            yr_copy_bytes(
                outputs[output_index] + (uint64_t)lo * element_bytes,
                input + ((uint64_t)source_axis_offset * inner + lo)
                    * element_bytes,
                (hi - lo) * element_bytes);
            yr_publish(
                (const void *)(outputs[output_index]
                    + (uint64_t)lo * element_bytes),
                (hi - lo) * element_bytes);
        }
        source_axis_offset += output_axis;
    }
    return YR_STATUS_OK;
}
#endif


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
    float *output,
    uint32_t flat_row_lo,
    uint32_t flat_row_hi)
{
    uint32_t batch_count;
    uint32_t row;
    uint32_t column;
    uint32_t reduction;
    uint32_t dimension;
    uint32_t m, k, n;
    uint32_t flat_row;
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
    (void)batch_count;
    for (flat_row = flat_row_lo; flat_row < flat_row_hi; ++flat_row) {
        const uint32_t batch = flat_row / m;
        const uint32_t left_base = batch * m * k;
        const uint32_t right_base = batch * k * n;
        row = flat_row - batch * m;
        for (column = 0u; column < n; ++column) {
            float accumulator = 0.0f;
            for (reduction = 0u; reduction < k; ++reduction) {
                accumulator +=
                    left[left_base + row * k + reduction]
                    * right[right_base + reduction * n + column];
            }
            output[(uint64_t)flat_row * n + column] = accumulator;
        }
    }
    return YR_STATUS_OK;
}


static uint32_t yr_softmax(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *output_desc,
    const float *input,
    float *output,
    uint32_t outer_lo,
    uint32_t outer_hi,
    uint32_t inner_lo,
    uint32_t inner_hi)
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
    if (outer_hi > outer) {
        outer_hi = outer;
    }
    if (inner_hi > inner) {
        inner_hi = inner;
    }
    for (outer_index = outer_lo; outer_index < outer_hi; ++outer_index) {
        for (inner_index = inner_lo; inner_index < inner_hi; ++inner_index) {
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
    uint8_t *output,
    uint32_t output_lo,
    uint32_t output_hi)
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
    for (output_index = output_lo; output_index < output_hi;
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
    float *output,
    uint32_t out_lo,
    uint32_t out_hi)
{
    int32_t axis;
    uint32_t outer;
    uint32_t inner;
    uint32_t axis_size;
    uint32_t flat;
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
    /*
     * Flat output index flat = outer_index * inner + inner_index, so a caller
     * range [out_lo, out_hi) selects a contiguous slice of the output that a
     * single hart owns. out_hi == 0 is treated as the whole output for the
     * single-hart callers.
     */
    if (out_hi == 0u) {
        out_hi = outer * inner;
    }
    for (flat = out_lo; flat < out_hi; ++flat) {
        const uint32_t outer_index = inner == 0u ? 0u : flat / inner;
        const uint32_t inner_index = inner == 0u ? 0u : flat % inner;
        const uint32_t base = outer_index * axis_size * inner + inner_index;
        float maximum = input[base];
        for (axis_index = 1u; axis_index < axis_size; ++axis_index) {
            const float value = input[base + axis_index * inner];
            if (value > maximum) {
                maximum = value;
            }
        }
        output[flat] = maximum;
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
#if YR_TOPK_FAST
                if (populated == k) {
                    const uint32_t last_offset =
                        (outer_index * k + (k - 1u)) * inner + inner_index;
                    const float last_value = values[last_offset];
                    const int64_t last_index = indices[last_offset];
                    if (!(candidate > last_value
                          || (candidate == last_value
                              && candidate_index < (uint32_t)last_index))) {
                        continue;
                    }
                }
#endif
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


#ifndef YR_SILU_FUSE
#define YR_SILU_FUSE 1
#endif

#ifndef YR_SILU_CONV_FUSE
#define YR_SILU_CONV_FUSE 0
#endif

/*
 * Split MatMul across harts by whole output rows. Each hart owns a contiguous
 * row range, and since a row is the last dimension of the output (a multiple of
 * sixteen for this graph's MatMuls) each hart's slice is cache-line aligned and
 * disjoint, the same safe pattern the parallel elementwise ops use. Reuses the
 * is_parallel_ew publish path with a row-aligned element range.
 */
#ifndef YR_MATMUL_MH
#define YR_MATMUL_MH 1
#endif

/*
 * Split MaxPool across harts by output channel. Each hart pools a contiguous
 * channel range whose output plane is a multiple of sixteen, so its slice is
 * cache-line aligned and disjoint. Shares the generic struct_stride path that
 * carries the per-unit output size so the is_parallel_ew publish covers exactly
 * the channels this hart wrote.
 */
#ifndef YR_MAXPOOL_MH
#define YR_MAXPOOL_MH 1
#endif

/*
 * Split Transpose across harts by output element. Each output element is an
 * independent scatter read followed by a contiguous write, so a hart owning a
 * cache-line-aligned output range writes a disjoint slice, the plain parallel
 * elementwise pattern. Float output only, so the is_parallel_ew publish stride
 * of one float per element is right.
 */
#ifndef YR_TRANSPOSE_MH
#define YR_TRANSPOSE_MH 1
#endif

/*
 * Split Concat across harts by output element. Each output element copies from
 * exactly one input at a position fixed by its own flat index, so a hart owning
 * a cache-line-aligned output range writes a disjoint slice, the plain parallel
 * elementwise pattern. Float output only, so the is_parallel_ew publish stride
 * of one float per element is right.
 */
#ifndef YR_CONCAT_MH
#define YR_CONCAT_MH 1
#endif

/*
 * Split Softmax across harts by outer group. Each outer group is an independent
 * softmax over the axis (with inner stride) whose output occupies a contiguous
 * block of axis_size*inner floats, so a hart owning a whole number of those
 * blocks writes a disjoint, cache-line-aligned slice. Shares the struct_stride
 * path with that block as the unit. A softmax with a single outer group (outer
 * == 1) stays on hart 0, correct but not split.
 */
#ifndef YR_SOFTMAX_MH
#define YR_SOFTMAX_MH 1
#endif


/*
 * Reshape across harts, off by default. Reshape materialises its output as a
 * flat one-to-one copy of the input (yr_reshape ends in yr_copy_tensor) and
 * otherwise runs on hart 0 alone; it is about 3 percent of the graph's output
 * traffic. Because it is a single output whose element i equals input element i,
 * each hart can copy its published cache-line-aligned element range straight
 * through the generic is_parallel_ew publish, no bespoke path needed. Enabled
 * only when input and output hold the same float element count in aligned
 * workspace; any other shape falls back to the serial yr_reshape, keeping the
 * output bit-identical.
 */
#ifndef YR_RESHAPE_MH
#define YR_RESHAPE_MH 1
#endif

/*
 * ReduceMax across harts, off by default. The head reduces [1,8400,80] to
 * [1,8400] (max over the class axis) on hart 0 alone, reading 2.7 MB in the
 * serial topk_selection tail. Each output element is an independent reduction,
 * so harts split the flat output range in cache-line-aligned slices through the
 * generic is_parallel_ew publish. Same max in the same order, bit-identical.
 */
#ifndef YR_REDUCEMAX_MH
#define YR_REDUCEMAX_MH 1
#endif

/*
 * TopK O(1) early reject. The selection keeps a sorted top-k list; a candidate
 * that cannot beat the current k-th (smallest kept) element can never enter, so
 * once the list is full reject such a candidate in one compare instead of
 * scanning all k positions. Bit-identical to the plain insertion (same kept
 * set, same order, same tie-break), it only skips the doomed full scan, which
 * is the whole cost when most candidates are rejected. Stays on hart 0.
 */
#ifndef YR_TOPK_FAST
#define YR_TOPK_FAST 1
#endif


#if YR_SILU_FUSE || YR_SILU_CONV_FUSE
/*
 * A SiLU is a Sigmoid whose output feeds a Mul that also takes the Sigmoid's
 * own input, giving out = x * sigmoid(x). When node sig_index is that Sigmoid
 * and the very next node is that Mul, the pair folds into a single yr_silu
 * pass written into the Mul output, with the Sigmoid writing nothing. This
 * checks the pattern on the static node list so every hart decides the same
 * way, and only when both nodes sit in one span (sig_index + 1 <= last_node)
 * so a stage boundary never splits a pair into a half-fused state. Each hart
 * still reaches the per-node barrier once for each of the two nodes, so the
 * fold changes no barrier count.
 */
static int yr_silu_pair(uint32_t sig_index, uint32_t last_node)
{
    const struct yr_node_desc *sig;
    const struct yr_node_desc *mul;
    uint32_t x_tensor, s_tensor;
    if (sig_index >= last_node) {
        return 0;
    }
    sig = &yr_nodes[sig_index];
    mul = &yr_nodes[sig_index + 1u];
    if (sig->op != YR_OP_SIGMOID || mul->op != YR_OP_MUL
        || sig->input_count != 1u || sig->output_count != 1u
        || mul->input_count != 2u || mul->output_count != 1u) {
        return 0;
    }
    x_tensor = sig->inputs[0];
    s_tensor = sig->outputs[0];
    return (mul->inputs[0] == x_tensor && mul->inputs[1] == s_tensor)
        || (mul->inputs[0] == s_tensor && mul->inputs[1] == x_tensor);
}
#endif

#if YR_SILU_CONV_FUSE
/*
 * True when this Conv is immediately followed by its own SiLU, that is node
 * conv_index + 1 is the Sigmoid and conv_index + 2 the Mul of a fuseable SiLU
 * pair whose x input is this Conv's output. All three sit in one span. Lets the
 * Conv apply the SiLU to its just-written output slice while it is still warm
 * in this hart's L1, publishing the Mul output so the Conv result never has to
 * go out to DRAM and come back through two separate elementwise nodes.
 */
static int yr_silu_conv_at(uint32_t conv_index, uint32_t last_node)
{
    const struct yr_node_desc *conv;
    if (conv_index + 2u > last_node) {
        return 0;
    }
    conv = &yr_nodes[conv_index];
    if (conv->op != YR_OP_CONV || conv->output_count < 1u) {
        return 0;
    }
    if (!yr_silu_pair(conv_index + 1u, last_node)) {
        return 0;
    }
    return yr_nodes[conv_index + 1u].inputs[0] == conv->outputs[0];
}
#endif


uint32_t yr_run_node_span(
    uint8_t *device_base,
    struct yr_result_header *result,
    uint32_t first_local_node,
    uint32_t last_local_node)
{
    uint32_t node_index;
    uint32_t status = YR_STATUS_OK;
#if YR_SILU_CONV_FUSE
    uint32_t silu_conv_done = 0xFFFFFFFFu;
#endif

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
        uint32_t ew_lo = 0u;
        uint32_t ew_hi = 0u;
        int silu_skip_publish = 0;
#if YR_MATMUL_MH
        int is_matmul_mh = 0;
#endif
#if YR_SOFTMAX_MH
        int sm_inner_mh = 0;
#endif
#if YR_SPLIT_MH
        int is_split_mh = 0;
#endif
        uint32_t struct_stride = 0u;
        /*
         * Nodes fall into three execution classes.
         *
         * Conv is split across all 16 harts by output channel; each hart
         * computes and publishes only its own channels, so there is no
         * redundant work and no shared output line to race on.
         *
         * Plain elementwise ops (SiLU's Sigmoid and Mul, plus Add/Sub) are
         * also split across all 16 harts, here by a cache-line-aligned slice
         * of the flat output (see yr_hart_elem_range and the is_parallel_ew
         * set below). Output element i depends only on input element i, so a
         * disjoint slice per hart is the same provably-safe pattern Conv uses.
         * This matters because a Sigmoid/Mul pair trails almost every Conv,
         * and running that pair on one hart while the other 15 wait was a
         * serial tail out of all proportion to its arithmetic.
         *
         * Everything else (Concat, Split, Transpose, Reshape, TopK, and the
         * rest) still runs on hart 0 alone. An earlier build ran these
         * redundantly on all 16 harts and each published only a cache-line
         * slice; that corrupted data partway through the graph, because L1
         * is minion local and not coherent and the unpublished lines evicted
         * later over live tensors. Partitioning those structural ops safely
         * is harder than for pure elementwise, so they stay single-hart until
         * measured to be worth it.
         *
         * Whichever class it is, every hart calls the per-node barrier exactly
         * once, so a hart that skips a node still waits for the hart(s) that
         * ran it before reading the next node's input.
         */
        int is_parallel_ew = 0;
        if ((node->op == YR_OP_SIGMOID || node->op == YR_OP_MUL
#if YR_MANIFEST_VERSION >= 2
             || node->op == YR_OP_ADD || node->op == YR_OP_SUB
#endif
            )
            && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *ew_out =
                &yr_tensors[node->outputs[0]];
            /*
             * Only split the node across harts when its output sits on whole,
             * aligned cache lines: a base that is a multiple of 64 bytes and a
             * size that is a multiple of 64. Then each hart's 16-float-aligned
             * slice covers complete lines that no other hart or neighbouring
             * tensor shares, so the disjoint evicts never race on a line, which
             * is the failure that made an earlier redundant version corrupt
             * data. Every parallel-elementwise output in the pinned graph
             * already satisfies this; the check keeps a future manifest that
             * did not from silently corrupting by leaving that node on the
             * single-hart path. Static manifest data, so every hart decides
             * identically.
             */
            if (ew_out->storage == YR_STORAGE_WORKSPACE
                && (ew_out->offset % 64u) == 0u
                && (ew_out->nbytes % 64u) == 0u) {
                is_parallel_ew = 1;
            }
        }
#if YR_MATMUL_MH
        if (node->op == YR_OP_MATMUL && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *mm_out =
                &yr_tensors[node->outputs[0]];
            const uint32_t mm_last = mm_out->dims[mm_out->rank - 1u];
            if (mm_out->storage == YR_STORAGE_WORKSPACE
                && (mm_out->offset % 64u) == 0u
                && (mm_out->nbytes % 64u) == 0u
                && mm_last != 0u && (mm_last % 16u) == 0u) {
                is_parallel_ew = 1;
                is_matmul_mh = 1;
            }
        }
#endif
#if YR_MAXPOOL_MH
        if (node->op == YR_OP_MAXPOOL && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *mp_out =
                &yr_tensors[node->outputs[0]];
            if (mp_out->rank == 4u && mp_out->dims[0] == 1u
                && mp_out->storage == YR_STORAGE_WORKSPACE
                && (mp_out->offset % 64u) == 0u
                && (mp_out->nbytes % 64u) == 0u
                && mp_out->dims[1] != 0u
                && ((mp_out->dims[2] * mp_out->dims[3]) % 16u) == 0u) {
                is_parallel_ew = 1;
                struct_stride = mp_out->dims[2] * mp_out->dims[3];
            }
        }
#endif
#if YR_TRANSPOSE_MH
        if (node->op == YR_OP_TRANSPOSE && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *tp_out =
                &yr_tensors[node->outputs[0]];
            if (tp_out->storage == YR_STORAGE_WORKSPACE
                && yr_tensor_dtype(tp_out) == YR_DTYPE_FLOAT
                && (tp_out->offset % 64u) == 0u
                && (tp_out->nbytes % 64u) == 0u) {
                is_parallel_ew = 1;
            }
        }
#endif
#if YR_CONCAT_MH
        if (node->op == YR_OP_CONCAT && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *cc_out =
                &yr_tensors[node->outputs[0]];
            if (cc_out->storage == YR_STORAGE_WORKSPACE
                && yr_tensor_dtype(cc_out) == YR_DTYPE_FLOAT
                && (cc_out->offset % 64u) == 0u
                && (cc_out->nbytes % 64u) == 0u) {
                is_parallel_ew = 1;
            }
        }
#endif
#if YR_SPLIT_MH
        /*
         * Split of the outer==1 shape (dims before the axis multiply to one).
         * every output is a whole number of cache lines in workspace, so each
         * hart can copy and publish a disjoint line-aligned slice of each
         * output. is_parallel_ew keeps all harts on the node and takes the
         * shared publish barrier; is_split_mh routes the dispatch to the
         * multi-output copy. Any other shape leaves both clear and stays serial.
         */
        if (node->op == YR_OP_SPLIT && node->output_count >= 1u
            && node->output_count <= 3u
            && node->input_count == 2u
            && node->inputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *sp_in = &yr_tensors[node->inputs[0]];
            const int32_t sp_axis =
                yr_normalize_axis(node->axis, sp_in->rank);
            int sp_ok = sp_axis >= 0
                && yr_tensor_dtype(sp_in) == YR_DTYPE_FLOAT;
            uint32_t sp_dim, sp_outer = 1u, sp_oi;
            for (sp_dim = 0u; sp_ok && sp_dim < (uint32_t)sp_axis; ++sp_dim) {
                sp_outer *= sp_in->dims[sp_dim];
            }
            if (sp_outer != 1u) {
                sp_ok = 0;
            }
            for (sp_oi = 0u; sp_ok && sp_oi < node->output_count; ++sp_oi) {
                if (node->outputs[sp_oi] >= YR_TENSOR_COUNT) {
                    sp_ok = 0;
                    break;
                }
                {
                    const struct yr_tensor_desc *sp_out =
                        &yr_tensors[node->outputs[sp_oi]];
                    if (sp_out->storage != YR_STORAGE_WORKSPACE
                        || yr_tensor_dtype(sp_out) != YR_DTYPE_FLOAT
                        || (sp_out->offset % 64u) != 0u
                        || (sp_out->nbytes % 64u) != 0u) {
                        sp_ok = 0;
                    }
                }
            }
            if (sp_ok) {
                is_parallel_ew = 1;
                is_split_mh = 1;
            }
        }
#endif
#if YR_RESHAPE_MH
        /*
         * Reshape is a flat one-to-one copy, so element i of the output equals
         * element i of the input. When the output is aligned float workspace and
         * element counts match, each hart copies its published range through the
         * generic is_parallel_ew path (the reshape dispatch does the ranged copy
         * in place of yr_reshape's full copy).
         */
        if (node->op == YR_OP_RESHAPE && node->output_count == 1u
            && node->input_count == 2u
            && node->inputs[0] < YR_TENSOR_COUNT
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *rs_in = &yr_tensors[node->inputs[0]];
            const struct yr_tensor_desc *rs_out = &yr_tensors[node->outputs[0]];
            if (rs_out->storage == YR_STORAGE_WORKSPACE
                && yr_tensor_dtype(rs_out) == YR_DTYPE_FLOAT
                && yr_tensor_dtype(rs_in) == YR_DTYPE_FLOAT
                && rs_in->elements == rs_out->elements
                && (rs_out->offset % 64u) == 0u
                && (rs_out->nbytes % 64u) == 0u) {
                is_parallel_ew = 1;
            }
        }
#endif
#if YR_REDUCEMAX_MH
        /*
         * ReduceMax output is one element per (outer, inner) reduction, all
         * independent, so harts split the flat output through the generic
         * is_parallel_ew publish (the dispatch passes the hart's ew range).
         */
        if (node->op == YR_OP_REDUCEMAX && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *rm_out = &yr_tensors[node->outputs[0]];
            if (rm_out->storage == YR_STORAGE_WORKSPACE
                && yr_tensor_dtype(rm_out) == YR_DTYPE_FLOAT
                && (rm_out->offset % 64u) == 0u
                && (rm_out->nbytes % 64u) == 0u) {
                is_parallel_ew = 1;
            }
        }
#endif
#if YR_SOFTMAX_MH
        if (node->op == YR_OP_SOFTMAX && node->output_count == 1u
            && node->outputs[0] < YR_TENSOR_COUNT) {
            const struct yr_tensor_desc *sm_out =
                &yr_tensors[node->outputs[0]];
            const int32_t sm_axis =
                yr_normalize_axis(node->axis, sm_out->rank);
            if (sm_axis >= 0
                && sm_out->storage == YR_STORAGE_WORKSPACE
                && yr_tensor_dtype(sm_out) == YR_DTYPE_FLOAT
                && (sm_out->offset % 64u) == 0u
                && (sm_out->nbytes % 64u) == 0u) {
                uint32_t sm_inner = 1u;
                uint32_t sm_outer = 1u;
                uint32_t sm_dim;
                uint32_t sm_block;
                for (sm_dim = (uint32_t)sm_axis + 1u;
                     sm_dim < sm_out->rank; ++sm_dim) {
                    sm_inner *= sm_out->dims[sm_dim];
                }
                for (sm_dim = 0u; sm_dim < (uint32_t)sm_axis; ++sm_dim) {
                    sm_outer *= sm_out->dims[sm_dim];
                }
                sm_block = sm_out->dims[sm_axis] * sm_inner;
                if (sm_outer > 1u && sm_block != 0u
                    && (sm_block % 16u) == 0u) {
                    /* Many outer groups: split by outer, each a contiguous
                     * cache-line-aligned block, published by the generic
                     * is_parallel_ew path. */
                    is_parallel_ew = 1;
                    struct_stride = sm_block;
                } else if (sm_outer == 1u && sm_inner >= 32u
                           && (sm_inner % 16u) == 0u) {
                    /* Single outer group with a wide inner: split by inner
                     * columns in whole cache lines. Each hart's output is
                     * axis_size disjoint line-aligned strips, published by the
                     * softmax dispatch below (silu_skip_publish suppresses the
                     * generic single-range publish). */
                    is_parallel_ew = 1;
                    sm_inner_mh = 1;
                }
            }
        }
#endif
        if (node->op != YR_OP_CONV && !is_parallel_ew && yr_hart_id() != 0u) {
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
        if (is_parallel_ew) {
#if YR_MATMUL_MH
            if (is_matmul_mh) {
                const uint32_t mm_last = out0->dims[out0->rank - 1u];
                uint32_t mm_r_lo, mm_r_hi;
                yr_hart_range(out0->elements / mm_last, &mm_r_lo, &mm_r_hi);
                ew_lo = mm_r_lo * mm_last;
                ew_hi = mm_r_hi * mm_last;
            } else
#endif
            if (struct_stride != 0u) {
                uint32_t s_lo, s_hi;
                yr_hart_range(out0->elements / struct_stride, &s_lo, &s_hi);
                ew_lo = s_lo * struct_stride;
                ew_hi = s_hi * struct_stride;
            } else {
                yr_hart_elem_range(out0->elements, &ew_lo, &ew_hi);
            }
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
#if YR_SILU_CONV_FUSE
            int silu_conv_here = 0;
            if (yr_silu_conv_at(node_index, last_local_node)) {
                const struct yr_tensor_desc *m_desc =
                    &yr_tensors[yr_nodes[node_index + 2u].outputs[0]];
                uint8_t *m_raw = yr_tensor_raw(device_base, m_desc);
                if (m_raw != (uint8_t *)0 && yr_same_shape(out0, m_desc)) {
                    /*
                     * Point the conv output at the Mul tensor so it writes M
                     * straight away. C is never materialized, which avoids
                     * leaving unpublished dirty C lines in this minion's L1
                     * that could later evict over a tensor reusing C's region.
                     */
                    out0_raw = m_raw;
                    silu_conv_here = 1;
                }
            }
#endif
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
                    const uint32_t plane = out0->dims[2] * out0->dims[3];
                    const uint32_t plane_bytes = plane * 4u;
                    const uint32_t batch_bytes = out0->dims[1] * plane_bytes;
                    uint32_t batch_index;
#if YR_SILU_CONV_FUSE
                    if (silu_conv_here) {
                        /*
                         * The conv wrote the Mul output tensor directly, so C
                         * was never materialized. Apply the SiLU in place on
                         * that slice, still warm in this hart's L1, then let
                         * the publish below evict M. The Sigmoid and Mul that
                         * follow skip their work and keep their barrier.
                         */
                        for (batch_index = 0u;
                             batch_index < out0->dims[0]; ++batch_index) {
                            const uint32_t base_i =
                                batch_index * out0->dims[1] * plane;
                            (void)yr_silu(
                                out0, out0, (const float *)out0_raw,
                                (float *)out0_raw, base_i + oc_lo * plane,
                                base_i + oc_hi * plane);
                        }
                        silu_conv_done = node_index + 2u;
                    }
#endif
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
            if (node->input_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
#if YR_SILU_CONV_FUSE
            if (silu_conv_done == node_index + 1u) {
                /*
                 * This Sigmoid's SiLU was already applied by the Conv two
                 * nodes back, which wrote the Mul output directly. Skip the
                 * work and the publish, keep the barrier so every hart stays
                 * in step.
                 */
                status = YR_STATUS_OK;
                silu_skip_publish = 1;
            } else
#endif
#if YR_SILU_FUSE
            if (yr_silu_pair(node_index, last_local_node)) {
                /*
                 * The next node is this Sigmoid's Mul, and yr_silu there does
                 * x * sigmoid(x) in one pass, so the sigmoid output is neither
                 * written nor published here. The barrier still runs below so
                 * every hart stays in step.
                 */
                status = YR_STATUS_OK;
                silu_skip_publish = 1;
            } else
#endif
            {
                status = yr_sigmoid(
                    in0, out0, (float *)in0_raw, (float *)out0_raw,
                    ew_lo, ew_hi);
            }
        } else if (node->op == YR_OP_MUL) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
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
#if YR_SILU_CONV_FUSE
            if (silu_conv_done == node_index) {
                /*
                 * The Conv three nodes back already wrote this Mul output as
                 * silu of its result. Skip work and publish, keep the barrier.
                 */
                status = YR_STATUS_OK;
                silu_skip_publish = 1;
            } else
#endif
#if YR_SILU_FUSE
            if (node_index > first_local_node
                && yr_silu_pair(node_index - 1u, last_local_node)) {
                /*
                 * Second half of a SiLU whose Sigmoid was folded away above.
                 * x is the Mul input that is the Sigmoid's own input; the
                 * other input is the sigmoid output, never written, so only x
                 * is read and yr_silu recomputes sigmoid(x) inline.
                 */
                const uint32_t x_tensor = yr_nodes[node_index - 1u].inputs[0];
                if (node->inputs[0] == x_tensor) {
                    status = yr_silu(in0, out0, (const float *)in0_raw,
                                     (float *)out0_raw, ew_lo, ew_hi);
                } else {
                    status = yr_silu(in1, out0, (const float *)in1_raw,
                                     (float *)out0_raw, ew_lo, ew_hi);
                }
            } else
#endif
            {
                status = yr_mul(
                    in0, in1, out0, (float *)in0_raw,
                    (float *)in1_raw, (float *)out0_raw, ew_lo, ew_hi);
            }
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
            {
                uint32_t cc_lo = 0u;
                uint32_t cc_hi = out0->elements;
                if (is_parallel_ew) {
                    cc_lo = ew_lo;
                    cc_hi = ew_hi;
                }
                status = yr_concat(
                    node, input_descs, out0, input_data, (float *)out0_raw,
                    cc_lo, cc_hi);
            }
#if YR_MANIFEST_VERSION >= 2
        } else if (node->op == YR_OP_ADD || node->op == YR_OP_SUB) {
            const struct yr_tensor_desc *in1;
            uint8_t *in1_raw;
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
            status = yr_add_sub(
                in0, in1, out0, (float *)in0_raw, (float *)in1_raw,
                (float *)out0_raw, node->op == YR_OP_SUB, ew_lo, ew_hi);
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
#if YR_SPLIT_MH
            if (is_split_mh) {
                /* Every hart copies and publishes its own slice of each output,
                 * so suppress the generic single-output publish below. */
                silu_skip_publish = 1;
                status = yr_split_mh(
                    node, in0, sizes_desc, output_descs, in0_raw,
                    (const int64_t *)sizes_raw, output_data);
            } else
#endif
            status = yr_split(
                node, in0, sizes_desc, output_descs, in0_raw,
                (const int64_t *)sizes_raw, output_data);
        } else if (node->op == YR_OP_MAXPOOL) {
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            {
                uint32_t mp_lo = 0u;
                uint32_t mp_hi = out0->dims[1];
                if (struct_stride != 0u) {
                    mp_lo = ew_lo / struct_stride;
                    mp_hi = ew_hi / struct_stride;
                }
                status = yr_maxpool(
                    node, in0, out0, (float *)in0_raw, (float *)out0_raw,
                    mp_lo, mp_hi);
            }
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
            {
                const uint32_t mm_last = out0->dims[out0->rank - 1u];
                uint32_t mm_lo = 0u;
                uint32_t mm_hi = mm_last != 0u ? out0->elements / mm_last : 0u;
#if YR_MATMUL_MH
                if (is_matmul_mh) {
                    mm_lo = ew_lo / mm_last;
                    mm_hi = ew_hi / mm_last;
                }
#endif
                status = yr_matmul(
                    in0, in1, out0, (float *)in0_raw, (float *)in1_raw,
                    (float *)out0_raw, mm_lo, mm_hi);
            }
        } else if (node->op == YR_OP_SOFTMAX) {
            uint32_t sm_o_lo = 0u;
            uint32_t sm_o_hi = 0xffffffffu;
            uint32_t sm_i_lo = 0u;
            uint32_t sm_i_hi = 0xffffffffu;
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
#if YR_SOFTMAX_MH
            if (is_parallel_ew && struct_stride != 0u) {
                sm_o_lo = ew_lo / struct_stride;
                sm_o_hi = ew_hi / struct_stride;
            } else if (sm_inner_mh) {
                const int32_t sm_ax =
                    yr_normalize_axis(node->axis, out0->rank);
                uint32_t sm_inner = 1u;
                uint32_t sm_dim;
                uint32_t sm_units;
                uint32_t r_lo;
                uint32_t r_hi;
                for (sm_dim = (uint32_t)sm_ax + 1u;
                     sm_dim < out0->rank; ++sm_dim) {
                    sm_inner *= out0->dims[sm_dim];
                }
                sm_units = sm_inner / 16u;
                yr_hart_range(sm_units, &r_lo, &r_hi);
                sm_i_lo = r_lo * 16u;
                sm_i_hi = r_hi * 16u;
                silu_skip_publish = 1;
            }
#endif
            status = yr_softmax(
                node, in0, out0, (float *)in0_raw, (float *)out0_raw,
                sm_o_lo, sm_o_hi, sm_i_lo, sm_i_hi);
#if YR_SOFTMAX_MH
            if (sm_inner_mh && status == YR_STATUS_OK && sm_i_hi > sm_i_lo) {
                const int32_t sm_ax =
                    yr_normalize_axis(node->axis, out0->rank);
                uint32_t sm_inner = 1u;
                uint32_t sm_outer = 1u;
                uint32_t sm_dim;
                uint32_t oo;
                uint32_t kk;
                uint32_t axsz = out0->dims[sm_ax];
                for (sm_dim = (uint32_t)sm_ax + 1u;
                     sm_dim < out0->rank; ++sm_dim) {
                    sm_inner *= out0->dims[sm_dim];
                }
                for (sm_dim = 0u; sm_dim < (uint32_t)sm_ax; ++sm_dim) {
                    sm_outer *= out0->dims[sm_dim];
                }
                for (oo = 0u; oo < sm_outer; ++oo) {
                    for (kk = 0u; kk < axsz; ++kk) {
                        const uint32_t sbase =
                            (oo * axsz + kk) * sm_inner + sm_i_lo;
                        yr_publish(
                            (const void *)(out0_raw
                                + (uint64_t)sbase * sizeof(float)),
                            (sm_i_hi - sm_i_lo) * (uint32_t)sizeof(float));
                    }
                }
            }
#endif
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
#if YR_RESHAPE_MH
            if (is_parallel_ew) {
                /* Flat 1:1 copy: this hart copies its published element range.
                 * The generic is_parallel_ew publish below evicts exactly it. */
                if (ew_hi > ew_lo) {
                    yr_copy_bytes(
                        out0_raw + (uint64_t)ew_lo * sizeof(float),
                        in0_raw + (uint64_t)ew_lo * sizeof(float),
                        (ew_hi - ew_lo) * (uint32_t)sizeof(float));
                }
                status = YR_STATUS_OK;
            } else
#endif
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
            {
                uint32_t tp_lo = 0u;
                uint32_t tp_hi = out0->elements;
                if (is_parallel_ew) {
                    tp_lo = ew_lo;
                    tp_hi = ew_hi;
                }
                status = yr_transpose(
                    node, in0, out0, in0_raw, out0_raw, tp_lo, tp_hi);
            }
        } else if (node->op == YR_OP_REDUCEMAX) {
            uint32_t rm_lo = 0u, rm_hi = 0u;
            if (node->input_count != 1u || node->output_count != 1u) {
                status = YR_STATUS_BAD_MANIFEST;
                goto fail;
            }
            if (is_parallel_ew) {
                rm_lo = ew_lo;
                rm_hi = ew_hi;
            }
            status = yr_reduce_max(
                node, in0, out0, (float *)in0_raw, (float *)out0_raw,
                rm_lo, rm_hi);
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
         * A parallel elementwise op wrote only its own cache-line-aligned
         * slice of the single output, so each hart evicts exactly that slice;
         * the slices are disjoint and line-aligned, so no two harts evict a
         * shared line. Every other non-Conv op ran on hart 0 alone, so hart 0
         * evicts each output's full range and no other hart has a slice to
         * publish. Either way the barrier must run on every hart whether or
         * not the status is OK, to match the unconditional barrier the
         * skipped harts already called; dropping it on failure would strand
         * them waiting for a call that never comes.
         */
        if (is_parallel_ew) {
            if (status == YR_STATUS_OK && ew_hi > ew_lo && !silu_skip_publish) {
                yr_publish(out0_raw + (uint64_t)ew_lo * sizeof(float),
                           (ew_hi - ew_lo) * (uint32_t)sizeof(float));
            }
            yr_hart_barrier();
        } else if (node->op != YR_OP_CONV) {
            if (status == YR_STATUS_OK && !silu_skip_publish) {
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
 * source that overrides yr_conv_tensor() below. A single-translation-unit build
 * that compiles that source alongside this one defines
 * YR_CONV_TENSOR_STRONG_PRESENT, because a strong definition cannot share a
 * translation unit with the weak one it replaces; the separate-compilation
 * builds leave the macro unset and keep resolving the override at link time.
 */
#ifndef YR_CONV_TENSOR_STRONG_PRESENT
__attribute__((weak))
void yr_conv_tensor_init(void)
{
}
#endif


/*
 * Default fast path, always declines. An ET-only source that defines this
 * symbol without the weak attribute overrides it at link time; the host
 * build and any ET build that does not list that source keep this stub, so
 * behavior stays exactly the portable scalar path. Guarded for the
 * single-translation-unit build for the same reason as yr_conv_tensor_init().
 */
#ifndef YR_CONV_TENSOR_STRONG_PRESENT
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
#endif
