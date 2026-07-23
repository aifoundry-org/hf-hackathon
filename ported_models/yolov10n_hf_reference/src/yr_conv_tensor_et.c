/*
 * ET-only tensor-unit fast path for 1x1 stride-1 Conv nodes.
 *
 * Linked only into the ET build (see build_et_slice.sh). Overrides the weak
 * yr_conv_tensor() stub in ref_runtime.c, which always declines and keeps
 * the portable scalar path in effect on host. Runs on a single hart only;
 * ref_runtime.c gates every call on yr_hart_count() == 1, so this file has
 * no hart partitioning of its own and needs no barrier.
 *
 * A 1x1 stride-1 Conv over NCHW tensors is just a per-pixel [OC x IC] times
 * [IC] matrix-vector product, batched over every spatial position, so it
 * maps directly onto the tensor unit's 16x16 multiply-accumulate array:
 * one weight tile of OC16 rows by IC16 columns, one activation tile of
 * IC16 rows by HW16 columns, tensor_fma-ed together and accumulated across
 * the IC16 tiles that make up the real input-channel count. Bias is added
 * afterwards in plain scalar code; no activation is fused here, because in
 * this graph SiLU is separate Sigmoid/Mul ONNX nodes dispatched elsewhere.
 *
 * Tile walk keeps a running pointer per loop level instead of recomputing
 * oc0*input_channels + ic0 style offsets on every iteration, the same
 * approach yr_conv() in ref_runtime.c uses for its own tap loops.
 */

#include <stdint.h>

#if __has_include("erbium/isa/syscall.h")
#include "erbium/isa/syscall.h"
#else
#include "erbium-soc1sim/isa/syscall.h"
#endif
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"
#include "erbium/isa/utils.h"

#include "ref_runtime.h"
#include "slice_manifest.h"

#define YR_TENSOR_TILE 16u

static int yr_tensor_scp_ready(void)
{
    static int checked = 0;
    static int ready = 0;
    if (!checked) {
        checked = 1;
        if (get_l1d_mode() == l1d_scp) {
            ready = 1;
        } else if (syscall(SYSCALL_CACHE_CONTROL, 1u, 1u, 0u) == 0) {
            ucache_control(1u, 0u, 0u);
            ready = (get_l1d_mode() == l1d_scp);
        }
    }
    return ready;
}

static void yr_tensor_clobber_fregs(void)
{
    __asm__ volatile("" ::
                         : "memory", "f0", "f1", "f2", "f3", "f4", "f5", "f6",
                           "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
                           "f15", "f16", "f17", "f18", "f19", "f20", "f21",
                           "f22", "f23", "f24", "f25", "f26", "f27", "f28",
                           "f29", "f30", "f31");
}

/*
 * Add bias to a computed OC16 x HW16 tile and write it out. No activation.
 *
 * Plain scalar loop, not the packed fbcx.ps/flq2/fadd.ps/fsq2 sequence the
 * epilogue used before: that packed form only ever updated 2 of every 16
 * elements (lanes 0 and 8), leaving the rest with the pre-bias FMA result.
 * The bias add here is a tiny fraction of the node's cost next to the
 * OC16/IC16/HW16 matmul above it, so there is nothing to gain from packing
 * it and real correctness risk in redoing that asm without being able to
 * verify the lane width on real hardware first.
 */
static void yr_tensor_bias_epilogue(
    float *output, const float *bias, uint32_t oc0, uint32_t HW, uint32_t hw0)
{
    uint32_t r, j;
    if (bias == (const float *)0) {
        return;
    }
    for (r = 0u; r < YR_TENSOR_TILE; ++r) {
        const float b = bias[oc0 + r];
        float *dst = output + (uint64_t)(oc0 + r) * HW + hw0;
        for (j = 0u; j < YR_TENSOR_TILE; ++j) {
            dst[j] += b;
        }
    }
}

/* True when the node is a plain, unpadded, unit-stride, ungrouped 1x1 Conv
 * whose channel and flattened-spatial extents all tile evenly into 16. */
static int yr_conv_tensor_1x1_applies(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *weight_desc,
    const struct yr_tensor_desc *output_desc)
{
    const int64_t unit_kernel = 1;
    uint32_t input_channels, output_channels, flat_pixels;

    if (input_desc->rank != 4u || weight_desc->rank != 4u
        || output_desc->rank != 4u) {
        return 0;
    }
    if ((int64_t)node->group != 1 || node->kernel_h != unit_kernel
        || node->kernel_w != unit_kernel || node->stride_h != unit_kernel
        || node->stride_w != unit_kernel || node->dilation_h != unit_kernel
        || node->dilation_w != unit_kernel
        || (node->pad_top | node->pad_left | node->pad_bottom
            | node->pad_right) != 0) {
        return 0;
    }
    if (input_desc->dims[0] != output_desc->dims[0]
        || input_desc->dims[2] != output_desc->dims[2]
        || input_desc->dims[3] != output_desc->dims[3]) {
        return 0;
    }
    input_channels = input_desc->dims[1];
    output_channels = output_desc->dims[1];
    flat_pixels = input_desc->dims[2] * input_desc->dims[3];
    if (input_channels == 0u || output_channels == 0u || flat_pixels == 0u) {
        return 0;
    }
    return input_channels % YR_TENSOR_TILE == 0u
        && output_channels % YR_TENSOR_TILE == 0u
        && flat_pixels % YR_TENSOR_TILE == 0u;
}

/*
 * Splits [0, count) into yr_hart_count() near-equal pieces, remainder going
 * to the lowest-indexed harts, and returns the piece owned by yr_hart_id().
 * Same partition formula as the static yr_hart_range() in ref_runtime.c,
 * duplicated here rather than exposed across translation units because it
 * is three lines and this file already keeps itself free of ref_runtime.c
 * internals. With a single hart this is [0, count), so the single-hart
 * tensor build partitions to its full range unchanged.
 */
static void yr_conv_tensor_hart_range(uint32_t count, uint32_t *lo, uint32_t *hi)
{
    const uint32_t harts = yr_hart_count();
    const uint32_t id = yr_hart_id();
    *lo = (count * id) / harts;
    *hi = (count * (id + 1u)) / harts;
}

/*
 * Walks OC16 x IC16 x HW16 tiles with a running pointer per level (weight,
 * activation, output all advance by their own tile stride each iteration)
 * instead of rederiving oc0/ic0/hw0 offsets on every pass, mirroring how
 * yr_conv()'s tap loops avoid recomputed indices in ref_runtime.c. Eviction
 * brackets the whole OC16 output slab once before and once after its HW16
 * sweep so every tensor_store into that slab lands before the slab is
 * published, matching the single evict pass yr_publish() expects upstream.
 *
 * oc_tile_lo/oc_tile_hi restrict the sweep to a sub-range of OC16 tiles so
 * multiple harts can split one node's output channels between them; the
 * caller (yr_conv_tensor()) works out that range per hart and reports it
 * back so ref_runtime.c publishes the matching slice.
 */
static void yr_conv_tensor_1x1_run(
    const float *input, const float *weight, const float *bias,
    float *output, uint32_t batches, uint32_t input_channels,
    uint32_t output_channels, uint32_t HW,
    uint32_t oc_tile_lo, uint32_t oc_tile_hi)
{
    const uint32_t ic_tiles = input_channels / YR_TENSOR_TILE;
    const uint32_t hw_tiles = HW / YR_TENSOR_TILE;
    const uint64_t weight_row_stride = (uint64_t)input_channels * sizeof(float);
    const uint64_t activation_row_stride = (uint64_t)HW * sizeof(float);
    const uint64_t oc_slab_stride = (uint64_t)YR_TENSOR_TILE * HW;
    uint32_t n;

    for (n = 0u; n < batches; ++n) {
        const float *batch_input = input + (uint64_t)n * input_channels * HW;
        float *batch_output = output + (uint64_t)n * output_channels * HW;
        const float *oc_weight_row =
            weight + (uint64_t)oc_tile_lo * YR_TENSOR_TILE * input_channels;
        float *oc_output_slab =
            batch_output + (uint64_t)oc_tile_lo * oc_slab_stride;
        uint32_t t_oc;

        for (t_oc = oc_tile_lo; t_oc < oc_tile_hi; ++t_oc) {
            const uint32_t oc0 = t_oc * YR_TENSOR_TILE;
            const float *hw_activation_col = batch_input;
            float *hw_output_col = oc_output_slab;
            uint32_t t_hw;

            evict((const void *)oc_output_slab,
                  oc_slab_stride * sizeof(float));
            WAIT_CACHEOPS;
            FENCE;

            for (t_hw = 0u; t_hw < hw_tiles; ++t_hw) {
                const float *ic_weight_tile = oc_weight_row;
                const float *ic_activation_tile = hw_activation_col;
                uint32_t t_ic;

                for (t_ic = 0u; t_ic < ic_tiles; ++t_ic) {
                    tensor_load(0u, 0u, 0u, 0u, 0u, (uint64_t)ic_weight_tile,
                                0u, 15u, weight_row_stride, 0u);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_load(0u, 0u, 0u, 0u, 1u,
                                (uint64_t)ic_activation_tile, 0u, 15u,
                                activation_row_stride, 1u);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_fma(0u, 3u, 15u, 15u, 0u, 0u, 0u, 0u, 1u, 0u, 0u,
                               0u, t_ic == 0u);
                    tensor_wait(TENSOR_FMA_WAIT);

                    ic_weight_tile += YR_TENSOR_TILE;
                    ic_activation_tile += (uint64_t)YR_TENSOR_TILE * HW;
                }

                tensor_store(0u, 0u, 3u, 15u, (uint64_t)hw_output_col, 0u,
                             activation_row_stride);
                tensor_wait(TENSOR_STORE_WAIT);
                yr_tensor_clobber_fregs();
                yr_tensor_bias_epilogue(
                    batch_output, bias, oc0, HW, t_hw * YR_TENSOR_TILE);

                hw_activation_col += YR_TENSOR_TILE;
                hw_output_col += YR_TENSOR_TILE;
            }

            evict((const void *)oc_output_slab,
                  oc_slab_stride * sizeof(float));
            WAIT_CACHEOPS;
            FENCE;

            oc_weight_row += (uint64_t)YR_TENSOR_TILE * input_channels;
            oc_output_slab += oc_slab_stride;
        }
    }
}

/*
 * ET-only tensor-unit path for 3x3 stride-1 "same" padding Conv nodes.
 *
 * A 3x3 tap only lines up with the tensor unit's 16x16 array when every one
 * of its 16 activation lanes needs the exact same (ic,row) source, which is
 * only true for the middle column (kx=1): shifting the input row by ky-1
 * turns that single column into the same [OC x IC] times [IC x HW16]
 * product the 1x1 path already runs. The kx={0,2} columns shift every lane
 * by one pixel relative to its neighbour, so they cannot share one tensor
 * load; those six of nine taps, plus the bias, are folded in afterwards
 * with a plain scalar accumulation on top of the tensor-computed partial
 * sum. Weights for the middle column are gathered once per node into
 * g_yr_conv3x3_center_weight so each tensor_load still reads a contiguous
 * IC16 row instead of the kernel's native ic-then-3x3 stride.
 */

#define YR_TENSOR_3X3_MAX_CHANNELS 128u

static float g_yr_conv3x3_center_weight
    [3][YR_TENSOR_3X3_MAX_CHANNELS * YR_TENSOR_3X3_MAX_CHANNELS]
    __attribute__((aligned(64)));

static int yr_conv_tensor_3x3_applies(
    const struct yr_node_desc *node,
    const struct yr_tensor_desc *input_desc,
    const struct yr_tensor_desc *weight_desc,
    const struct yr_tensor_desc *output_desc)
{
    const int64_t same_pad = 1;
    uint32_t input_channels, output_channels, row_width;

    if (input_desc->rank != 4u || weight_desc->rank != 4u
        || output_desc->rank != 4u) {
        return 0;
    }
    if ((int64_t)node->group != 1 || node->kernel_h != 3
        || node->kernel_w != 3 || node->stride_h != 1 || node->stride_w != 1
        || node->dilation_h != 1 || node->dilation_w != 1
        || node->pad_top != same_pad || node->pad_left != same_pad
        || node->pad_bottom != same_pad || node->pad_right != same_pad) {
        return 0;
    }
    if (input_desc->dims[0] != output_desc->dims[0]
        || input_desc->dims[2] != output_desc->dims[2]
        || input_desc->dims[3] != output_desc->dims[3]) {
        return 0;
    }
    input_channels = input_desc->dims[1];
    output_channels = output_desc->dims[1];
    row_width = input_desc->dims[3];
    if (input_channels == 0u || output_channels == 0u || row_width == 0u
        || input_desc->dims[2] == 0u) {
        return 0;
    }
    return input_channels % YR_TENSOR_TILE == 0u
        && output_channels % YR_TENSOR_TILE == 0u
        && row_width % YR_TENSOR_TILE == 0u
        && input_channels <= YR_TENSOR_3X3_MAX_CHANNELS
        && output_channels <= YR_TENSOR_3X3_MAX_CHANNELS;
}

/*
 * Gather the kx=1 column of every (oc,ic) pair into three contiguous
 * [OC x IC] planes, one per ky, so the tensor unit can read an IC16 row
 * without striding through the other eight taps of each 3x3 filter.
 *
 * The three planes are written with plain stores from this hart, unlike
 * the node's own weight buffer, which the launcher DMAs straight into
 * device DRAM before the kernel ever starts. tensor_load reads below this
 * level, so each plane needs an explicit evict once it is fully written,
 * or the tensor unit sees whatever this address held before the repack
 * rather than the values just stored here.
 */
static void yr_conv_tensor_repack_3x3_center(
    const float *weight, uint32_t output_channels, uint32_t input_channels)
{
    const uint64_t plane_bytes = (uint64_t)output_channels * input_channels
        * sizeof(float);
    uint32_t ky;
    for (ky = 0u; ky < 3u; ++ky) {
        float *plane = g_yr_conv3x3_center_weight[ky];
        const float *tap_base = weight + ky * 3u + 1u;
        uint32_t oc;
        for (oc = 0u; oc < output_channels; ++oc) {
            float *dst_row = plane + (uint64_t)oc * input_channels;
            const float *src_row = tap_base + (uint64_t)oc * input_channels * 9u;
            uint32_t ic;
            for (ic = 0u; ic < input_channels; ++ic) {
                dst_row[ic] = src_row[ic * 9u];
            }
        }
        evict((const void *)plane, plane_bytes);
    }
    WAIT_CACHEOPS;
    FENCE;
}

/*
 * Adds bias plus the kx={0,2} taps (all three ky rows, boundary-checked)
 * on top of whatever the tensor unit already stored for the kx=1 column,
 * for one OC16 x 16-wide row segment. Plain nested loops, no packed asm.
 */
static void yr_conv_tensor_3x3_sides_and_bias(
    const float *channel_input, const float *weight, const float *bias,
    float *channel_output, uint32_t input_channels, uint32_t H,
    uint32_t row_width, uint32_t oc0, uint32_t oh, uint32_t ow0)
{
    const uint32_t plane = H * row_width;
    uint32_t lane;

    for (lane = 0u; lane < YR_TENSOR_TILE; ++lane) {
        const uint32_t oc = oc0 + lane;
        const float bias_value =
            (bias != (const float *)0) ? bias[oc] : 0.0f;
        float *out_row = channel_output + (uint64_t)oc * plane + oh * row_width;
        uint32_t col;

        for (col = 0u; col < YR_TENSOR_TILE; ++col) {
            const uint32_t ow = ow0 + col;
            float accumulator = bias_value;
            int32_t ky;

            for (ky = 0; ky < 3; ++ky) {
                const int32_t ih = (int32_t)oh + ky - 1;
                uint32_t side;
                if (ih < 0 || ih >= (int32_t)H) {
                    continue;
                }
                for (side = 0u; side < 2u; ++side) {
                    const int32_t kx = (int32_t)(side * 2u);
                    const int32_t iw = (int32_t)ow + kx - 1;
                    const float *taps;
                    uint32_t ic;
                    if (iw < 0 || iw >= (int32_t)row_width) {
                        continue;
                    }
                    taps = weight + (uint64_t)oc * input_channels * 9u
                        + (uint32_t)ky * 3u + (uint32_t)kx;
                    for (ic = 0u; ic < input_channels; ++ic) {
                        accumulator += taps[ic * 9u]
                            * channel_input[(uint64_t)ic * plane
                                + (uint32_t)ih * row_width
                                + (uint32_t)iw];
                    }
                }
            }
            out_row[ow] += accumulator;
        }
    }
}

static void yr_conv_tensor_3x3_run(
    const float *input, const float *weight, const float *bias,
    float *output, uint32_t batches, uint32_t input_channels,
    uint32_t output_channels, uint32_t H, uint32_t row_width)
{
    const uint32_t oc_tiles = output_channels / YR_TENSOR_TILE;
    const uint32_t ic_tiles = input_channels / YR_TENSOR_TILE;
    const uint32_t ow_tiles = row_width / YR_TENSOR_TILE;
    const uint64_t weight_row_stride = (uint64_t)input_channels * sizeof(float);
    const uint64_t channel_plane_stride =
        (uint64_t)H * row_width * sizeof(float);
    const uint32_t plane = H * row_width;
    uint32_t n;

    yr_conv_tensor_repack_3x3_center(weight, output_channels, input_channels);

    for (n = 0u; n < batches; ++n) {
        const float *batch_input = input + (uint64_t)n * input_channels * plane;
        float *batch_output = output + (uint64_t)n * output_channels * plane;
        uint32_t t_oc;

        for (t_oc = 0u; t_oc < oc_tiles; ++t_oc) {
            const uint32_t oc0 = t_oc * YR_TENSOR_TILE;
            float *oc_output_slab = batch_output + (uint64_t)oc0 * plane;
            uint32_t oh;

            evict((const void *)oc_output_slab, channel_plane_stride * YR_TENSOR_TILE);
            WAIT_CACHEOPS;
            FENCE;

            for (oh = 0u; oh < H; ++oh) {
                uint32_t t_ow;

                for (t_ow = 0u; t_ow < ow_tiles; ++t_ow) {
                    const uint32_t ow0 = t_ow * YR_TENSOR_TILE;
                    float *ctile = oc_output_slab + oh * row_width + ow0;
                    int center_first = 1;
                    int32_t ky;

                    for (ky = 0; ky < 3; ++ky) {
                        const int32_t ih = (int32_t)oh + ky - 1;
                        const float *ky_weight_row;
                        uint32_t t_ic;
                        if (ih < 0 || ih >= (int32_t)H) {
                            continue;
                        }
                        ky_weight_row = g_yr_conv3x3_center_weight[ky]
                            + (uint64_t)oc0 * input_channels;

                        for (t_ic = 0u; t_ic < ic_tiles; ++t_ic) {
                            const uint32_t ic0 = t_ic * YR_TENSOR_TILE;
                            const float *wtile = ky_weight_row + ic0;
                            const float *atile = batch_input
                                + (uint64_t)ic0 * plane
                                + (uint32_t)ih * row_width + ow0;

                            tensor_load(0u, 0u, 0u, 0u, 0u, (uint64_t)wtile,
                                        0u, 15u, weight_row_stride, 0u);
                            tensor_wait(TENSOR_LOAD_WAIT_0);
                            tensor_load(0u, 0u, 0u, 0u, 1u, (uint64_t)atile,
                                        0u, 15u, channel_plane_stride, 1u);
                            tensor_wait(TENSOR_LOAD_WAIT_0);
                            tensor_fma(0u, 3u, 15u, 15u, 0u, 0u, 0u, 0u, 1u,
                                       0u, 0u, 0u, center_first);
                            tensor_wait(TENSOR_FMA_WAIT);
                            center_first = 0;
                        }
                    }

                    tensor_store(0u, 0u, 3u, 15u, (uint64_t)ctile, 0u,
                                 channel_plane_stride);
                    tensor_wait(TENSOR_STORE_WAIT);
                    yr_tensor_clobber_fregs();

                    yr_conv_tensor_3x3_sides_and_bias(
                        batch_input, weight, bias, batch_output,
                        input_channels, H, row_width, oc0, oh, ow0);
                }
            }

            evict((const void *)oc_output_slab, channel_plane_stride * YR_TENSOR_TILE);
            WAIT_CACHEOPS;
            FENCE;
        }
    }
}

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
    (void)bias_desc;
    *hart_oc_lo = 0u;
    *hart_oc_hi = 0u;
    if (!yr_tensor_scp_ready()) {
        return 0u;
    }
    if (yr_conv_tensor_1x1_applies(node, input_desc, weight_desc, output_desc)) {
        const uint32_t output_channels = output_desc->dims[1];
        const uint32_t oc_tiles = output_channels / YR_TENSOR_TILE;
        uint32_t tile_lo, tile_hi;
        yr_conv_tensor_hart_range(oc_tiles, &tile_lo, &tile_hi);
        if (tile_hi > tile_lo) {
            yr_conv_tensor_1x1_run(
                input, weight, bias, output, input_desc->dims[0],
                input_desc->dims[1], output_channels,
                input_desc->dims[2] * input_desc->dims[3],
                tile_lo, tile_hi);
            *hart_oc_lo = tile_lo * YR_TENSOR_TILE;
            *hart_oc_hi = tile_hi * YR_TENSOR_TILE;
        }
        return 1u;
    }
    /*
     * The 3x3 path runs its whole node on one hart with no partitioning of
     * its own (see yr_conv_tensor_3x3_run), and measured 19x slower there
     * than the plain scalar yr_conv() fallback across 16 harts for the same
     * node, because its per-tap scalar "sides" accumulation has poor memory
     * locality. Splitting its output channels the way the 1x1 path now does
     * would still leave that slow scalar core in place, so on any build
     * with more than one hart this declines and lets the faster scalar path
     * handle it instead.
     */
    if (yr_hart_count() == 1u
        && yr_conv_tensor_3x3_applies(
            node, input_desc, weight_desc, output_desc)) {
        yr_conv_tensor_3x3_run(
            input, weight, bias, output, input_desc->dims[0],
            input_desc->dims[1], output_desc->dims[1], input_desc->dims[2],
            input_desc->dims[3]);
        *hart_oc_lo = 0u;
        *hart_oc_hi = output_desc->dims[1];
        return 1u;
    }
    return 0u;
}
