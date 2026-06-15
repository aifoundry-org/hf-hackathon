#pragma once

#include <ggml.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "expected.hpp"
#include "inner/helpers.hpp"

inline constexpr size_t kOptionalValueAbsent = std::numeric_limits<size_t>::max();

inline std::optional<ggml_type> OnnxTypeToGGML(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return GGML_TYPE_F32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return GGML_TYPE_F16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return GGML_TYPE_I8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return GGML_TYPE_I32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return GGML_TYPE_I64;
    default: return std::nullopt;
  }
}

struct TensorMetadata {
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
};

// Binding for a single folded-constant element that was populated with
// kUnknownShapeDimSentinel at meta-eval time: the true value is dim `axis` of
// the runtime tensor named `source_name`. The EP resolves these from runtime
// input metadata when materializing the compute graph.
struct DynamicDimBinding {
  std::string source_name;
  int64_t axis;
};

struct ConstantTensor {
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
  std::vector<uint8_t> data;
  // Per-element dynamic bindings, parallel to `data` in element order. Empty
  // means fully-static. When non-empty, size == total element count; each slot
  // is either nullopt (the data holds a concrete value) or a binding (the data
  // holds the sentinel, to be substituted at inference time).
  std::vector<std::optional<DynamicDimBinding>> dim_bindings;
};

using ConstantValueMap = std::unordered_map<std::string, ConstantTensor>;
using SupportResult = ExpectedVoid<std::string>;

// Forward declaration: full definition lives in meta_eval.hpp. Passed by
// pointer through support and compile predicates so they can consult both the
// folded constant map and the separately-propagated inferred-shape map.
struct MetaAnalysis;

TensorMetadata getTensorMetadata(Ort::ConstValueInfo value_info);
TensorMetadata getTensorMetadata(Ort::ConstValue value);
Expected<TensorMetadata, std::string> try_get_tensor_metadata(Ort::ConstValueInfo value_info);
SupportResult get_node_support(Ort::ConstNode node, const MetaAnalysis* meta);
SupportResult support_error(std::string message);
SupportResult support_ok();

// True only for plain tensor-typed value infos. Sequence/Optional/Map types
// make GetTensorTypeAndShapeInfo() invalid — any support-check path that
// forwards a value info into getTensorMetadata must gate on this first.
bool isTensorTyped(Ort::ConstValueInfo value_info);

inline bool shapeIsFullyStatic(const TensorMetadata& tensor) {
  return std::all_of(tensor.dims.begin(), tensor.dims.end(), [](int64_t dim) { return dim >= 0; });
}

inline bool rankSupportedByGGML(const TensorMetadata& tensor) {
  return tensor.dims.size() <= GGML_MAX_DIMS;
}

inline bool shapeIsFullyStatic(const std::vector<int64_t>& dims) {
  return std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim >= 0; });
}

inline void AssertShapeMatchesGGML(const std::vector<int64_t>& expected_onnx_dims,
                            const ggml_tensor* tensor,
                            const std::string& tensor_name) {
  GGONNX_NOT_NULL(tensor, "ggml tensor must not be null for shape check");
  const std::vector<int64_t> actual_onnx_dims = ToOnnxDims(tensor);
  if (ToPaddedGGMLDims(actual_onnx_dims) != ToPaddedGGMLDims(expected_onnx_dims)) {
    throw std::runtime_error("shape mismatch for tensor '" + tensor_name + "': ONNX " +
                             FormatDims(expected_onnx_dims) + " vs GGML " +
                             FormatDims(actual_onnx_dims));
  }
}

// broadcastSupportedByGGML
// @brief Returns true the requested broadcast in ONNX format is supported by GGML's
//   tensor broadcasting rules.
// @param lhs_dims The dimensions of the left-hand side tensor.
// @param rhs_dims The dimensions of the right-hand side tensor.
// @return True if the broadcast is supported, false otherwise.
inline bool broadcastSupportedByGGML(const std::vector<int64_t>& lhs_dims,
                                    const std::vector<int64_t>& rhs_dims) {
  const std::array<int64_t, GGML_MAX_DIMS> lhs_ggml_dims = ToPaddedGGMLDims(lhs_dims);
  const std::array<int64_t, GGML_MAX_DIMS> rhs_ggml_dims = ToPaddedGGMLDims(rhs_dims);

  auto can_repeat = [](const std::array<int64_t, GGML_MAX_DIMS>& src,
                       const std::array<int64_t, GGML_MAX_DIMS>& dst) {
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
      // special path to support ONNX non-specified dimensions
      if (src[i] <= 0 || dst[i] <= 0) {
        continue;
      }
      if (dst[i] % src[i] != 0) {
        return false;
      }
    }
    return true;
  };

  return can_repeat(lhs_ggml_dims, rhs_ggml_dims);
}

struct NodeDesc {
  struct NoAttrs {};
  struct GRUAttrs {
    int64_t hidden_size{};
  };
  struct LSTMAttrs {
    int64_t hidden_size{};
  };
  struct AlphaAttrs {
    float alpha{};
  };
  struct AxisAttrs {
    int64_t axis{};
  };
  struct CastAttrs {
    ggml_type target_type{GGML_TYPE_F32};
  };
  struct Conv2DAttrs {
    int spatial_rank{2};
    int s0{1}, s1{1};
    int p0{0}, p1{0};
    int d0{1}, d1{1};
    int crop0_begin{0}, crop1_begin{0};
    int out_w{0}, out_h{0};
    bool is_depthwise{false};
  };
  struct ClipAttrs {
    float min{-std::numeric_limits<float>::infinity()};
    float max{std::numeric_limits<float>::infinity()};
  };
  struct GemmAttrs {
    float alpha{1.0f};
    float beta{1.0f};
    bool trans_a{false};
    bool trans_b{false};
  };
  struct ReshapeAttrs {
    std::vector<int64_t> onnx_dims;  // fully static target shape (Reshape/Flatten)
  };
  // Squeeze/Unsqueeze: when axes are known at compile time we store them so
  // EmitSqueezeNode can derive the correct output shape from the actual runtime
  // input (handles If-branch nodes where ORT's static shape inference may be
  // wrong). When axes are only available as a runtime input we fall back to the
  // baked output dims from ORT's inference, which is always correct for the
  // main graph.
  struct SqueezeAttrs {
    std::vector<int64_t> onnx_axes;        // set when axes are a compile-time constant
    std::vector<int64_t> baked_onnx_dims;  // fallback: ORT-inferred output shape
    std::vector<int64_t> input_onnx_dims;  // compile-time input shape (preserves ONNX rank)
    bool is_unsqueeze{false};
  };
  struct Pool2DAttrs {
    ggml_op_pool op{GGML_OP_POOL_MAX};
    bool is_global{false};  // kernel derived from input spatial dims at emit time
    int k0{1}, k1{1};
    int s0{1}, s1{1};
    int p0{0}, p1{0};
    int crop0_begin{0}, crop1_begin{0};
    int out_w{0}, out_h{0};
  };
  struct PadAttrs {
    enum class Mode { Reflect, Constant };
    Mode mode{Mode::Reflect};
    // Pads in ggml axis order (axis 0 = fastest-varying). Entries may be
    // negative in constant mode (crop). Reflect mode is always non-negative
    // and bounded by src size per axis.
    std::array<int, GGML_MAX_DIMS> pad_begin{0, 0, 0, 0};
    std::array<int, GGML_MAX_DIMS> pad_end{0, 0, 0, 0};
  };
  struct ConvTransposeAttrs {
    int stride{1};
    int pad_w{0};
    int pad_h{0};
  };
  struct ExpandAttrs {
    std::vector<int64_t> onnx_dims;  // fully static target shape, broadcasted
  };
  struct InstanceNormAttrs {
    float epsilon{1e-5f};
  };
  struct NormAttrs {
    // Resolved non-negative axis: normalize over trailing (rank - axis) ONNX
    // dims. Scale (and bias, if present) must have shape matching X.shape[axis:].
    int axis{0};
    int onnx_rank{0};
    float epsilon{1e-5f};
    bool has_bias{false};
  };
  struct BatchNormAttrs {
    float epsilon{1e-5f};
    // ONNX input rank (2..4 supported). The channel axis in ONNX is always 1,
    // which in ggml's reversed layout sits at ggml axis (rank-2). Stored at
    // compile time because the emit-time tensor alone can't distinguish "rank
    // 4 with N=1, H=1" from "rank 2".
    int onnx_rank{4};
  };
  struct UpsampleAttrs {
    int scale_w{1};
    int scale_h{1};
  };
  struct TransposeAttrs {
    // Permutation expressed in GGML axis order (padded to GGML_MAX_DIMS). Axis j
    // of the output is taken from axis ggml_perm[j] of the input — the same
    // convention ggml_permute() uses.
    std::array<int, GGML_MAX_DIMS> ggml_perm{0, 1, 2, 3};
  };
  struct SliceAttrs {
    // Slice lowers to ggml_view_4d. For step==1 axes the view is rectangular;
    // for step>1 axes the view multiplies the source stride by `step`. ggml
    // forces nb[0] = type_size, so step>1 on ggml axis 0 (= last ONNX axis) is
    // rejected by the support predicate.
    //
    // Per-ggml-axis start/end/step held as the raw constant values from the
    // ONNX inputs (start/end may be negative, INT64_MAX/MIN for "to-end"). The
    // input data's per-axis length (`dim`) isn't necessarily known at compile
    // time — we resolve negative indices, clamping, and output length from
    // data->ne[axis] at emit time. `sliced` flags which axes participate; the
    // others pass through unchanged.
    std::array<int64_t, GGML_MAX_DIMS> raw_start{0, 0, 0, 0};
    std::array<int64_t, GGML_MAX_DIMS> raw_end{0, 0, 0, 0};
    std::array<int64_t, GGML_MAX_DIMS> raw_step{1, 1, 1, 1};
    std::array<bool, GGML_MAX_DIMS> sliced{false, false, false, false};
    int onnx_rank = 0;
  };
  struct SplitAttrs {
    int ggml_axis{0};
    std::vector<int64_t> lengths;  // per-output length along ggml_axis
  };
  struct DepthToSpaceAttrs {
    int blocksize{1};
    bool crd{false};  // false = DCR (ONNX default), true = CRD
  };
  struct QKVSplitAttrs {
    // Fuses Reshape([B*nw, M*M, 3C] -> [B*nw, M*M, 3, heads, head_dim]) ->
    // Transpose(perm=[2,0,3,1,4]) -> Split(axis=0, splits=[1,1,1]) ->
    // 3x Squeeze(axes=[0]). Each output lands at ggml rank-4
    // ne=[head_dim, num_tokens, num_heads, num_batch] — the canonical attention
    // Q/K/V layout — without ever materializing the rank-5 intermediates.
    int64_t num_heads{};
    int64_t head_dim{};
    int64_t num_tokens{};
    int64_t num_batch{};
  };
  struct GenericShuffleAttrs {
    // General fused Reshape(4D->XD)->Transpose->Reshape(XD->4D) triple where
    // X > GGML_MAX_DIMS. The permutation is coalesced into ≤4 axis groups by
    // merging consecutive ONNX axes that remain adjacent post-transpose (i.e.
    // inv_perm[i+1] == inv_perm[i]+1). Each group's axes are flattened into a
    // single dimension and the inter-group permutation is expressed as a rank-4
    // GGML permute, avoiding the rank-X intermediate entirely.
    //
    // grouped_ggml_dims: product-of-group sizes in GGML axis order (reversed
    //   from ONNX, padded to GGML_MAX_DIMS with 1s).
    // ggml_perm: the inter-group permutation in GGML axis order.
    // output_onnx_dims: dims of the trailing Reshape's output (rank 1..4).
    std::array<int64_t, GGML_MAX_DIMS> grouped_ggml_dims{1, 1, 1, 1};
    std::array<int, GGML_MAX_DIMS> ggml_perm{0, 1, 2, 3};
    std::vector<int64_t> output_onnx_dims;
  };
  struct ReduceAttrs {
    // Number of axes being reduced.
    int trailing_count{0};
    bool keepdims{true};
    // GGML permutation to move the reduction axes to positions [0, k-1].
    // Identity ([0,1,2,3]) when they are already the leading GGML axes
    // (= trailing ONNX axes). inv_perm is the inverse permutation for
    // the keepdims restore step.
    std::array<int, GGML_MAX_DIMS> perm{0, 1, 2, 3};
    std::array<int, GGML_MAX_DIMS> inv_perm{0, 1, 2, 3};
  };
  struct MatMulAttrs {
    // When true, tag the ggml_mul_mat output with GGML_PREC_F32 so backends
    // that default to fp16 accumulation (e.g. Vulkan) use fp32 instead.
    // Driven by the `ep.ggonnx.matmul_precision` provider option.
    bool force_f32{false};
  };
  struct RangeAttrs {
    // ONNX Range(start, limit, delta). All three inputs must be compile-time
    // scalar constants so we can lower to ggml_arange, which takes floats.
    float start{0.0f};
    float limit{0.0f};
    float delta{1.0f};
    // ggml_arange always produces F32; for integer-typed Range outputs we emit
    // a trailing ggml_cast to this type.
    ggml_type target_type{GGML_TYPE_F32};
  };

  using Attrs = std::variant<NoAttrs,
                             GRUAttrs,
                             LSTMAttrs,
                             AlphaAttrs,
                             AxisAttrs,
                             CastAttrs,
                             Conv2DAttrs,
                             ClipAttrs,
                             GemmAttrs,
                             ReshapeAttrs,
                             SqueezeAttrs,
                             Pool2DAttrs,
                             PadAttrs,
                             InstanceNormAttrs,
                             NormAttrs,
                             BatchNormAttrs,
                             UpsampleAttrs,
                             TransposeAttrs,
                             SliceAttrs,
                             SplitAttrs,
                             ReduceAttrs,
                             ConvTransposeAttrs,
                             ExpandAttrs,
                             DepthToSpaceAttrs,
                             GenericShuffleAttrs,
                             QKVSplitAttrs,
                             MatMulAttrs,
                             RangeAttrs>;

  std::string op_type;
  std::string domain;
  std::string name;
  std::vector<size_t> inputs;
  std::vector<size_t> outputs;
  Attrs attrs{};
};

using EmitOutputs = std::vector<ggml_tensor*>;
using EmitResult = std::optional<EmitOutputs>;
using EmitNodeFn = EmitResult (*)(ggml_context* ctx,
                                  const NodeDesc& node,
                                  const std::vector<ggml_tensor*>& values);
using CompileAttrsFn = void (*)(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta);

// Layout hint for constant-initializer inputs materialized at compile time.
// AS_IS: ggml ne = reversed ONNX dims, data byte-identical.
// MATMUL_WEIGHT_TRANSPOSED: swap the last two ONNX dims before storing. For a
//   [K, N] ONNX weight this yields ggml ne=[K, N] with contiguous data, which
//   is what ggml_mul_mat(weight, activation) wants — so the runtime transpose
//   + cont vanishes. Extends naturally to batched [..., K, N].
enum class ConstantLayout {
  AS_IS,
  MATMUL_WEIGHT_TRANSPOSED,
};

// Per-op hook: given a compiled node and an input index that is a constant
// initializer, return the desired layout. Nullable on OpDefinition; nullptr
// means "always AS_IS".
using ConstantLayoutFn = ConstantLayout (*)(const NodeDesc& node, size_t input_idx);

struct OpDefinition {
  SupportResult (*support)(Ort::ConstNode node, const MetaAnalysis* meta);
  CompileAttrsFn compile_attrs;
  EmitNodeFn emit;
  ConstantLayoutFn constant_layout;
};

const OpDefinition* FindOpDefinition(std::string_view domain, std::string_view op_type);

// Graph-level fusion plan. Populated in AnalyzeCompileTimeConstants.
struct FusionPlan {
  // Anchor node key (the Transpose) -> fused attrs for any
  // Reshape(4D->XD)->Transpose->Reshape(XD->4D) triple where X > GGML_MAX_DIMS
  // and the permutation coalesces to ≤4 axis groups.
  std::unordered_map<std::string, NodeDesc::GenericShuffleAttrs> generic_shuffle_anchors;
  // Anchor node key (the Split) -> fused QKV split attrs. Produces three
  // rank-4 outputs from one rank-3 input, bypassing the rank-5 intermediates.
  std::unordered_map<std::string, NodeDesc::QKVSplitAttrs> qkv_split_anchors;
  // Anchor node key (the rank-5 Add) -> rank-reduced mask ONNX dims. Fuses
  // Reshape(rank-4 -> rank-5) -> Add(rank-5, constant mask) -> Reshape(back).
  // The mask is materialized at its rank-reduced shape — its bytes are
  // unchanged because only size-1 ONNX axes get dropped.
  std::unordered_map<std::string, std::vector<int64_t>> window_mask_add_anchors;
  // Keyed by anchor node key; the value names to wire as synthetic
  // input(s)/output(s). Both vectors have N entries depending on the fusion
  // — __WindowShuffle: 1 input, 1 output; __QKVSplit: 1 input, 3 outputs;
  // __WindowMaskAdd: 2 inputs (X, mask), 1 output.
  struct AnchorIO {
    std::vector<std::string> input_values;
    std::vector<std::string> output_values;
    std::string anchor_node_name;  // for the synthetic NodeDesc.name
  };
  std::unordered_map<std::string, AnchorIO> anchor_io;
  // Node keys consumed by a fusion (skipped entirely during partition compile —
  // their high-rank outputs never become ggml values).
  std::unordered_set<std::string> consumed_nodes;
  // Value-name -> rank-reduced ONNX dims. Used when a constant initializer
  // that ORT exposes at rank > 4 is actually consumed by a fusion that only
  // needs the rank-reduced view (e.g. __WindowMaskAdd). The byte layout is
  // unchanged — only size-1 ONNX axes are dropped — so the existing constant
  // materializer can copy straight through.
  std::unordered_map<std::string, std::vector<int64_t>> constant_override_dims;
  // Keyed by Conv node key. A zero-constant Pad whose sole consumer is this
  // Conv has been absorbed: its H/W padding is folded into the Conv's p0/p1,
  // the Pad is in consumed_nodes, and data_input_name is the Pad's data input
  // (bypassing the Pad's output so no padded tensor is materialised).
  struct AbsorbedPad {
    int p0;  // extra W padding (ggml axis 0)
    int p1;  // extra H padding (ggml axis 1)
    std::string data_input_name;
  };
  std::unordered_map<std::string, AbsorbedPad> absorbed_pads;
  // Keyed by Conv node key. The sole consumer BatchNormalization has been
  // folded into the Conv's compile-time weight/bias constants. The Conv should
  // wire its output directly to output_value_name and use the replacement
  // weight/bias constants by name. The BN node is consumed_nodes.
  struct FoldedBatchNorm {
    std::string output_value_name;
    std::string weight_input_name;
    std::string bias_input_name;
  };
  std::unordered_map<std::string, FoldedBatchNorm> folded_batchnorms;
};
