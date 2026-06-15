#include "meta_eval.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"

namespace {

using ggonnx::ort_internal::GetOrtApi;
using ggonnx::ort_internal::THROW_ON_ERROR;

bool IsNodeCompilable(Ort::ConstNode node, const ConstantValueMap& constants);

// ===========================================================================
//  GraphIndex — central authority for graph-structure queries.
//
//  Built once per AnalyzeCompileTimeConstants call; fusion detectors, the
//  main fold loop, and EvalShapeTensorGather all consult it. Holds:
//    - producer edge:           ProducerOf(name) -> node producing it
//    - consumer edge:           Consumers(name) -> node(s) consuming it
//    - consumer op types:       ConsumerOps(name) -> ["Reshape", "Gather"]
//    - graph outputs:           IsGraphOutput(name)
//    - topological position:    TopoIndex(node_key)
//
//  All queries are O(1). Graph outputs are tracked separately from node
//  consumers: callers that want "node uses + graph-output use" compose
//  ConsumerCount(name) + IsGraphOutput(name).
// ===========================================================================
struct GraphIndex {
  std::vector<Ort::ConstNode> nodes;  // topological order from GetNodes()
  std::unordered_map<std::string, Ort::ConstNode> producer_by_output;
  std::unordered_map<std::string, std::vector<Ort::ConstNode>> consumers_by_name;
  std::unordered_map<std::string, std::vector<std::string>> consumer_ops_by_name;
  std::unordered_set<std::string> graph_output_names;
  std::unordered_map<std::string, size_t> topo_index_by_key;

  explicit GraphIndex(const Ort::ConstGraph& graph) {
    const auto ns = graph.GetNodes();
    nodes.assign(ns.begin(), ns.end());
    for (size_t i = 0; i < nodes.size(); ++i) {
      topo_index_by_key.emplace(NodeKey(nodes[i]), i);
      const std::string op_type = std::string(nodes[i].GetOperatorType());
      for (Ort::ConstValueInfo out : nodes[i].GetOutputs()) {
        if (out != nullptr) producer_by_output.emplace(out.GetName(), nodes[i]);
      }
      auto record_use = [&](Ort::ConstValueInfo in) {
        if (in == nullptr) return;
        const std::string name = in.GetName();
        consumers_by_name[name].push_back(nodes[i]);
        consumer_ops_by_name[name].push_back(op_type);
      };
      for (Ort::ConstValueInfo in : nodes[i].GetInputs()) record_use(in);
      for (Ort::ConstValueInfo in : nodes[i].GetImplicitInputs()) record_use(in);
    }
    for (Ort::ConstValueInfo out : graph.GetOutputs()) {
      if (out != nullptr) graph_output_names.emplace(out.GetName());
    }
  }

  Ort::ConstNode ProducerOf(const std::string& name) const {
    const auto it = producer_by_output.find(name);
    return it == producer_by_output.end() ? Ort::ConstNode{nullptr} : it->second;
  }

  size_t ConsumerCount(const std::string& name) const {
    const auto it = consumers_by_name.find(name);
    return it == consumers_by_name.end() ? 0 : it->second.size();
  }

  const std::vector<Ort::ConstNode>& Consumers(const std::string& name) const {
    static const std::vector<Ort::ConstNode> empty;
    const auto it = consumers_by_name.find(name);
    return it == consumers_by_name.end() ? empty : it->second;
  }

  const std::vector<std::string>& ConsumerOps(const std::string& name) const {
    static const std::vector<std::string> empty;
    const auto it = consumer_ops_by_name.find(name);
    return it == consumer_ops_by_name.end() ? empty : it->second;
  }

  bool IsGraphOutput(const std::string& name) const {
    return graph_output_names.count(name) > 0;
  }

  std::optional<size_t> TopoIndex(const std::string& node_key) const {
    const auto it = topo_index_by_key.find(node_key);
    if (it == topo_index_by_key.end()) return std::nullopt;
    return it->second;
  }

  // Unique consumer of `name`, or null if count != 1.
  Ort::ConstNode OnlyConsumer(const std::string& name) const {
    const auto& cs = Consumers(name);
    return cs.size() == 1 ? cs.front() : Ort::ConstNode{nullptr};
  }

  // Like OnlyConsumer but also verifies op type.
  Ort::ConstNode OnlyConsumerOf(const std::string& name, std::string_view op_type) const {
    Ort::ConstNode c = OnlyConsumer(name);
    if (c == nullptr || std::string_view(c.GetOperatorType()) != op_type) {
      return Ort::ConstNode{nullptr};
    }
    return c;
  }
};

// Fusion topological-span safety check: every node between the min- and
// max-topo-index members must itself be a member, already folded, already
// claimed by an earlier fusion, or compilable. Otherwise an unsupported
// node would split the ORT partition and break ensure_value.
bool FusionSpanSafe(const GraphIndex& gi,
                    std::initializer_list<std::string> member_keys,
                    const ConstantValueMap& constants,
                    const std::unordered_set<std::string>& folded_nodes,
                    const FusionPlan& plan) {
  const std::unordered_set<std::string> members(member_keys.begin(), member_keys.end());
  size_t lo = std::numeric_limits<size_t>::max(), hi = 0;
  for (const std::string& k : member_keys) {
    const auto idx = gi.TopoIndex(k);
    if (!idx) return false;
    lo = std::min(lo, *idx);
    hi = std::max(hi, *idx);
  }
  if (lo > hi) return false;
  for (size_t i = lo; i <= hi; ++i) {
    const std::string key = NodeKey(gi.nodes[i]);
    if (members.count(key)) continue;
    if (folded_nodes.count(key)) continue;
    if (plan.consumed_nodes.count(key)) continue;
    if (plan.generic_shuffle_anchors.count(key)) continue;
    if (plan.qkv_split_anchors.count(key)) continue;
    if (plan.window_mask_add_anchors.count(key)) continue;
    if (IsNodeCompilable(gi.nodes[i], constants)) continue;
    return false;
  }
  return true;
}

constexpr size_t kMaxFoldedElements = 256;
// Sentinel used inside compile-time shape tensors for dimensions that ORT
// knows exist but cannot statically resolve. It must survive int64<->int32
// shape casts without colliding with ONNX's meaningful shape placeholders 0/-1.
// Keep the local definition for internal use; the header also exposes it.
static_assert(static_cast<int64_t>(std::numeric_limits<int32_t>::min()) == kUnknownShapeDimSentinel,
              "sentinel mismatch between header and meta_eval.cpp");
// Raw initializers live on disk/in ORT memory already; loading them into the
// constants map is a bounded duplication, not an unbounded compute. The
// tighter kMaxFoldedElements still guards derived folds (Tile/Expand/etc.)
// so they can't synthesize huge tensors.
constexpr size_t kMaxInitializerElements = 1 << 20;  // 1M elements (~4 MB f32)

size_t ByteWidth(ONNXTensorElementDataType element_type) {
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return sizeof(uint8_t);
  if (const auto g = OnnxTypeToGGML(element_type)) return ggml_type_size(*g);
  throw std::runtime_error("unsupported compile-time constant dtype");
}

// Invokes `fn(T{})` with the C++ arithmetic type corresponding to `et`.
// Returns fn's result for supported dtypes; returns {} (default-constructed)
// for unsupported ones so callers can keep returning std::nullopt uniformly.
// Add a new dtype once here and every typed evaluator (Binary/Concat/Slice/
// Tile/Equal/...) picks it up automatically.
template <typename F>
auto DispatchByElementType(ONNXTensorElementDataType et, F&& fn)
    -> decltype(fn(float{})) {
  switch (et) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return fn(float{});
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:  return fn(int8_t{});
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return fn(int32_t{});
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return fn(int64_t{});
    default: return {};
  }
}

size_t ElementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "compile-time constant has invalid dynamic shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

bool IsFoldableDType(ONNXTensorElementDataType element_type) {
  return OnnxTypeToGGML(element_type).has_value() ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool FitsFoldLimit(const std::vector<int64_t>& dims) {
  return ElementCount(dims) <= kMaxFoldedElements;
}

bool FitsInitializerLimit(const std::vector<int64_t>& dims) {
  return ElementCount(dims) <= kMaxInitializerElements;
}

template <typename T>
std::vector<T> ReadTypedData(const ConstantTensor& tensor) {
  const size_t count = ElementCount(tensor.dims);
  GGONNX_ASSERT(tensor.data.size() == count * sizeof(T),
                "compile-time constant byte size mismatch");
  std::vector<T> out(count);
  if (!out.empty()) {
    std::memcpy(out.data(), tensor.data.data(), tensor.data.size());
  }
  return out;
}

// Build a per-element bindings vector for a shape tensor produced from
// `source_name`, where element `i` corresponds to `source_dims[axis_for[i]]`.
// Returns {} if no dim at any selected axis is dynamic (fully-static result).
std::vector<std::optional<DynamicDimBinding>> MakeShapeBindings(
    const std::vector<int64_t>& source_dims,
    const std::string& source_name,
    const std::vector<size_t>& axis_for) {
  bool any_dynamic = false;
  for (size_t axis : axis_for) {
    if (axis < source_dims.size() && source_dims[axis] < 0) {
      any_dynamic = true;
      break;
    }
  }
  if (!any_dynamic) return {};
  std::vector<std::optional<DynamicDimBinding>> out(axis_for.size());
  for (size_t pos = 0; pos < axis_for.size(); ++pos) {
    const size_t axis = axis_for[pos];
    if (axis < source_dims.size() && source_dims[axis] < 0) {
      out[pos] = DynamicDimBinding{source_name, static_cast<int64_t>(axis)};
    }
  }
  return out;
}

// Normalize a possibly-empty dim_bindings vector to length `n` (all nullopt).
void NormalizeBindings(std::vector<std::optional<DynamicDimBinding>>& bindings, size_t n) {
  if (bindings.empty()) {
    bindings.assign(n, std::nullopt);
  } else {
    GGONNX_ASSERT(bindings.size() == n, "dim_bindings size mismatch");
  }
}

bool ContainsUnknownShapeDim(const ConstantTensor& tensor) {
  if (tensor.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const auto values = ReadTypedData<int64_t>(tensor);
    return std::find(values.begin(), values.end(), kUnknownShapeDimSentinel) != values.end();
  }
  if (tensor.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const auto values = ReadTypedData<int32_t>(tensor);
    return std::find(values.begin(), values.end(),
                     static_cast<int32_t>(kUnknownShapeDimSentinel)) != values.end();
  }
  return false;
}


template <typename T>
ConstantTensor MakeConstant(std::vector<int64_t> dims, std::vector<T> values,
                            ONNXTensorElementDataType element_type) {
  GGONNX_ASSERT(ElementCount(dims) == values.size(),
                "compile-time constant element count mismatch");
  ConstantTensor tensor;
  tensor.element_type = element_type;
  tensor.dims = std::move(dims);
  tensor.data.resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
  }
  return tensor;
}

ConstantTensor LoadInitializer(Ort::ConstValue value) {
  const TensorMetadata meta = getTensorMetadata(value);
  GGONNX_ASSERT(IsFoldableDType(meta.element_type),
                "unsupported initializer dtype for compile-time folding");
  GGONNX_ASSERT(FitsInitializerLimit(meta.dims),
                "initializer too large for compile-time folding");
  const size_t bytes = ElementCount(meta.dims) * ByteWidth(meta.element_type);
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null tensor data for initializer");

  ConstantTensor tensor;
  tensor.element_type = meta.element_type;
  tensor.dims = meta.dims;
  tensor.data.resize(bytes);
  if (bytes > 0) {
    std::memcpy(tensor.data.data(), src, bytes);
  }
  return tensor;
}

std::optional<ConstantTensor> LookupConstant(const ConstantValueMap& constants, Ort::ConstValueInfo value) {
  if (value == nullptr) return std::nullopt;
  auto it = constants.find(value.GetName());
  if (it == constants.end()) return std::nullopt;
  return it->second;
}

std::optional<std::vector<int64_t>> ReadConstantInt64Input(const ConstantValueMap& constants,
                                                           Ort::ConstNode node,
                                                           size_t input_idx) {
  const auto inputs = node.GetInputs();
  if (input_idx >= inputs.size() || inputs[input_idx] == nullptr) return std::nullopt;
  const auto tensor_opt = LookupConstant(constants, inputs[input_idx]);
  if (!tensor_opt || tensor_opt->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return std::nullopt;
  }
  return ReadTypedData<int64_t>(*tensor_opt);
}

std::optional<bool> ReadConstantBool(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) return std::nullopt;
  Ort::ConstValue value{nullptr};
  Ort::ThrowOnError(value_info.GetInitializer(value));
  if (value == nullptr) return std::nullopt;
  const TensorMetadata meta = getTensorMetadata(value);
  if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return std::nullopt;
  GGONNX_ASSERT(ElementCount(meta.dims) == 1, "bool constant must be scalar");
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null bool tensor data");
  return *static_cast<const bool*>(src);
}

template <typename T>
std::vector<int64_t> MakeStrides(const std::vector<T>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * static_cast<int64_t>(dims[static_cast<size_t>(i + 1)]);
  }
  return strides;
}

template <typename T>
ConstantTensor CastConstant(const ConstantTensor& input, ONNXTensorElementDataType to_type) {
  const size_t count = ElementCount(input.dims);
  if (input.element_type == to_type) return input;

  // Integer-to-integer casts preserve per-element positions and the sentinel
  // value INT32_MIN (fits both int32 and int64 losslessly); we propagate
  // bindings. Casts involving floats lose the sentinel meaning so we don't
  // propagate (and in practice those shouldn't occur for shape tensors).

  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const auto values = ReadTypedData<int64_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int32_t>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const auto values = ReadTypedData<float>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int32_t>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    auto result = MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      result.dim_bindings = input.dim_bindings;
    }
    return result;
  }
  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    std::vector<int64_t> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto values = ReadTypedData<int32_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int64_t>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const auto values = ReadTypedData<float>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int64_t>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    auto result = MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      result.dim_bindings = input.dim_bindings;
    }
    return result;
  }
  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto values = ReadTypedData<int32_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const auto values = ReadTypedData<int64_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    return MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
  }
  throw std::runtime_error("unsupported cast destination dtype");
}

template <typename T>
ConstantTensor EvalSliceTyped(const ConstantTensor& data,
                              const std::vector<int64_t>& starts,
                              const std::vector<int64_t>& ends,
                              std::vector<int64_t> axes,
                              const std::vector<int64_t>& steps,
                              ONNXTensorElementDataType element_type) {
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  GGONNX_ASSERT(rank >= 0, "invalid slice rank");
  if (axes.empty()) {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) axes[i] = static_cast<int64_t>(i);
  }

  std::vector<int64_t> begin = std::vector<int64_t>(data.dims.size(), 0);
  std::vector<int64_t> stride = std::vector<int64_t>(data.dims.size(), 1);
  std::vector<int64_t> out_dims = data.dims;

  auto clamp = [](int64_t value, int64_t lo, int64_t hi) {
    return std::max(lo, std::min(hi, value));
  };

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0) axis += rank;
    GGONNX_ASSERT(axis >= 0 && axis < rank, "slice axis out of range");
    const int64_t dim = data.dims[static_cast<size_t>(axis)];
    const int64_t step = steps.empty() ? 1 : steps[i];
    GGONNX_ASSERT(step != 0, "slice step must not be zero");

    int64_t start = starts[i];
    int64_t end = ends[i];
    if (step > 0) {
      if (start < 0) start += dim;
      if (end < 0) end += dim;
      start = clamp(start, 0, dim);
      end = clamp(end, 0, dim);
      out_dims[static_cast<size_t>(axis)] = end > start ? (end - start + step - 1) / step : 0;
    } else {
      if (start < 0) start += dim;
      if (end < 0) end += dim;
      start = clamp(start, -1, dim - 1);
      end = clamp(end, -1, dim - 1);
      out_dims[static_cast<size_t>(axis)] = start > end ? (start - end + (-step) - 1) / (-step) : 0;
    }
    begin[static_cast<size_t>(axis)] = start;
    stride[static_cast<size_t>(axis)] = step;
  }

  GGONNX_ASSERT(FitsFoldLimit(out_dims), "slice output too large for compile-time folding");
  const auto input = ReadTypedData<T>(data);
  const size_t out_count = ElementCount(out_dims);
  std::vector<T> out(out_count);
  const auto in_strides = MakeStrides(data.dims);
  const auto out_strides = MakeStrides(out_dims);
  std::vector<int64_t> coord(out_dims.size(), 0);

  for (size_t linear = 0; linear < out_count; ++linear) {
    int64_t input_linear = 0;
    for (size_t axis = 0; axis < out_dims.size(); ++axis) {
      const int64_t idx = begin[axis] + coord[axis] * stride[axis];
      input_linear += idx * in_strides[axis];
    }
    out[linear] = input[static_cast<size_t>(input_linear)];
    for (int axis = static_cast<int>(coord.size()) - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return MakeConstant(std::move(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalShape(const ConstantValueMap& /*constants*/, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const TensorMetadata meta = getTensorMetadata(inputs[0]);
  // ORT nested subgraphs may omit shape metadata entirely; GetShape() then
  // comes back as [] even for non-scalar tensors. Treat [] as "unknown" here
  // instead of folding Shape(x) to an empty constant.
  if (meta.dims.empty()) return std::nullopt;
  std::vector<int64_t> values = meta.dims;
  for (int64_t& dim : values) {
    if (dim < 0) dim = kUnknownShapeDimSentinel;
  }
  auto result = MakeConstant<int64_t>({static_cast<int64_t>(values.size())}, std::move(values),
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  // Bind each position p → axis p of this Shape op's input, so the EP can
  // substitute the real dim at inference time.
  std::vector<size_t> axis_for(meta.dims.size());
  for (size_t i = 0; i < axis_for.size(); ++i) axis_for[i] = i;
  result.dim_bindings = MakeShapeBindings(meta.dims, std::string(inputs[0].GetName()), axis_for);
  return result;
}

std::optional<ConstantTensor> EvalCast(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("to", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t to = 0;
  Ort::ThrowOnError(attr.GetValue(to));
  return CastConstant<int64_t>(*input, static_cast<ONNXTensorElementDataType>(to));
}

std::optional<ConstantTensor> EvalIdentity(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  return LookupConstant(constants, inputs[0]);
}

std::optional<ConstantTensor> EvalConcat(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty()) return std::nullopt;
  std::vector<ConstantTensor> tensors;
  tensors.reserve(inputs.size());
  for (auto input : inputs) {
    if (input == nullptr) return std::nullopt;
    const auto tensor = LookupConstant(constants, input);
    if (!tensor) return std::nullopt;
    tensors.push_back(*tensor);
  }
  const ONNXTensorElementDataType element_type = tensors[0].element_type;
  for (const auto& tensor : tensors) {
    if (tensor.element_type != element_type) return std::nullopt;
  }
  // If any input carries dynamic-dim bindings, leave the Concat unfolded so
  // the EP can run it as a real GGML node with the sentinel-input values
  // resolved from runtime input shapes. Folding-with-bindings is only useful
  // when the result will itself flow into another op that knows how to consume
  // bindings — graph outputs memcpy raw bytes and would leak the sentinel.
  for (const auto& tensor : tensors) {
    if (!tensor.dim_bindings.empty()) return std::nullopt;
  }
  bool any_bindings = false;

  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("axis", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t axis = 0;
  Ort::ThrowOnError(attr.GetValue(axis));

  const std::vector<int64_t>& base_shape = tensors[0].dims;
  if (axis < 0) axis += static_cast<int64_t>(base_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(base_shape.size())) return std::nullopt;
  std::vector<int64_t> out_dims = base_shape;
  out_dims[static_cast<size_t>(axis)] = 0;
  for (const auto& tensor : tensors) {
    if (tensor.dims.size() != base_shape.size()) return std::nullopt;
    for (size_t i = 0; i < tensor.dims.size(); ++i) {
      if (i == static_cast<size_t>(axis)) continue;
      if (tensor.dims[i] != base_shape[i]) return std::nullopt;
    }
    out_dims[static_cast<size_t>(axis)] += tensor.dims[static_cast<size_t>(axis)];
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  // Bindings can be concatenated flat only when the concat axis is the
  // outermost one (axis==0) — then each input contributes a contiguous block
  // of elements to the output in input order. For inner-axis concats the
  // elements interleave, which we don't currently reconstruct.
  if (any_bindings && axis != 0) return std::nullopt;
  auto concat_bindings = [&]() {
    std::vector<std::optional<DynamicDimBinding>> out;
    if (!any_bindings) return out;
    for (const auto& tensor : tensors) {
      const size_t n = ElementCount(tensor.dims);
      if (tensor.dim_bindings.empty()) {
        out.insert(out.end(), n, std::nullopt);
      } else {
        GGONNX_ASSERT(tensor.dim_bindings.size() == n,
                      "dim_bindings length mismatch in Concat input");
        out.insert(out.end(), tensor.dim_bindings.begin(), tensor.dim_bindings.end());
      }
    }
    return out;
  };

  return DispatchByElementType(element_type, [&](auto tag) -> std::optional<ConstantTensor> {
    using T = decltype(tag);
    std::vector<T> out;
    for (const auto& tensor : tensors) {
      auto values = ReadTypedData<T>(tensor);
      out.insert(out.end(), values.begin(), values.end());
    }
    auto result = MakeConstant(std::move(out_dims), std::move(out), element_type);
    // bindings only exist on int32/int64 shape tensors (their values can be
    // INT32_MIN). For other dtypes concat_bindings() yields empty.
    result.dim_bindings = concat_bindings();
    return result;
  });
}

std::optional<ConstantTensor> EvalUnsqueeze(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;

  std::optional<std::vector<int64_t>> axes;
  if (inputs.size() >= 2) axes = ReadConstantInt64Input(constants, node, 1);
  if (!axes.has_value()) {
    Ort::ConstOpAttr attr{nullptr};
    const Ort::Status status = node.GetAttributeByName("axes", attr);
    if (!status.IsOK() || attr == nullptr) return std::nullopt;
    axes.emplace();
    Ort::ThrowOnError(attr.GetValueArray(*axes));
  }

  std::vector<int64_t> out_dims = input->dims;
  std::sort(axes->begin(), axes->end());
  const int64_t out_rank = static_cast<int64_t>(out_dims.size() + axes->size());
  for (int64_t axis : *axes) {
    if (axis < 0) axis += out_rank;
    if (axis < 0 || axis > static_cast<int64_t>(out_dims.size())) return std::nullopt;
    out_dims.insert(out_dims.begin() + axis, 1);
  }
  ConstantTensor out = *input;
  out.dims = std::move(out_dims);
  return out;
}

std::optional<ConstantTensor> EvalSqueeze(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;

  std::vector<int64_t> out_dims = input->dims;
  std::optional<std::vector<int64_t>> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr) axes = ReadConstantInt64Input(constants, node, 1);
  if (!axes.has_value()) {
    Ort::ConstOpAttr attr{nullptr};
    const Ort::Status status = node.GetAttributeByName("axes", attr);
    if (status.IsOK() && attr != nullptr) {
      axes.emplace();
      Ort::ThrowOnError(attr.GetValueArray(*axes));
    }
  }

  if (axes.has_value()) {
    std::vector<int64_t> normalized = *axes;
    std::sort(normalized.rbegin(), normalized.rend());
    for (int64_t axis : normalized) {
      if (axis < 0) axis += static_cast<int64_t>(out_dims.size());
      if (axis < 0 || axis >= static_cast<int64_t>(out_dims.size())) return std::nullopt;
      if (out_dims[static_cast<size_t>(axis)] != 1) return std::nullopt;
      out_dims.erase(out_dims.begin() + axis);
    }
  } else {
    out_dims.erase(std::remove(out_dims.begin(), out_dims.end(), 1), out_dims.end());
  }

  ConstantTensor out = *input;
  out.dims = std::move(out_dims);
  return out;
}

template <typename T>
std::optional<ConstantTensor> EvalBinaryTyped(const ConstantTensor& lhs,
                                              const ConstantTensor& rhs,
                                              const std::string& op_type,
                                              ONNXTensorElementDataType element_type) {
  if (lhs.element_type != element_type || rhs.element_type != element_type) return std::nullopt;
  const size_t lhs_count = ElementCount(lhs.dims);
  const size_t rhs_count = ElementCount(rhs.dims);
  const bool lhs_scalar = lhs_count == 1;
  const bool rhs_scalar = rhs_count == 1;
  if (!lhs_scalar && !rhs_scalar && lhs.dims != rhs.dims) return std::nullopt;
  const std::vector<int64_t> out_dims = lhs_scalar ? rhs.dims : lhs.dims;
  const size_t out_count = std::max(lhs_count, rhs_count);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  const auto lhs_values = ReadTypedData<T>(lhs);
  const auto rhs_values = ReadTypedData<T>(rhs);
  std::vector<T> out(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    const T a = lhs_values[lhs_scalar ? 0 : i];
    const T b = rhs_values[rhs_scalar ? 0 : i];
    if (op_type == "Add") out[i] = a + b;
    else if (op_type == "Sub") out[i] = a - b;
    else if (op_type == "Mul") out[i] = a * b;
    else if (op_type == "Div") out[i] = a / b;
    else return std::nullopt;
  }
  return MakeConstant(std::vector<int64_t>(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalBinary(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto lhs = LookupConstant(constants, inputs[0]);
  const auto rhs = LookupConstant(constants, inputs[1]);
  if (!lhs || !rhs || lhs->element_type != rhs->element_type) return std::nullopt;
  if (ContainsUnknownShapeDim(*lhs) || ContainsUnknownShapeDim(*rhs)) return std::nullopt;
  const std::string op_type = node.GetOperatorType();
  return DispatchByElementType(lhs->element_type, [&](auto tag) -> std::optional<ConstantTensor> {
    using T = decltype(tag);
    return EvalBinaryTyped<T>(*lhs, *rhs, op_type, lhs->element_type);
  });
}

std::optional<ConstantTensor> EvalReshape(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  const auto shape = LookupConstant(constants, inputs[1]);
  if (!data || !shape || shape->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return std::nullopt;
  const auto target_shape_values = ReadTypedData<int64_t>(*shape);
  std::vector<int64_t> out_dims = target_shape_values;
  int64_t infer_axis = -1;
  int64_t known_product = 1;
  for (size_t i = 0; i < out_dims.size(); ++i) {
    if (out_dims[i] == 0) out_dims[i] = data->dims[i];
    if (out_dims[i] == -1) {
      if (infer_axis >= 0) return std::nullopt;
      infer_axis = static_cast<int64_t>(i);
      continue;
    }
    if (out_dims[i] < -1 || out_dims[i] == kUnknownShapeDimSentinel) return std::nullopt;
    known_product *= out_dims[i];
  }
  const int64_t total = static_cast<int64_t>(ElementCount(data->dims));
  if (infer_axis >= 0) {
    GGONNX_ASSERT(known_product != 0, "reshape known product must be non-zero");
    out_dims[static_cast<size_t>(infer_axis)] = total / known_product;
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  if (ElementCount(out_dims) != ElementCount(data->dims)) return std::nullopt;
  ConstantTensor out = *data;
  out.dims = std::move(out_dims);
  return out;
}

std::optional<ConstantTensor> EvalSlice(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() < 3 || inputs[0] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const auto starts = ReadConstantInt64Input(constants, node, 1);
  const auto ends = ReadConstantInt64Input(constants, node, 2);
  if (!starts || !ends || starts->size() != ends->size()) return std::nullopt;
  std::vector<int64_t> axes;
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes_opt = ReadConstantInt64Input(constants, node, 3);
    if (!axes_opt) return std::nullopt;
    axes = *axes_opt;
  }
  std::vector<int64_t> steps;
  if (inputs.size() >= 5 && inputs[4] != nullptr) {
    const auto steps_opt = ReadConstantInt64Input(constants, node, 4);
    if (!steps_opt) return std::nullopt;
    steps = *steps_opt;
  }
  return DispatchByElementType(data->element_type, [&](auto tag) -> std::optional<ConstantTensor> {
    using T = decltype(tag);
    return EvalSliceTyped<T>(*data, *starts, *ends, std::move(axes), steps, data->element_type);
  });
}

template <typename T>
ConstantTensor EvalTileTyped(const ConstantTensor& data,
                             const std::vector<int64_t>& repeats,
                             ONNXTensorElementDataType element_type) {
  GGONNX_ASSERT(repeats.size() == data.dims.size(),
                "tile repeats rank must match input rank");
  std::vector<int64_t> out_dims(data.dims.size(), 0);
  for (size_t i = 0; i < repeats.size(); ++i) {
    GGONNX_ASSERT(repeats[i] >= 0, "tile repeats must be non-negative");
    out_dims[i] = data.dims[i] * repeats[i];
  }
  GGONNX_ASSERT(FitsFoldLimit(out_dims), "tile output too large for compile-time folding");

  const auto input = ReadTypedData<T>(data);
  const size_t out_count = ElementCount(out_dims);
  std::vector<T> out(out_count);
  const auto in_strides = MakeStrides(data.dims);
  std::vector<int64_t> coord(out_dims.size(), 0);

  for (size_t linear = 0; linear < out_count; ++linear) {
    int64_t input_linear = 0;
    for (size_t axis = 0; axis < out_dims.size(); ++axis) {
      const int64_t src_idx = data.dims[axis] == 0 ? 0 : (coord[axis] % data.dims[axis]);
      input_linear += src_idx * in_strides[axis];
    }
    out[linear] = input[static_cast<size_t>(input_linear)];
    for (int axis = static_cast<int>(coord.size()) - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return MakeConstant(std::move(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalTile(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const auto repeats = ReadConstantInt64Input(constants, node, 1);
  if (!repeats) return std::nullopt;
  return DispatchByElementType(data->element_type, [&](auto tag) -> std::optional<ConstantTensor> {
    using T = decltype(tag);
    return EvalTileTyped<T>(*data, *repeats, data->element_type);
  });
}

std::optional<std::vector<ConstantTensor>> EvalArangeLoop(const ConstantValueMap& constants,
                                                          Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 2) return std::nullopt;
  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) return std::nullopt;

  const auto trip_count_tensor = LookupConstant(constants, inputs[0]);
  const auto start_tensor = LookupConstant(constants, inputs[2]);
  const auto cond = ReadConstantBool(inputs[1]);
  if (!trip_count_tensor || !start_tensor || !cond || !*cond) return std::nullopt;
  if (trip_count_tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return std::nullopt;
  if (ElementCount(trip_count_tensor->dims) != 1 || ElementCount(start_tensor->dims) != 1) return std::nullopt;
  const int64_t trip_count = ReadTypedData<int64_t>(*trip_count_tensor)[0];
  if (trip_count < 0 || trip_count > static_cast<int64_t>(kMaxFoldedElements)) return std::nullopt;

  Ort::ConstGraph body{nullptr};
  for (const auto& subgraph : node.GetSubgraphs()) {
    if (subgraph.attr_name == "body") {
      body = subgraph.sub_graph;
      break;
    }
  }
  if (body == nullptr) return std::nullopt;
  const auto body_inputs = body.GetInputs();
  const auto body_outputs = body.GetOutputs();
  const auto body_nodes = body.GetNodes();
  if (body_inputs.size() != 3 || body_outputs.size() != 3) return std::nullopt;

  std::unordered_map<std::string, std::string> identity_input_by_output;
  auto resolve_alias = [&identity_input_by_output](const std::string& value_name)
      -> std::optional<std::string> {
    std::string current = value_name;
    std::unordered_set<std::string> seen;
    while (true) {
      if (!seen.insert(current).second) return std::nullopt;
      auto it = identity_input_by_output.find(current);
      if (it == identity_input_by_output.end()) return current;
      current = it->second;
    }
  };
  auto lookup_constant_name = [&constants](const std::string& value_name)
      -> std::optional<ConstantTensor> {
    auto it = constants.find(value_name);
    if (it == constants.end()) return std::nullopt;
    return it->second;
  };

  Ort::ConstNode add_node{nullptr};
  for (Ort::ConstNode body_node : body_nodes) {
    const std::string op_type = body_node.GetOperatorType();
    if (op_type == "Add") {
      if (add_node != nullptr) return std::nullopt;
      add_node = body_node;
    } else if (op_type == "Identity") {
      const auto node_inputs = body_node.GetInputs();
      const auto node_outputs = body_node.GetOutputs();
      if (node_inputs.size() != 1 || node_outputs.size() != 1 || node_inputs[0] == nullptr || node_outputs[0] == nullptr) {
        return std::nullopt;
      }
      identity_input_by_output.emplace(node_outputs[0].GetName(), node_inputs[0].GetName());
    } else {
      return std::nullopt;
    }
  }
  if (add_node == nullptr) return std::nullopt;
  const auto cond_root = resolve_alias(body_outputs[0].GetName());
  const auto prev_root = resolve_alias(body_outputs[2].GetName());
  if (!cond_root || !prev_root) return std::nullopt;
  if (*cond_root != body_inputs[1].GetName() || *prev_root != body_inputs[2].GetName()) return std::nullopt;

  const auto add_inputs = add_node.GetInputs();
  const auto add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 || add_outputs[0] == nullptr) {
    return std::nullopt;
  }
  const auto current_root = resolve_alias(body_outputs[1].GetName());
  if (!current_root || *current_root != add_outputs[0].GetName()) return std::nullopt;

  std::optional<ConstantTensor> delta_tensor;
  bool saw_prev = false;
  for (size_t i = 0; i < 2; ++i) {
    if (add_inputs[i] == nullptr) return std::nullopt;
    const auto add_input_root = resolve_alias(add_inputs[i].GetName());
    if (!add_input_root) return std::nullopt;
    if (*add_input_root == body_inputs[2].GetName()) {
      saw_prev = true;
      continue;
    }
    delta_tensor = lookup_constant_name(*add_input_root);
  }
  if (!saw_prev || !delta_tensor) return std::nullopt;
  if (delta_tensor->element_type != start_tensor->element_type ||
      ElementCount(delta_tensor->dims) != 1) {
    return std::nullopt;
  }

  if (start_tensor->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t start = ReadTypedData<int32_t>(*start_tensor)[0];
    const int32_t delta = ReadTypedData<int32_t>(*delta_tensor)[0];
    std::vector<int32_t> scan(static_cast<size_t>(trip_count));
    for (int64_t i = 0; i < trip_count; ++i) scan[static_cast<size_t>(i)] = start + static_cast<int32_t>(i) * delta;
    const int32_t final_value = start + static_cast<int32_t>(trip_count) * delta;
    return std::vector<ConstantTensor>{
        MakeConstant<int32_t>({}, {final_value}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
        MakeConstant<int32_t>({trip_count}, std::move(scan), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
    };
  }
  if (start_tensor->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t start = ReadTypedData<int64_t>(*start_tensor)[0];
    const int64_t delta = ReadTypedData<int64_t>(*delta_tensor)[0];
    std::vector<int64_t> scan(static_cast<size_t>(trip_count));
    for (int64_t i = 0; i < trip_count; ++i) scan[static_cast<size_t>(i)] = start + i * delta;
    const int64_t final_value = start + trip_count * delta;
    return std::vector<ConstantTensor>{
        MakeConstant<int64_t>({}, {final_value}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
        MakeConstant<int64_t>({trip_count}, std::move(scan), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
    };
  }
  return std::nullopt;
}

std::optional<ConstantTensor> EvalConstant(const ConstantValueMap& /*constants*/,
                                            Ort::ConstNode node) {
  // Constant ops usually carry a `value` tensor attribute; other attr forms
  // (value_float, value_int, value_ints, ...) are rare in practice, so we only
  // handle the tensor form here.
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("value", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  Ort::Value value{nullptr};
  const Ort::Status vstatus = attr.GetTensorAttributeAsOrtValue(value);
  if (!vstatus.IsOK() || !value) return std::nullopt;
  const auto tshape = value.GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType et = tshape.GetElementType();
  if (!IsFoldableDType(et)) return std::nullopt;
  const std::vector<int64_t> dims = tshape.GetShape();
  if (!FitsFoldLimit(dims)) return std::nullopt;
  const size_t count = ElementCount(dims);
  const size_t elem = ByteWidth(et);
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null Constant tensor data");
  ConstantTensor out;
  out.element_type = et;
  out.dims = dims;
  out.data.resize(count * elem);
  if (count > 0) std::memcpy(out.data.data(), src, count * elem);
  return out;
}

std::optional<ConstantTensor> EvalTranspose(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const int64_t rank = static_cast<int64_t>(data->dims.size());
  std::vector<int64_t> perm(rank);
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("perm", attr);
  if (status.IsOK() && attr != nullptr) {
    Ort::ThrowOnError(attr.GetValueArray(perm));
    if (static_cast<int64_t>(perm.size()) != rank) return std::nullopt;
  } else {
    for (int64_t i = 0; i < rank; ++i) perm[static_cast<size_t>(i)] = rank - 1 - i;
  }
  std::vector<int64_t> out_dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t p = perm[static_cast<size_t>(i)];
    if (p < 0 || p >= rank) return std::nullopt;
    out_dims[static_cast<size_t>(i)] = data->dims[static_cast<size_t>(p)];
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  const size_t elem = ByteWidth(data->element_type);
  const size_t count = ElementCount(out_dims);
  const auto in_strides = MakeStrides(data->dims);
  const auto out_strides = MakeStrides(out_dims);

  ConstantTensor out;
  out.element_type = data->element_type;
  out.dims = out_dims;
  out.data.resize(count * elem);
  std::vector<int64_t> coord(rank, 0);
  for (size_t linear = 0; linear < count; ++linear) {
    int64_t src_linear = 0;
    for (int64_t i = 0; i < rank; ++i) {
      src_linear += coord[static_cast<size_t>(i)] * in_strides[static_cast<size_t>(perm[static_cast<size_t>(i)])];
    }
    std::memcpy(out.data.data() + linear * elem,
                data->data.data() + static_cast<size_t>(src_linear) * elem, elem);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return out;
}

std::optional<ConstantTensor> EvalGather(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  const auto indices = LookupConstant(constants, inputs[1]);
  if (!data || !indices) return std::nullopt;
  if (indices->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      indices->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return std::nullopt;
  }
  const int64_t data_rank = static_cast<int64_t>(data->dims.size());
  if (data_rank == 0) return std::nullopt;

  int64_t axis = 0;
  if (auto attr = [&]() -> std::optional<int64_t> {
        Ort::ConstOpAttr a{nullptr};
        const Ort::Status s = node.GetAttributeByName("axis", a);
        if (!s.IsOK() || a == nullptr) return std::nullopt;
        int64_t v = 0;
        Ort::ThrowOnError(a.GetValue(v));
        return v;
      }()) {
    axis = *attr;
  }
  if (axis < 0) axis += data_rank;
  if (axis < 0 || axis >= data_rank) return std::nullopt;

  std::vector<int64_t> idx_values;
  if (indices->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    idx_values = ReadTypedData<int64_t>(*indices);
  } else {
    const auto v32 = ReadTypedData<int32_t>(*indices);
    idx_values.assign(v32.begin(), v32.end());
  }
  const int64_t axis_len = data->dims[static_cast<size_t>(axis)];
  for (int64_t& idx : idx_values) {
    if (idx < 0) idx += axis_len;
    if (idx < 0 || idx >= axis_len) return std::nullopt;
  }

  std::vector<int64_t> out_dims;
  out_dims.reserve(data->dims.size() + indices->dims.size() - 1);
  for (int64_t i = 0; i < axis; ++i) out_dims.push_back(data->dims[static_cast<size_t>(i)]);
  for (int64_t d : indices->dims) out_dims.push_back(d);
  for (int64_t i = axis + 1; i < data_rank; ++i) out_dims.push_back(data->dims[static_cast<size_t>(i)]);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  // Element stride along the gather axis.
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < data_rank; ++i) inner *= data->dims[static_cast<size_t>(i)];
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= data->dims[static_cast<size_t>(i)];
  const size_t elem_size = ByteWidth(data->element_type);
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_size;

  ConstantTensor out;
  out.element_type = data->element_type;
  out.dims = std::move(out_dims);
  out.data.resize(ElementCount(out.dims) * elem_size);
  uint8_t* dst = out.data.data();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t k : idx_values) {
      const size_t src_offset =
          (static_cast<size_t>(o) * static_cast<size_t>(axis_len) + static_cast<size_t>(k)) *
          inner_bytes;
      std::memcpy(dst, data->data.data() + src_offset, inner_bytes);
      dst += inner_bytes;
    }
  }
  // Propagate per-element bindings alongside the data copy. Gather is a
  // permutation/selection at the element level, so this mirrors the memcpy
  // loop above but operates on one optional<binding> per element.
  if (!data->dim_bindings.empty()) {
    const size_t inner_sz = static_cast<size_t>(inner);
    out.dim_bindings.assign(ElementCount(out.dims), std::nullopt);
    size_t d = 0;
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t k : idx_values) {
        const size_t src = (static_cast<size_t>(o) * static_cast<size_t>(axis_len) +
                            static_cast<size_t>(k)) * inner_sz;
        for (size_t q = 0; q < inner_sz; ++q) {
          out.dim_bindings[d++] = data->dim_bindings[src + q];
        }
      }
    }
  }
  return out;
}

std::optional<ConstantTensor> EvalShapeTensorGather(
    const ConstantValueMap& constants,
    const GraphIndex& graph_index,
    Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;

  int64_t axis = 0;
  if (auto attr = [&]() -> std::optional<int64_t> {
        Ort::ConstOpAttr a{nullptr};
        const Ort::Status s = node.GetAttributeByName("axis", a);
        if (!s.IsOK() || a == nullptr) return std::nullopt;
        int64_t v = 0;
        Ort::ThrowOnError(a.GetValue(v));
        return v;
      }()) {
    axis = *attr;
  }
  if (axis != 0 && axis != -1) return std::nullopt;

  Ort::ConstNode producer = graph_index.ProducerOf(std::string(inputs[0].GetName()));
  if (producer == nullptr || producer.GetOperatorType() != "Shape") return std::nullopt;
  const std::string producer_domain = producer.GetDomain();
  if (!producer_domain.empty() && producer_domain != "ai.onnx") return std::nullopt;

  const auto shape_inputs = producer.GetInputs();
  if (shape_inputs.size() != 1 || shape_inputs[0] == nullptr) return std::nullopt;
  const TensorMetadata source_meta = getTensorMetadata(shape_inputs[0]);
  const int64_t rank = static_cast<int64_t>(source_meta.dims.size());
  if (rank == 0) return std::nullopt;

  const auto indices_tensor = LookupConstant(constants, inputs[1]);
  if (!indices_tensor.has_value()) return std::nullopt;
  const TensorMetadata indices_meta{
      .element_type = indices_tensor->element_type,
      .dims = indices_tensor->dims,
  };
  if (indices_meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      indices_meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return std::nullopt;
  }

  std::vector<int64_t> raw_indices;
  if (indices_meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    raw_indices = ReadTypedData<int64_t>(*indices_tensor);
  } else {
    const auto v32 = ReadTypedData<int32_t>(*indices_tensor);
    raw_indices.assign(v32.begin(), v32.end());
  }

  std::vector<int64_t> gathered;
  std::vector<size_t> picked_axes;
  gathered.reserve(raw_indices.size());
  picked_axes.reserve(raw_indices.size());
  for (int64_t idx : raw_indices) {
    if (idx < 0) idx += rank;
    if (idx < 0 || idx >= rank) return std::nullopt;
    const int64_t dim = source_meta.dims[static_cast<size_t>(idx)];
    gathered.push_back(dim < 0 ? kUnknownShapeDimSentinel : dim);
    picked_axes.push_back(static_cast<size_t>(idx));
  }

  auto result = MakeConstant<int64_t>(indices_meta.dims, std::move(gathered),
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  result.dim_bindings =
      MakeShapeBindings(source_meta.dims, std::string(shape_inputs[0].GetName()), picked_axes);
  return result;
}

std::optional<ConstantTensor> EvalConstantOfShape(const ConstantValueMap& constants,
                                                  Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto shape_tensor = LookupConstant(constants, inputs[0]);
  if (!shape_tensor ||
      (shape_tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
       shape_tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
    return std::nullopt;
  }
  std::vector<int64_t> shape_values;
  if (shape_tensor->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    shape_values = ReadTypedData<int64_t>(*shape_tensor);
  } else {
    const auto shape_values_i32 = ReadTypedData<int32_t>(*shape_tensor);
    shape_values.assign(shape_values_i32.begin(), shape_values_i32.end());
  }
  for (int64_t d : shape_values) {
    if (d < 0) return std::nullopt;
  }
  if (!FitsFoldLimit(shape_values)) return std::nullopt;

  // Default value is float 0. An optional "value" attribute supplies a single-
  // element tensor whose dtype determines the output dtype.
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<uint8_t> scalar_bytes(sizeof(float), 0);

  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("value", attr);
  if (status.IsOK() && attr != nullptr) {
    Ort::Value value{nullptr};
    const Ort::Status v_status = attr.GetTensorAttributeAsOrtValue(value);
    if (v_status.IsOK() && value) {
      const auto tshape = value.GetTensorTypeAndShapeInfo();
      const ONNXTensorElementDataType et = tshape.GetElementType();
      if (!IsFoldableDType(et)) return std::nullopt;
      element_type = et;
      const void* src = nullptr;
      THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
      GGONNX_NOT_NULL(src, "ORT returned null ConstantOfShape value data");
      const size_t elem = ByteWidth(element_type);
      scalar_bytes.assign(static_cast<const uint8_t*>(src),
                          static_cast<const uint8_t*>(src) + elem);
    }
  }

  const size_t elem_size = ByteWidth(element_type);
  const size_t count = ElementCount(shape_values);
  ConstantTensor out;
  out.element_type = element_type;
  out.dims = shape_values;
  out.data.resize(count * elem_size);
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(out.data.data() + i * elem_size, scalar_bytes.data(), elem_size);
  }
  return out;
}

template <typename T>
std::optional<ConstantTensor> EvalEqualTyped(const ConstantTensor& lhs, const ConstantTensor& rhs) {
  const size_t lhs_count = ElementCount(lhs.dims);
  const size_t rhs_count = ElementCount(rhs.dims);
  const bool lhs_scalar = lhs_count == 1;
  const bool rhs_scalar = rhs_count == 1;
  if (!lhs_scalar && !rhs_scalar && lhs.dims != rhs.dims) return std::nullopt;
  const std::vector<int64_t> out_dims = lhs_scalar ? rhs.dims : lhs.dims;
  const size_t out_count = std::max(lhs_count, rhs_count);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  const auto a = ReadTypedData<T>(lhs);
  const auto b = ReadTypedData<T>(rhs);
  std::vector<uint8_t> out(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    out[i] = a[lhs_scalar ? 0 : i] == b[rhs_scalar ? 0 : i] ? 1u : 0u;
  }
  ConstantTensor tensor;
  tensor.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  tensor.dims = out_dims;
  tensor.data.resize(out.size());
  std::memcpy(tensor.data.data(), out.data(), out.size());
  return tensor;
}

std::optional<ConstantTensor> EvalEqual(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto lhs = LookupConstant(constants, inputs[0]);
  const auto rhs = LookupConstant(constants, inputs[1]);
  if (!lhs || !rhs || lhs->element_type != rhs->element_type) return std::nullopt;
  if (ContainsUnknownShapeDim(*lhs) || ContainsUnknownShapeDim(*rhs)) return std::nullopt;
  return DispatchByElementType(lhs->element_type, [&](auto tag) -> std::optional<ConstantTensor> {
    using T = decltype(tag);
    return EvalEqualTyped<T>(*lhs, *rhs);
  });
}

std::optional<ConstantTensor> EvalWhere(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return std::nullopt;
  }
  const auto cond = LookupConstant(constants, inputs[0]);
  const auto lhs = LookupConstant(constants, inputs[1]);
  const auto rhs = LookupConstant(constants, inputs[2]);
  if (!cond || !lhs || !rhs) return std::nullopt;
  if (ContainsUnknownShapeDim(*cond) || ContainsUnknownShapeDim(*lhs) ||
      ContainsUnknownShapeDim(*rhs)) {
    return std::nullopt;
  }
  if (cond->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return std::nullopt;
  if (lhs->element_type != rhs->element_type) return std::nullopt;
  // Limit broadcasting: all three equal-shape, or any scalar.
  const size_t c = ElementCount(cond->dims);
  const size_t l = ElementCount(lhs->dims);
  const size_t r = ElementCount(rhs->dims);
  const size_t out_count = std::max({c, l, r});
  std::vector<int64_t> out_dims;
  if (c == out_count) out_dims = cond->dims;
  else if (l == out_count) out_dims = lhs->dims;
  else out_dims = rhs->dims;
  auto pick = [&](size_t i, const ConstantTensor& t, size_t n) -> size_t {
    return n == 1 ? 0 : i;
  };
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  const size_t elem = ByteWidth(lhs->element_type);
  ConstantTensor out;
  out.element_type = lhs->element_type;
  out.dims = out_dims;
  out.data.resize(out_count * elem);
  for (size_t i = 0; i < out_count; ++i) {
    const uint8_t cval = cond->data[pick(i, *cond, c)];
    const uint8_t* src = (cval ? lhs->data.data() + pick(i, *lhs, l) * elem
                               : rhs->data.data() + pick(i, *rhs, r) * elem);
    std::memcpy(out.data.data() + i * elem, src, elem);
  }
  return out;
}

std::optional<ConstantTensor> EvalRange(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return std::nullopt;
  }
  const auto start = LookupConstant(constants, inputs[0]);
  const auto limit = LookupConstant(constants, inputs[1]);
  const auto delta = LookupConstant(constants, inputs[2]);
  if (!start || !limit || !delta) return std::nullopt;
  if (start->element_type != limit->element_type || start->element_type != delta->element_type) {
    return std::nullopt;
  }
  if (ElementCount(start->dims) != 1 || ElementCount(limit->dims) != 1 || ElementCount(delta->dims) != 1) {
    return std::nullopt;
  }

  const auto et = start->element_type;
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float s = ReadTypedData<float>(*start)[0];
    const float l = ReadTypedData<float>(*limit)[0];
    const float d = ReadTypedData<float>(*delta)[0];
    if (d == 0.0f) return std::nullopt;
    const int64_t n = std::max<int64_t>(static_cast<int64_t>(std::ceil((l - s) / d)), 0);
    if (n > static_cast<int64_t>(kMaxFoldedElements)) return std::nullopt;
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = s + static_cast<float>(i) * d;
    return MakeConstant<float>({n}, std::move(out), et);
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t s = ReadTypedData<int32_t>(*start)[0];
    const int32_t l = ReadTypedData<int32_t>(*limit)[0];
    const int32_t d = ReadTypedData<int32_t>(*delta)[0];
    if (d == 0) return std::nullopt;
    const int64_t n = std::max<int64_t>(static_cast<int64_t>(std::ceil(double(l - s) / double(d))), 0);
    if (n > static_cast<int64_t>(kMaxFoldedElements)) return std::nullopt;
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = s + static_cast<int32_t>(i) * d;
    return MakeConstant<int32_t>({n}, std::move(out), et);
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t s = ReadTypedData<int64_t>(*start)[0];
    const int64_t l = ReadTypedData<int64_t>(*limit)[0];
    const int64_t d = ReadTypedData<int64_t>(*delta)[0];
    if (d == 0) return std::nullopt;
    const int64_t n = std::max<int64_t>(static_cast<int64_t>(std::ceil(double(l - s) / double(d))), 0);
    if (n > static_cast<int64_t>(kMaxFoldedElements)) return std::nullopt;
    std::vector<int64_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = s + i * d;
    return MakeConstant<int64_t>({n}, std::move(out), et);
  }
  return std::nullopt;
}

std::optional<ConstantTensor> EvalUnaryMath(const ConstantValueMap& constants,
                                            Ort::ConstNode node) {
  const std::string op_type = node.GetOperatorType();
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input || input->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return std::nullopt;
  if (ContainsUnknownShapeDim(*input) || !FitsFoldLimit(input->dims)) return std::nullopt;
  std::vector<float> out = ReadTypedData<float>(*input);
  for (float& v : out) {
    if (op_type == "Ceil") v = std::ceil(v);
    else if (op_type == "Round") v = std::round(v);
    else return std::nullopt;
  }
  return MakeConstant<float>(std::vector<int64_t>(input->dims), std::move(out), input->element_type);
}

std::optional<ConstantTensor> EvalReduceMin(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data || ContainsUnknownShapeDim(*data)) return std::nullopt;
  if (data->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      data->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      data->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }

  std::vector<int64_t> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto a = ReadConstantInt64Input(constants, node, 1);
    if (!a) return std::nullopt;
    axes = *a;
  } else {
    Ort::ConstOpAttr attr{nullptr};
    if (node.GetAttributeByName("axes", attr).IsOK() && attr != nullptr) {
      Ort::ThrowOnError(attr.GetValueArray(axes));
    } else {
      axes.resize(data->dims.size());
      for (size_t i = 0; i < axes.size(); ++i) axes[i] = static_cast<int64_t>(i);
    }
  }
  const bool keepdims = [&]() {
    Ort::ConstOpAttr attr{nullptr};
    int64_t v = 1;
    if (node.GetAttributeByName("keepdims", attr).IsOK() && attr != nullptr) attr.GetValue(v);
    return v != 0;
  }();

  const int64_t rank = static_cast<int64_t>(data->dims.size());
  std::vector<int64_t> norm_axes = axes;
  for (int64_t& a : norm_axes) {
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) return std::nullopt;
  }
  std::sort(norm_axes.begin(), norm_axes.end());
  norm_axes.erase(std::unique(norm_axes.begin(), norm_axes.end()), norm_axes.end());

  // Keep this simple: fold only full reduction or 1-axis reductions for now.
  if (!(norm_axes.size() == rank || norm_axes.size() == 1)) return std::nullopt;

  auto make_out_dims = [&]() {
    std::vector<int64_t> out_dims = data->dims;
    if (keepdims) {
      for (int64_t a : norm_axes) out_dims[static_cast<size_t>(a)] = 1;
    } else {
      for (auto it = norm_axes.rbegin(); it != norm_axes.rend(); ++it) {
        out_dims.erase(out_dims.begin() + *it);
      }
      if (out_dims.empty()) out_dims = {};
    }
    return out_dims;
  };

  if (norm_axes.size() == rank) {
    if (data->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const auto vals = ReadTypedData<float>(*data);
      const float v = *std::min_element(vals.begin(), vals.end());
      return MakeConstant<float>(make_out_dims(), {v}, data->element_type);
    }
    if (data->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto vals = ReadTypedData<int32_t>(*data);
      const int32_t v = *std::min_element(vals.begin(), vals.end());
      return MakeConstant<int32_t>(make_out_dims(), {v}, data->element_type);
    }
    const auto vals = ReadTypedData<int64_t>(*data);
    const int64_t v = *std::min_element(vals.begin(), vals.end());
    return MakeConstant<int64_t>(make_out_dims(), {v}, data->element_type);
  }

  const int64_t axis = norm_axes[0];
  const auto strides = MakeStrides(data->dims);
  const int64_t axis_stride = strides[static_cast<size_t>(axis)];
  const int64_t axis_len = data->dims[static_cast<size_t>(axis)];
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= data->dims[static_cast<size_t>(i)];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < rank; ++i) inner *= data->dims[static_cast<size_t>(i)];
  const auto out_dims = make_out_dims();
  if (!FitsFoldLimit(out_dims.empty() ? std::vector<int64_t>{1} : out_dims)) return std::nullopt;

  if (data->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const auto vals = ReadTypedData<float>(*data);
    std::vector<float> out(static_cast<size_t>(outer * inner));
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t in = 0; in < inner; ++in) {
        float best = vals[static_cast<size_t>(o * axis_len * inner + in)];
        for (int64_t a = 1; a < axis_len; ++a) {
          best = std::min(best, vals[static_cast<size_t>(o * axis_len * inner + a * inner + in)]);
        }
        out[static_cast<size_t>(o * inner + in)] = best;
      }
    }
    return MakeConstant<float>(out_dims, std::move(out), data->element_type);
  }
  if (data->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const auto vals = ReadTypedData<int32_t>(*data);
    std::vector<int32_t> out(static_cast<size_t>(outer * inner));
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t in = 0; in < inner; ++in) {
        int32_t best = vals[static_cast<size_t>(o * axis_len * inner + in)];
        for (int64_t a = 1; a < axis_len; ++a) {
          best = std::min(best, vals[static_cast<size_t>(o * axis_len * inner + a * inner + in)]);
        }
        out[static_cast<size_t>(o * inner + in)] = best;
      }
    }
    return MakeConstant<int32_t>(out_dims, std::move(out), data->element_type);
  }
  const auto vals = ReadTypedData<int64_t>(*data);
  std::vector<int64_t> out(static_cast<size_t>(outer * inner));
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t in = 0; in < inner; ++in) {
      int64_t best = vals[static_cast<size_t>(o * axis_len * inner + in)];
      for (int64_t a = 1; a < axis_len; ++a) {
        best = std::min(best, vals[static_cast<size_t>(o * axis_len * inner + a * inner + in)]);
      }
      out[static_cast<size_t>(o * inner + in)] = best;
    }
  }
  return MakeConstant<int64_t>(out_dims, std::move(out), data->element_type);
}

}  // namespace

bool ConstantContainsSentinel(const ConstantTensor& tensor) {
  return ContainsUnknownShapeDim(tensor);
}

std::string NodeKey(Ort::ConstNode node) {
  const auto outputs = node.GetOutputs();
  for (auto output : outputs) {
    if (output != nullptr) return output.GetName();
  }
  const std::string name = node.GetName();
  if (!name.empty()) return name;
  return node.GetOperatorType();
}

namespace {

std::optional<std::vector<int64_t>> ReadTransposePerm(Ort::ConstNode node) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("perm", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  std::vector<int64_t> values;
  Ort::ThrowOnError(attr.GetValueArray(values));
  return values;
}

// Resolve a Reshape target tensor (with 0 and -1 placeholders) against the
// input's concrete dims. Returns nullopt if resolution would leave dynamic
// entries. Element-count conservation is not enforced here — the ORT graph
// already guarantees it at export time for valid models.
std::optional<std::vector<int64_t>> ResolveReshapeTarget(const std::vector<int64_t>& target,
                                                         const std::vector<int64_t>& input_dims) {
  std::vector<int64_t> out = target;
  int64_t infer_axis = -1;
  int64_t known = 1;
  int64_t total = 1;
  for (int64_t d : input_dims) {
    if (d < 0) return std::nullopt;
    total *= d;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == 0) {
      if (i >= input_dims.size() || input_dims[i] < 0) return std::nullopt;
      out[i] = input_dims[i];
    }
    if (out[i] == -1) {
      if (infer_axis >= 0) return std::nullopt;
      infer_axis = static_cast<int64_t>(i);
      continue;
    }
    if (out[i] <= 0) return std::nullopt;
    known *= out[i];
  }
  if (infer_axis >= 0) {
    if (known == 0) return std::nullopt;
    out[static_cast<size_t>(infer_axis)] = total / known;
  }
  return out;
}

// Detects any Reshape(4D->XD)->Transpose->Reshape(XD->4D) triple where X >
// GGML_MAX_DIMS and the permutation coalesces to ≤GGML_MAX_DIMS axis groups.
//
// Coalescing rule: consecutive ONNX input axes i and i+1 can be merged into
// one group when they remain adjacent in the output with the same order, i.e.
// inv_perm[i+1] == inv_perm[i]+1. We greedily merge maximal such runs, giving
// N_groups groups. If N_groups ≤ GGML_MAX_DIMS each group is flattened to one
// dim and the inter-group reorder becomes a plain rank-4 ggml_permute.
//
// The GGML permutation formula (for GGML axis j, fastest-varying first):
//   ONNX output group out_g = N_groups-1-j
//   ONNX input group  in_g  = coalesced_perm[out_g]
//   ggml_perm[j] = N_groups-1 - coalesced_perm[N_groups-1-j]
void DetectShuffleTripleFusions(const GraphIndex& gi,
                                const ConstantValueMap& constants,
                                FusionPlan& plan) {
  // Treats graph-output use as one additional consumer, matching the original
  // intent of rejecting fusions where the intermediate also leaves the graph.
  auto effective_consumers = [&](const std::string& name) {
    return gi.ConsumerCount(name) + (gi.IsGraphOutput(name) ? 1 : 0);
  };
  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) {
    return claimed.insert(key).second;
  };

  for (Ort::ConstNode trans : gi.nodes) {
    if (trans.GetOperatorType() != "Transpose") continue;
    const auto perm_opt = ReadTransposePerm(trans);
    if (!perm_opt) continue;
    const std::vector<int64_t>& perm = *perm_opt;
    const int X = static_cast<int>(perm.size());
    if (X <= GGML_MAX_DIMS) continue;  // individual ops already handle rank ≤4

    const auto trans_inputs = trans.GetInputs();
    const auto trans_outputs = trans.GetOutputs();
    if (trans_inputs.size() != 1 || trans_outputs.size() != 1) continue;
    if (trans_inputs[0] == nullptr || trans_outputs[0] == nullptr) continue;

    const std::string r1_out = std::string(trans_inputs[0].GetName());
    Ort::ConstNode r1 = gi.ProducerOf(r1_out);
    if (r1 == nullptr || r1.GetOperatorType() != "Reshape") continue;
    if (effective_consumers(r1_out) != 1) continue;

    const std::string trans_out_name = std::string(trans_outputs[0].GetName());
    if (effective_consumers(trans_out_name) != 1) continue;

    Ort::ConstNode r2 = gi.OnlyConsumerOf(trans_out_name, "Reshape");
    if (r2 == nullptr) continue;

    const auto r1_inputs = r1.GetInputs();
    const auto r2_inputs = r2.GetInputs();
    const auto r2_outputs = r2.GetOutputs();
    if (r1_inputs.size() != 2 || r2_inputs.size() != 2 || r2_outputs.size() != 1) continue;
    if (r1_inputs[0] == nullptr || r1_inputs[1] == nullptr ||
        r2_inputs[0] == nullptr || r2_inputs[1] == nullptr || r2_outputs[0] == nullptr) {
      continue;
    }

    // r1 input must be fully static with rank ≤ GGML_MAX_DIMS.
    const TensorMetadata r1_in_meta = getTensorMetadata(r1_inputs[0]);
    if (!shapeIsFullyStatic(r1_in_meta) ||
        r1_in_meta.dims.size() > static_cast<size_t>(GGML_MAX_DIMS)) continue;

    // r1 target must resolve to exactly rank X (the intermediate rank).
    auto r1_target = ReadConstantInt64Input(constants, r1, 1);
    if (!r1_target || static_cast<int>(r1_target->size()) != X) continue;
    auto r1_resolved = ResolveReshapeTarget(*r1_target, r1_in_meta.dims);
    if (!r1_resolved || static_cast<int>(r1_resolved->size()) != X) continue;
    bool any_nonpos = false;
    for (int64_t d : *r1_resolved) { if (d <= 0) { any_nonpos = true; break; } }
    if (any_nonpos) continue;

    // Compute inv_perm.
    std::vector<int> inv_perm(X);
    for (int j = 0; j < X; ++j) inv_perm[static_cast<int>(perm[j])] = j;

    // Greedily merge consecutive input axes that remain adjacent post-transpose.
    // Each group is a contiguous run [start..end) of input axes.
    struct Group { int start; int end; };
    std::vector<Group> groups;
    groups.push_back({0, 1});
    for (int i = 1; i < X; ++i) {
      if (inv_perm[i] == inv_perm[i - 1] + 1) {
        groups.back().end = i + 1;
      } else {
        groups.push_back({i, i + 1});
      }
    }
    const int N_groups = static_cast<int>(groups.size());
    if (N_groups > GGML_MAX_DIMS) continue;

    // Compute group product dims (ONNX order, slowest first).
    std::vector<int64_t> group_onnx_dims(N_groups);
    for (int g = 0; g < N_groups; ++g) {
      int64_t prod = 1;
      for (int ax = groups[g].start; ax < groups[g].end; ++ax)
        prod *= (*r1_resolved)[ax];
      group_onnx_dims[g] = prod;
    }

    // Determine where each input group lands in the output.
    // The first axis of group g goes to output position inv_perm[groups[g].start].
    // Sort groups by that position to get the coalesced ONNX perm:
    //   coalesced_perm[out_g] = in_g (input group that produces output group out_g).
    std::vector<int> group_out_pos(N_groups);
    for (int g = 0; g < N_groups; ++g) group_out_pos[g] = inv_perm[groups[g].start];
    std::vector<int> order(N_groups);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return group_out_pos[a] < group_out_pos[b]; });
    // order[out_g] = in_g
    std::vector<int> coalesced_perm(N_groups);
    for (int out_g = 0; out_g < N_groups; ++out_g) coalesced_perm[out_g] = order[out_g];

    // Convert to GGML axis order (reversed from ONNX, padded to GGML_MAX_DIMS).
    // ggml_perm[j] = N_groups-1 - coalesced_perm[N_groups-1-j]
    NodeDesc::GenericShuffleAttrs attrs;
    attrs.grouped_ggml_dims.fill(1);
    attrs.ggml_perm = {0, 1, 2, 3};
    for (int g = 0; g < N_groups; ++g) {
      int ggml_axis = N_groups - 1 - g;  // ONNX group g → GGML axis (N-1-g)
      attrs.grouped_ggml_dims[ggml_axis] = group_onnx_dims[g];
    }
    for (int j = 0; j < N_groups; ++j) {
      attrs.ggml_perm[j] = N_groups - 1 - coalesced_perm[N_groups - 1 - j];
    }

    // Compute r2's input dims from the transposed intermediate shape.
    std::vector<int64_t> r2_in_dims(X);
    for (int j = 0; j < X; ++j) r2_in_dims[j] = (*r1_resolved)[static_cast<int>(perm[j])];

    auto r2_target = ReadConstantInt64Input(constants, r2, 1);
    std::optional<std::vector<int64_t>> r2_resolved;
    if (r2_target) {
      r2_resolved = ResolveReshapeTarget(*r2_target, r2_in_dims);
    } else {
      const TensorMetadata r2_out_meta = getTensorMetadata(r2_outputs[0]);
      if (shapeIsFullyStatic(r2_out_meta)) r2_resolved = r2_out_meta.dims;
    }
    if (!r2_resolved || r2_resolved->empty() ||
        r2_resolved->size() > static_cast<size_t>(GGML_MAX_DIMS)) continue;
    for (int64_t d : *r2_resolved) { if (d <= 0) { r2_resolved.reset(); break; } }
    if (!r2_resolved) continue;

    auto prod_fn = [](const std::vector<int64_t>& v) {
      int64_t p = 1; for (int64_t d : v) p *= d; return p;
    };
    if (prod_fn(*r1_resolved) != prod_fn(r1_in_meta.dims) ||
        prod_fn(*r2_resolved) != prod_fn(*r1_resolved)) {
      continue;
    }

    const std::string r1_key = NodeKey(r1);
    const std::string r2_key = NodeKey(r2);
    const std::string anchor_key = NodeKey(trans);
    if (!claim(r1_key) || !claim(r2_key) || !claim(anchor_key)) continue;

    attrs.output_onnx_dims = *r2_resolved;

    FusionPlan::AnchorIO io;
    io.input_values.push_back(std::string(r1_inputs[0].GetName()));
    io.output_values.push_back(std::string(r2_outputs[0].GetName()));
    io.anchor_node_name = std::string(trans.GetName());

    plan.generic_shuffle_anchors.emplace(anchor_key, std::move(attrs));
    plan.anchor_io.emplace(anchor_key, std::move(io));
    plan.consumed_nodes.insert(r1_key);
    plan.consumed_nodes.insert(r2_key);
  }
}

std::optional<int64_t> ReadInt64Attr(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t value{};
  Ort::ThrowOnError(attr.GetValue(value));
  return value;
}

bool IsNodeCompilable(Ort::ConstNode node, const ConstantValueMap& constants) {
  // Fusion detection runs mid-analysis before inferred_shapes is populated,
  // so hand the support predicates a lightweight MetaAnalysis holding only the
  // already-folded constants.
  MetaAnalysis partial;
  partial.constants = constants;
  return get_node_support(node, &partial).has_value();
}

void DetectQKVSplitFusions(const GraphIndex& gi,
                           const ConstantValueMap& constants,
                           const std::unordered_set<std::string>& folded_nodes,
                           FusionPlan& plan) {
  // Matches the pre-L2 attention QKV split that ORT's CPU EP would later fuse
  // into a Split kernel. Our EP sees the raw form:
  //   Reshape(x=[..., 3C] -> [B*nw, M*M, 3, heads, head_dim]) ->
  //     Transpose(perm=[2,0,3,1,4]) ->
  //     3x Gather(axis=0, scalar index in {0,1,2})
  // Every rank-5 intermediate has exactly one consumer, so the fusion is safe
  // to discard them. Emits three rank-4 ne=[head_dim, M*M, heads, B*nw].
  auto effective_consumers = [&](const std::string& name) {
    return gi.ConsumerCount(name) + (gi.IsGraphOutput(name) ? 1 : 0);
  };
  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) { return claimed.insert(key).second; };

  for (Ort::ConstNode trans : gi.nodes) {
    if (trans.GetOperatorType() != "Transpose") continue;
    const auto perm = ReadTransposePerm(trans);
    const std::vector<int64_t> expected_perm = {2, 0, 3, 1, 4};
    if (!perm || *perm != expected_perm) continue;

    const auto trans_inputs = trans.GetInputs();
    const auto trans_outputs = trans.GetOutputs();
    if (trans_inputs.size() != 1 || trans_outputs.size() != 1) continue;
    if (trans_inputs[0] == nullptr || trans_outputs[0] == nullptr) continue;

    // Upstream Reshape producing the rank-5 [B*nw, M*M, 3, heads, head_dim].
    const std::string r_out_name = std::string(trans_inputs[0].GetName());
    if (effective_consumers(r_out_name) != 1) continue;
    Ort::ConstNode reshape = gi.ProducerOf(r_out_name);
    if (reshape == nullptr || reshape.GetOperatorType() != "Reshape") continue;

    const auto r_inputs = reshape.GetInputs();
    const auto r_outputs = reshape.GetOutputs();
    if (r_inputs.size() != 2 || r_outputs.size() != 1) continue;
    if (r_inputs[0] == nullptr || r_outputs[0] == nullptr) continue;

    const TensorMetadata r_in_meta = getTensorMetadata(r_inputs[0]);
    if (!shapeIsFullyStatic(r_in_meta)) continue;

    std::vector<int64_t> r_target_dims;
    if (auto r_target = ReadConstantInt64Input(constants, reshape, 1)) {
      auto resolved = ResolveReshapeTarget(*r_target, r_in_meta.dims);
      if (!resolved) continue;
      r_target_dims = *resolved;
    } else {
      const TensorMetadata r_out_meta = getTensorMetadata(r_outputs[0]);
      if (!shapeIsFullyStatic(r_out_meta)) continue;
      r_target_dims = r_out_meta.dims;
    }
    if (r_target_dims.size() != 5) continue;
    for (int64_t d : r_target_dims) {
      if (d <= 0) { r_target_dims.clear(); break; }
    }
    if (r_target_dims.empty()) continue;
    if (r_target_dims[2] != 3) continue;

    // Downstream: exactly three consumers of the Transpose output, each a
    // Gather(axis=0) with a scalar int64 index in {0,1,2} — one per Q/K/V.
    const std::string trans_out_name = std::string(trans_outputs[0].GetName());
    if (effective_consumers(trans_out_name) != 3) continue;

    std::array<Ort::ConstNode, 3> gathers{Ort::ConstNode{nullptr}, Ort::ConstNode{nullptr},
                                          Ort::ConstNode{nullptr}};
    bool seen[3] = {false, false, false};
    bool ok = true;
    for (Ort::ConstNode n : gi.Consumers(trans_out_name)) {
      if (n.GetOperatorType() != "Gather") { ok = false; break; }
      const int64_t gaxis = ReadInt64Attr(n, "axis").value_or(0);
      if (gaxis != 0) { ok = false; break; }
      const auto g_inputs = n.GetInputs();
      const auto g_outputs = n.GetOutputs();
      if (g_inputs.size() != 2 || g_outputs.size() != 1) { ok = false; break; }
      const auto idx = ReadConstantInt64Input(constants, n, 1);
      if (!idx || idx->size() != 1) { ok = false; break; }
      const int64_t which = (*idx)[0];
      if (which < 0 || which > 2 || seen[which]) { ok = false; break; }
      seen[which] = true;
      gathers[static_cast<size_t>(which)] = n;
    }
    if (!ok || !seen[0] || !seen[1] || !seen[2]) continue;

    const std::string r_key = NodeKey(reshape);
    const std::string t_key = NodeKey(trans);
    const std::string g0_key = NodeKey(gathers[0]);
    const std::string g1_key = NodeKey(gathers[1]);
    const std::string g2_key = NodeKey(gathers[2]);

    // Partition safety: every node between the fusion members must be a
    // member, folded, claimed by an earlier fusion, or compilable. Otherwise
    // an unsupported node (e.g. a Softmax we haven't implemented) would
    // split the partition and leave the rank-5 Transpose output exposed.
    if (!FusionSpanSafe(gi, {r_key, t_key, g0_key, g1_key, g2_key},
                        constants, folded_nodes, plan)) continue;

    if (!claim(r_key) || !claim(t_key) || !claim(g0_key) || !claim(g1_key) || !claim(g2_key)) {
      continue;
    }

    NodeDesc::QKVSplitAttrs attrs;
    attrs.num_batch = r_target_dims[0];
    attrs.num_tokens = r_target_dims[1];
    attrs.num_heads = r_target_dims[3];
    attrs.head_dim = r_target_dims[4];

    // Anchor on the Transpose — it's the unique node in the pattern (the
    // Gathers don't share keys). Synthesizing outputs in Q/K/V order requires
    // indexing the `gathers` array by the scalar index value.
    const std::string anchor_key = t_key;

    FusionPlan::AnchorIO io;
    io.input_values.push_back(std::string(r_inputs[0].GetName()));
    for (size_t i = 0; i < 3; ++i) {
      io.output_values.push_back(std::string(gathers[i].GetOutputs()[0].GetName()));
    }
    io.anchor_node_name = std::string(trans.GetName());

    plan.qkv_split_anchors.emplace(anchor_key, std::move(attrs));
    plan.anchor_io.emplace(anchor_key, std::move(io));
    plan.consumed_nodes.insert(r_key);
    plan.consumed_nodes.insert(g0_key);
    plan.consumed_nodes.insert(g1_key);
    plan.consumed_nodes.insert(g2_key);
  }
}

void DetectWindowMaskAddFusions(const GraphIndex& gi,
                                const ConstantValueMap& constants,
                                const std::unordered_set<std::string>& folded_nodes,
                                FusionPlan& plan) {
  // Matches the SW-MSA mask-apply round-trip:
  //   Reshape(X: rank<=4 -> rank>4) -> Add(X', constant_mask) -> Reshape(back)
  // where the trailing Reshape's output shape equals the leading Reshape's
  // input shape. Because the mask is a compile-time constant, we can drop its
  // leading size-1 ONNX axes to rank-reduce it to <=4 without moving any
  // bytes; the Add then becomes a plain rank-4 broadcast add directly on X.
  auto effective_consumers = [&](const std::string& name) {
    return gi.ConsumerCount(name) + (gi.IsGraphOutput(name) ? 1 : 0);
  };
  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) { return claimed.insert(key).second; };

  for (Ort::ConstNode add : gi.nodes) {
    if (add.GetOperatorType() != "Add") continue;
    const auto add_inputs = add.GetInputs();
    const auto add_outputs = add.GetOutputs();
    if (add_inputs.size() != 2 || add_outputs.size() != 1) continue;
    if (add_inputs[0] == nullptr || add_inputs[1] == nullptr || add_outputs[0] == nullptr) continue;

    // One side must be the rank-5 Reshape output, the other a constant mask.
    // The mask can be in either slot — probe both orderings.
    for (size_t score_slot = 0; score_slot < 2; ++score_slot) {
      const size_t mask_slot = 1 - score_slot;
      Ort::ConstValueInfo score_vi = add_inputs[score_slot];
      Ort::ConstValueInfo mask_vi = add_inputs[mask_slot];

      // Mask must be a tracked compile-time constant.
      const std::string mask_name = std::string(mask_vi.GetName());
      auto mask_it = constants.find(mask_name);
      if (mask_it == constants.end()) continue;
      if (mask_it->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
      std::vector<int64_t> mask_dims = mask_it->second.dims;

      // Leading Reshape must produce the score side.
      const std::string lead_out_name = std::string(score_vi.GetName());
      if (effective_consumers(lead_out_name) != 1) continue;
      Ort::ConstNode lead = gi.ProducerOf(lead_out_name);
      if (lead == nullptr || lead.GetOperatorType() != "Reshape") continue;
      const auto lead_inputs = lead.GetInputs();
      if (lead_inputs.size() < 1 || lead_inputs[0] == nullptr) continue;

      const TensorMetadata lead_in_meta = getTensorMetadata(lead_inputs[0]);
      if (!shapeIsFullyStatic(lead_in_meta)) continue;
      if (!rankSupportedByGGML(lead_in_meta)) continue;  // X must already fit in ggml

      const TensorMetadata lead_out_meta = getTensorMetadata(score_vi);
      if (!shapeIsFullyStatic(lead_out_meta)) continue;
      if (rankSupportedByGGML(lead_out_meta)) continue;  // fusion only pays off at rank>4

      // Trailing Reshape must be the single consumer of the Add's output and
      // must land back at X's shape.
      const std::string add_out_name = std::string(add_outputs[0].GetName());
      if (effective_consumers(add_out_name) != 1) continue;
      Ort::ConstNode trail = gi.OnlyConsumerOf(add_out_name, "Reshape");
      if (trail == nullptr) continue;
      const auto trail_outputs = trail.GetOutputs();
      if (trail_outputs.size() != 1 || trail_outputs[0] == nullptr) continue;
      const TensorMetadata trail_out_meta = getTensorMetadata(trail_outputs[0]);
      if (!shapeIsFullyStatic(trail_out_meta)) continue;
      if (trail_out_meta.dims != lead_in_meta.dims) continue;

      // Rank-reduce the mask by dropping leading size-1 ONNX axes (these are
      // pure broadcast axes — dropping them preserves element ordering).
      std::vector<int64_t> reduced = mask_dims;
      while (reduced.size() > 4 && !reduced.empty() && reduced.front() == 1) {
        reduced.erase(reduced.begin());
      }
      if (reduced.size() > 4) continue;
      if (!broadcastSupportedByGGML(reduced, lead_in_meta.dims)) continue;

      // Partition-safety span check: only members, compilable nodes, and
      // folded nodes may sit between the three members in topological order.
      const std::string lead_key = NodeKey(lead);
      const std::string anchor_key = NodeKey(add);
      const std::string trail_key = NodeKey(trail);
      if (!FusionSpanSafe(gi, {lead_key, anchor_key, trail_key},
                          constants, folded_nodes, plan)) continue;

      if (!claim(lead_key) || !claim(anchor_key) || !claim(trail_key)) continue;

      FusionPlan::AnchorIO io;
      io.input_values.push_back(std::string(lead_inputs[0].GetName()));
      io.input_values.push_back(mask_name);
      io.output_values.push_back(std::string(trail_outputs[0].GetName()));
      io.anchor_node_name = std::string(add.GetName());

      plan.constant_override_dims.emplace(mask_name, reduced);
      plan.window_mask_add_anchors.emplace(anchor_key, std::move(reduced));
      plan.anchor_io.emplace(anchor_key, std::move(io));
      plan.consumed_nodes.insert(lead_key);
      plan.consumed_nodes.insert(trail_key);
      break;  // don't probe the swapped ordering once we've fused
    }
  }
}

// Reads the pads vector from a Pad node (attribute or constant input[1]).
static std::optional<std::vector<int64_t>> ReadPadNodePads(
    Ort::ConstNode node, const ConstantValueMap& constants) {
  // opset 9: pads is an int64 attribute.
  Ort::ConstOpAttr attr{nullptr};
  if (node.GetAttributeByName("pads", attr).IsOK() && attr != nullptr) {
    std::vector<int64_t> v;
    if (attr.GetValueArray(v).IsOK()) return v;
  }
  // opset 11+: pads is input[1].
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto it = constants.find(std::string(inputs[1].GetName()));
    if (it != constants.end() &&
        it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const size_t count = it->second.data.size() / sizeof(int64_t);
      std::vector<int64_t> v(count);
      std::memcpy(v.data(), it->second.data.data(), it->second.data.size());
      return v;
    }
  }
  return std::nullopt;
}

// Folds zero-constant Pad → Conv pairs: the Pad's H/W padding is added to the
// Conv's p0/p1 and the Pad node is eliminated. Applies only when the Pad is
// constant-mode with value 0, pads only H/W symmetrically, and the Conv has
// no padding of its own and has exactly one consumer of the Pad's output.
void DetectPadConvFusions(const GraphIndex& gi,
                          const ConstantValueMap& constants,
                          FusionPlan& plan) {
  for (Ort::ConstNode conv : gi.nodes) {
    const std::string op = conv.GetOperatorType();
    if (op != "Conv") continue;
    const std::string dom = conv.GetDomain();
    if (!dom.empty() && dom != "ai.onnx") continue;

    const auto conv_inputs = conv.GetInputs();
    if (conv_inputs.empty() || conv_inputs[0] == nullptr) continue;

    // Conv must have no explicit padding (VALID or zero pads).
    const std::string auto_pad =
        [&]() -> std::string {
          Ort::ConstOpAttr a{nullptr};
          if (!conv.GetAttributeByName("auto_pad", a).IsOK() || a == nullptr) return "NOTSET";
          std::string s;
          a.GetValue(s);
          return s;
        }();
    if (auto_pad != "NOTSET" && auto_pad != "VALID") continue;
    bool conv_has_nonzero_pads = false;
    {
      Ort::ConstOpAttr a{nullptr};
      if (conv.GetAttributeByName("pads", a).IsOK() && a != nullptr) {
        std::vector<int64_t> cp;
        if (a.GetValueArray(cp).IsOK()) {
          for (int64_t p : cp) if (p != 0) { conv_has_nonzero_pads = true; break; }
        }
      }
    }
    if (conv_has_nonzero_pads) continue;

    // Find the Pad node producing the Conv's data input.
    const std::string data_input_name = conv_inputs[0].GetName();
    Ort::ConstNode pad = gi.ProducerOf(data_input_name);
    if (pad == nullptr || std::string(pad.GetOperatorType()) != "Pad") continue;
    const std::string pad_dom = pad.GetDomain();
    if (!pad_dom.empty() && pad_dom != "ai.onnx") continue;

    // Pad must be constant mode with value 0.
    std::string mode = "constant";
    {
      Ort::ConstOpAttr a{nullptr};
      if (pad.GetAttributeByName("mode", a).IsOK() && a != nullptr) a.GetValue(mode);
    }
    if (mode != "constant") continue;

    const auto pad_inputs = pad.GetInputs();
    if (pad_inputs.size() >= 3 && pad_inputs[2] != nullptr) {
      auto it = constants.find(std::string(pad_inputs[2].GetName()));
      if (it == constants.end()) continue;
      if (it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float val = 0.f;
        std::memcpy(&val, it->second.data.data(), sizeof(float));
        if (val != 0.f) continue;
      }
    }

    // Read the pads vector; must be 4D (8 elements).
    const auto pads = ReadPadNodePads(pad, constants);
    if (!pads.has_value() || pads->size() != 8) continue;

    // No N or C padding, symmetric H and W padding only.
    if ((*pads)[0] != 0 || (*pads)[1] != 0 || (*pads)[4] != 0 || (*pads)[5] != 0) continue;
    if ((*pads)[2] != (*pads)[6] || (*pads)[3] != (*pads)[7]) continue;
    const int h_pad = static_cast<int>((*pads)[2]);
    const int w_pad = static_cast<int>((*pads)[3]);

    // The Pad output must be consumed only by this Conv.
    if (gi.ConsumerCount(data_input_name) != 1) continue;

    // Don't absorb a Pad that another fusion already claimed.
    const std::string pad_key = NodeKey(pad);
    if (plan.consumed_nodes.count(pad_key) > 0) continue;

    plan.consumed_nodes.insert(pad_key);
    FusionPlan::AbsorbedPad absorbed;
    absorbed.p0 = w_pad;
    absorbed.p1 = h_pad;
    absorbed.data_input_name = std::string(pad_inputs[0].GetName());
    plan.absorbed_pads.emplace(NodeKey(conv), std::move(absorbed));
  }
}

void DetectConvBatchNormFusions(const GraphIndex& gi,
                                ConstantValueMap& constants,
                                FusionPlan& plan) {
  for (Ort::ConstNode bn : gi.nodes) {
    const std::string op = bn.GetOperatorType();
    if (op != "BatchNormalization") continue;
    const std::string dom = bn.GetDomain();
    if (!dom.empty() && dom != "ai.onnx") continue;
    const auto bn_inputs = bn.GetInputs();
    const auto bn_outputs = bn.GetOutputs();
    if (bn_inputs.size() != 5 || bn_outputs.size() != 1 ||
        bn_inputs[0] == nullptr || bn_inputs[1] == nullptr || bn_inputs[2] == nullptr ||
        bn_inputs[3] == nullptr || bn_inputs[4] == nullptr || bn_outputs[0] == nullptr) {
      continue;
    }

    const std::string conv_out_name = bn_inputs[0].GetName();
    if (gi.ConsumerCount(conv_out_name) != 1) continue;
    if (gi.IsGraphOutput(conv_out_name)) continue;

    Ort::ConstNode conv = gi.ProducerOf(conv_out_name);
    if (conv == nullptr || std::string(conv.GetOperatorType()) != "Conv") continue;
    const std::string conv_dom = conv.GetDomain();
    if (!conv_dom.empty() && conv_dom != "ai.onnx") continue;

    const std::string conv_key = NodeKey(conv);
    const std::string bn_key = NodeKey(bn);
    if (plan.consumed_nodes.count(conv_key) > 0 || plan.consumed_nodes.count(bn_key) > 0) continue;

    const auto conv_inputs = conv.GetInputs();
    if ((conv_inputs.size() != 2 && conv_inputs.size() != 3) ||
        conv_inputs[1] == nullptr) {
      continue;
    }

    const std::string weight_name = std::string(conv_inputs[1].GetName());
    auto wit = constants.find(weight_name);
    if (wit == constants.end() ||
        wit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      continue;
    }
    const TensorMetadata w_meta = getTensorMetadata(conv_inputs[1]);
    if (w_meta.dims.empty() || w_meta.dims[0] <= 0) continue;
    const size_t out_channels = static_cast<size_t>(w_meta.dims[0]);

    const std::string scale_name = std::string(bn_inputs[1].GetName());
    const std::string beta_name = std::string(bn_inputs[2].GetName());
    const std::string mean_name = std::string(bn_inputs[3].GetName());
    const std::string var_name = std::string(bn_inputs[4].GetName());
    auto sit = constants.find(scale_name);
    auto bit = constants.find(beta_name);
    auto mit = constants.find(mean_name);
    auto vit = constants.find(var_name);
    if (sit == constants.end() || bit == constants.end() || mit == constants.end() || vit == constants.end()) {
      continue;
    }
    if (sit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        bit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        mit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        vit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      continue;
    }
    if (gi.ConsumerCount(weight_name) != 1 ||
        gi.ConsumerCount(scale_name) != 1 ||
        gi.ConsumerCount(beta_name) != 1 ||
        gi.ConsumerCount(mean_name) != 1 ||
        gi.ConsumerCount(var_name) != 1) {
      continue;
    }

    const std::vector<float> scale = ReadTypedData<float>(sit->second);
    const std::vector<float> beta = ReadTypedData<float>(bit->second);
    const std::vector<float> mean = ReadTypedData<float>(mit->second);
    const std::vector<float> var = ReadTypedData<float>(vit->second);
    if (scale.size() != out_channels || beta.size() != out_channels ||
        mean.size() != out_channels || var.size() != out_channels) {
      continue;
    }

    std::vector<float> conv_bias(out_channels, 0.0f);
    std::string folded_bias_name = beta_name;
    if (conv_inputs.size() == 3 && conv_inputs[2] != nullptr) {
      folded_bias_name = std::string(conv_inputs[2].GetName());
      auto cbit = constants.find(folded_bias_name);
      if (cbit == constants.end() ||
          cbit->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          gi.ConsumerCount(folded_bias_name) != 1) {
        continue;
      }
      conv_bias = ReadTypedData<float>(cbit->second);
      if (conv_bias.size() != out_channels) continue;
    }

    std::vector<float> folded_w = ReadTypedData<float>(wit->second);
    if (folded_w.empty() || folded_w.size() % out_channels != 0) continue;
    const size_t kernel_elems_per_oc = folded_w.size() / out_channels;
    std::vector<float> folded_b(out_channels, 0.0f);

    float epsilon = 1e-5f;
    {
      Ort::ConstOpAttr attr{nullptr};
      if (bn.GetAttributeByName("epsilon", attr).IsOK() && attr != nullptr) {
        attr.GetValue(epsilon);
      }
    }
    for (size_t oc = 0; oc < out_channels; ++oc) {
      const float inv_std = 1.0f / std::sqrt(var[oc] + epsilon);
      const float a = scale[oc] * inv_std;
      const size_t base = oc * kernel_elems_per_oc;
      for (size_t i = 0; i < kernel_elems_per_oc; ++i) {
        folded_w[base + i] *= a;
      }
      folded_b[oc] = beta[oc] + (conv_bias[oc] - mean[oc]) * a;
    }

    wit->second = MakeConstant(std::vector<int64_t>(wit->second.dims),
                               std::move(folded_w),
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    constants[folded_bias_name] =
        MakeConstant(std::vector<int64_t>{static_cast<int64_t>(out_channels)},
                     std::move(folded_b),
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    plan.consumed_nodes.insert(bn_key);
    FusionPlan::FoldedBatchNorm folded;
    folded.output_value_name = std::string(bn_outputs[0].GetName());
    folded.weight_input_name = weight_name;
    folded.bias_input_name = folded_bias_name;
    plan.folded_batchnorms.emplace(conv_key, std::move(folded));
  }
}

// Look up a tensor's shape in our maps + ORT's declared shape. Returns
// nullopt if the tensor is not known statically. Internal helper used by
// shape-propagation rules below — distinct from the public ResolveShape(),
// which works off an Ort::ConstValueInfo.
std::optional<std::vector<int64_t>> LookupInferredShape(
    const std::string& name,
    const MetaAnalysis& analysis,
    const std::unordered_map<std::string, std::vector<int64_t>>& ort_static_shapes) {
  auto it = analysis.inferred_shapes.find(name);
  if (it != analysis.inferred_shapes.end()) return it->second;
  auto oit = ort_static_shapes.find(name);
  if (oit != ort_static_shapes.end()) return oit->second;
  auto cit = analysis.constants.find(name);
  if (cit != analysis.constants.end()) return cit->second.dims;
  return std::nullopt;
}

bool IsShapeFullyStatic(const std::vector<int64_t>& dims) {
  for (int64_t d : dims) {
    if (d < 0) return false;
  }
  return true;
}

// Extract `pads` for Pad node across opset versions. Opset >= 11 passes pads
// as input[1]; earlier opsets used a `pads` attribute. Returns nullopt if not
// resolvable as a fully-known int64 vector.
std::optional<std::vector<int64_t>> ReadPadPadsConstant(
    Ort::ConstNode node, const ConstantValueMap& constants) {
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto pads = ReadConstantInt64Input(constants, node, 1);
    if (pads.has_value()) return pads;
  }
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("pads", attr);
  if (status.IsOK() && attr != nullptr) {
    std::vector<int64_t> values;
    Ort::ThrowOnError(attr.GetValueArray(values));
    return values;
  }
  return std::nullopt;
}

// Shape rule: Pad.
// out[d] = in[d] + pads[d] + pads[d + rank]
// pads layout is ONNX-standard: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
std::optional<std::vector<int64_t>> InferPadOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_dims,
    const ConstantValueMap& constants) {
  const auto pads = ReadPadPadsConstant(node, constants);
  if (!pads.has_value()) return std::nullopt;
  const size_t rank = input_dims.size();
  if (pads->size() != rank * 2) return std::nullopt;
  std::vector<int64_t> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim = input_dims[i] + (*pads)[i] + (*pads)[i + rank];
    if (dim < 0) return std::nullopt;
    out[i] = dim;
  }
  return out;
}

// Shape rule: Reshape. Reuses ResolveReshapeTarget, but only when both the
// target shape constant and the input's concrete shape are known. `allowzero`
// (opset >= 14) disables the 0-copy-from-input rule — we currently don't
// fold when allowzero=1; treat it as not-yet-implemented.
std::optional<std::vector<int64_t>> InferReshapeOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& target) {
  // Reject if any target entry is the dynamic-dim sentinel (from a folded
  // Shape of a dynamic-dim tensor).
  for (int64_t v : target) {
    if (v == kUnknownShapeDimSentinel) return std::nullopt;
  }
  // Respect allowzero when present: a 0 then means literal zero, not copy.
  if (const auto allowzero = ReadInt64Attr(node, "allowzero");
      allowzero.has_value() && *allowzero != 0) {
    return std::nullopt;
  }
  return ResolveReshapeTarget(target, input_dims);
}

// Shape rule: Transpose.
std::optional<std::vector<int64_t>> InferTransposeOutputShape(
    Ort::ConstNode node, const std::vector<int64_t>& input_dims) {
  const int64_t rank = static_cast<int64_t>(input_dims.size());
  std::vector<int64_t> perm;
  if (const auto p = ReadTransposePerm(node)) {
    perm = *p;
  } else {
    perm.resize(rank);
    for (int64_t i = 0; i < rank; ++i) perm[i] = rank - 1 - i;
  }
  if (static_cast<int64_t>(perm.size()) != rank) return std::nullopt;
  std::vector<int64_t> out(rank);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t ax = perm[i];
    if (ax < 0 || ax >= rank) return std::nullopt;
    out[i] = input_dims[ax];
  }
  return out;
}

// Read axes for Squeeze/Unsqueeze across opset versions: attribute pre-opset
// 13, tensor input[1] on >= 13. Returns nullopt if the axes aren't known.
// Empty optional vector means "axes omitted" (only valid for Squeeze).
std::optional<std::vector<int64_t>> ReadAxesAttrOrInput(
    Ort::ConstNode node, const ConstantValueMap& constants) {
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto axes = ReadConstantInt64Input(constants, node, 1);
    if (axes.has_value()) return axes;
    return std::nullopt;
  }
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("axes", attr);
  if (status.IsOK() && attr != nullptr) {
    std::vector<int64_t> values;
    Ort::ThrowOnError(attr.GetValueArray(values));
    return values;
  }
  // Squeeze allows axes to be absent — caller distinguishes.
  return std::vector<int64_t>{};
}

std::optional<std::vector<int64_t>> InferUnsqueezeOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_dims,
    const ConstantValueMap& constants) {
  const auto axes_opt = ReadAxesAttrOrInput(node, constants);
  if (!axes_opt.has_value()) return std::nullopt;
  if (axes_opt->empty()) return std::nullopt;  // Unsqueeze requires axes.
  const int64_t out_rank = static_cast<int64_t>(input_dims.size() + axes_opt->size());
  std::vector<int64_t> axes = *axes_opt;
  for (auto& a : axes) if (a < 0) a += out_rank;
  std::unordered_set<int64_t> axes_set(axes.begin(), axes.end());
  if (axes_set.size() != axes.size()) return std::nullopt;
  std::vector<int64_t> out;
  out.reserve(out_rank);
  size_t in_cursor = 0;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (axes_set.count(i)) out.push_back(1);
    else {
      if (in_cursor >= input_dims.size()) return std::nullopt;
      out.push_back(input_dims[in_cursor++]);
    }
  }
  return out;
}

std::optional<std::vector<int64_t>> InferSqueezeOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_dims,
    const ConstantValueMap& constants) {
  const auto axes_opt = ReadAxesAttrOrInput(node, constants);
  if (!axes_opt.has_value()) return std::nullopt;
  const int64_t in_rank = static_cast<int64_t>(input_dims.size());
  std::unordered_set<int64_t> squeeze;
  if (axes_opt->empty()) {
    // Squeeze every size-1 axis.
    for (int64_t i = 0; i < in_rank; ++i) {
      if (input_dims[i] == 1) squeeze.insert(i);
    }
  } else {
    for (int64_t a : *axes_opt) {
      if (a < 0) a += in_rank;
      if (a < 0 || a >= in_rank) return std::nullopt;
      if (input_dims[a] != 1) return std::nullopt;
      squeeze.insert(a);
    }
  }
  std::vector<int64_t> out;
  out.reserve(in_rank - static_cast<int64_t>(squeeze.size()));
  for (int64_t i = 0; i < in_rank; ++i) {
    if (!squeeze.count(i)) out.push_back(input_dims[i]);
  }
  return out;
}

// Shape rule: Concat. All input shapes must be known, axes resolved and
// non-concat dims identical; output dim on the concat axis is the sum.
std::optional<std::vector<int64_t>> InferConcatOutputShape(
    Ort::ConstNode node,
    const std::vector<std::optional<std::vector<int64_t>>>& input_shapes) {
  if (input_shapes.empty()) return std::nullopt;
  for (const auto& s : input_shapes) if (!s.has_value()) return std::nullopt;
  const int64_t rank = static_cast<int64_t>(input_shapes[0]->size());
  auto axis_opt = ReadInt64Attr(node, "axis");
  if (!axis_opt.has_value()) return std::nullopt;
  int64_t axis = *axis_opt;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return std::nullopt;
  std::vector<int64_t> out = *input_shapes[0];
  for (size_t i = 1; i < input_shapes.size(); ++i) {
    const auto& s = *input_shapes[i];
    if (static_cast<int64_t>(s.size()) != rank) return std::nullopt;
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis) out[d] += s[d];
      else if (s[d] != out[d]) return std::nullopt;
    }
  }
  return out;
}

// Shape rule: Split. Produces one fully-static shape per output when the input
// shape and split sizes are known.
std::optional<std::vector<std::vector<int64_t>>> InferSplitOutputShapes(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_shape,
    const ConstantValueMap& constants) {
  const auto outputs = node.GetOutputs();
  if (outputs.empty()) return std::nullopt;
  const int64_t rank = static_cast<int64_t>(input_shape.size());
  if (rank == 0) return std::nullopt;

  int64_t axis = ReadInt64Attr(node, "axis").value_or(0);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return std::nullopt;
  const int64_t axis_dim = input_shape[static_cast<size_t>(axis)];
  if (axis_dim < 0) return std::nullopt;

  const auto inputs = node.GetInputs();
  std::vector<int64_t> split_sizes;
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto splits = ReadConstantInt64Input(constants, node, 1);
    if (!splits.has_value()) return std::nullopt;
    split_sizes = std::move(*splits);
  } else {
    Ort::ConstOpAttr attr{nullptr};
    const Ort::Status status = node.GetAttributeByName("split", attr);
    if (status.IsOK() && attr != nullptr) {
      Ort::ThrowOnError(attr.GetValueArray(split_sizes));
    } else {
      const int64_t n = static_cast<int64_t>(outputs.size());
      if (n <= 0 || axis_dim % n != 0) return std::nullopt;
      split_sizes.assign(static_cast<size_t>(n), axis_dim / n);
    }
  }
  if (split_sizes.size() != outputs.size()) return std::nullopt;
  int64_t total = 0;
  for (int64_t s : split_sizes) {
    if (s < 0) return std::nullopt;
    total += s;
  }
  if (total != axis_dim) return std::nullopt;

  std::vector<std::vector<int64_t>> out_shapes;
  out_shapes.reserve(outputs.size());
  for (int64_t s : split_sizes) {
    auto out = input_shape;
    out[static_cast<size_t>(axis)] = s;
    out_shapes.push_back(std::move(out));
  }
  return out_shapes;
}

// Shape rule: Slice (float or int tensors). Mirrors CompileSliceAttributes
// clamping. All of starts/ends/axes/steps must be constants.
std::optional<std::vector<int64_t>> InferSliceOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& input_dims,
    const ConstantValueMap& constants) {
  const auto starts = ReadConstantInt64Input(constants, node, 1);
  const auto ends = ReadConstantInt64Input(constants, node, 2);
  if (!starts || !ends || starts->size() != ends->size() || starts->empty()) return std::nullopt;
  const int64_t rank = static_cast<int64_t>(input_dims.size());
  std::vector<int64_t> axes;
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    auto a = ReadConstantInt64Input(constants, node, 3);
    if (!a) return std::nullopt;
    axes = *a;
  } else {
    axes.resize(starts->size());
    for (size_t i = 0; i < starts->size(); ++i) axes[i] = static_cast<int64_t>(i);
  }
  std::vector<int64_t> steps(starts->size(), 1);
  if (inputs.size() == 5 && inputs[4] != nullptr) {
    auto s = ReadConstantInt64Input(constants, node, 4);
    if (!s) return std::nullopt;
    steps = *s;
  }
  std::vector<int64_t> out = input_dims;
  auto clamp = [](int64_t v, int64_t lo, int64_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  };
  for (size_t k = 0; k < axes.size(); ++k) {
    int64_t ax = axes[k];
    if (ax < 0) ax += rank;
    if (ax < 0 || ax >= rank) return std::nullopt;
    const int64_t dim = input_dims[ax];
    int64_t s = (*starts)[k];
    int64_t e = (*ends)[k];
    const int64_t step = steps[k];
    if (step == 0) return std::nullopt;
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    if (step > 0) { s = clamp(s, 0, dim); e = clamp(e, 0, dim); }
    else          { s = clamp(s, 0, dim - 1); e = clamp(e, -1, dim - 1); }
    int64_t length = 0;
    if (step > 0 && e > s) length = (e - s + step - 1) / step;
    else if (step < 0 && s > e) length = (s - e + (-step) - 1) / (-step);
    out[ax] = length;
  }
  return out;
}

// Shape rule: Gather. out = in.shape[:axis] ++ indices.shape ++ in.shape[axis+1:]
std::optional<std::vector<int64_t>> InferGatherOutputShape(
    Ort::ConstNode node,
    const std::vector<int64_t>& data_dims,
    const std::vector<int64_t>& indices_dims) {
  const int64_t rank = static_cast<int64_t>(data_dims.size());
  int64_t axis = ReadInt64Attr(node, "axis").value_or(0);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return std::nullopt;
  std::vector<int64_t> out;
  out.reserve(rank - 1 + indices_dims.size());
  for (int64_t d = 0; d < axis; ++d) out.push_back(data_dims[d]);
  for (int64_t d : indices_dims) out.push_back(d);
  for (int64_t d = axis + 1; d < rank; ++d) out.push_back(data_dims[d]);
  return out;
}

// Shape rule: Expand. out = broadcast(in, shape-arg). Follows ONNX bidirectional
// broadcasting — dims are prefixed with 1 to match the longer rank, then each
// axis takes max(lhs, rhs).
std::optional<std::vector<int64_t>> InferExpandOutputShape(
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& target_dims) {
  for (int64_t v : target_dims) {
    if (v == kUnknownShapeDimSentinel || v < 0) return std::nullopt;
  }
  const size_t rank = std::max(input_dims.size(), target_dims.size());
  std::vector<int64_t> a(rank, 1), b(rank, 1);
  std::copy(input_dims.begin(), input_dims.end(),
            a.begin() + (rank - input_dims.size()));
  std::copy(target_dims.begin(), target_dims.end(),
            b.begin() + (rank - target_dims.size()));
  std::vector<int64_t> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (a[i] == 1) out[i] = b[i];
    else if (b[i] == 1) out[i] = a[i];
    else if (a[i] == b[i]) out[i] = a[i];
    else return std::nullopt;
  }
  return out;
}

// Shape rule: Tile. out[d] = in[d] * repeats[d].
std::optional<std::vector<int64_t>> InferTileOutputShape(
    Ort::ConstNode /*node*/,
    const std::vector<int64_t>& input_dims,
    const ConstantValueMap& constants,
    Ort::ConstNode raw_node) {
  const auto repeats = ReadConstantInt64Input(constants, raw_node, 1);
  if (!repeats.has_value()) return std::nullopt;
  if (repeats->size() != input_dims.size()) return std::nullopt;
  std::vector<int64_t> out(input_dims.size());
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if ((*repeats)[i] < 0) return std::nullopt;
    out[i] = input_dims[i] * (*repeats)[i];
  }
  return out;
}

// Shape rule: ConstantOfShape. input[0] is the shape arg (must be in constants).
std::optional<std::vector<int64_t>> InferConstantOfShapeOutputShape(
    const std::vector<int64_t>& shape) {
  for (int64_t v : shape) {
    if (v == kUnknownShapeDimSentinel || v < 0) return std::nullopt;
  }
  return shape;
}

// NumPy-style broadcast over an arbitrary number of input shapes. Returns the
// broadcast result shape, or nullopt if any input is missing or shapes are
// incompatible.
std::optional<std::vector<int64_t>> InferBroadcastShape(
    const std::vector<std::optional<std::vector<int64_t>>>& shapes) {
  if (shapes.empty()) return std::nullopt;
  size_t out_rank = 0;
  for (const auto& s : shapes) {
    if (!s.has_value()) return std::nullopt;
    out_rank = std::max(out_rank, s->size());
  }
  std::vector<int64_t> out(out_rank, 1);
  for (const auto& s : shapes) {
    const size_t pad = out_rank - s->size();
    for (size_t i = 0; i < s->size(); ++i) {
      const int64_t d = (*s)[i];
      if (d < 0) return std::nullopt;
      int64_t& o = out[pad + i];
      if (o == 1) o = d;
      else if (d != 1 && d != o) return std::nullopt;
    }
  }
  return out;
}

// Range output shape: ceil(max(0, limit - start) / delta). Requires all three
// scalar inputs to be known constants.
std::optional<std::vector<int64_t>> InferRangeOutputShape(
    Ort::ConstNode node,
    const ConstantValueMap& constants) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 3) return std::nullopt;
  auto read_scalar = [&](size_t i) -> std::optional<double> {
    auto t = LookupConstant(constants, inputs[i]);
    if (!t.has_value() || ElementCount(t->dims) != 1) return std::nullopt;
    if (t->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return static_cast<double>(*reinterpret_cast<const int64_t*>(t->data.data()));
    }
    if (t->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return static_cast<double>(*reinterpret_cast<const int32_t*>(t->data.data()));
    }
    if (t->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return static_cast<double>(*reinterpret_cast<const float*>(t->data.data()));
    }
    return std::nullopt;
  };
  auto start = read_scalar(0);
  auto limit = read_scalar(1);
  auto delta = read_scalar(2);
  if (!start || !limit || !delta || *delta == 0.0) return std::nullopt;
  const double n = std::max(0.0, std::ceil((*limit - *start) / *delta));
  return std::vector<int64_t>{static_cast<int64_t>(n)};
}

std::optional<std::vector<int64_t>> ResolveShapeRuleInt64Input(
    const ConstantValueMap& constants,
    Ort::ConstNode node,
    size_t input_idx,
    const std::function<std::optional<std::vector<int64_t>>(const std::string&)>& lookup_shape) {
  const auto inputs = node.GetInputs();
  if (input_idx >= inputs.size() || inputs[input_idx] == nullptr) return std::nullopt;
  const auto tensor = LookupConstant(constants, inputs[input_idx]);
  if (!tensor.has_value() ||
      tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return std::nullopt;
  }

  const size_t count = ElementCount(tensor->dims);
  std::vector<int64_t> values(count, 0);
  if (count != 0) {
    const int64_t* src = reinterpret_cast<const int64_t*>(tensor->data.data());
    GGONNX_NOT_NULL(src, "resolved INT64 shape-rule input data must not be null");
    values.assign(src, src + count);
  }
  if (tensor->dim_bindings.empty()) return values;

  GGONNX_ASSERT(tensor->dim_bindings.size() == count,
                "dim_bindings length mismatch in shape-rule input");
  for (size_t i = 0; i < count; ++i) {
    const auto& binding = tensor->dim_bindings[i];
    if (!binding.has_value()) continue;
    const auto source_shape = lookup_shape(binding->source_name);
    if (!source_shape.has_value()) return std::nullopt;
    if (binding->axis < 0 || static_cast<size_t>(binding->axis) >= source_shape->size()) {
      return std::nullopt;
    }
    values[i] = (*source_shape)[static_cast<size_t>(binding->axis)];
  }
  return values;
}

// ===========================================================================
//  MetaOp registry — single source of truth for per-op folding + shape rules.
//
//  TryFoldNode and PropagateInferredShapes both dispatch through this table.
//  Adding an op means adding one row to kMetaOps: the registry is iterable
//  (GetMetaOpRegistry) and lookup is a single hash probe (FindMetaOp).
//
//  Entries are sparse: either function pointer may be null. A fold-only op
//  leaves shape_fn null; a shape-only op leaves fold_fn null.
// ===========================================================================

struct ShapeRuleResult {
  std::optional<std::vector<int64_t>> out_shape;              // single-output rules
  std::optional<std::vector<std::vector<int64_t>>> per_output; // multi-output (Split)
};

// Per-node scratch bundle passed to every shape rule. Built once per node by
// PropagateInferredShapes; rules read from it and never mutate.
struct ShapeRuleCtx {
  Ort::ConstNode node;
  const std::vector<Ort::ConstValueInfo>& inputs;
  const ConstantValueMap& constants;
  // Look up the fully-static shape of input[idx]; nullopt if not resolvable.
  std::function<std::optional<std::vector<int64_t>>(size_t)> get_in_shape;
  // Resolve shape of a tensor by name (constant-valued shape tensors etc.).
  std::function<std::optional<std::vector<int64_t>>(const std::string&)> lookup_shape_by_name;
};

using FoldFn = std::optional<ConstantTensor> (*)(const ConstantValueMap&, Ort::ConstNode);
using ShapeFn = ShapeRuleResult (*)(const ShapeRuleCtx&);

struct MetaOp {
  std::string_view op_type;  // all ops currently live in domain ""
  FoldFn fold_fn;            // nullable
  ShapeFn shape_fn;          // nullable
};

// --- Shape rule adapters --------------------------------------------------
// Each adapter unpacks ShapeRuleCtx into the call signature of the
// corresponding Infer* function. Kept tiny and data-only so the registry
// table below reads as a flat index of supported ops.

ShapeRuleResult ShapeRulePad(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferPadOutputShape(c.node, *in, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleReshape(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  auto target = ResolveShapeRuleInt64Input(c.constants, c.node, 1, c.lookup_shape_by_name);
  if (in && target) r.out_shape = InferReshapeOutputShape(c.node, *in, *target);
  return r;
}
ShapeRuleResult ShapeRuleTranspose(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferTransposeOutputShape(c.node, *in);
  return r;
}
ShapeRuleResult ShapeRuleUnsqueeze(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferUnsqueezeOutputShape(c.node, *in, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleSqueeze(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferSqueezeOutputShape(c.node, *in, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleSlice(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferSliceOutputShape(c.node, *in, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleExpand(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  auto target = ResolveShapeRuleInt64Input(c.constants, c.node, 1, c.lookup_shape_by_name);
  if (in && target) r.out_shape = InferExpandOutputShape(*in, *target);
  return r;
}
ShapeRuleResult ShapeRuleTile(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.out_shape = InferTileOutputShape(c.node, *in, c.constants, c.node);
  return r;
}
ShapeRuleResult ShapeRuleIdentityShape(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  r.out_shape = c.get_in_shape(0);
  return r;
}
ShapeRuleResult ShapeRuleGather(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto data_shape = c.get_in_shape(0);
  auto idx_shape = c.get_in_shape(1);
  if (data_shape && idx_shape) r.out_shape = InferGatherOutputShape(c.node, *data_shape, *idx_shape);
  return r;
}
ShapeRuleResult ShapeRuleConcat(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  std::vector<std::optional<std::vector<int64_t>>> shapes;
  shapes.reserve(c.inputs.size());
  for (size_t i = 0; i < c.inputs.size(); ++i) shapes.push_back(c.get_in_shape(i));
  r.out_shape = InferConcatOutputShape(c.node, shapes);
  return r;
}
ShapeRuleResult ShapeRuleSplit(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto in = c.get_in_shape(0);
  if (in) r.per_output = InferSplitOutputShapes(c.node, *in, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleBroadcast(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  std::vector<std::optional<std::vector<int64_t>>> shapes;
  shapes.reserve(c.inputs.size());
  for (size_t i = 0; i < c.inputs.size(); ++i) shapes.push_back(c.get_in_shape(i));
  r.out_shape = InferBroadcastShape(shapes);
  return r;
}
ShapeRuleResult ShapeRuleRange(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  r.out_shape = InferRangeOutputShape(c.node, c.constants);
  return r;
}
ShapeRuleResult ShapeRuleConstantOfShape(const ShapeRuleCtx& c) {
  ShapeRuleResult r;
  auto shape = ResolveShapeRuleInt64Input(c.constants, c.node, 0, c.lookup_shape_by_name);
  if (shape) r.out_shape = InferConstantOfShapeOutputShape(*shape);
  return r;
}

// --- The registry table ---------------------------------------------------
// Rows are ordered by category: trivial / shape-manip / elementwise /
// reductions-and-generators. Broadcasted elementwise ops share one shape fn
// (ShapeRuleBroadcast) across 13 op types — spelled out explicitly instead
// of being hidden in an if-chain cluster.

const std::vector<MetaOp>& GetMetaOpRegistry() {
  static const std::vector<MetaOp> kMetaOps = {
      // Trivial / pass-through
      {"Constant",        EvalConstant,        nullptr},
      {"Shape",           EvalShape,           nullptr},
      {"Cast",            EvalCast,            ShapeRuleIdentityShape},
      {"Identity",        EvalIdentity,        ShapeRuleIdentityShape},
      // Shape-manipulating folds and shape rules
      {"Reshape",         EvalReshape,         ShapeRuleReshape},
      {"Transpose",       EvalTranspose,       ShapeRuleTranspose},
      {"Unsqueeze",       EvalUnsqueeze,       ShapeRuleUnsqueeze},
      {"Squeeze",         EvalSqueeze,         ShapeRuleSqueeze},
      {"Slice",           EvalSlice,           ShapeRuleSlice},
      {"Expand",          nullptr,             ShapeRuleExpand},
      {"Tile",            EvalTile,            ShapeRuleTile},
      {"Concat",          EvalConcat,          ShapeRuleConcat},
      {"Gather",          EvalGather,          ShapeRuleGather},
      {"Split",           nullptr,             ShapeRuleSplit},
      {"Pad",             nullptr,             ShapeRulePad},
      // Elementwise — broadcast shape
      {"Add",             EvalBinary,          ShapeRuleBroadcast},
      {"Sub",             EvalBinary,          ShapeRuleBroadcast},
      {"Mul",             EvalBinary,          ShapeRuleBroadcast},
      {"Div",             EvalBinary,          ShapeRuleBroadcast},
      {"Max",             nullptr,             ShapeRuleBroadcast},
      {"Min",             nullptr,             ShapeRuleBroadcast},
      {"Pow",             nullptr,             ShapeRuleBroadcast},
      {"Equal",           EvalEqual,           ShapeRuleBroadcast},
      {"Greater",         nullptr,             ShapeRuleBroadcast},
      {"GreaterOrEqual",  nullptr,             ShapeRuleBroadcast},
      {"Less",            nullptr,             ShapeRuleBroadcast},
      {"LessOrEqual",     nullptr,             ShapeRuleBroadcast},
      {"And",             nullptr,             ShapeRuleBroadcast},
      {"Or",              nullptr,             ShapeRuleBroadcast},
      {"Where",           EvalWhere,           ShapeRuleBroadcast},
      // Unary math
      {"Ceil",            EvalUnaryMath,       nullptr},
      {"Round",           EvalUnaryMath,       nullptr},
      // Reductions and generators
      {"ReduceMin",       EvalReduceMin,       nullptr},
      {"Range",           EvalRange,           ShapeRuleRange},
      {"ConstantOfShape", EvalConstantOfShape, ShapeRuleConstantOfShape},
  };
  return kMetaOps;
}

const MetaOp* FindMetaOp(std::string_view op_type) {
  static const std::unordered_map<std::string_view, const MetaOp*> kByName = [] {
    std::unordered_map<std::string_view, const MetaOp*> m;
    const auto& ops = GetMetaOpRegistry();
    m.reserve(ops.size());
    for (const auto& op : ops) m.emplace(op.op_type, &op);
    return m;
  }();
  const auto it = kByName.find(op_type);
  return it == kByName.end() ? nullptr : it->second;
}

// Table-driven fold dispatch. Replaces the former if-chain — a new fold op
// only requires setting fold_fn on its registry row.
std::optional<ConstantTensor> TryFoldNode(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto* op = FindMetaOp(node.GetOperatorType());
  if (!op || op->fold_fn == nullptr) return std::nullopt;
  return op->fold_fn(constants, node);
}

// --- Fusion passes --------------------------------------------------------
// Isolated behind a named list so they iterate like every other pass. The
// five Detect* bodies are unchanged — the wrappers below only adapt their
// varying signatures to the uniform (graph, analysis) shape.

struct FusionPass {
  std::string_view name;
  void (*run)(const GraphIndex&, MetaAnalysis&);
};

void RunShuffleTripleFusions(const GraphIndex& gi, MetaAnalysis& a) {
  DetectShuffleTripleFusions(gi, a.constants, a.fusions);
}
void RunWindowMaskAddFusions(const GraphIndex& gi, MetaAnalysis& a) {
  DetectWindowMaskAddFusions(gi, a.constants, a.folded_nodes, a.fusions);
}
void RunQKVSplitFusions(const GraphIndex& gi, MetaAnalysis& a) {
  DetectQKVSplitFusions(gi, a.constants, a.folded_nodes, a.fusions);
}
void RunPadConvFusions(const GraphIndex& gi, MetaAnalysis& a) {
  DetectPadConvFusions(gi, a.constants, a.fusions);
}
void RunConvBatchNormFusions(const GraphIndex& gi, MetaAnalysis& a) {
  DetectConvBatchNormFusions(gi, a.constants, a.fusions);
}

// Order matters: WindowMaskAdd must precede QKVSplit so the mask-apply
// rank-5 bubble is fused out before QKVSplit's span check runs (if run
// after, the bubble inflates the span past the match threshold).
inline constexpr std::array<FusionPass, 5> kFusionPasses = {{
    {"ShuffleTriple", RunShuffleTripleFusions},
    {"WindowMaskAdd", RunWindowMaskAddFusions},
    {"QKVSplit",      RunQKVSplitFusions},
    {"PadConv",       RunPadConvFusions},
    {"ConvBatchNorm", RunConvBatchNormFusions},
}};

void PropagateInferredShapes(const Ort::ConstGraph& ort_graph, MetaAnalysis& analysis) {
  const bool trace = std::getenv("GGONNX_TRACE_SHAPE_PROP") != nullptr;
  std::unordered_map<std::string, size_t> rule_fired;
  std::unordered_map<std::string, size_t> rule_attempted;
  // Seed with ORT's fully-static shapes from inputs and value_info.
  std::unordered_map<std::string, std::vector<int64_t>> ort_static;
  auto ingest = [&](Ort::ConstValueInfo vi) {
    if (vi == nullptr) return;
    const std::string name = vi.GetName();
    if (ort_static.count(name)) return;
    // Fall back to the initializer body if this tensor has one; weight
    // tensors often carry a full shape on the value but only a partial
    // one on the value_info. Initializer shapes are authoritative — even
    // an empty-dim initializer is a genuine rank-0 scalar.
    try {
      Ort::ConstValue value{nullptr};
      if (vi.GetInitializer(value).IsOK() && value != nullptr) {
        const TensorMetadata meta = getTensorMetadata(value);
        if (IsShapeFullyStatic(meta.dims)) {
          ort_static.emplace(name, meta.dims);
          return;
        }
      }
    } catch (...) {}
    // Value_info declared shape. Skip empty-dim declarations on non-initializer
    // value_infos: ORT's shape-inference frequently emits `shape: []` for
    // tensors whose true shape it failed to trace (Loop/Scan bodies, etc.),
    // which is indistinguishable from a genuine rank-0 scalar at this API
    // level. Trusting it lets a bogus scalar propagate into Reshape(-1) and
    // collapse a real [L] tensor to [1].
    try {
      const TensorMetadata meta = getTensorMetadata(vi);
      if (!meta.dims.empty() && IsShapeFullyStatic(meta.dims)) {
        ort_static.emplace(name, meta.dims);
      }
    } catch (...) {}
  };
  for (Ort::ConstValueInfo vi : ort_graph.GetInputs()) ingest(vi);
  for (Ort::ConstValueInfo vi : ort_graph.GetOutputs()) ingest(vi);
  for (Ort::ConstValueInfo vi : ort_graph.GetInitializers()) ingest(vi);
  // Ingest intermediate tensors via every node's output value_info — that's
  // where ORT's shape inference populates concrete dims for non-IO tensors.
  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    for (Ort::ConstValueInfo vi : node.GetOutputs()) ingest(vi);
    // Also ingest node inputs — many initializers surface only through
    // node input value_infos, not graph-level GetInitializers().
    for (Ort::ConstValueInfo vi : node.GetInputs()) ingest(vi);
  }

  auto get_in_shape = [&](size_t idx,
                          const std::vector<Ort::ConstValueInfo>& inputs)
      -> std::optional<std::vector<int64_t>> {
    if (idx >= inputs.size() || inputs[idx] == nullptr) return std::nullopt;
    return LookupInferredShape(inputs[idx].GetName(), analysis, ort_static);
  };
  auto lookup_shape_by_name =
      [&](const std::string& name) -> std::optional<std::vector<int64_t>> {
    return LookupInferredShape(name, analysis, ort_static);
  };

  // Record a single shape against all outputs that still need one.
  auto record_single = [&](const std::vector<Ort::ConstValueInfo>& outputs,
                           const std::vector<int64_t>& shape) {
    if (!IsShapeFullyStatic(shape)) return;
    for (Ort::ConstValueInfo output : outputs) {
      if (output == nullptr) continue;
      const std::string out_name = output.GetName();
      if (ort_static.count(out_name) || analysis.inferred_shapes.count(out_name)) continue;
      analysis.inferred_shapes[out_name] = shape;
    }
  };
  auto record_per_output = [&](const std::vector<Ort::ConstValueInfo>& outputs,
                               const std::vector<std::vector<int64_t>>& shapes) {
    if (shapes.size() != outputs.size()) return;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i] == nullptr || !IsShapeFullyStatic(shapes[i])) continue;
      const std::string out_name = outputs[i].GetName();
      if (ort_static.count(out_name) || analysis.inferred_shapes.count(out_name)) continue;
      analysis.inferred_shapes[out_name] = shapes[i];
    }
  };

  // Walk nodes in topological order (GetNodes() returns topo order) and
  // dispatch each through the MetaOp registry.
  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    try {
      const std::string op_type = std::string(node.GetOperatorType());
      const auto* meta_op = FindMetaOp(op_type);
      if (meta_op == nullptr || meta_op->shape_fn == nullptr) continue;

      const auto inputs_rng = node.GetInputs();
      const std::vector<Ort::ConstValueInfo> inputs(inputs_rng.begin(), inputs_rng.end());
      const auto outputs_rng = node.GetOutputs();
      const std::vector<Ort::ConstValueInfo> outputs(outputs_rng.begin(), outputs_rng.end());
      if (outputs.empty()) continue;
      bool needs_shape = false;
      for (Ort::ConstValueInfo output : outputs) {
        if (output == nullptr) continue;
        const std::string out_name = output.GetName();
        if (ort_static.count(out_name) == 0 && analysis.inferred_shapes.count(out_name) == 0) {
          needs_shape = true;
          break;
        }
      }
      if (!needs_shape) continue;

      ShapeRuleCtx ctx{
          node, inputs, analysis.constants,
          [&](size_t idx) { return get_in_shape(idx, inputs); },
          lookup_shape_by_name,
      };
      ++rule_attempted[op_type];
      const ShapeRuleResult result = meta_op->shape_fn(ctx);
      const bool fired = result.out_shape.has_value() || result.per_output.has_value();
      if (fired) ++rule_fired[op_type];
      if (result.per_output.has_value()) {
        record_per_output(outputs, *result.per_output);
      } else if (result.out_shape.has_value()) {
        record_single(outputs, *result.out_shape);
      }
    } catch (...) {
      continue;
    }
  }
  if (trace) {
    std::fprintf(stderr, "[ggonnx][shape-prop] ort_static=%zu inferred=%zu;",
                 ort_static.size(), analysis.inferred_shapes.size());
    for (const auto& [op, count] : rule_attempted) {
      std::fprintf(stderr, " %s=%zu/%zu", op.c_str(), rule_fired[op], count);
    }
    std::fprintf(stderr, "\n");
  }
}

}  // namespace

std::optional<std::vector<int64_t>> ResolveShape(Ort::ConstValueInfo vi,
                                                 const MetaAnalysis& meta) {
  if (vi == nullptr) return std::nullopt;
  const TensorMetadata ort_meta = getTensorMetadata(vi);
  // Empty-dim declarations on non-initializer value_infos are ORT "I don't
  // know" in disguise — see PropagateInferredShapes::ingest for the rationale.
  // For initializers we trust empty dims as a genuine rank-0 scalar.
  const bool dims_trustworthy = !ort_meta.dims.empty() || [&]() {
    try {
      Ort::ConstValue value{nullptr};
      return vi.GetInitializer(value).IsOK() && value != nullptr;
    } catch (...) { return false; }
  }();
  if (dims_trustworthy && IsShapeFullyStatic(ort_meta.dims)) return ort_meta.dims;
  auto cit = meta.constants.find(std::string(vi.GetName()));
  if (cit != meta.constants.end() && IsShapeFullyStatic(cit->second.dims)) {
    return cit->second.dims;
  }
  auto it = meta.inferred_shapes.find(std::string(vi.GetName()));
  if (it != meta.inferred_shapes.end()) return it->second;
  return std::nullopt;
}

MetaAnalysis AnalyzeCompileTimeConstants(const OrtGraph* graph) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  const Ort::ConstGraph ort_graph{graph};
  MetaAnalysis analysis;
  const GraphIndex gi{ort_graph};

  for (Ort::ConstValueInfo input : ort_graph.GetInputs()) {
    if (input == nullptr || !input.IsConstantInitializer()) continue;
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(input.GetInitializer(value));
    if (value == nullptr) continue;
    const TensorMetadata meta = getTensorMetadata(value);
    if (!IsFoldableDType(meta.element_type) || !FitsInitializerLimit(meta.dims)) continue;
    analysis.constants.emplace(input.GetName(), LoadInitializer(value));
  }

  for (Ort::ConstValueInfo init : ort_graph.GetInitializers()) {
    if (init == nullptr || !init.IsConstantInitializer()) continue;
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(init.GetInitializer(value));
    if (value == nullptr) continue;
    const TensorMetadata meta = getTensorMetadata(value);
    if (!IsFoldableDType(meta.element_type) || !FitsInitializerLimit(meta.dims)) continue;
    analysis.constants.emplace(init.GetName(), LoadInitializer(value));
  }

  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    const auto outputs = node.GetOutputs();
    if (outputs.empty() || outputs[0] == nullptr) continue;
    try {
      if (node.GetOperatorType() == "Loop") {
        const auto folded_outputs = EvalArangeLoop(analysis.constants, node);
        if (folded_outputs.has_value()) {
          GGONNX_ASSERT(folded_outputs->size() == outputs.size(),
                        "folded Loop output count mismatch");
          analysis.folded_nodes.insert(NodeKey(node));
          for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i] == nullptr) continue;
            analysis.constants[outputs[i].GetName()] = (*folded_outputs)[i];
          }
          continue;
        }
      }
      const auto folded = TryFoldNode(analysis.constants, node);
      const auto shape_gather_folded =
          folded.has_value() ? folded : EvalShapeTensorGather(analysis.constants, gi, node);
      if (!shape_gather_folded.has_value()) continue;
      if (!folded.has_value() && node.GetOperatorType() == "Gather") {
        bool shape_only_consumers = true;
        for (Ort::ConstValueInfo output : outputs) {
          if (output == nullptr) continue;
          static const std::unordered_set<std::string> kShapeOnlyConsumers = {
              "Reshape", "Expand", "Unsqueeze", "Squeeze", "Concat", "Slice",
          };
          for (const std::string& op_type : gi.ConsumerOps(std::string(output.GetName()))) {
            if (kShapeOnlyConsumers.count(op_type) == 0) {
              shape_only_consumers = false;
              break;
            }
          }
          if (!shape_only_consumers) break;
        }
        if (!shape_only_consumers) continue;
      }
      analysis.folded_nodes.insert(NodeKey(node));
      if (!folded.has_value() && node.GetOperatorType() == "Gather") {
        const auto inputs = node.GetInputs();
        if (!inputs.empty() && inputs[0] != nullptr) {
          const std::string shape_output_name = inputs[0].GetName();
          Ort::ConstNode shape_producer = gi.ProducerOf(shape_output_name);
          if (shape_producer != nullptr &&
              shape_producer.GetOperatorType() == "Shape" &&
              gi.ConsumerCount(shape_output_name) == 1 &&
              !gi.IsGraphOutput(shape_output_name)) {
            analysis.folded_nodes.insert(NodeKey(shape_producer));
          }
        }
      }
      for (Ort::ConstValueInfo output : outputs) {
        if (output == nullptr) continue;
        analysis.constants[output.GetName()] = *shape_gather_folded;
      }
    } catch (...) {
      continue;
    }
  }
  PropagateInferredShapes(ort_graph, analysis);

  for (const FusionPass& pass : kFusionPasses) pass.run(gi, analysis);

  // A folded constant containing kUnknownShapeDimSentinel is symbolic — it only
  // makes sense as input to further constant folders that understand the
  // sentinel. If we let such a node be claimed by the EP, its sentinel value
  // would be memcpy'd straight into a downstream Reshape/Expand shape input,
  // which sees -2147483648 and raises. Drop these from folded_nodes; the
  // constants map keeps the symbolic value for analytic propagation.
  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    if (node == nullptr) continue;
    const std::string key = NodeKey(node);
    if (analysis.folded_nodes.count(key) == 0) continue;
    bool leaks_sentinel = false;
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output == nullptr) continue;
      const auto it = analysis.constants.find(output.GetName());
      if (it == analysis.constants.end()) continue;
      if (ConstantContainsSentinel(it->second)) {
        leaks_sentinel = true;
        break;
      }
    }
    if (leaks_sentinel) analysis.folded_nodes.erase(key);
  }

  // A Shape node is co-folded into folded_nodes when its sole consumer is a
  // folded Gather (see the shape-gather insertion above). If the Gather just
  // got un-folded by the sentinel filter, the Shape is now orphaned — its
  // output has no producer in the partition and nothing writes it. Un-fold
  // any Shape whose consumers aren't all folded.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      if (node == nullptr) continue;
      const std::string key = NodeKey(node);
      if (analysis.folded_nodes.count(key) == 0) continue;
      if (analysis.constants.count(std::string(node.GetOutputs()[0].GetName())) > 0) continue;
      bool all_consumers_folded = true;
      for (Ort::ConstValueInfo output : node.GetOutputs()) {
        if (output == nullptr) continue;
        const std::string output_name = output.GetName();
        if (gi.IsGraphOutput(output_name)) {
          all_consumers_folded = false;
          break;
        }
        for (Ort::ConstNode consumer : gi.Consumers(output_name)) {
          if (analysis.folded_nodes.count(NodeKey(consumer)) == 0) {
            all_consumers_folded = false;
            break;
          }
        }
        if (!all_consumers_folded) break;
      }
      if (!all_consumers_folded) {
        analysis.folded_nodes.erase(key);
        changed = true;
      }
    }
  }

  return analysis;
}
