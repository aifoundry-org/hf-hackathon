#include "ops.hpp"
#include "meta_eval.hpp"
#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

static bool isGGMLFloatType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

std::string describe_node(Ort::ConstNode node) {
  if (node == nullptr) return "<null-node>";
  std::ostringstream oss;
  oss << "'" << node.GetOperatorType() << "'";
  const std::string name = node.GetName();
  if (!name.empty()) oss << " (" << name << ")";
  return oss.str();
}

bool support_trace_enabled() {
  const char* trace = std::getenv("GGONNX_TRACE_SUPPORT");
  return trace != nullptr && std::string(trace) != "0" && std::string(trace) != "";
}

SupportResult normalize_support_result(Ort::ConstNode node, SupportResult result) {
  if (result.has_value()) return result;
  if (!result.error().empty()) return result;
  return support_error("support predicate rejected " + describe_node(node));
}

void trace_support_decision(Ort::ConstNode node, const SupportResult& result) {
  if (!support_trace_enabled() || result.has_value()) return;
  std::fprintf(stderr, "[ggonnx][support] reject %s: %s\n",
               describe_node(node).c_str(), result.error().c_str());
}

#define SUPPORT_CHECK(cond, msg) \
  do { \
    if (!(cond)) return support_error(msg); \
  } while (false)

void ComputeErf(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void* /*userdata*/) {
  GGONNX_NOT_NULL(dst, "Erf custom op destination must not be null");
  GGONNX_NOT_NULL(src, "Erf custom op source must not be null");

  const int64_t n = ggml_nelements(src);
  const int64_t start = (n * ith) / nth;
  const int64_t end = (n * (ith + 1)) / nth;
  float* dst_data = ggml_get_data_f32(dst);
  float* src_data = ggml_get_data_f32(src);
  GGONNX_NOT_NULL(dst_data, "Erf custom op destination data must not be null");
  GGONNX_NOT_NULL(src_data, "Erf custom op source data must not be null");
  for (int64_t i = start; i < end; ++i) {
    dst_data[i] = std::erf(src_data[i]);
  }
}

std::optional<ConstantTensor> LookupCompileTimeConstant(Ort::ConstValueInfo value_info,
                                                         const ConstantValueMap* constants) {
  if (value_info == nullptr) return std::nullopt;
  if (value_info.IsConstantInitializer()) {
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(value_info.GetInitializer(value));
    if (value != nullptr) {
      const TensorMetadata meta = getTensorMetadata(value);
      const auto tensor_info = value.GetTensorTypeAndShapeInfo();
      const auto gtype = OnnxTypeToGGML(meta.element_type);
      if (!gtype.has_value()) return std::nullopt;
      const size_t bytes = tensor_info.GetElementCount() * ggml_type_size(*gtype);
      const void* raw_data = nullptr;
      ggonnx::ort_internal::THROW_ON_ERROR(
          ggonnx::ort_internal::GetOrtApi().GetTensorData(value, &raw_data));
      GGONNX_NOT_NULL(raw_data, "ORT returned null tensor data for constant");
      ConstantTensor tensor;
      tensor.element_type = meta.element_type;
      tensor.dims = meta.dims;
      tensor.data.resize(bytes);
      if (bytes > 0) {
        std::memcpy(tensor.data.data(), raw_data, bytes);
      }
      return tensor;
    }
  }
  if (constants == nullptr) return std::nullopt;
  auto it = constants->find(value_info.GetName());
  if (it == constants->end()) return std::nullopt;
  return it->second;
}

}  // namespace

SupportResult support_ok() {
  return SupportResult::success();
}

SupportResult support_error(std::string message) {
  return SupportResult::failure(std::move(message));
}

TensorMetadata getTensorMetadata(Ort::ConstValueInfo value_info) {
  const auto tensor_info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();

  TensorMetadata result;
  result.element_type = tensor_info.GetElementType();
  result.dims = tensor_info.GetShape();
  return result;
}

Expected<TensorMetadata, std::string> try_get_tensor_metadata(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) {
    return Expected<TensorMetadata, std::string>::failure("value info is null");
  }
  if (!isTensorTyped(value_info)) {
    return Expected<TensorMetadata, std::string>::failure("value is not tensor-typed");
  }
  try {
    return getTensorMetadata(value_info);
  } catch (const Ort::Exception& ex) {
    return Expected<TensorMetadata, std::string>::failure(ex.what());
  } catch (const std::exception& ex) {
    return Expected<TensorMetadata, std::string>::failure(ex.what());
  }
}

bool isTensorTyped(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) return false;
  try {
    return value_info.TypeInfo().GetONNXType() == ONNX_TYPE_TENSOR;
  } catch (const Ort::Exception&) {
    return false;
  }
}

TensorMetadata getTensorMetadata(Ort::ConstValue value) {
  const auto tensor_info = value.GetTensorTypeAndShapeInfo();

  TensorMetadata result;
  result.element_type = tensor_info.GetElementType();
  result.dims = tensor_info.GetShape();
  return result;
}

template <typename T>
std::optional<T> readNodeAttribute(Ort::ConstNode node, const char* attribute_name) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName(attribute_name, attr);
  if (!status.IsOK() || attr == nullptr) {
    return std::nullopt;
  }

  T value{};
  if constexpr (std::is_same_v<T, std::vector<std::string>> ||
                std::is_same_v<T, std::vector<int64_t>> ||
                std::is_same_v<T, std::vector<float>>) {
    T values;
    Ort::ThrowOnError(attr.GetValueArray(values));
    return values;
  } else {
    Ort::ThrowOnError(attr.GetValue(value));
    return value;
  }
}

size_t elementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (const int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "tensor element count requested for dynamic or invalid shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

template <typename T>
std::optional<std::vector<T>> readConstantInputArray(Ort::ConstNode node,
                                                     size_t input_idx,
                                                     ONNXTensorElementDataType element_type,
                                                     const ConstantValueMap* constants) {
  const auto inputs = node.GetInputs();
  if (input_idx >= inputs.size() || inputs[input_idx] == nullptr) {
    return std::nullopt;
  }

  const Ort::ConstValueInfo input = inputs[input_idx];
  const auto constant = LookupCompileTimeConstant(input, constants);
  if (!constant.has_value()) return std::nullopt;
  const TensorMetadata meta{.element_type = constant->element_type, .dims = constant->dims};
  if (meta.element_type != element_type) {
    return std::nullopt;
  }

  const size_t count = elementCount(meta.dims);
  if (count == 0) {
    return std::vector<T>{};
  }

  const T* typed_data = reinterpret_cast<const T*>(constant->data.data());
  GGONNX_NOT_NULL(typed_data, "ORT returned null tensor data for constant");
  return std::vector<T>(typed_data, typed_data + count);
}

std::optional<std::vector<int64_t>> readPadVector(Ort::ConstNode node, const ConstantValueMap* constants) {
  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    return pads;
  }
  if (const auto paddings = readNodeAttribute<std::vector<int64_t>>(node, "paddings")) {
    return paddings;
  }
  return readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
}

bool inferIntegerSpatialScale(const TensorMetadata& x,
                              const TensorMetadata& y,
                              int* scale_h,
                              int* scale_w) {
  GGONNX_NOT_NULL(scale_h, "scale_h output must not be null");
  GGONNX_NOT_NULL(scale_w, "scale_w output must not be null");
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!shapeIsFullyStatic(x) || !shapeIsFullyStatic(y)) {
    return false;
  }
  if (y.dims[0] != x.dims[0] || y.dims[1] != x.dims[1]) {
    return false;
  }
  if (x.dims[2] <= 0 || x.dims[3] <= 0 || y.dims[2] <= 0 || y.dims[3] <= 0) {
    return false;
  }
  if (y.dims[2] % x.dims[2] != 0 || y.dims[3] % x.dims[3] != 0) {
    return false;
  }

  const int64_t inferred_h = y.dims[2] / x.dims[2];
  const int64_t inferred_w = y.dims[3] / x.dims[3];
  if (inferred_h <= 0 || inferred_w <= 0) {
    return false;
  }

  *scale_h = static_cast<int>(inferred_h);
  *scale_w = static_cast<int>(inferred_w);
  return true;
}

SupportResult IsSupportedElementwiseBinaryNode(Ort::ConstNode node, std::string_view op_type,
                                       const MetaAnalysis* /*meta*/) {
  SUPPORT_CHECK(op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div" ||
                    op_type == "Max" || op_type == "Min" || op_type == "Mod",
                "unexpected binary op registration");
  // Mod has two semantics; we only handle fmod=1 (C fmod, sign-of-dividend)
  // with float inputs. The default fmod=0 is Python-style and is typically
  // paired with integer inputs, which we don't support here anyway.
  if (op_type == "Mod") {
    const int64_t fmod_attr = readNodeAttribute<int64_t>(node, "fmod").value_or(0);
    SUPPORT_CHECK(fmod_attr == 1, "Mod: only fmod=1 (C fmod semantics) is supported");
  }

  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const auto implicit_inputs = node.GetImplicitInputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();
  const size_t num_implicit_inputs = implicit_inputs.size();
  SUPPORT_CHECK(num_inputs == 2 && num_outputs == 1 && num_implicit_inputs == 0,
                "binary ops require exactly 2 inputs, 1 output, and no implicit inputs");

  GGONNX_ASSERT(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "ORT returned null binary op input/output metadata");

  const TensorMetadata lhs = getTensorMetadata(inputs[0]);
  const TensorMetadata rhs = getTensorMetadata(inputs[1]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);

  if (!isGGMLFloatType(lhs.element_type) ||
      rhs.element_type != lhs.element_type ||
      out.element_type != lhs.element_type) {
    return support_error("binary op tensors must all share the same float element type");
  }
  // Max/Min are synthesized via ggml_scale which is F32-only on CPU.
  if ((op_type == "Max" || op_type == "Min") &&
      lhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("Max/Min are only supported for float32 tensors");
  }
  // Mod's fmod=1 path uses ggml_unary(TRUNC) which is float-only.
  if (op_type == "Mod" && lhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("Mod: only float32 supported (fmod=1 path)");
  }
  SUPPORT_CHECK(rankSupportedByGGML(lhs) && rankSupportedByGGML(rhs) && rankSupportedByGGML(out),
                "binary op rank exceeds GGML_MAX_DIMS");
  // Either operand may broadcast to the output shape — check both sides against
  // the output rather than against each other. The older lhs→rhs check only
  // accepted broadcasts where rhs was the larger side.
  SUPPORT_CHECK(broadcastSupportedByGGML(lhs.dims, out.dims),
                "left operand cannot broadcast to output shape");
  SUPPORT_CHECK(broadcastSupportedByGGML(rhs.dims, out.dims),
                "right operand cannot broadcast to output shape");

  // GGML binary ops require operand_a to BE the output shape and operand_b to
  // broadcast into it. ONNX allows both operands to broadcast into a third
  // (output) shape. EmitElementwiseBinaryNode bridges the gap by inserting a
  // ggml_repeat_4d on whichever side doesn't already match the output. So the
  // sufficient condition here is just "both operands broadcast into output",
  // which the two checks above already enforce. No further restriction is
  // needed for either commutative or non-commutative ops.

  return support_ok();
}

SupportResult IsSupportedElementwiseBinaryOpNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  return IsSupportedElementwiseBinaryNode(node, node.GetOperatorType(), meta);
}

SupportResult IsSupportedGRUNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const auto implicit_inputs = node.GetImplicitInputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();
  const size_t num_implicit_inputs = implicit_inputs.size();
  if (num_inputs < 3 || num_outputs == 0 || num_implicit_inputs != 0) {
    return false;
  }

  std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  if (direction != "forward") {
    return false;
  }

  const int64_t layout = readNodeAttribute<int64_t>(node, "layout").value_or(0);
  if (layout != 0) {
    return false;
  }

  const int64_t linear_before_reset =
      readNodeAttribute<int64_t>(node, "linear_before_reset").value_or(0);
  if (linear_before_reset != 0) {
    return false;
  }

  if (readNodeAttribute<float>(node, "clip").has_value()) {
    return false;
  }

  if (const auto activations = readNodeAttribute<std::vector<std::string>>(node, "activations")) {
    if (activations->size() != 2 || (*activations)[0] != "Sigmoid" || (*activations)[1] != "Tanh") {
      return false;
    }
  }

  if (inputs.size() < 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return false;
  }
  if (inputs.size() > 4 && inputs[4] != nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata r = getTensorMetadata(inputs[2]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      r.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (w.dims.size() != 3 || r.dims.size() != 3) {
    return false;
  }
  if (!x.dims.empty() && x.dims.size() != 3) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(r)) {
    return false;
  }

  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  if (!hidden_size.has_value()) {
    return false;
  }
  if (*hidden_size <= 0 || w.dims[0] != 1 || r.dims[0] != 1 || w.dims[1] != *hidden_size * 3 ||
      r.dims[1] != *hidden_size * 3 || w.dims[2] <= 0 || r.dims[2] != *hidden_size) {
    return false;
  }
  if (!x.dims.empty() && x.dims[2] >= 0 && x.dims[2] != w.dims[2]) {
    return false;
  }

  if (inputs.size() > 3 && inputs[3] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[3]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 2 || b.dims[0] != 1 ||
        b.dims[1] != *hidden_size * 6) {
      return false;
    }
  }

  if (inputs.size() > 5 && inputs[5] != nullptr) {
    const TensorMetadata initial_h = getTensorMetadata(inputs[5]);
    if (initial_h.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || initial_h.dims.size() != 3 ||
        initial_h.dims[0] != 1 || initial_h.dims[2] != *hidden_size) {
      return false;
    }
    if (x.dims[1] >= 0 && initial_h.dims[1] >= 0 && x.dims[1] != initial_h.dims[1]) {
      return false;
    }
  }

  for (Ort::ConstValueInfo output : outputs) {
    if (output == nullptr) {
      continue;
    }
    const TensorMetadata meta = getTensorMetadata(output);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !rankSupportedByGGML(meta)) {
      return false;
    }
  }

  return true;
}

void CompileGRUAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  GGONNX_ASSERT(direction == "forward", "only forward GRU direction is supported");
  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                "GRU hidden_size attribute must be present and positive");
  compiled_node->attrs = NodeDesc::GRUAttrs{.hidden_size = *hidden_size};
}

EmitResult EmitElementwiseBinaryNode(ggml_context* ctx,
                                     const NodeDesc& node,
                                     const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  if (node.inputs.size() != 2 || node.outputs.size() != 1) {
    throw std::runtime_error("compiled binary op node has invalid arity");
  }
  GGONNX_ASSERT(node.inputs[0] < values.size() && node.inputs[1] < values.size(),
                "compiled binary op node input index out of range");
  GGONNX_ASSERT(node.outputs[0] < values.size(), "compiled binary op node output index out of range");

  ggml_tensor* lhs = values[node.inputs[0]];
  ggml_tensor* rhs = values[node.inputs[1]];
  if (lhs == nullptr || rhs == nullptr) {
    throw std::runtime_error("compiled binary op node missing GGML inputs");
  }

  const std::string_view op_type(node.op_type);
  const bool commutative =
      (op_type == "Add" || op_type == "Mul" || op_type == "Max" || op_type == "Min");

  // ONNX elementwise binary uses NumPy broadcasting: both operands broadcast
  // into a third (output) shape. ggml's binary kernels are stricter — the
  // first operand IS the output shape and the second only broadcasts into it.
  // For ONNX patterns where neither side is the output (e.g. [L] * [L,1] ->
  // [L,L] for relative-position attention bias), we materialize the missing
  // side via ggml_repeat_4d. The support check verified both inputs broadcast
  // into the output shape per ONNX rules, so out_ne computed here is well
  // defined; for the common case where one side already matches we skip the
  // repeat and emit the same code as before.
  int64_t out_ne[GGML_MAX_DIMS];
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    out_ne[i] = std::max(lhs->ne[i], rhs->ne[i]);
  }
  auto matches = [&](const ggml_tensor* t) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      if (t->ne[i] != out_ne[i]) return false;
    }
    return true;
  };
  if (commutative && !matches(lhs) && matches(rhs)) {
    std::swap(lhs, rhs);
  }
  if (!matches(lhs)) {
    lhs = ggml_repeat_4d(ctx, lhs, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
  }
  if (!ggml_can_repeat(rhs, lhs)) {
    // Defensive: rhs should broadcast into the output by support-check
    // construction, but symbolic-dim leniency in capability can let an edge
    // case through. Materialize rhs explicitly rather than tripping the
    // ggml_add/mul internal assert.
    rhs = ggml_repeat_4d(ctx, rhs, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
  }
  // ggml_is_contiguous short-circuits to true when ne[0]==1 even if nb[0] is
  // larger than the element size (permuted view). CPU binary kernels still
  // assert nb00 == sizeof(elem), so check nb[0] directly and materialize.
  auto needs_cont = [](const ggml_tensor* t) {
    return t->nb[0] != ggml_type_size(t->type);
  };
  if (needs_cont(lhs)) lhs = ggml_cont(ctx, lhs);
  if (needs_cont(rhs)) rhs = ggml_cont(ctx, rhs);
  if (op_type == "Add") {
    return EmitOutputs{ggml_add(ctx, lhs, rhs)};
  } else if (op_type == "Sub") {
    return EmitOutputs{ggml_sub(ctx, lhs, rhs)};
  } else if (op_type == "Mul") {
    return EmitOutputs{ggml_mul(ctx, lhs, rhs)};
  } else if (op_type == "Div") {
    return EmitOutputs{ggml_div(ctx, lhs, rhs)};
  } else if (op_type == "Max" || op_type == "Min") {
    // ggml has no elementwise max/min, so synthesize:
    //   max(a, b) = (a + b + |a - b|) / 2
    //   min(a, b) = (a + b - |a - b|) / 2
    // Broadcast is handled by the underlying add/sub — matches the support check.
    ggml_tensor* sum = ggml_add(ctx, lhs, rhs);
    ggml_tensor* diff = ggml_sub(ctx, lhs, rhs);
    ggml_tensor* abs_diff = ggml_abs(ctx, diff);
    ggml_tensor* combined = (op_type == "Max") ? ggml_add(ctx, sum, abs_diff)
                                               : ggml_sub(ctx, sum, abs_diff);
    return EmitOutputs{ggml_scale(ctx, combined, 0.5f)};
  } else if (op_type == "Mod") {
    // Float fmod semantics: r = a - trunc(a / b) * b. We only register Mod
    // with fmod=1; fmod=0 (Python-style, sign of divisor) and integer types
    // are rejected by the support predicate.
    ggml_tensor* q = ggml_div(ctx, lhs, rhs);
    ggml_tensor* t = ggml_unary(ctx, q, GGML_UNARY_OP_TRUNC);
    return EmitOutputs{ggml_sub(ctx, lhs, ggml_mul(ctx, t, rhs))};
  }

  return std::nullopt;
}

EmitResult EmitGRUNode(ggml_context* ctx,
                       const NodeDesc& node,
                       const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() >= 3, "compiled GRU node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::GRUAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled GRU node missing GRU attributes");
  GGONNX_ASSERT(attrs->hidden_size > 0, "compiled GRU node must have positive hidden_size");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  ggml_tensor* r = values[node.inputs[2]];
  ggml_tensor* b = (node.inputs.size() > 3 && node.inputs[3] != kOptionalValueAbsent) ? values[node.inputs[3]] : nullptr;
  ggml_tensor* initial_h =
      (node.inputs.size() > 5 && node.inputs[5] != kOptionalValueAbsent) ? values[node.inputs[5]] : nullptr;
  if (x == nullptr || w == nullptr || r == nullptr) {
    throw std::runtime_error("compiled GRU node missing required GGML inputs");
  }

  const int64_t input_size = x->ne[0];
  const int64_t batch_size = x->ne[1];
  const int64_t seq_length = x->ne[2];
  const int64_t hidden_size = attrs->hidden_size;
  GGONNX_ASSERT(w->ne[0] == input_size && w->ne[1] == hidden_size * 3 && w->ne[2] == 1,
                "compiled GRU W tensor shape mismatch");
  GGONNX_ASSERT(r->ne[0] == hidden_size && r->ne[1] == hidden_size * 3 && r->ne[2] == 1,
                "compiled GRU R tensor shape mismatch");

  auto matrix_slice_rows = [&](ggml_tensor* matrix, int64_t row_offset, int64_t row_count) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        matrix,
                        matrix->ne[0],
                        row_count,
                        matrix->nb[1],
                        static_cast<size_t>(row_offset) * matrix->nb[1]);
  };
  auto vector_slice = [&](ggml_tensor* vector, int64_t offset, int64_t length) -> ggml_tensor* {
    return ggml_view_1d(
        ctx, vector, length, static_cast<size_t>(offset) * ggml_element_size(vector));
  };
  auto timestep_slice = [&](ggml_tensor* sequence, int64_t step) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        sequence,
                        sequence->ne[0],
                        sequence->ne[1],
                        sequence->nb[1],
                        static_cast<size_t>(step) * sequence->nb[2]);
  };
  ggml_tensor* wb = nullptr;
  ggml_tensor* rb = nullptr;
  if (b != nullptr) {
    GGONNX_ASSERT(b->ne[0] == hidden_size * 6 && b->ne[1] == 1, "compiled GRU bias tensor shape mismatch");
    wb = vector_slice(b, 0, hidden_size * 3);
    rb = vector_slice(b, hidden_size * 3, hidden_size * 3);
  }

  ggml_tensor* y = nullptr;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_size, batch_size, 1, seq_length);
  }

  ggml_tensor* w_z = matrix_slice_rows(w, 0, hidden_size);

  ggml_tensor* h_t = nullptr;
  if (initial_h != nullptr) {
    GGONNX_ASSERT(initial_h->ne[0] == hidden_size && initial_h->ne[1] == batch_size && initial_h->ne[2] == 1,
                  "compiled GRU initial_h tensor shape mismatch");
    h_t = ggml_view_2d(ctx, initial_h, initial_h->ne[0], initial_h->ne[1], initial_h->nb[1], 0);
  } else {
    // Graph build runs with ggml_init(no_alloc=true), so ggml_new_tensor_2d
    // returns a descriptor with data=nullptr until the backend allocator
    // runs later — we can't memset it. Derive the zero tensor from a matmul
    // against a scaled-to-zero timestep slice so ggml handles storage and
    // the output has the right [hidden_size, batch_size] shape.
    h_t = ggml_scale(ctx, ggml_mul_mat(ctx, w_z, timestep_slice(x, 0)), 0.0f);
  }
  ggml_tensor* w_r = matrix_slice_rows(w, hidden_size, hidden_size);
  ggml_tensor* w_h = matrix_slice_rows(w, hidden_size * 2, hidden_size);
  ggml_tensor* r_z = matrix_slice_rows(r, 0, hidden_size);
  ggml_tensor* r_r = matrix_slice_rows(r, hidden_size, hidden_size);
  ggml_tensor* r_h = matrix_slice_rows(r, hidden_size * 2, hidden_size);
  ggml_tensor* wb_z = wb != nullptr ? vector_slice(wb, 0, hidden_size) : nullptr;
  ggml_tensor* wb_r = wb != nullptr ? vector_slice(wb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* wb_h = wb != nullptr ? vector_slice(wb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* rb_z = rb != nullptr ? vector_slice(rb, 0, hidden_size) : nullptr;
  ggml_tensor* rb_r = rb != nullptr ? vector_slice(rb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* rb_h = rb != nullptr ? vector_slice(rb, hidden_size * 2, hidden_size) : nullptr;

  for (int64_t step = 0; step < seq_length; ++step) {
    ggml_tensor* x_t = timestep_slice(x, step);

    ggml_tensor* z = ggml_add(ctx, ggml_mul_mat(ctx, w_z, x_t), ggml_mul_mat(ctx, r_z, h_t));
    if (wb_z != nullptr) {
      z = ggml_add(ctx, z, wb_z);
    }
    if (rb_z != nullptr) {
      z = ggml_add(ctx, z, rb_z);
    }
    z = ggml_sigmoid(ctx, z);

    ggml_tensor* r_gate = ggml_add(ctx, ggml_mul_mat(ctx, w_r, x_t), ggml_mul_mat(ctx, r_r, h_t));
    if (wb_r != nullptr) {
      r_gate = ggml_add(ctx, r_gate, wb_r);
    }
    if (rb_r != nullptr) {
      r_gate = ggml_add(ctx, r_gate, rb_r);
    }
    r_gate = ggml_sigmoid(ctx, r_gate);

    ggml_tensor* h_candidate = ggml_mul_mat(ctx, w_h, x_t);
    ggml_tensor* recurrent_term = ggml_mul_mat(ctx, r_h, ggml_mul(ctx, r_gate, h_t));
    h_candidate = ggml_add(ctx, h_candidate, recurrent_term);
    if (wb_h != nullptr) {
      h_candidate = ggml_add(ctx, h_candidate, wb_h);
    }
    if (rb_h != nullptr) {
      h_candidate = ggml_add(ctx, h_candidate, rb_h);
    }
    h_candidate = ggml_tanh(ctx, h_candidate);

    h_t = ggml_add(ctx, h_candidate, ggml_mul(ctx, z, ggml_sub(ctx, h_t, h_candidate)));

    if (y != nullptr) {
      y = ggml_set(ctx, y, h_t, y->nb[1], y->nb[2], y->nb[3], static_cast<size_t>(step) * y->nb[3]);
    }
  }

  EmitOutputs outputs;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    outputs.push_back(y);
  }
  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, h_t, hidden_size, batch_size, 1));
  }
  return outputs;
}

SupportResult IsSupportedLSTMNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const auto implicit_inputs = node.GetImplicitInputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();
  const size_t num_implicit_inputs = implicit_inputs.size();
  if (num_inputs < 3 || num_outputs == 0 || num_implicit_inputs != 0) {
    return false;
  }

  std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  if (direction != "forward") {
    return false;
  }

  const int64_t layout = readNodeAttribute<int64_t>(node, "layout").value_or(0);
  if (layout != 0) {
    return false;
  }

  const int64_t input_forget = readNodeAttribute<int64_t>(node, "input_forget").value_or(0);
  if (input_forget != 0) {
    return false;
  }

  if (readNodeAttribute<float>(node, "clip").has_value()) {
    return false;
  }

  if (const auto activations = readNodeAttribute<std::vector<std::string>>(node, "activations")) {
    if (activations->size() != 3 || (*activations)[0] != "Sigmoid" ||
        (*activations)[1] != "Tanh" || (*activations)[2] != "Tanh") {
      return false;
    }
  }

  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return false;
  }
  // sequence_lens (input 4) must be absent
  if (inputs.size() > 4 && inputs[4] != nullptr) {
    return false;
  }
  // P (peepholes, input 7) must be absent
  if (inputs.size() > 7 && inputs[7] != nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata r = getTensorMetadata(inputs[2]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      r.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 3 || w.dims.size() != 3 || r.dims.size() != 3) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(r)) {
    return false;
  }

  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  if (!hidden_size.has_value()) {
    return false;
  }
  if (*hidden_size <= 0 || w.dims[0] != 1 || r.dims[0] != 1 || w.dims[1] != *hidden_size * 4 ||
      r.dims[1] != *hidden_size * 4 || w.dims[2] <= 0 || r.dims[2] != *hidden_size) {
    return false;
  }
  if (x.dims[2] >= 0 && x.dims[2] != w.dims[2]) {
    return false;
  }

  if (inputs.size() > 3 && inputs[3] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[3]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 2 ||
        b.dims[0] != 1 || b.dims[1] != *hidden_size * 8) {
      return false;
    }
  }

  auto check_initial_state = [&](size_t idx) -> bool {
    if (inputs.size() <= idx || inputs[idx] == nullptr) {
      return true;
    }
    const TensorMetadata s = getTensorMetadata(inputs[idx]);
    if (s.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    if (!s.dims.empty() && (s.dims.size() != 3 || s.dims[0] != 1 || s.dims[2] != *hidden_size)) {
      return false;
    }
    if (!x.dims.empty() && !s.dims.empty() && x.dims[1] >= 0 && s.dims[1] >= 0 && x.dims[1] != s.dims[1]) {
      return false;
    }
    return true;
  };
  if (!check_initial_state(5) || !check_initial_state(6)) {
    return false;
  }

  for (Ort::ConstValueInfo output : outputs) {
    if (output == nullptr) {
      continue;
    }
    const TensorMetadata meta = getTensorMetadata(output);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !rankSupportedByGGML(meta)) {
      return false;
    }
  }

  return true;
}

void CompileLSTMAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  GGONNX_ASSERT(direction == "forward", "only forward LSTM direction is supported");
  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                "LSTM hidden_size attribute must be present and positive");
  compiled_node->attrs = NodeDesc::LSTMAttrs{.hidden_size = *hidden_size};
}

EmitResult EmitLSTMNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() >= 3, "compiled LSTM node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::LSTMAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled LSTM node missing LSTM attributes");
  GGONNX_ASSERT(attrs->hidden_size > 0, "compiled LSTM node must have positive hidden_size");

  auto input_or_null = [&](size_t idx) -> ggml_tensor* {
    if (node.inputs.size() <= idx || node.inputs[idx] == kOptionalValueAbsent) {
      return nullptr;
    }
    return values[node.inputs[idx]];
  };

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  ggml_tensor* r = values[node.inputs[2]];
  ggml_tensor* b = input_or_null(3);
  ggml_tensor* initial_h = input_or_null(5);
  ggml_tensor* initial_c = input_or_null(6);
  if (x == nullptr || w == nullptr || r == nullptr) {
    throw std::runtime_error("compiled LSTM node missing required GGML inputs");
  }

  const int64_t input_size = x->ne[0];
  const int64_t batch_size = x->ne[1];
  const int64_t seq_length = x->ne[2];
  const int64_t hidden_size = attrs->hidden_size;
  GGONNX_ASSERT(w->ne[0] == input_size && w->ne[1] == hidden_size * 4 && w->ne[2] == 1,
                "compiled LSTM W tensor shape mismatch");
  GGONNX_ASSERT(r->ne[0] == hidden_size && r->ne[1] == hidden_size * 4 && r->ne[2] == 1,
                "compiled LSTM R tensor shape mismatch");

  auto matrix_slice_rows = [&](ggml_tensor* matrix, int64_t row_offset, int64_t row_count) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        matrix,
                        matrix->ne[0],
                        row_count,
                        matrix->nb[1],
                        static_cast<size_t>(row_offset) * matrix->nb[1]);
  };
  auto vector_slice = [&](ggml_tensor* vector, int64_t offset, int64_t length) -> ggml_tensor* {
    return ggml_view_1d(
        ctx, vector, length, static_cast<size_t>(offset) * ggml_element_size(vector));
  };
  auto timestep_slice = [&](ggml_tensor* sequence, int64_t step) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        sequence,
                        sequence->ne[0],
                        sequence->ne[1],
                        sequence->nb[1],
                        static_cast<size_t>(step) * sequence->nb[2]);
  };

  // ONNX LSTM gate order in W/R/B: i, o, f, c.
  ggml_tensor* wb = nullptr;
  ggml_tensor* rb = nullptr;
  if (b != nullptr) {
    GGONNX_ASSERT(b->ne[0] == hidden_size * 8 && b->ne[1] == 1,
                  "compiled LSTM bias tensor shape mismatch");
    wb = vector_slice(b, 0, hidden_size * 4);
    rb = vector_slice(b, hidden_size * 4, hidden_size * 4);
  }

  // Shared zero tensor for the absent initial_h / initial_c cases. Graph build
  // runs with no_alloc=true, so ggml_new_tensor_2d returns data=nullptr and
  // memset would dereference null — derive zeros via a scaled matmul instead.
  auto make_zero_hb = [&]() -> ggml_tensor* {
    ggml_tensor* w_i_slice = matrix_slice_rows(w, 0, hidden_size);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w_i_slice, timestep_slice(x, 0)), 0.0f);
  };

  ggml_tensor* h_t = nullptr;
  if (initial_h != nullptr) {
    GGONNX_ASSERT(initial_h->ne[0] == hidden_size && initial_h->ne[1] == batch_size &&
                      initial_h->ne[2] == 1,
                  "compiled LSTM initial_h tensor shape mismatch");
    h_t = ggml_view_2d(ctx, initial_h, initial_h->ne[0], initial_h->ne[1], initial_h->nb[1], 0);
  } else {
    h_t = make_zero_hb();
  }

  ggml_tensor* c_t = nullptr;
  if (initial_c != nullptr) {
    GGONNX_ASSERT(initial_c->ne[0] == hidden_size && initial_c->ne[1] == batch_size &&
                      initial_c->ne[2] == 1,
                  "compiled LSTM initial_c tensor shape mismatch");
    c_t = ggml_view_2d(ctx, initial_c, initial_c->ne[0], initial_c->ne[1], initial_c->nb[1], 0);
  } else {
    c_t = make_zero_hb();
  }

  ggml_tensor* y = nullptr;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_size, batch_size, 1, seq_length);
  }

  ggml_tensor* w_i = matrix_slice_rows(w, 0, hidden_size);
  ggml_tensor* w_o = matrix_slice_rows(w, hidden_size, hidden_size);
  ggml_tensor* w_f = matrix_slice_rows(w, hidden_size * 2, hidden_size);
  ggml_tensor* w_c = matrix_slice_rows(w, hidden_size * 3, hidden_size);
  ggml_tensor* r_i = matrix_slice_rows(r, 0, hidden_size);
  ggml_tensor* r_o = matrix_slice_rows(r, hidden_size, hidden_size);
  ggml_tensor* r_f = matrix_slice_rows(r, hidden_size * 2, hidden_size);
  ggml_tensor* r_c = matrix_slice_rows(r, hidden_size * 3, hidden_size);
  ggml_tensor* wb_i = wb != nullptr ? vector_slice(wb, 0, hidden_size) : nullptr;
  ggml_tensor* wb_o = wb != nullptr ? vector_slice(wb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* wb_f = wb != nullptr ? vector_slice(wb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* wb_c = wb != nullptr ? vector_slice(wb, hidden_size * 3, hidden_size) : nullptr;
  ggml_tensor* rb_i = rb != nullptr ? vector_slice(rb, 0, hidden_size) : nullptr;
  ggml_tensor* rb_o = rb != nullptr ? vector_slice(rb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* rb_f = rb != nullptr ? vector_slice(rb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* rb_c = rb != nullptr ? vector_slice(rb, hidden_size * 3, hidden_size) : nullptr;

  auto gate_preact = [&](ggml_tensor* wg,
                         ggml_tensor* rg,
                         ggml_tensor* x_t,
                         ggml_tensor* wbg,
                         ggml_tensor* rbg) -> ggml_tensor* {
    ggml_tensor* acc = ggml_add(ctx, ggml_mul_mat(ctx, wg, x_t), ggml_mul_mat(ctx, rg, h_t));
    if (wbg != nullptr) {
      acc = ggml_add(ctx, acc, wbg);
    }
    if (rbg != nullptr) {
      acc = ggml_add(ctx, acc, rbg);
    }
    return acc;
  };

  for (int64_t step = 0; step < seq_length; ++step) {
    ggml_tensor* x_t = timestep_slice(x, step);

    ggml_tensor* i_gate = ggml_sigmoid(ctx, gate_preact(w_i, r_i, x_t, wb_i, rb_i));
    ggml_tensor* f_gate = ggml_sigmoid(ctx, gate_preact(w_f, r_f, x_t, wb_f, rb_f));
    ggml_tensor* c_tilde = ggml_tanh(ctx, gate_preact(w_c, r_c, x_t, wb_c, rb_c));

    c_t = ggml_add(ctx, ggml_mul(ctx, f_gate, c_t), ggml_mul(ctx, i_gate, c_tilde));

    ggml_tensor* o_gate = ggml_sigmoid(ctx, gate_preact(w_o, r_o, x_t, wb_o, rb_o));
    h_t = ggml_mul(ctx, o_gate, ggml_tanh(ctx, c_t));

    if (y != nullptr) {
      y = ggml_set(ctx, y, h_t, y->nb[1], y->nb[2], y->nb[3], static_cast<size_t>(step) * y->nb[3]);
    }
  }

  EmitOutputs outputs;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    outputs.push_back(y);
  }
  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, h_t, hidden_size, batch_size, 1));
  }
  if (node.outputs.size() > 2 && node.outputs[2] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, c_t, hidden_size, batch_size, 1));
  }
  return outputs;
}

// Shape-preserving unary float op with no attributes. Covers Relu, Sigmoid, Tanh, Neg,
// Abs, Sqrt, Exp, Log, Erf, Softplus, Elu (ONNX default alpha=1.0 only).
SupportResult IsSupportedUnaryFloatNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  const bool in_float = in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                        in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  const bool out_float = out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                         out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (!in_float || !out_float) {
    return false;
  }
  // Erf uses a custom kernel that casts raw data to float* — not safe for F16.
  if (node.GetOperatorType() == "Erf" &&
      (in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
       out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }

  // ONNX Elu supports an alpha attribute; GGML's ggml_elu is fixed at alpha=1.0.
  if (node.GetOperatorType() == "Elu") {
    const float alpha = readNodeAttribute<float>(node, "alpha").value_or(1.0f);
    if (alpha != 1.0f) {
      return false;
    }
  }
  return true;
}

EmitResult EmitUnaryFloatNode(ggml_context* ctx,
                              const NodeDesc& node,
                              const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled unary op node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled unary op node missing GGML input");
  // ggml CPU unary kernels assert nb00 == sizeof(src0_t) unconditionally. A
  // permuted view whose ne[0]==1 slips past ggml_is_contiguous (that helper
  // short-circuits on the blck_size match) but still trips the kernel assert,
  // so check nb[0] directly.
  if (x->nb[0] != ggml_type_size(x->type)) x = ggml_cont(ctx, x);

  const std::string_view op(node.op_type);
  if (op == "Abs")      return EmitOutputs{ggml_abs(ctx, x)};
  if (op == "Relu")     return EmitOutputs{ggml_relu(ctx, x)};
  if (op == "Sigmoid")  return EmitOutputs{ggml_sigmoid(ctx, x)};
  if (op == "Tanh")     return EmitOutputs{ggml_tanh(ctx, x)};
  if (op == "Neg")      return EmitOutputs{ggml_neg(ctx, x)};
  if (op == "Sqrt")     return EmitOutputs{ggml_sqrt(ctx, x)};
  if (op == "Exp")      return EmitOutputs{ggml_exp(ctx, x)};
  if (op == "Log")      return EmitOutputs{ggml_log(ctx, x)};
  if (op == "Erf")      return EmitOutputs{ggml_map_custom1(ctx, ggml_cont(ctx, x), ComputeErf,
                                                            GGML_N_TASKS_MAX, nullptr)};
  if (op == "Reciprocal") {
    // 1/x as ggml_div(ones, x). Build ones same-shape-as-x via scale_bias(x*0 + 1)
    // so the op stays on whatever backend owns x (Vulkan / CPU / ...).
    ggml_tensor* ones = ggml_scale_bias(ctx, x, 0.0f, 1.0f);
    return EmitOutputs{ggml_div(ctx, ones, x)};
  }
  if (op == "Softplus") return EmitOutputs{ggml_softplus(ctx, x)};
  if (op == "Elu")      return EmitOutputs{ggml_elu(ctx, x)};
  return std::nullopt;
}

SupportResult IsSupportedLeakyReluNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  return IsSupportedUnaryFloatNode(node, meta);
}

void CompileLeakyReluAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const float alpha = readNodeAttribute<float>(node, "alpha").value_or(0.01f);
  compiled_node->attrs = NodeDesc::AlphaAttrs{.alpha = alpha};
}

EmitResult EmitLeakyReluNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::AlphaAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled LeakyRelu node missing alpha attribute");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled LeakyRelu node missing GGML input");
  return EmitOutputs{ggml_leaky_relu(ctx, x, attrs->alpha, /*inplace=*/false)};
}

// ONNX PRelu: Y[i] = X[i] if X[i] >= 0 else slope[i] * X[i]. Slope is
// unidirectionally broadcastable to X. ggml has no native PRelu, so we emit
// Y = relu(X) - slope * relu(-X), which evaluates to relu(X) when X >= 0 and
// to slope * X when X < 0 (since -relu(-X) == X on that branch).
SupportResult IsSupportedPReluNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata slope = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (!isGGMLFloatType(x.element_type) ||
      slope.element_type != x.element_type ||
      y.element_type != x.element_type) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(slope) || !rankSupportedByGGML(y)) {
    return false;
  }
  // ggml's broadcast is per-dim "dst[i] % src[i] == 0" in reversed layout,
  // which subsumes ONNX unidirectional broadcasting of slope -> X.
  if (!broadcastSupportedByGGML(slope.dims, x.dims)) {
    return false;
  }
  return true;
}

EmitResult EmitPReluNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled PRelu node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* slope = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled PRelu node missing GGML X input");
  GGONNX_NOT_NULL(slope, "compiled PRelu node missing GGML slope input");
  ggml_tensor* relu_x = ggml_relu(ctx, x);
  ggml_tensor* relu_neg_x = ggml_relu(ctx, ggml_neg(ctx, x));
  ggml_tensor* scaled = ggml_mul(ctx, relu_neg_x, slope);
  return EmitOutputs{ggml_sub(ctx, relu_x, scaled)};
}

// ONNX Clip: Y = min(max(X, min_val), max_val). Pre-opset-11 reads min/max from
// node attributes; opset 11+ reads them from inputs[1] and inputs[2] (both
// optional, defaulting to -inf / +inf). ggml_clamp takes the two bounds as
// plain floats, so we require the input forms to be compile-time scalars.
// Read a Clip bound (min or max) as a float, accepting F32 or F16 scalar constants.
static std::optional<float> readClipBound(Ort::ConstNode node, size_t input_idx,
                                          const ConstantValueMap* constants) {
  if (const auto v = readConstantInputArray<float>(node, input_idx,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants)) {
    if (v->size() == 1) return (*v)[0];
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<ggml_fp16_t>(node, input_idx,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, constants)) {
    if (v->size() == 1) return ggml_fp16_to_fp32((*v)[0]);
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<float> readPowExponent(Ort::ConstNode node, const ConstantValueMap* constants) {
  if (const auto v = readConstantInputArray<float>(node, 1,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants)) {
    if (v->size() == 1) return (*v)[0];
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<ggml_fp16_t>(node, 1,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, constants)) {
    if (v->size() == 1) return ggml_fp16_to_fp32((*v)[0]);
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<int32_t>(node, 1,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, constants)) {
    if (v->size() == 1) return static_cast<float>((*v)[0]);
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<int64_t>(node, 1,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants)) {
    if (v->size() == 1) return static_cast<float>((*v)[0]);
    return std::nullopt;
  }
  return std::nullopt;
}

static std::optional<int64_t> read_constant_input_scalar_int64(Ort::ConstNode node,
                                                               size_t input_idx,
                                                               const ConstantValueMap* constants) {
  if (const auto v = readConstantInputArray<int64_t>(node, input_idx,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants)) {
    if (v->size() == 1) return (*v)[0];
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<int32_t>(node, input_idx,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, constants)) {
    if (v->size() == 1) return static_cast<int64_t>((*v)[0]);
    return std::nullopt;
  }
  return std::nullopt;
}

SupportResult IsSupportedPowNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 2 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Pow requires exactly 2 inputs, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "Pow input/output metadata must not be null");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(isGGMLFloatType(x.element_type) && y.element_type == x.element_type,
                "Pow only supports float/float16 tensors with matching output type");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y),
                "Pow rank exceeds GGML_MAX_DIMS");
  const auto exponent = readPowExponent(node, constants);
  SUPPORT_CHECK(exponent.has_value(), "Pow exponent must be a compile-time scalar constant");
  SUPPORT_CHECK(std::fabs(*exponent - 2.0f) <= 1e-6f,
                "Pow only supports exponent 2");
  return support_ok();
}

EmitResult EmitPowNode(ggml_context* ctx,
                       const NodeDesc& node,
                       const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled Pow node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Pow node missing GGML base input");
  if (x->nb[0] != ggml_type_size(x->type)) x = ggml_cont(ctx, x);
  return EmitOutputs{ggml_sqr(ctx, x)};
}

SupportResult IsSupportedCumSumNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 2 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "CumSum requires exactly 2 inputs, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "CumSum input/output metadata must not be null");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "CumSum only supports float32 tensors");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y),
                "CumSum rank exceeds GGML_MAX_DIMS");
  SUPPORT_CHECK(!x.dims.empty(), "CumSum does not support scalar tensors");
  const auto axis = read_constant_input_scalar_int64(node, 1, constants);
  SUPPORT_CHECK(axis.has_value(), "CumSum axis must be a compile-time scalar integer");
  const int64_t rank = static_cast<int64_t>(x.dims.size());
  const int64_t normalized = *axis < 0 ? *axis + rank : *axis;
  SUPPORT_CHECK(normalized == rank - 1,
                "CumSum only supports the trailing ONNX axis");
  SUPPORT_CHECK(!readNodeAttribute<int64_t>(node, "exclusive").value_or(0),
                "CumSum does not support exclusive=1");
  SUPPORT_CHECK(!readNodeAttribute<int64_t>(node, "reverse").value_or(0),
                "CumSum does not support reverse=1");
  return support_ok();
}

EmitResult EmitCumSumNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled CumSum node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled CumSum node missing GGML input");
  return EmitOutputs{ggml_cumsum(ctx, x)};
}

// ONNX Range(start, limit, delta): emits a 1-D tensor with
//   N = max(ceil((limit - start) / delta), 0) elements.
// ggml_arange takes start/stop/step as floats, so all three inputs must be
// compile-time scalar constants. Integer outputs are produced by casting the
// F32 arange result.
static std::optional<float> readConstantScalarAsFloat(Ort::ConstNode node,
                                                      size_t input_idx,
                                                      const ConstantValueMap* constants) {
  if (const auto v = readConstantInputArray<float>(node, input_idx,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants)) {
    if (v->size() == 1) return (*v)[0];
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<ggml_fp16_t>(node, input_idx,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, constants)) {
    if (v->size() == 1) return ggml_fp16_to_fp32((*v)[0]);
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<int32_t>(node, input_idx,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, constants)) {
    if (v->size() == 1) return static_cast<float>((*v)[0]);
    return std::nullopt;
  }
  if (const auto v = readConstantInputArray<int64_t>(node, input_idx,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants)) {
    if (v->size() == 1) return static_cast<float>((*v)[0]);
    return std::nullopt;
  }
  return std::nullopt;
}

SupportResult IsSupportedRangeNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 3 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Range requires exactly 3 inputs, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                    outputs[0] != nullptr,
                "Range input/output metadata must not be null");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  const auto target_gtype = OnnxTypeToGGML(out.element_type);
  SUPPORT_CHECK(target_gtype.has_value(), "Range: unsupported output element type");
  SUPPORT_CHECK(out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
                    out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                "Range: output type must be float/int32/int64");
  const auto start = readConstantScalarAsFloat(node, 0, constants);
  const auto limit = readConstantScalarAsFloat(node, 1, constants);
  const auto delta = readConstantScalarAsFloat(node, 2, constants);
  SUPPORT_CHECK(start.has_value() && limit.has_value() && delta.has_value(),
                "Range: start/limit/delta must be compile-time scalar constants");
  SUPPORT_CHECK(*delta != 0.0f, "Range: delta must be non-zero");
  return support_ok();
}

void CompileRangeAttributes(Ort::ConstNode node, NodeDesc* compiled_node,
                            const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto outputs = node.GetOutputs();
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  const auto start = readConstantScalarAsFloat(node, 0, constants);
  const auto limit = readConstantScalarAsFloat(node, 1, constants);
  const auto delta = readConstantScalarAsFloat(node, 2, constants);
  GGONNX_ASSERT(start.has_value() && limit.has_value() && delta.has_value(),
                "Range compile: start/limit/delta must be constant scalars");
  NodeDesc::RangeAttrs attrs;
  attrs.start = *start;
  attrs.limit = *limit;
  attrs.delta = *delta;
  attrs.target_type = OnnxTypeToGGML(out.element_type).value_or(GGML_TYPE_F32);
  compiled_node->attrs = attrs;
}

EmitResult EmitRangeNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& /*values*/) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::RangeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Range node missing attributes");
  ggml_tensor* out = ggml_arange(ctx, attrs->start, attrs->limit, attrs->delta);
  GGONNX_NOT_NULL(out, "ggml_arange returned null");
  if (attrs->target_type != GGML_TYPE_F32) {
    out = ggml_cast(ctx, out, attrs->target_type);
    GGONNX_NOT_NULL(out, "ggml_cast for Range output returned null");
  }
  return EmitOutputs{out};
}

SupportResult IsSupportedClipNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) return false;
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (!isGGMLFloatType(x.element_type) || y.element_type != x.element_type) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;

  // Opset 11+: min/max come in as scalar tensors. We need them as compile-time
  // constants so ggml_clamp can take float literals.
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr) continue;  // optional, absent
    if (!readClipBound(node, i, constants).has_value()) return false;
  }
  return true;
}

void CompileClipAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::ClipAttrs attrs;
  // Opset <11: attributes. Opset >=11: inputs.
  if (const auto min_attr = readNodeAttribute<float>(node, "min")) attrs.min = *min_attr;
  if (const auto max_attr = readNodeAttribute<float>(node, "max")) attrs.max = *max_attr;
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    const auto v = readClipBound(node, 1, constants);
    if (v) attrs.min = *v;
  }
  if (inputs.size() >= 3 && inputs[2] != nullptr) {
    const auto v = readClipBound(node, 2, constants);
    if (v) attrs.max = *v;
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitClipNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ClipAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Clip node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Clip node missing GGML input");
  // ggml_clamp returns an in-place view that mutates its input; duplicate first
  // so other consumers of x keep their original values.
  ggml_tensor* src = ggml_dup(ctx, x);
  return EmitOutputs{ggml_clamp(ctx, src, attrs->min, attrs->max)};
}

// ONNX Softmax (opset >= 13): softmax along `axis` (default -1). GGML's ggml_soft_max
// operates along ne[0], which is the last ONNX dim.. so we only accept axis=-1 or
// axis=rank-1. Pre-opset-13 Softmax with its "coerce to 2D" semantics is rejected.
SupportResult IsSupportedSoftmaxNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 1 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Softmax requires exactly 1 input, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr,
                "Softmax input/output metadata must not be null");
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  SUPPORT_CHECK(in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Softmax only supports float32 tensors");
  SUPPORT_CHECK(rankSupportedByGGML(in), "Softmax rank exceeds GGML_MAX_DIMS");
  SUPPORT_CHECK(!in.dims.empty(), "Softmax does not support scalar tensors");
  const int64_t rank = static_cast<int64_t>(in.dims.size());
  const int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(-1);
  const int64_t normalized = axis < 0 ? axis + rank : axis;
  SUPPORT_CHECK(normalized == rank - 1,
                "Softmax only supports the trailing ONNX axis");
  return support_ok();
}

void CompileSoftmaxAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  compiled_node->attrs =
      NodeDesc::AxisAttrs{.axis = readNodeAttribute<int64_t>(node, "axis").value_or(-1)};
}

EmitResult EmitSoftmaxNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Softmax node missing GGML input");
  return EmitOutputs{ggml_soft_max(ctx, x)};
}

// ONNX MatMul: A[..., M, K] @ B[..., K, N] -> [..., M, N], with numpy-style batch
// broadcasting. In GGML dim ordering (reversed), A is [K, M, batch...] (ne[0]=K)
// and B is [N, K, batch...] (ne[0]=N). ggml_mul_mat(w, x) requires w->ne[0]==x->ne[0]
// (the shared K), so we transpose+materialize B to get [K, N, batch...] and pass
// that as the "weight" operand. Result shape is [N, M, batch...] which is the
// reversed ONNX [..., M, N]. We require batch dims to match exactly for now.
SupportResult IsSupportedMatMulNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 2 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "MatMul requires exactly 2 inputs, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "MatMul input/output metadata must not be null");
  const TensorMetadata a = getTensorMetadata(inputs[0]);
  const TensorMetadata b = getTensorMetadata(inputs[1]);
  const TensorMetadata c = getTensorMetadata(outputs[0]);
  if (a.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      c.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("MatMul only supports float32 tensors");
  }
  SUPPORT_CHECK(a.dims.size() >= 2 && b.dims.size() >= 2,
                "MatMul requires both operands to have rank >= 2");
  SUPPORT_CHECK(a.dims.size() == b.dims.size(),
                "MatMul batch-rank broadcasting is not supported yet");
  SUPPORT_CHECK(rankSupportedByGGML(a) && rankSupportedByGGML(b) && rankSupportedByGGML(c),
                "MatMul rank exceeds GGML_MAX_DIMS");
  // Inner dims must agree when both are concrete.
  const int64_t a_k = a.dims[a.dims.size() - 1];
  const int64_t b_k = b.dims[b.dims.size() - 2];
  SUPPORT_CHECK(!(a_k >= 0 && b_k >= 0 && a_k != b_k),
                "MatMul inner dimensions do not match");
  // Batch dims must match exactly (concrete or symbolic-identical not checked — conservative).
  for (size_t i = 0; i + 2 < a.dims.size(); ++i) {
    if (a.dims[i] >= 0 && b.dims[i] >= 0 && a.dims[i] != b.dims[i]) {
      return support_error("MatMul batch dimensions must match exactly");
    }
  }
  return support_ok();
}

void CompileMatMulAttributes(Ort::ConstNode /*node*/, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  // MatMulAttrs.force_f32 is set by the EP after compile_attrs; default false here.
  compiled_node->attrs = NodeDesc::MatMulAttrs{};
}

ConstantLayout MatMulConstantLayout(const NodeDesc& /*node*/, size_t input_idx) {
  // ONNX MatMul: A[...,M,K] @ B[...,K,N]. Only B (input 1) benefits from being
  // stored pre-transposed so that ggml_mul_mat(B, A) can skip the runtime
  // transpose + ggml_cont.
  return input_idx == 1 ? ConstantLayout::MATMUL_WEIGHT_TRANSPOSED
                        : ConstantLayout::AS_IS;
}

EmitResult EmitMatMulNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled MatMul node has invalid arity");
  ggml_tensor* a = values[node.inputs[0]];
  ggml_tensor* b = values[node.inputs[1]];
  GGONNX_NOT_NULL(a, "compiled MatMul node missing GGML A input");
  GGONNX_NOT_NULL(b, "compiled MatMul node missing GGML B input");
  // Fast path: B is a pre-transposed constant (compile-time materialized with
  // ne[0]==K). Detected by shape match against A's contraction dim. Otherwise
  // we pay the runtime transpose+cont. ggml_mul_mat requires !is_transposed on
  // its first operand and a unit innermost stride on the second, so a plain
  // ggml_transpose view is not accepted by the CPU kernel.
  ggml_tensor* b_eff = (b->ne[0] == a->ne[0])
                           ? b
                           : ggml_cont(ctx, ggml_transpose(ctx, b));
  ggml_tensor* out = ggml_mul_mat(ctx, b_eff, a);
  if (std::get<NodeDesc::MatMulAttrs>(node.attrs).force_f32) {
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
  }
  return EmitOutputs{out};
}

// ONNX Conv (2D only for now): X[N,C,H,W] @ W[OC,IC/group,KH,KW] (+ optional B[OC]).
// GGML's reverse-dim convention lines up with ggml_conv_2d_direct's expected layouts:
// X -> ne=[W,H,C,N] and W -> ne=[KW,KH,IC,OC]. No transposes needed.
// Limitations: group=1 for regular conv; auto_pad is lowered via direct padding
// when symmetric, or symmetric-overpad + output crop for one-sided SAME_*.
namespace {

struct AutoPadAxis {
  int begin{0};
  int end{0};
  int symmetric{0};
  int crop_begin{0};
  int output{0};
};

AutoPadAxis ResolveAutoPadAxis(int64_t input,
                               int64_t kernel,
                               int64_t stride,
                               int64_t dilation,
                               int64_t output,
                               bool same_upper) {
  GGONNX_ASSERT(input > 0 && kernel > 0 && stride > 0 && dilation > 0 && output > 0,
                "auto_pad requires positive concrete dims");
  const int64_t effective_kernel = dilation * (kernel - 1) + 1;
  const int64_t total_pad =
      std::max<int64_t>(0, (output - 1) * stride + effective_kernel - input);
  const int64_t begin = same_upper ? (total_pad / 2) : ((total_pad + 1) / 2);
  const int64_t end = total_pad - begin;
  const int64_t symmetric = std::max(begin, end);
  const int64_t crop_begin = symmetric - begin;
  return AutoPadAxis{
      .begin = static_cast<int>(begin),
      .end = static_cast<int>(end),
      .symmetric = static_cast<int>(symmetric),
      .crop_begin = static_cast<int>(crop_begin),
      .output = static_cast<int>(output),
  };
}

ggml_tensor* CropSpatialOutputIfNeeded(ggml_context* ctx,
                                       ggml_tensor* x,
                                       int crop0_begin,
                                       int crop1_begin,
                                       int out_w,
                                       int out_h) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_NOT_NULL(x, "crop source tensor must not be null");
  GGONNX_ASSERT(out_w > 0 && out_h > 0, "crop target shape must be positive");
  if (crop0_begin == 0 && crop1_begin == 0 &&
      x->ne[0] == out_w && x->ne[1] == out_h) {
    return x;
  }
  GGONNX_ASSERT(crop0_begin >= 0 && crop1_begin >= 0, "crop offset must be non-negative");
  GGONNX_ASSERT(crop0_begin + out_w <= x->ne[0] && crop1_begin + out_h <= x->ne[1],
                "crop exceeds spatial output bounds");
  ggml_tensor* cropped = ggml_view_4d(
      ctx, x, out_w, out_h, x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3],
      crop1_begin * x->nb[1] + crop0_begin * x->nb[0]);
  return ggml_cont(ctx, cropped);
}

}  // namespace

SupportResult IsSupportedConvNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK((inputs.size() == 2 || inputs.size() == 3) && outputs.size() == 1 &&
                    node.GetImplicitInputs().size() == 0,
                "Conv requires 2 or 3 inputs, 1 output, and no implicit inputs");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "Conv input/output metadata must not be null");

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("Conv only supports float32 tensors");
  }
  SUPPORT_CHECK(w.dims.size() == 3 || w.dims.size() == 4,
                "Conv only supports 1D and 2D kernels");
  const size_t spatial_rank = w.dims.size() - 2;
  SUPPORT_CHECK(!x.dims.empty() && x.dims.size() == spatial_rank + 2 &&
                    !y.dims.empty() && y.dims.size() == spatial_rank + 2,
                "Conv input/output rank does not match the kernel rank");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(w) && rankSupportedByGGML(y),
                "Conv rank exceeds GGML_MAX_DIMS");

  // Accept group == 1 (regular conv) and depthwise (group == C_in == C_out with
  // weight shape [C, 1, kH, kW]). Depthwise maps to ggml_conv_2d_dw_direct.
  const int64_t group = readNodeAttribute<int64_t>(node, "group").value_or(1);
  if (group != 1) {
    SUPPORT_CHECK(!x.dims.empty() && x.dims[1] >= 0, "depthwise Conv requires static input channels");
    SUPPORT_CHECK(w.dims[0] >= 0 && w.dims[1] >= 0, "depthwise Conv requires static weight channels");
    SUPPORT_CHECK(!y.dims.empty() && y.dims[1] >= 0, "depthwise Conv requires static output channels");
    SUPPORT_CHECK(group == x.dims[1], "depthwise Conv requires group == input channels");
    SUPPORT_CHECK(w.dims[0] == group, "depthwise Conv requires output channels == group");
    SUPPORT_CHECK(w.dims[1] == 1, "depthwise Conv requires weight IC/group == 1");
    SUPPORT_CHECK(y.dims[1] == group, "depthwise Conv requires output channels == group");
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID" &&
      auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") {
    return support_error("Conv auto_pad must be NOTSET, VALID, SAME_UPPER, or SAME_LOWER");
  }
  const bool same_auto_pad = auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER";
  if (same_auto_pad) {
    SUPPORT_CHECK(spatial_rank != 1, "Conv1D does not support SAME_* auto_pad");
    if (x.dims[2] < 0 || x.dims[3] < 0 || y.dims[2] < 0 || y.dims[3] < 0 ||
        w.dims[2] < 0 || w.dims[3] < 0) {
      return support_error("SAME_* Conv requires concrete spatial dimensions");
    }
  }

  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (spatial_rank == 1) {
      if (pads->size() != 2) return false;
      if ((*pads)[0] != (*pads)[1]) return false;
      if (auto_pad == "VALID" && (*pads)[0] != 0) return false;
    } else {
      if (pads->size() != 4) return false;
      if ((*pads)[0] != (*pads)[2] || (*pads)[1] != (*pads)[3]) return false;
      if (auto_pad == "VALID" && ((*pads)[0] != 0 || (*pads)[1] != 0)) return false;
    }
  }
  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (strides->size() != spatial_rank) return false;
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (dilations->size() != spatial_rank) return false;
  }
  if (const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape")) {
    if (kernel_shape->size() != spatial_rank) return false;
    if (spatial_rank == 1) {
      if (w.dims[2] >= 0 && (*kernel_shape)[0] != w.dims[2]) return false;
    } else {
      if (w.dims[2] >= 0 && (*kernel_shape)[0] != w.dims[2]) return false;
      if (w.dims[3] >= 0 && (*kernel_shape)[1] != w.dims[3]) return false;
    }
  }

  SUPPORT_CHECK(!(group == 1 && !x.dims.empty() && x.dims[1] >= 0 && w.dims[1] >= 0 &&
                  x.dims[1] != w.dims[1]),
                "Conv input channels must match weight channels");

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[2]);
    SUPPORT_CHECK(b.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && b.dims.size() == 1,
                  "Conv bias must be a float32 1D tensor");
    SUPPORT_CHECK(!(b.dims[0] >= 0 && w.dims[0] >= 0 && b.dims[0] != w.dims[0]),
                  "Conv bias length must match output channels");
  }

  return support_ok();
}

void CompileConvAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Conv2DAttrs attrs;
  const auto inputs = node.GetInputs();
  GGONNX_ASSERT(inputs.size() >= 2 && inputs[1] != nullptr,
                "Conv must have a weight input");
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  GGONNX_ASSERT(w.dims.size() == 3 || w.dims.size() == 4,
                "Conv compile requires 1D or 2D weight rank");
  attrs.spatial_rank = static_cast<int>(w.dims.size() - 2);

  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (attrs.spatial_rank == 1) {
      attrs.s0 = static_cast<int>((*strides)[0]);
    } else {
      // ONNX [stride_h, stride_w] -> ggml (s0=width, s1=height).
      attrs.s0 = static_cast<int>((*strides)[1]);
      attrs.s1 = static_cast<int>((*strides)[0]);
    }
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (attrs.spatial_rank == 1) {
      attrs.d0 = static_cast<int>((*dilations)[0]);
    } else {
      attrs.d0 = static_cast<int>((*dilations)[1]);
      attrs.d1 = static_cast<int>((*dilations)[0]);
    }
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad == "NOTSET") {
    if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
      if (attrs.spatial_rank == 1) {
        attrs.p0 = static_cast<int>((*pads)[0]);
      } else {
        // pads=[h_begin, w_begin, h_end, w_end], already validated symmetric.
        attrs.p0 = static_cast<int>((*pads)[1]);
        attrs.p1 = static_cast<int>((*pads)[0]);
      }
    }
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    const TensorMetadata x = getTensorMetadata(inputs[0]);
    const TensorMetadata y = getTensorMetadata(node.GetOutputs()[0]);
    const bool same_upper = auto_pad == "SAME_UPPER";
    const int kw = static_cast<int>(w.dims[3]);
    const int kh = static_cast<int>(w.dims[2]);
    const AutoPadAxis w_axis = ResolveAutoPadAxis(
        x.dims[3], kw, attrs.s0, attrs.d0, y.dims[3], same_upper);
    const AutoPadAxis h_axis = ResolveAutoPadAxis(
        x.dims[2], kh, attrs.s1, attrs.d1, y.dims[2], same_upper);
    attrs.p0 = w_axis.symmetric;
    attrs.p1 = h_axis.symmetric;
    attrs.crop0_begin = w_axis.crop_begin;
    attrs.crop1_begin = h_axis.crop_begin;
    attrs.out_w = w_axis.output;
    attrs.out_h = h_axis.output;
  }
  // VALID => pads remain 0.

  attrs.is_depthwise = readNodeAttribute<int64_t>(node, "group").value_or(1) != 1;
  compiled_node->attrs = attrs;
}

EmitResult EmitConvNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 || node.inputs.size() == 3,
                "compiled Conv node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::Conv2DAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Conv node missing Conv2D attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled Conv node missing GGML X input");
  GGONNX_NOT_NULL(w, "compiled Conv node missing GGML W input");

  ggml_tensor* out = nullptr;
  if (attrs->spatial_rank == 1) {
    // Lower ONNX Conv1D as a degenerate Conv2D so we stay on the same direct
    // kernel family as Conv2D. Layout:
    //   X [N,C,W]      -> [N,C,1,W]
    //   W [OC,IC,K]    -> [OC,IC,1,K]
    //   Y [N,OC,OW]    -> [N,OC,1,OW] -> squeeze H
    // ggml_reshape_* asserts contiguous; transposed/strided producers (common
    // upstream of attention/embedding paths) need an explicit cont first.
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (!ggml_is_contiguous(w)) w = ggml_cont(ctx, w);
    ggml_tensor* x_4d = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    ggml_tensor* w_4d = ggml_reshape_4d(ctx, w, w->ne[0], 1, w->ne[1], w->ne[2]);
    ggml_tensor* out_4d = attrs->is_depthwise
        ? ggml_conv_2d_dw_direct(ctx, w_4d, x_4d,
                                 attrs->s0, 1,
                                 attrs->p0, 0,
                                 attrs->d0, 1)
        : ggml_conv_2d_direct(ctx, w_4d, x_4d,
                              attrs->s0, 1,
                              attrs->p0, 0,
                              attrs->d0, 1);
    out = ggml_reshape_3d(ctx, out_4d, out_4d->ne[0], out_4d->ne[2], out_4d->ne[3]);
  } else {
    out = attrs->is_depthwise
        ? ggml_conv_2d_dw_direct(ctx, w, x,
                                 attrs->s0, attrs->s1,
                                 attrs->p0, attrs->p1,
                                 attrs->d0, attrs->d1)
        : ggml_conv_2d_direct(ctx, w, x,
                              attrs->s0, attrs->s1,
                              attrs->p0, attrs->p1,
                              attrs->d0, attrs->d1);
  }

  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "compiled Conv node missing GGML B input");
    if (attrs->spatial_rank == 1) {
      // bias ne=[OC] -> [1,OC,1] to broadcast across W,N of conv output ne=[OW,OC,N].
      ggml_tensor* bias_3d = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
      out = ggml_add(ctx, out, bias_3d);
    } else {
      // bias ne=[OC] -> [1,1,OC,1] to broadcast across W,H,N of conv output ne=[OW,OH,OC,N].
      ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
      out = ggml_add(ctx, out, bias_4d);
    }
  }

  if (attrs->spatial_rank == 2 && attrs->out_w > 0 && attrs->out_h > 0) {
    out = CropSpatialOutputIfNeeded(
        ctx, out, attrs->crop0_begin, attrs->crop1_begin, attrs->out_w, attrs->out_h);
  }

  return EmitOutputs{out};
}

// ONNX ConvTranspose: 2D only, square stride, symmetric pads. Maps to
// ggml_conv_transpose_2d_p0 (no built-in padding) followed by a center crop
// when ONNX pads are non-zero, since output = (in-1)*s - 2p + kernel.
SupportResult IsSupportedConvTransposeNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || w.dims.size() != 4 || y.dims.size() != 4) return false;
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(y)) return false;

  if (readNodeAttribute<int64_t>(node, "group").value_or(1) != 1) return false;
  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET") return false;
  if (readNodeAttribute<std::vector<int64_t>>(node, "output_padding").has_value()) {
    const auto op = *readNodeAttribute<std::vector<int64_t>>(node, "output_padding");
    for (int64_t v : op) if (v != 0) return false;
  }
  if (readNodeAttribute<std::vector<int64_t>>(node, "output_shape").has_value()) return false;

  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (dilations->size() != 2 || (*dilations)[0] != 1 || (*dilations)[1] != 1) return false;
  }

  const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides");
  if (!strides || strides->size() != 2) return false;
  if ((*strides)[0] != (*strides)[1]) return false;

  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (pads->size() != 4) return false;
    if ((*pads)[0] != (*pads)[2] || (*pads)[1] != (*pads)[3]) return false;
    if ((*pads)[0] < 0 || (*pads)[1] < 0) return false;
  }

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[2]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 1) return false;
  }
  return true;
}

void CompileConvTransposeAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::ConvTransposeAttrs attrs;
  const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides");
  GGONNX_ASSERT(strides && strides->size() == 2, "ConvTranspose needs 2D strides");
  attrs.stride = static_cast<int>((*strides)[0]);
  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    attrs.pad_h = static_cast<int>((*pads)[0]);
    attrs.pad_w = static_cast<int>((*pads)[1]);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitConvTransposeNode(ggml_context* ctx,
                                 const NodeDesc& node,
                                 const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ConvTransposeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled ConvTranspose node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "ConvTranspose missing X input");
  GGONNX_NOT_NULL(w, "ConvTranspose missing W input");

  ggml_tensor* out = ggml_conv_transpose_2d_p0(ctx, w, x, attrs->stride);
  if (attrs->pad_w != 0 || attrs->pad_h != 0) {
    // Emulate ONNX symmetric padding by cropping p pixels from each spatial edge.
    const int64_t new_w = out->ne[0] - 2 * attrs->pad_w;
    const int64_t new_h = out->ne[1] - 2 * attrs->pad_h;
    GGONNX_ASSERT(new_w > 0 && new_h > 0, "ConvTranspose crop produced non-positive size");
    ggml_tensor* cropped = ggml_view_4d(
        ctx, out, new_w, new_h, out->ne[2], out->ne[3],
        out->nb[1], out->nb[2], out->nb[3],
        static_cast<size_t>(attrs->pad_w) * out->nb[0] +
            static_cast<size_t>(attrs->pad_h) * out->nb[1]);
    out = ggml_cont(ctx, cropped);
  }
  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "ConvTranspose missing B input");
    ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    out = ggml_add(ctx, out, bias_4d);
  }
  return EmitOutputs{out};
}

// ONNX Expand: broadcast input to a given shape. We require the shape input to
// be a compile-time constant so the target dims land in the compiled attrs —
// ORT's shape inference typically leaves the output shape symbolic when the
// shape tensor comes from a Where/Equal chain.
SupportResult IsSupportedExpandNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 2 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Expand: wrong input/output count");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "Expand: null input or output");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Expand: non-float type");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y), "Expand: rank exceeds GGML_MAX_DIMS");
  const auto resolved_in = (meta != nullptr) ? ResolveShape(inputs[0], *meta) : std::nullopt;
  const auto resolved_out = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
  SUPPORT_CHECK(resolved_in.has_value(), "Expand: input has dynamic shape");
  SUPPORT_CHECK(resolved_out.has_value(), "Expand: output has dynamic shape");
  SUPPORT_CHECK(rankSupportedByGGML({ONNXTensorElementDataType{}, *resolved_out}),
                "Expand: resolved output rank exceeds GGML_MAX_DIMS");

  // Align to the longer rank (ONNX Expand broadcasts leading dims).
  const size_t out_rank = std::max(resolved_in->size(), resolved_out->size());
  std::vector<int64_t> padded_x(out_rank, 1);
  std::vector<int64_t> padded_t(out_rank, 1);
  std::copy_backward(resolved_in->begin(), resolved_in->end(), padded_x.end());
  std::copy_backward(resolved_out->begin(), resolved_out->end(), padded_t.end());
  for (size_t i = 0; i < out_rank; ++i) {
    SUPPORT_CHECK(padded_x[i] == 1 || padded_t[i] == 1 || padded_x[i] == padded_t[i],
                  "Expand: incompatible broadcast dims at axis " + std::to_string(i) +
                      " (x=" + std::to_string(padded_x[i]) + " t=" + std::to_string(padded_t[i]) + ")");
    SUPPORT_CHECK(padded_t[i] >= 0,
                  "Expand: target shape has negative/sentinel dim at axis " + std::to_string(i) +
                      " (value=" + std::to_string(padded_t[i]) + ")");
  }
  return support_ok();
}

void CompileExpandAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr, "Expand must have a single output");
  const auto resolved = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
  GGONNX_ASSERT(resolved.has_value(),
                "Expand compile: output shape must be resolvable (support predicate invariant)");
  compiled_node->attrs = NodeDesc::ExpandAttrs{.onnx_dims = *resolved};
}

EmitResult EmitExpandNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ExpandAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Expand node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "Expand missing input");

  const std::array<int64_t, GGML_MAX_DIMS> ggml_ne = ToPaddedGGMLDims(attrs->onnx_dims);
  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  // ggml_repeat_4d handles integer-multiple broadcasting; source dims of 1
  // become N = target / 1 = target, which is what ONNX Expand means for those
  // axes.
  ggml_tensor* out = ggml_repeat_4d(ctx, src, ggml_ne[0], ggml_ne[1], ggml_ne[2], ggml_ne[3]);
  return EmitOutputs{out};
}

// ONNX Gemm: Y = alpha * A' @ B' + beta * C, where A' = transA ? A^T : A (shape [M,K])
// and B' = transB ? B^T : B (shape [K,N]). Maps to ggml_mul_mat after making both
// operands have the shared K dim at ne[0].
SupportResult IsSupportedGemmNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata a = getTensorMetadata(inputs[0]);
  const TensorMetadata b = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (a.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (a.dims.size() != 2 || b.dims.size() != 2 || y.dims.size() != 2) {
    return false;
  }

  const bool trans_a = readNodeAttribute<int64_t>(node, "transA").value_or(0) != 0;
  const bool trans_b = readNodeAttribute<int64_t>(node, "transB").value_or(0) != 0;
  const int64_t a_k = trans_a ? a.dims[0] : a.dims[1];
  const int64_t b_k = trans_b ? b.dims[1] : b.dims[0];
  if (a_k >= 0 && b_k >= 0 && a_k != b_k) {
    return false;
  }

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata c = getTensorMetadata(inputs[2]);
    if (c.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (!rankSupportedByGGML(c)) return false;
    // Must broadcast to [M, N] == y.dims.
    if (!broadcastSupportedByGGML(c.dims, y.dims)) return false;
  }

  return true;
}

void CompileGemmAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::GemmAttrs attrs;
  attrs.alpha = readNodeAttribute<float>(node, "alpha").value_or(1.0f);
  attrs.beta = readNodeAttribute<float>(node, "beta").value_or(1.0f);
  attrs.trans_a = readNodeAttribute<int64_t>(node, "transA").value_or(0) != 0;
  attrs.trans_b = readNodeAttribute<int64_t>(node, "transB").value_or(0) != 0;
  compiled_node->attrs = attrs;
}

ConstantLayout GemmConstantLayout(const NodeDesc& node, size_t input_idx) {
  // Only B benefits from pre-transpose, and only when transB is false — with
  // transB the ONNX B is already [N, K] which lands as ggml ne=[K, N] with no
  // runtime transpose needed.
  if (input_idx != 1) return ConstantLayout::AS_IS;
  const auto* attrs = std::get_if<NodeDesc::GemmAttrs>(&node.attrs);
  if (attrs == nullptr || attrs->trans_b) return ConstantLayout::AS_IS;
  return ConstantLayout::MATMUL_WEIGHT_TRANSPOSED;
}

EmitResult EmitGemmNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::GemmAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Gemm node missing attributes");

  ggml_tensor* a = values[node.inputs[0]];
  ggml_tensor* b = values[node.inputs[1]];
  GGONNX_NOT_NULL(a, "compiled Gemm node missing GGML A input");
  GGONNX_NOT_NULL(b, "compiled Gemm node missing GGML B input");

  // Build a_eff with K at ne[0]; then require the same of b_eff. A plain view
  // from ggml_transpose violates ggml_mul_mat's invariants, so we materialize
  // when needed. The b_eff branch collapses to a no-op either when transB=1 or
  // when B was pre-transposed as a constant.
  ggml_tensor* a_eff = attrs->trans_a ? ggml_cont(ctx, ggml_transpose(ctx, a)) : a;
  const int64_t K = a_eff->ne[0];
  ggml_tensor* b_eff = (b->ne[0] == K) ? b : ggml_cont(ctx, ggml_transpose(ctx, b));

  ggml_tensor* out = ggml_mul_mat(ctx, b_eff, a_eff);
  if (attrs->alpha != 1.0f) {
    out = ggml_scale(ctx, out, attrs->alpha);
  }

  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* c = values[node.inputs[2]];
    GGONNX_NOT_NULL(c, "compiled Gemm node missing GGML C input");
    if (attrs->beta != 1.0f) {
      c = ggml_scale(ctx, c, attrs->beta);
    }
    out = ggml_add(ctx, out, c);
  }

  return EmitOutputs{out};
}

// TODO(channel-shuffle): fuse the Reshape→Transpose→Reshape triple that ShuffleNet
// (and similar) use for channel shuffle. The 5D intermediate ([N,g,k,H,W] with
// perm=[0,2,1,3,4]) cannot be held in ggml (GGML_MAX_DIMS=4), but the whole
// triple collapses to a pure 4D view+permute+cont+reshape because H,W are
// pass-through. Implemented as __ChannelShuffle — see EmitChannelShuffleNode.

// ONNX Reshape: data + shape input. We require the output shape to be fully static
// (resolved by shape inference) and snapshot it at compile time — the runtime shape
// tensor is ignored.
SupportResult IsSupportedReshapeNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 2 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Reshape: wrong input/output count");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr, "Reshape: null input or output");
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Reshape: non-float type (in=" + std::to_string(in.element_type) +
                    " out=" + std::to_string(out.element_type) + ")");
  SUPPORT_CHECK(rankSupportedByGGML(in) && rankSupportedByGGML(out),
                "Reshape: rank exceeds GGML_MAX_DIMS");
  // Accept either ORT's static output shape or one we propagated ourselves.
  const auto resolved = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
  SUPPORT_CHECK(resolved.has_value(), "Reshape: output has dynamic shape");
  return support_ok();
}

void CompileReshapeAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr,
                "Reshape must have a single output");
  const auto resolved = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
  GGONNX_ASSERT(resolved.has_value(),
                "Reshape compile: output shape must be resolvable (support predicate invariant)");
  compiled_node->attrs = NodeDesc::ReshapeAttrs{.onnx_dims = *resolved};
}

EmitResult EmitReshapeNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ReshapeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Reshape node missing attributes");

  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Reshape node missing GGML data input");

  const std::array<int64_t, GGML_MAX_DIMS> target = ToPaddedGGMLDims(attrs->onnx_dims);
  // Guard: ggml_reshape_4d aborts if element count changes. This can happen for
  // nodes inside If/Loop branches where ORT's static shape inference at compile
  // time disagrees with the actual runtime input shape. Throw here so the error
  // surfaces as a catchable ORT status rather than a process abort.
  if (ggml_nelements(data) != target[0] * target[1] * target[2] * target[3]) {
    std::string dims_str;
    for (size_t i = 0; i < attrs->onnx_dims.size(); ++i) {
      if (i) dims_str += ",";
      dims_str += std::to_string(attrs->onnx_dims[i]);
    }
    std::string in_dims;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      if (i) in_dims += ",";
      in_dims += std::to_string(data->ne[i]);
    }
    throw std::runtime_error(
        "reshape element count mismatch at '" + node.name + "': input ne=[" +
        in_dims + "] (" + std::to_string(ggml_nelements(data)) +
        " elements) but target shape [" + dims_str + "] has " +
        std::to_string(target[0] * target[1] * target[2] * target[3]));
  }
  ggml_tensor* src = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
  ggml_tensor* out = ggml_reshape_4d(ctx, src, target[0], target[1], target[2], target[3]);
  return EmitOutputs{out};
}

// ONNX Flatten: collapses input dims around `axis` into a 2D tensor. We reuse
// the Reshape machinery — the output shape is fully determined by shape
// inference, so we snapshot it and emit a ggml reshape.
SupportResult IsSupportedFlattenNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }
  if (!shapeIsFullyStatic(out.dims)) {
    return false;
  }
  return true;
}

// ONNX Squeeze/Unsqueeze: remove or insert size-1 axes. We store the axes and
// derive the actual output shape from the runtime input tensor in EmitSqueezeNode,
// so the emit is correct even when ORT's static shape inference for If-branch
// subgraphs is inconsistent with the actual runtime input shape.
SupportResult IsSupportedSqueezeNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK((inputs.size() == 1 || inputs.size() == 2) && outputs.size() == 1 &&
                    node.GetImplicitInputs().size() == 0,
                "Squeeze/Unsqueeze: unexpected arity");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr,
                "Squeeze/Unsqueeze: null input/output");
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(in.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    out.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Squeeze/Unsqueeze: only float supported");
  SUPPORT_CHECK(rankSupportedByGGML(in) && rankSupportedByGGML(out),
                "Squeeze/Unsqueeze: rank unsupported by ggml");
  // Some ORT If-subgraph values arrive with no static rank at all. GGML can
  // still represent the runtime tensor, but Squeeze/Unsqueeze axis semantics
  // become ambiguous enough to mis-lower those nodes. Leave them on CPU rather
  // than speculating and risking a native crash or wrong shape.
  SUPPORT_CHECK(!in.dims.empty(), "Squeeze/Unsqueeze: input has no static rank");
  // If axes are a runtime input (not a constant) we fall back to baked output
  // dims — those must be resolvable either via ORT or our shape propagation.
  const bool has_const_axes =
      (inputs.size() == 2 && inputs[1] != nullptr &&
       readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants).has_value()) ||
      readNodeAttribute<std::vector<int64_t>>(node, "axes").has_value();
  if (!has_const_axes) {
    const auto resolved = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
    SUPPORT_CHECK(resolved.has_value(),
                  "Squeeze/Unsqueeze: dynamic axes and unresolved output shape");
  }
  return true;
}

void CompileSqueezeAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const bool is_unsqueeze = std::string_view(node.GetOperatorType()) == "Unsqueeze";
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const int64_t in_rank = static_cast<int64_t>(in.dims.size());

  NodeDesc::SqueezeAttrs attrs;
  attrs.is_unsqueeze = is_unsqueeze;
  attrs.input_onnx_dims = in.dims;

  // Axes: opset 13+ passes them as the second input (a constant int64 tensor).
  // Older opsets encode them as an attribute. When axes are a runtime input
  // (e.g. fed from outside the graph) readConstantInputArray returns nullopt and
  // we fall back to baked_onnx_dims from ORT's static inference, which is always
  // correct for the main graph (unlike If-branch subgraphs).
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto v = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (v) attrs.onnx_axes = *v;
  } else if (const auto a = readNodeAttribute<std::vector<int64_t>>(node, "axes")) {
    attrs.onnx_axes = *a;
  }
  // Normalize negative axes.
  if (!attrs.onnx_axes.empty()) {
    const int64_t ref_rank = is_unsqueeze
        ? in_rank + static_cast<int64_t>(attrs.onnx_axes.size())
        : in_rank;
    for (auto& ax : attrs.onnx_axes) {
      if (ax < 0) ax += ref_rank;
    }
  }
  // Always bake the ORT-inferred output shape as a fallback for cases where axes
  // are not a compile-time constant.
  const auto outputs = node.GetOutputs();
  if (outputs.size() == 1 && outputs[0] != nullptr) {
    attrs.baked_onnx_dims = getTensorMetadata(outputs[0]).dims;
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitSqueezeNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::SqueezeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Squeeze/Unsqueeze node missing attributes");
  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Squeeze/Unsqueeze node missing GGML data input");

  std::vector<int64_t> out_dims;
  if (!attrs->onnx_axes.empty()) {
    std::vector<int64_t> in_dims;
    if (!attrs->input_onnx_dims.empty()) {
      // Use compile-time rank to preserve leading ONNX size-1 dims that GGML's
      // runtime rank view drops (e.g. ONNX [1,3,1,5] becomes GGML ne={5,1,3,1}
      // and ggml_n_dims=3, so ToOnnxDims() would incorrectly report [3,1,5]).
      const int in_onnx_rank = static_cast<int>(attrs->input_onnx_dims.size());
      in_dims.resize(static_cast<size_t>(in_onnx_rank));
      for (int i = 0; i < in_onnx_rank; ++i) {
        in_dims[i] = data->ne[in_onnx_rank - 1 - i];
      }
    } else {
      // Some nested If-subgraph values in ORT lose their static rank metadata.
      // Fall back to the runtime tensor rank and validate axes explicitly so a
      // bad model raises a clear ORT runtime error instead of segfaulting.
      in_dims = ToOnnxDims(data);
    }
    if (attrs->is_unsqueeze) {
      out_dims = in_dims;
      std::vector<int64_t> sorted = attrs->onnx_axes;
      std::sort(sorted.begin(), sorted.end());
      for (int64_t ax : sorted) {
        GGONNX_ASSERT(ax >= 0 && ax <= static_cast<int64_t>(out_dims.size()),
                      "Unsqueeze axis " + std::to_string(ax) +
                          " is out of bounds for input rank " +
                          std::to_string(in_dims.size()));
        out_dims.insert(out_dims.begin() + ax, 1);
      }
    } else {
      std::unordered_set<int64_t> ax_set(attrs->onnx_axes.begin(), attrs->onnx_axes.end());
      for (int64_t ax : attrs->onnx_axes) {
        GGONNX_ASSERT(ax >= 0 && ax < static_cast<int64_t>(in_dims.size()),
                      "Squeeze axis " + std::to_string(ax) +
                          " is out of bounds for input rank " +
                          std::to_string(in_dims.size()));
      }
      for (int64_t i = 0; i < static_cast<int64_t>(in_dims.size()); ++i) {
        if (!ax_set.count(i)) out_dims.push_back(in_dims[i]);
      }
    }
  } else {
    // Axes are a runtime input: fall back to ORT's baked output shape (correct
    // for main-graph nodes where shape inference is reliable).
    out_dims = attrs->baked_onnx_dims;
  }
  GGONNX_ASSERT(out_dims.size() <= GGML_MAX_DIMS,
                "Squeeze/Unsqueeze output rank exceeds GGML_MAX_DIMS");
  GGONNX_ASSERT(ggml_nelements(data) == static_cast<int64_t>(elementCount(out_dims)),
                "Squeeze/Unsqueeze element count mismatch: input has " +
                    std::to_string(ggml_nelements(data)) + " elements but output shape " +
                    FormatDims(out_dims) + " has " + std::to_string(elementCount(out_dims)));

  const std::array<int64_t, GGML_MAX_DIMS> target = ToPaddedGGMLDims(out_dims);
  ggml_tensor* src = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
  return EmitOutputs{ggml_reshape_4d(ctx, src, target[0], target[1], target[2], target[3])};
}

void CompileFlattenAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr,
                "Flatten must have a single output");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  compiled_node->attrs = NodeDesc::ReshapeAttrs{.onnx_dims = out.dims};
}

// ONNX MaxPool / AveragePool (2D only). ONNX layout X[N,C,H,W] -> ggml ne=[W,H,C,N],
// so ggml_pool_2d's k0/s0/p0 apply to W and k1/s1/p1 apply to H — ONNX kernel_shape/
// strides/pads are [h, w] so we swap. GGML's AveragePool divides by the full kernel
// area regardless of how many in-bounds samples were summed (i.e. count_include_pad=1),
// so we reject AveragePool with non-zero padding unless count_include_pad=1.
SupportResult IsSupportedPool2DNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.empty() || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (outputs.size() > 1) {
    return false;  // MaxPool Indices output not supported.
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) {
    return false;
  }

  if (readNodeAttribute<int64_t>(node, "ceil_mode").value_or(0) != 0) {
    return false;
  }
  if (readNodeAttribute<int64_t>(node, "storage_order").value_or(0) != 0) {
    return false;
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    for (int64_t d : *dilations) {
      if (d != 1) return false;
    }
  }

  const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape");
  if (!kernel_shape.has_value() || kernel_shape->size() != 2) {
    return false;
  }
  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (strides->size() != 2) return false;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID" &&
      auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") {
    return false;
  }
  if ((auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") &&
      (x.dims[2] < 0 || x.dims[3] < 0 || y.dims[2] < 0 || y.dims[3] < 0)) {
    return false;
  }

  std::array<int64_t, 4> pads{0, 0, 0, 0};
  if (const auto pads_attr = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (pads_attr->size() != 4) return false;
    if ((*pads_attr)[0] != (*pads_attr)[2] || (*pads_attr)[1] != (*pads_attr)[3]) return false;
    if (auto_pad == "VALID" && ((*pads_attr)[0] != 0 || (*pads_attr)[1] != 0)) return false;
    for (size_t i = 0; i < 4; ++i) pads[i] = (*pads_attr)[i];
  }

  if (node.GetOperatorType() == "AveragePool") {
    // GGML's AveragePool divides by the full kernel area regardless of how many
    // in-bounds samples were summed — i.e. count_include_pad=1 semantics. If
    // ONNX wants count_include_pad=0 and the op will actually pad, reject so
    // ORT falls back to CPU. SAME_UPPER/SAME_LOWER imply padding whenever the
    // kernel is larger than the stride, so treat them as padded.
    const bool has_explicit_padding = pads[0] != 0 || pads[1] != 0;
    const bool has_auto_padding =
        (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER");
    const int64_t count_include_pad =
        readNodeAttribute<int64_t>(node, "count_include_pad").value_or(0);
    if ((has_explicit_padding || has_auto_padding) && count_include_pad != 1) {
      return false;
    }
  }

  return true;
}

void CompilePool2DAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Pool2DAttrs attrs;
  attrs.op = node.GetOperatorType() == "MaxPool" ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;

  const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape");
  GGONNX_ASSERT(kernel_shape.has_value() && kernel_shape->size() == 2,
                "Pool2D kernel_shape must be present and 2D");
  attrs.k0 = static_cast<int>((*kernel_shape)[1]);
  attrs.k1 = static_cast<int>((*kernel_shape)[0]);

  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    attrs.s0 = static_cast<int>((*strides)[1]);
    attrs.s1 = static_cast<int>((*strides)[0]);
  } else {
    attrs.s0 = 1;
    attrs.s1 = 1;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad == "NOTSET") {
    if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
      attrs.p0 = static_cast<int>((*pads)[1]);
      attrs.p1 = static_cast<int>((*pads)[0]);
    }
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    const TensorMetadata x = getTensorMetadata(node.GetInputs()[0]);
    const TensorMetadata y = getTensorMetadata(node.GetOutputs()[0]);
    const bool same_upper = auto_pad == "SAME_UPPER";
    const AutoPadAxis w_axis = ResolveAutoPadAxis(
        x.dims[3], (*kernel_shape)[1], attrs.s0, 1, y.dims[3], same_upper);
    const AutoPadAxis h_axis = ResolveAutoPadAxis(
        x.dims[2], (*kernel_shape)[0], attrs.s1, 1, y.dims[2], same_upper);
    attrs.p0 = w_axis.symmetric;
    attrs.p1 = h_axis.symmetric;
    attrs.crop0_begin = w_axis.crop_begin;
    attrs.crop1_begin = h_axis.crop_begin;
    attrs.out_w = w_axis.output;
    attrs.out_h = h_axis.output;
  }
  compiled_node->attrs = attrs;
}

SupportResult IsSupportedGlobalPoolNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) {
    return false;
  }
  // Spatial dims must be concrete so the kernel is known at emit time.
  if (x.dims[2] < 0 || x.dims[3] < 0) {
    return false;
  }
  return true;
}

void CompileGlobalPoolAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Pool2DAttrs attrs;
  attrs.op = node.GetOperatorType() == "GlobalMaxPool" ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
  attrs.is_global = true;
  compiled_node->attrs = attrs;
}

EmitResult EmitPool2DNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled Pool2D node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::Pool2DAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Pool2D node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Pool2D node missing GGML input");

  int k0 = attrs->k0, k1 = attrs->k1;
  int s0 = attrs->s0, s1 = attrs->s1;
  int p0 = attrs->p0, p1 = attrs->p1;
  if (attrs->is_global) {
    k0 = static_cast<int>(x->ne[0]);
    k1 = static_cast<int>(x->ne[1]);
    s0 = k0;
    s1 = k1;
    p0 = 0;
    p1 = 0;
  }

  ggml_tensor* out = ggml_pool_2d(ctx, x, attrs->op, k0, k1, s0, s1,
                                  static_cast<float>(p0), static_cast<float>(p1));
  if (!attrs->is_global && attrs->out_w > 0 && attrs->out_h > 0) {
    out = CropSpatialOutputIfNeeded(
        ctx, out, attrs->crop0_begin, attrs->crop1_begin, attrs->out_w, attrs->out_h);
  }
  return EmitOutputs{out};
}

SupportResult IsSupportedPadNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(!inputs.empty() && inputs.size() <= 4, "Pad: unexpected input arity");
  SUPPORT_CHECK(outputs.size() == 1, "Pad: unexpected output arity");
  SUPPORT_CHECK(node.GetImplicitInputs().size() == 0, "Pad: implicit inputs not supported");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr, "Pad: null input/output");

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Pad: only float supported");
  SUPPORT_CHECK(!x.dims.empty() && x.dims.size() <= 4 && y.dims.size() == x.dims.size(),
                "Pad: rank must match between input/output and be in [1,4]");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y), "Pad: rank unsupported by ggml");

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("constant");
  SUPPORT_CHECK(mode == "reflect" || mode == "constant",
                std::string("Pad: mode ") + mode + " not supported");

  const auto pads = readPadVector(node, constants);
  SUPPORT_CHECK(pads.has_value(), "Pad: pads not statically known");
  SUPPORT_CHECK(pads->size() == x.dims.size() * 2, "Pad: pads length != 2*rank");

  // Per-ONNX-axis validation. Pad can be nonzero on any subset of axes; reflect
  // mode additionally requires non-negative pads strictly smaller than the src
  // length along each active axis.
  const int rank = static_cast<int>(x.dims.size());
  int active_axes = 0;
  for (int a = 0; a < rank; ++a) {
    const int64_t begin = (*pads)[a];
    const int64_t end = (*pads)[a + rank];
    if (begin == 0 && end == 0) continue;
    ++active_axes;
    SUPPORT_CHECK(mode != "reflect" || (begin >= 0 && end >= 0),
                  "Pad: reflect mode requires non-negative pads");
    if (x.dims[a] < 0) continue;  // unresolved dim: trust the export
    SUPPORT_CHECK(begin + end + x.dims[a] > 0, "Pad: non-positive output dim");
    SUPPORT_CHECK(mode != "reflect" || (begin < x.dims[a] && end < x.dims[a]),
                  "Pad: reflect pad >= src length");
  }
  // Reflect support is 1D in ggml, so we pad one axis at a time; ≤ 2 active
  // axes keeps the emitted sequence short. Constant mode uses ggml_pad_ext in
  // one shot and already tolerates any axis set — keep the same cap for
  // simplicity.
  SUPPORT_CHECK(active_axes <= 2, "Pad: >2 active axes");

  if (mode == "constant") {
    // Only zero fill is supported. Accept an absent constant input or one
    // that folds to a single zero float.
    const auto node_inputs = node.GetInputs();
    if (node_inputs.size() >= 3 && node_inputs[2] != nullptr) {
      const auto v = readConstantInputArray<float>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants);
      SUPPORT_CHECK(v.has_value(), "Pad: constant_value not statically known");
      SUPPORT_CHECK(v->size() <= 1, "Pad: constant_value must be scalar");
      SUPPORT_CHECK(v->empty() || (*v)[0] == 0.0f, "Pad: non-zero constant_value");
    }
  }

  return true;
}

void CompilePadAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto pads = readPadVector(node, constants);
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const int rank = static_cast<int>(x.dims.size());
  GGONNX_ASSERT(pads.has_value() && static_cast<int>(pads->size()) == rank * 2,
                "Pad node must provide 2*rank pads");
  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("constant");

  NodeDesc::PadAttrs attrs;
  attrs.mode = mode == "reflect" ? NodeDesc::PadAttrs::Mode::Reflect
                                 : NodeDesc::PadAttrs::Mode::Constant;
  // ONNX axis a maps to ggml axis (rank-1-a).
  for (int a = 0; a < rank; ++a) {
    const int ggml_axis = rank - 1 - a;
    attrs.pad_begin[ggml_axis] = static_cast<int>((*pads)[a]);
    attrs.pad_end[ggml_axis] = static_cast<int>((*pads)[a + rank]);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitPadNode(ggml_context* ctx,
                       const NodeDesc& node,
                       const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(!node.inputs.empty() && node.outputs.size() == 1,
                "compiled Pad node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::PadAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Pad node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Pad node missing GGML input");

  ggml_tensor* out = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  if (attrs->mode == NodeDesc::PadAttrs::Mode::Constant) {
    // Crop phase (negative pads) via a 4D view + cont.
    std::array<int, GGML_MAX_DIMS> crop_begin{};
    std::array<int, GGML_MAX_DIMS> crop_end{};
    bool any_crop = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      crop_begin[i] = std::max(0, -attrs->pad_begin[i]);
      crop_end[i] = std::max(0, -attrs->pad_end[i]);
      any_crop = any_crop || crop_begin[i] || crop_end[i];
    }
    if (any_crop) {
      std::array<int64_t, GGML_MAX_DIMS> new_ne{};
      for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        new_ne[i] = out->ne[i] - crop_begin[i] - crop_end[i];
        GGONNX_ASSERT(new_ne[i] > 0, "Pad crop produced non-positive size");
      }
      size_t offset = 0;
      for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        offset += static_cast<size_t>(crop_begin[i]) * out->nb[i];
      }
      ggml_tensor* cropped = ggml_view_4d(
          ctx, out, new_ne[0], new_ne[1], new_ne[2], new_ne[3],
          out->nb[1], out->nb[2], out->nb[3], offset);
      out = ggml_cont(ctx, cropped);
    }
    // Positive-pad phase: ggml_pad_ext fills with zeros on any axis set.
    std::array<int, GGML_MAX_DIMS> pos_begin{};
    std::array<int, GGML_MAX_DIMS> pos_end{};
    bool any_pad = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      pos_begin[i] = std::max(0, attrs->pad_begin[i]);
      pos_end[i] = std::max(0, attrs->pad_end[i]);
      any_pad = any_pad || pos_begin[i] || pos_end[i];
    }
    if (any_pad) {
      out = ggml_pad_ext(ctx, out,
                         pos_begin[0], pos_end[0],
                         pos_begin[1], pos_end[1],
                         pos_begin[2], pos_end[2],
                         pos_begin[3], pos_end[3]);
    }
    return EmitOutputs{out};
  }
  // Reflect mode: ggml_pad_reflect_1d works on ggml axis 0 only. For any other
  // active axis, rotate it into axis 0 with ggml_permute + ggml_cont, pad,
  // and rotate back.
  for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
    const int begin = attrs->pad_begin[axis];
    const int end = attrs->pad_end[axis];
    if (begin == 0 && end == 0) continue;
    if (axis == 0) {
      out = ggml_pad_reflect_1d(ctx, out, begin, end);
      continue;
    }
    std::array<int, GGML_MAX_DIMS> perm{0, 1, 2, 3};
    std::swap(perm[0], perm[axis]);
    ggml_tensor* rotated =
        ggml_cont(ctx, ggml_permute(ctx, out, perm[0], perm[1], perm[2], perm[3]));
    rotated = ggml_pad_reflect_1d(ctx, rotated, begin, end);
    out = ggml_cont(ctx, ggml_permute(ctx, rotated, perm[0], perm[1], perm[2], perm[3]));
  }

  return EmitOutputs{out};
}

SupportResult IsSupportedInstanceNormalizationNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata scale = getTensorMetadata(inputs[1]);
  const TensorMetadata bias = getTensorMetadata(inputs[2]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      scale.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      bias.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4 || scale.dims.size() != 1 || bias.dims.size() != 1) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y) ||
      !rankSupportedByGGML(scale) || !rankSupportedByGGML(bias)) {
    return false;
  }
  if (x.dims[1] >= 0 && scale.dims[0] >= 0 && x.dims[1] != scale.dims[0]) {
    return false;
  }
  if (scale.dims[0] >= 0 && bias.dims[0] >= 0 && scale.dims[0] != bias.dims[0]) {
    return false;
  }
  return true;
}

void CompileInstanceNormalizationAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  compiled_node->attrs = NodeDesc::InstanceNormAttrs{
      .epsilon = readNodeAttribute<float>(node, "epsilon").value_or(1e-5f),
  };
}

EmitResult EmitInstanceNormalizationNode(ggml_context* ctx,
                                         const NodeDesc& node,
                                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 3 && node.outputs.size() == 1,
                "compiled InstanceNormalization node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::InstanceNormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled InstanceNormalization node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  ggml_tensor* bias = values[node.inputs[2]];
  GGONNX_NOT_NULL(x, "compiled InstanceNormalization node missing GGML input");
  GGONNX_NOT_NULL(scale, "compiled InstanceNormalization node missing scale input");
  GGONNX_NOT_NULL(bias, "compiled InstanceNormalization node missing bias input");

  GGONNX_ASSERT(x->ne[2] > 0, "InstanceNormalization channel dimension must be positive");
  ggml_tensor* norm = ggml_group_norm(ctx, x, static_cast<int>(x->ne[2]), attrs->epsilon);
  ggml_tensor* scale_4d = ggml_reshape_4d(ctx, scale, 1, 1, scale->ne[0], 1);
  ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
  return EmitOutputs{ggml_add(ctx, ggml_mul(ctx, norm, scale_4d), bias_4d)};
}

// Shared support predicate for LayerNormalization (with optional bias and
// optional Mean/InvStdDev outputs) and RMSNormalization (no bias, single
// output). ONNX normalizes over dims [axis..rank-1]; scale (and bias if
// present) must match that shape. We lower by flattening those trailing ONNX
// dims into GGML axis 0 so ggml_norm / ggml_rms_norm apply.
static SupportResult IsSupportedNormLikeNode(Ort::ConstNode node, bool allow_bias,
                                              bool allow_extra_outputs) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const size_t num_inputs = inputs.size();
  SUPPORT_CHECK(num_inputs >= 2 && num_inputs <= (allow_bias ? 3u : 2u),
                "Norm: unexpected input arity");
  const size_t max_outputs = allow_extra_outputs ? 3u : 1u;
  SUPPORT_CHECK(outputs.size() >= 1 && outputs.size() <= max_outputs,
                "Norm: unexpected output arity");
  SUPPORT_CHECK(node.GetImplicitInputs().size() == 0, "Norm: implicit inputs not supported");
  SUPPORT_CHECK(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "Norm: null input/output metadata");

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata scale = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    scale.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Norm: non-float dtype");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y) && rankSupportedByGGML(scale),
                "Norm: rank exceeds GGML_MAX_DIMS");
  SUPPORT_CHECK(shapeIsFullyStatic(x.dims), "Norm: input shape not fully static");
  SUPPORT_CHECK(shapeIsFullyStatic(scale.dims), "Norm: scale shape not fully static");

  const int rank = static_cast<int>(x.dims.size());
  SUPPORT_CHECK(rank > 0, "Norm: scalar input");
  int axis = static_cast<int>(readNodeAttribute<int64_t>(node, "axis").value_or(-1));
  if (axis < 0) axis += rank;
  SUPPORT_CHECK(axis >= 0 && axis < rank, "Norm: axis out of range");

  // Scale shape must match x.shape[axis:].
  const int trailing = rank - axis;
  SUPPORT_CHECK(static_cast<int>(scale.dims.size()) == trailing,
                "Norm: scale rank does not match trailing X dims");
  for (int i = 0; i < trailing; ++i) {
    SUPPORT_CHECK(scale.dims[i] == x.dims[axis + i],
                  "Norm: scale dim mismatch with X trailing dims");
  }
  if (allow_bias && num_inputs == 3 && inputs[2] != nullptr) {
    const TensorMetadata bias = getTensorMetadata(inputs[2]);
    SUPPORT_CHECK(bias.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  "Norm: bias dtype must match X");
    SUPPORT_CHECK(shapeIsFullyStatic(bias.dims) && bias.dims == scale.dims,
                  "Norm: bias shape must match scale shape");
  }
  return support_ok();
}

static void CompileNormLikeAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const int rank = static_cast<int>(x.dims.size());
  int axis = static_cast<int>(readNodeAttribute<int64_t>(node, "axis").value_or(-1));
  if (axis < 0) axis += rank;
  NodeDesc::NormAttrs attrs;
  attrs.axis = axis;
  attrs.onnx_rank = rank;
  attrs.epsilon = readNodeAttribute<float>(node, "epsilon").value_or(1e-5f);
  attrs.has_bias = inputs.size() == 3 && inputs[2] != nullptr;
  compiled_node->attrs = attrs;
}

SupportResult IsSupportedLayerNormalizationNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  return IsSupportedNormLikeNode(node, /*allow_bias=*/true, /*allow_extra_outputs=*/true);
}

SupportResult IsSupportedRMSNormalizationNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  return IsSupportedNormLikeNode(node, /*allow_bias=*/false, /*allow_extra_outputs=*/false);
}

void CompileLayerNormalizationAttributes(Ort::ConstNode node,
                                         NodeDesc* compiled_node,
                                         const MetaAnalysis* /*meta*/) {
  CompileNormLikeAttributes(node, compiled_node);
}

void CompileRMSNormalizationAttributes(Ort::ConstNode node,
                                       NodeDesc* compiled_node,
                                       const MetaAnalysis* /*meta*/) {
  CompileNormLikeAttributes(node, compiled_node);
}

// Emit helper: apply `norm_fn` (ggml_norm or ggml_rms_norm) over the trailing
// (rank - axis) ONNX dims by flattening them into GGML axis 0, then broadcast
// the scale and (optionally) add bias.
static EmitResult EmitNormLike(ggml_context* ctx,
                               const NodeDesc& node,
                               const std::vector<ggml_tensor*>& values,
                               ggml_tensor* (*norm_fn)(ggml_context*, ggml_tensor*, float)) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::NormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Norm node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled Norm node missing X input");
  GGONNX_NOT_NULL(scale, "compiled Norm node missing scale input");

  // Flatten x into 2D [trailing_prod, leading_prod] so ggml_norm's "normalize
  // along rows" (axis 0) covers exactly X.shape[axis:]. leading_prod may be 1
  // if axis == 0 (normalize the whole tensor).
  int64_t trailing_prod = 1;
  for (int i = attrs->axis; i < attrs->onnx_rank; ++i) trailing_prod *= x->ne[attrs->onnx_rank - 1 - i];
  int64_t leading_prod = 1;
  for (int i = 0; i < attrs->axis; ++i) leading_prod *= x->ne[attrs->onnx_rank - 1 - i];

  ggml_tensor* x_cont = x;
  if (!ggml_is_contiguous(x_cont) || x_cont->nb[0] != ggml_type_size(x_cont->type)) {
    x_cont = ggml_cont(ctx, x_cont);
  }
  ggml_tensor* flat = ggml_reshape_2d(ctx, x_cont, trailing_prod, leading_prod);
  ggml_tensor* normed = norm_fn(ctx, flat, attrs->epsilon);

  // Reshape back to the original GGML shape of x (= reversed ONNX dims, padded).
  int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = 0; i < attrs->onnx_rank; ++i) out_ne[i] = x->ne[i];
  ggml_tensor* restored =
      ggml_reshape_4d(ctx, normed, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);

  // Broadcast scale: scale's ONNX shape == x.shape[axis:], i.e. GGML axes [0..trailing-1]
  // match; pad with 1s on the remaining leading-GGML axes.
  const int trailing = attrs->onnx_rank - attrs->axis;
  int64_t scale_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = 0; i < trailing; ++i) scale_ne[i] = scale->ne[i];
  ggml_tensor* scale_bc =
      ggml_reshape_4d(ctx, scale, scale_ne[0], scale_ne[1], scale_ne[2], scale_ne[3]);
  ggml_tensor* out = ggml_mul(ctx, restored, scale_bc);

  if (attrs->has_bias) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "compiled Norm node missing bias input");
    int64_t bias_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < trailing; ++i) bias_ne[i] = bias->ne[i];
    ggml_tensor* bias_bc =
        ggml_reshape_4d(ctx, bias, bias_ne[0], bias_ne[1], bias_ne[2], bias_ne[3]);
    out = ggml_add(ctx, out, bias_bc);
  }
  return EmitOutputs{out};
}

EmitResult EmitLayerNormalizationNode(ggml_context* ctx,
                                      const NodeDesc& node,
                                      const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::NormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled LayerNorm node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled LayerNorm node missing X input");
  GGONNX_NOT_NULL(scale, "compiled LayerNorm node missing scale input");

  // Same flatten as EmitNormLike but we compute mean/var/inv_std manually so
  // we can emit Mean and InvStdDev as additional outputs (ONNX LayerNorm has
  // up to 3 outputs; ORT requires all declared ones).
  const int rank = attrs->onnx_rank;
  const int trailing = rank - attrs->axis;
  int64_t trailing_prod = 1;
  for (int i = 0; i < trailing; ++i) trailing_prod *= x->ne[i];
  int64_t leading_prod = 1;
  for (int i = trailing; i < rank; ++i) leading_prod *= x->ne[i];

  ggml_tensor* x_cont = x;
  if (!ggml_is_contiguous(x_cont) || x_cont->nb[0] != ggml_type_size(x_cont->type)) {
    x_cont = ggml_cont(ctx, x_cont);
  }
  ggml_tensor* flat = ggml_reshape_2d(ctx, x_cont, trailing_prod, leading_prod);

  // mean over ggml axis 0 (= normalized region); ggml_mean keepdims.
  ggml_tensor* mean_flat = ggml_mean(ctx, flat);  // [1, leading_prod]
  // centered = x - mean (broadcast on axis 0).
  ggml_tensor* centered = ggml_sub(ctx, flat, mean_flat);
  ggml_tensor* sq = ggml_mul(ctx, centered, centered);
  ggml_tensor* var_flat = ggml_mean(ctx, sq);  // [1, leading_prod]
  ggml_tensor* var_plus_eps = ggml_scale_bias(ctx, var_flat, 1.0f, attrs->epsilon);
  ggml_tensor* sqrt_v = ggml_sqrt(ctx, var_plus_eps);
  // inv_std = 1 / sqrt_v via ones/sqrt_v. ones built from sqrt_v via scale_bias.
  ggml_tensor* ones = ggml_scale_bias(ctx, sqrt_v, 0.0f, 1.0f);
  ggml_tensor* inv_std_flat = ggml_div(ctx, ones, sqrt_v);

  // y_flat = centered * inv_std (broadcast).
  ggml_tensor* y_flat = ggml_mul(ctx, centered, inv_std_flat);

  // Reshape y back to original x GGML shape.
  int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = 0; i < rank; ++i) out_ne[i] = x->ne[i];
  ggml_tensor* y = ggml_reshape_4d(ctx, y_flat, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);

  // Scale + optional bias broadcast.
  int64_t scale_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = 0; i < trailing; ++i) scale_ne[i] = scale->ne[i];
  ggml_tensor* scale_bc =
      ggml_reshape_4d(ctx, scale, scale_ne[0], scale_ne[1], scale_ne[2], scale_ne[3]);
  y = ggml_mul(ctx, y, scale_bc);
  if (attrs->has_bias) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "compiled LayerNorm node missing bias input");
    int64_t bias_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < trailing; ++i) bias_ne[i] = bias->ne[i];
    ggml_tensor* bias_bc =
        ggml_reshape_4d(ctx, bias, bias_ne[0], bias_ne[1], bias_ne[2], bias_ne[3]);
    y = ggml_add(ctx, y, bias_bc);
  }

  EmitOutputs out;
  out.push_back(y);
  // Mean and InvStdDev: reshape flat [1, leading_prod] to full rank with
  // trailing (normalized) ggml axes set to 1 and leading axes matching x.
  if (node.outputs.size() > 1) {
    int64_t m_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = trailing; i < rank; ++i) m_ne[i] = x->ne[i];
    ggml_tensor* mean_out = ggml_reshape_4d(ctx, mean_flat, m_ne[0], m_ne[1], m_ne[2], m_ne[3]);
    out.push_back(mean_out);
    if (node.outputs.size() > 2) {
      ggml_tensor* inv_std_out =
          ggml_reshape_4d(ctx, inv_std_flat, m_ne[0], m_ne[1], m_ne[2], m_ne[3]);
      out.push_back(inv_std_out);
    }
  }
  return out;
}

EmitResult EmitRMSNormalizationNode(ggml_context* ctx,
                                    const NodeDesc& node,
                                    const std::vector<ggml_tensor*>& values) {
  return EmitNormLike(ctx, node, values, ggml_rms_norm);
}

// ONNX Dropout: at inference (training_mode absent or compile-time-constant
// False) the data output is just the input. Mask output (an all-true bool
// tensor) isn't supported here — we only claim the single-output form.
SupportResult IsSupportedDropoutNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(!inputs.empty() && inputs.size() <= 3 && outputs.size() == 1 &&
                    node.GetImplicitInputs().size() == 0,
                "Dropout: unsupported arity (only inference single-output form)");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr,
                "Dropout: null data input/output");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Dropout: non-float dtype");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y),
                "Dropout: rank exceeds GGML_MAX_DIMS");
  // If training_mode is provided, it must be a compile-time-constant False.
  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const auto v = readConstantInputArray<uint8_t>(
        node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, constants);
    SUPPORT_CHECK(v.has_value() && !v->empty() && (*v)[0] == 0,
                  "Dropout: training_mode must be constant-false for inference");
  }
  return support_ok();
}

EmitResult EmitDropoutNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Dropout node missing input");
  return EmitOutputs{ggml_cont(ctx, x)};
}

SupportResult IsSupportedUpsampleNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const std::string op_type = node.GetOperatorType();
  const bool is_resize = op_type == "Resize";
  if ((!is_resize && (inputs.empty() || inputs.size() > 2)) ||
      (is_resize && (inputs.empty() || inputs.size() > 4)) ||
      outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) {
    return false;
  }

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("nearest");
  if (mode != "nearest") {
    return false;
  }

  if (is_resize) {
    const std::string coordinate_transformation_mode =
        readNodeAttribute<std::string>(node, "coordinate_transformation_mode").value_or("half_pixel");
    if (coordinate_transformation_mode != "asymmetric") {
      return false;
    }
    const std::string nearest_mode =
        readNodeAttribute<std::string>(node, "nearest_mode").value_or("round_prefer_floor");
    if (nearest_mode != "floor" && nearest_mode != "round_prefer_floor") {
      return false;
    }
  }

  int scale_h = 0;
  int scale_w = 0;
  if (!inferIntegerSpatialScale(x, y, &scale_h, &scale_w)) {
    return false;
  }
  if (scale_h != scale_w) {
    return false;
  }

  if (op_type == "Upsample" && inputs.size() == 1) {
    const auto scales = readNodeAttribute<std::vector<float>>(node, "scales");
    if (!scales.has_value() || scales->size() != 4) {
      return false;
    }
    if ((*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
        (*scales)[2] != static_cast<float>(scale_h) ||
        (*scales)[3] != static_cast<float>(scale_w)) {
      return false;
    }
  }

  if (op_type == "Upsample" && inputs.size() == 2) {
    const auto scales =
        readConstantInputArray<float>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants);
    if (scales.has_value() &&
        (scales->size() != 4 || (*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
         (*scales)[2] != static_cast<float>(scale_h) || (*scales)[3] != static_cast<float>(scale_w))) {
      return false;
    }
  }

  if (is_resize && inputs.size() >= 3 && inputs[2] != nullptr) {
    const auto scales =
        readConstantInputArray<float>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, constants);
    if (scales.has_value() &&
        (scales->size() != 4 || (*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
         (*scales)[2] != static_cast<float>(scale_h) || (*scales)[3] != static_cast<float>(scale_w))) {
      return false;
    }
  }

  if (is_resize && inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto sizes =
        readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (sizes.has_value() &&
        (sizes->size() != 4 || (*sizes)[0] != y.dims[0] || (*sizes)[1] != y.dims[1] ||
         (*sizes)[2] != y.dims[2] || (*sizes)[3] != y.dims[3])) {
      return false;
    }
  }

  if (!is_resize && inputs.size() == 1 && scale_h <= 0) {
    return false;
  }

  return true;
}

void CompileUpsampleAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(!inputs.empty() && inputs[0] != nullptr && outputs.size() == 1 && outputs[0] != nullptr,
                "Upsample/Resize must have one data input and one output");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  int scale_h = 0;
  int scale_w = 0;
  GGONNX_ASSERT(inferIntegerSpatialScale(x, y, &scale_h, &scale_w),
                "Upsample/Resize must have static integer spatial scale");
  compiled_node->attrs = NodeDesc::UpsampleAttrs{
      .scale_w = scale_w,
      .scale_h = scale_h,
  };
}

EmitResult EmitUpsampleNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(!node.inputs.empty() && node.outputs.size() == 1,
                "compiled Upsample/Resize node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::UpsampleAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Upsample node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Upsample node missing GGML input");
  GGONNX_ASSERT(attrs->scale_w == attrs->scale_h,
                "GGML nearest upsample requires equal width/height scale");
  return EmitOutputs{ggml_upscale(ctx, x, attrs->scale_w, GGML_SCALE_MODE_NEAREST)};
}

// Identity: logical copy. We emit a view of the input so the result has a
// distinct ggml_tensor* (graph build asserts on reused output slots) while
// sharing storage and staying cheap.
SupportResult IsSupportedIdentityNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  // Identity is polymorphic in ONNX — it also accepts sequence/optional
  // types. Feeding one of those into getTensorMetadata() crashes because
  // GetTensorTypeAndShapeInfo() is only valid for tensor-typed TypeInfos.
  if (!isTensorTyped(inputs[0]) || !isTensorTyped(outputs[0])) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }
  return true;
}

EmitResult EmitIdentityNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Identity node missing GGML input");
  // ggml_view_tensor() has op==NONE, so if the partition output is a view-only
  // chain the graph allocator skips the input buffer entirely. ggml_cont is a
  // real op and keeps the graph traversal honest.
  return EmitOutputs{ggml_cont(ctx, x)};
}

// ONNX Cast: convert `input` elementwise to the `to` dtype. Meta-eval folds
// compile-time constants via CastConstant; this handles the runtime path by
// emitting ggml_cast. Same-dtype casts degrade to ggml_cont (Identity).
SupportResult IsSupportedCastNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(inputs.size() == 1 && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Cast: wrong input/output count");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr, "Cast: null input/output");
  SUPPORT_CHECK(isTensorTyped(inputs[0]) && isTensorTyped(outputs[0]),
                "Cast: non-tensor-typed input/output");
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(OnnxTypeToGGML(in.element_type).has_value(),
                "Cast: unsupported input dtype " + std::to_string(in.element_type));
  SUPPORT_CHECK(OnnxTypeToGGML(out.element_type).has_value(),
                "Cast: unsupported output dtype " + std::to_string(out.element_type));
  SUPPORT_CHECK(rankSupportedByGGML(in) && rankSupportedByGGML(out),
                "Cast: rank exceeds GGML_MAX_DIMS");
  // ONNX `saturate` (default 1) only matters for float8 targets, which we
  // don't support anyway; any value is fine otherwise.
  const ggml_type src_g = *OnnxTypeToGGML(in.element_type);
  const ggml_type dst_g = *OnnxTypeToGGML(out.element_type);
  // ggml-cpu's ggml_compute_forward_dup only implements a small set of
  // (src,dst) pairs. Everything else — including I64 as src or dst, I8 as
  // anything but a same-type copy — aborts. Keep the allowlist tight and
  // defer the rest to CPU.
  auto supported_pair = [](ggml_type s, ggml_type d) {
    if (s == d) return true;
    if ((s == GGML_TYPE_F32 && d == GGML_TYPE_F16) ||
        (s == GGML_TYPE_F16 && d == GGML_TYPE_F32)) return true;
    if ((s == GGML_TYPE_F32 && d == GGML_TYPE_I32) ||
        (s == GGML_TYPE_I32 && d == GGML_TYPE_F32)) return true;
    return false;
  };
  SUPPORT_CHECK(supported_pair(src_g, dst_g),
                "Cast: ggml_cast doesn't implement this (src,dst) pair");
  return support_ok();
}

void CompileCastAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  const auto gtype = OnnxTypeToGGML(out.element_type);
  GGONNX_ASSERT(gtype.has_value(), "Cast target dtype has no GGML representation");
  compiled_node->attrs = NodeDesc::CastAttrs{.target_type = *gtype};
}

EmitResult EmitCastNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::CastAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Cast node missing CastAttrs");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Cast node missing GGML input");
  if (x->type == attrs->target_type) {
    return EmitOutputs{ggml_cont(ctx, x)};
  }
  return EmitOutputs{ggml_cast(ctx, x, attrs->target_type)};
}

// ONNX Transpose: permutes dims by `perm` (default = reverse). GGML dim order is
// reversed vs ONNX, so the ggml axis that feeds output GGML axis j is
// R-1 - perm[R-1 - j]. Axes beyond the ONNX rank are padded identity.
SupportResult IsSupportedTransposeNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }
  const size_t rank = in.dims.size();
  if (rank == 0 || rank != out.dims.size()) {
    return false;
  }
  if (const auto perm = readNodeAttribute<std::vector<int64_t>>(node, "perm")) {
    if (perm->size() != rank) return false;
    std::vector<int> seen(rank, 0);
    for (int64_t p : *perm) {
      if (p < 0 || static_cast<size_t>(p) >= rank) return false;
      if (seen[p]) return false;
      seen[p] = 1;
    }
  }
  return true;
}

void CompileTransposeAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const size_t rank = getTensorMetadata(inputs[0]).dims.size();

  std::vector<int64_t> perm;
  if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "perm")) {
    perm = *attr;
  } else {
    // ONNX default: reverse dims.
    perm.resize(rank);
    for (size_t i = 0; i < rank; ++i) perm[i] = static_cast<int64_t>(rank - 1 - i);
  }

  // Invert the ONNX permutation: inv[perm[i]] = i. Semantically inv maps an
  // input ONNX dim to the output ONNX dim it ends up at — which matches
  // ggml_permute's axis_j = "new position of input axis j".
  std::vector<int64_t> inv(rank);
  for (size_t i = 0; i < rank; ++i) {
    inv[static_cast<size_t>(perm[i])] = static_cast<int64_t>(i);
  }

  NodeDesc::TransposeAttrs attrs;
  // Identity for padded GGML axes beyond input rank.
  for (int i = 0; i < GGML_MAX_DIMS; ++i) attrs.ggml_perm[i] = i;
  for (size_t j = 0; j < rank; ++j) {
    const size_t onnx_in_dim = rank - 1 - j;
    const size_t onnx_out_dim = static_cast<size_t>(inv[onnx_in_dim]);
    attrs.ggml_perm[j] = static_cast<int>(rank - 1 - onnx_out_dim);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitTransposeNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::TransposeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Transpose node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Transpose node missing GGML input");
  ggml_tensor* permuted = ggml_permute(ctx,
                                       x,
                                       attrs->ggml_perm[0],
                                       attrs->ggml_perm[1],
                                       attrs->ggml_perm[2],
                                       attrs->ggml_perm[3]);
  // Keep Transpose as a view. Downstream emitters that truly require
  // contiguous storage materialize on demand; eager cont here would pay a
  // heavy pack for chains that can consume the view directly.
  return EmitOutputs{permuted};
}

// ONNX Concat: join N inputs along `axis`. GGML's ggml_concat takes two tensors
// and a GGML-axis integer, so we translate ONNX axis -> (rank-1-axis) and fold
// left-to-right.
SupportResult IsSupportedConcatNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  SUPPORT_CHECK(!inputs.empty() && outputs.size() == 1 && node.GetImplicitInputs().size() == 0,
                "Concat: wrong input/output count");
  SUPPORT_CHECK(outputs[0] != nullptr, "Concat: null output");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(OnnxTypeToGGML(out.element_type).has_value(),
                "Concat: unsupported output type " + std::to_string(out.element_type));
  SUPPORT_CHECK(rankSupportedByGGML(out), "Concat: output rank exceeds GGML_MAX_DIMS");
  const size_t rank = out.dims.size();
  SUPPORT_CHECK(rank > 0, "Concat: scalar output");

  const auto axis_attr = readNodeAttribute<int64_t>(node, "axis");
  SUPPORT_CHECK(axis_attr.has_value(), "Concat: missing axis attribute");
  int64_t axis = *axis_attr;
  if (axis < 0) axis += static_cast<int64_t>(rank);
  SUPPORT_CHECK(axis >= 0 && axis < static_cast<int64_t>(rank), "Concat: axis out of range");

  for (Ort::ConstValueInfo input : inputs) {
    SUPPORT_CHECK(input != nullptr, "Concat: null input");
    const TensorMetadata meta = getTensorMetadata(input);
    SUPPORT_CHECK(meta.element_type == out.element_type,
                  "Concat: input type " + std::to_string(meta.element_type) +
                      " != output type " + std::to_string(out.element_type));
    SUPPORT_CHECK(meta.dims.size() == rank, "Concat: input rank mismatch");
    SUPPORT_CHECK(rankSupportedByGGML(meta), "Concat: input rank exceeds GGML_MAX_DIMS");
    if (constants != nullptr) {
      auto it = constants->find(std::string(input.GetName()));
      if (it != constants->end() && !it->second.dim_bindings.empty()) {
        return support_error("Concat: shape-bound constant inputs are not supported");
      }
    }
  }
  return support_ok();
}

void CompileConcatAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  const size_t rank = getTensorMetadata(outputs[0]).dims.size();
  const int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  const int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  // Store as GGML axis index so emit is a pure translation.
  compiled_node->attrs = NodeDesc::AxisAttrs{.axis = static_cast<int64_t>(rank) - 1 - normalized};
}

EmitResult EmitConcatNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::AxisAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Concat node missing attributes");
  GGONNX_ASSERT(!node.inputs.empty(), "compiled Concat node has no inputs");
  const int ggml_axis = static_cast<int>(attrs->axis);

  ggml_tensor* acc = values[node.inputs[0]];
  GGONNX_NOT_NULL(acc, "compiled Concat node missing GGML input 0");
  if (!ggml_is_contiguous(acc)) acc = ggml_cont(ctx, acc);

  for (size_t i = 1; i < node.inputs.size(); ++i) {
    ggml_tensor* rhs = values[node.inputs[i]];
    GGONNX_NOT_NULL(rhs, "compiled Concat node missing GGML input");
    if (!ggml_is_contiguous(rhs)) rhs = ggml_cont(ctx, rhs);
    acc = ggml_concat(ctx, acc, rhs, ggml_axis);
  }
  return EmitOutputs{acc};
}

// ONNX Slice (opset >= 10): data, starts, ends, axes?, steps?. We require
// constant int64 initializers for the index inputs so the view shape can be
// resolved at compile time. Step==1 lowers to a rectangular ggml_view_4d (a
// zero-copy aliased view); step>1 multiplies the per-axis stride in that
// same view. The one case we reject is step>1 on ggml axis 0 (= the last
// ONNX axis), because ggml_view_*d forces nb[0]=type_size and there is no
// way to express a non-unit innermost stride. Negative steps would need a
// real reverse op (or ggml_get_rows with a constant index tensor wired into
// the EP's constant-upload path) and aren't supported either.
SupportResult IsSupportedSliceNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5) {
    return support_error("Slice: expected 3..5 inputs, got " + std::to_string(inputs.size()));
  }
  if (outputs.size() != 1) {
    return support_error("Slice: expected 1 output, got " + std::to_string(outputs.size()));
  }
  if (node.GetImplicitInputs().size() != 0) {
    return support_error("Slice: implicit inputs not supported");
  }
  if (inputs[0] == nullptr) return support_error("Slice: data input is null");
  if (outputs[0] == nullptr) return support_error("Slice: output is null");

  const TensorMetadata data = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (data.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("Slice: data dtype is " +
                         std::to_string(data.element_type) + ", only FLOAT supported");
  }
  if (out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return support_error("Slice: output dtype is " +
                         std::to_string(out.element_type) + ", only FLOAT supported");
  }
  if (!rankSupportedByGGML(data)) {
    return support_error("Slice: data rank " + std::to_string(data.dims.size()) +
                         " exceeds GGML_MAX_DIMS");
  }
  if (!rankSupportedByGGML(out)) {
    return support_error("Slice: output rank " + std::to_string(out.dims.size()) +
                         " exceeds GGML_MAX_DIMS");
  }
  if (data.dims.size() == 0) {
    return support_error("Slice: scalar data not supported");
  }
  // Require fully-static input dims so we can prove the sliced output is
  // non-degenerate at compile time. Runtime-0-element slices cascade into
  // downstream ggml ops whose allocator can't size a zero tensor (tripping
  // ggml_backend_tensor_alloc's buffer==NULL assert). If we ever gain a
  // way to short-circuit downstream propagation for zero-sized intermediates
  // we can revisit.
  const auto resolved = (meta != nullptr) ? ResolveShape(inputs[0], *meta) : std::nullopt;
  if (!resolved.has_value()) {
    return support_error("Slice: data has dynamic dims");
  }
  // We require constant int64 starts/ends (and axes/steps if present) so the
  // view can be materialized at compile time. Dynamic Slice falls back to CPU.
  const auto starts = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
  if (!starts) return support_error("Slice: 'starts' is not a constant int64 initializer");
  const auto ends = readConstantInputArray<int64_t>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
  if (!ends) return support_error("Slice: 'ends' is not a constant int64 initializer");
  if (starts->size() != ends->size()) {
    return support_error("Slice: starts/ends length mismatch");
  }
  if (starts->empty()) return support_error("Slice: empty starts/ends");

  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes = readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (!axes) return support_error("Slice: 'axes' is not a constant int64 initializer");
    if (axes->size() != starts->size()) {
      return support_error("Slice: axes length does not match starts");
    }
  }
  if (inputs.size() == 5 && inputs[4] != nullptr) {
    const auto steps = readConstantInputArray<int64_t>(node, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (!steps) return support_error("Slice: 'steps' is not a constant int64 initializer");
    if (steps->size() != starts->size()) {
      return support_error("Slice: steps length does not match starts");
    }
    // Resolve axes for the per-step axis check below.
    std::vector<int64_t> axes_resolved;
    if (inputs.size() >= 4 && inputs[3] != nullptr) {
      const auto axes = readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
      axes_resolved = *axes;
    } else {
      axes_resolved.resize(starts->size());
      for (size_t i = 0; i < starts->size(); ++i) axes_resolved[i] = static_cast<int64_t>(i);
    }
    const int64_t rank = static_cast<int64_t>(data.dims.size());
    for (size_t k = 0; k < steps->size(); ++k) {
      const int64_t s = (*steps)[k];
      if (s == 0) {
        return support_error("Slice: step=0 is illegal");
      }
      if (s < 0) {
        return support_error("Slice: negative step=" + std::to_string(s) +
                             " not supported (would require reverse / get_rows path)");
      }
      if (s == 1) continue;
      int64_t onnx_axis = axes_resolved[k];
      if (onnx_axis < 0) onnx_axis += rank;
      // ggml axis 0 == last ONNX axis; ggml_view_*d forces nb[0]=type_size so
      // we can't express a strided innermost axis as a view.
      if (onnx_axis == rank - 1) {
        return support_error("Slice: step=" + std::to_string(s) +
                             " on the last ONNX axis is not view-expressible "
                             "(ggml_view forces nb[0]=type_size)");
      }
    }
  }
  return support_ok();
}

void CompileSliceAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata data = getTensorMetadata(inputs[0]);
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  GGONNX_ASSERT(rank > 0, "Slice compile: rank must be > 0 (predicate invariant)");

  const auto starts = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
  const auto ends = readConstantInputArray<int64_t>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
  GGONNX_ASSERT(starts.has_value() && ends.has_value(),
                "Slice compile: starts/ends must be constant int64 initializers");

  std::vector<int64_t> axes;
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes_opt = readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    GGONNX_ASSERT(axes_opt.has_value(), "Slice compile: axes must be a constant int64 initializer");
    axes = *axes_opt;
  } else {
    axes.resize(starts->size());
    for (size_t i = 0; i < starts->size(); ++i) axes[i] = static_cast<int64_t>(i);
  }

  std::vector<int64_t> steps;
  if (inputs.size() == 5 && inputs[4] != nullptr) {
    const auto steps_opt = readConstantInputArray<int64_t>(node, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    GGONNX_ASSERT(steps_opt.has_value(), "Slice compile: steps must be a constant int64 initializer");
    steps = *steps_opt;
  } else {
    steps.assign(starts->size(), 1);
  }

  NodeDesc::SliceAttrs attrs;
  attrs.onnx_rank = static_cast<int>(rank);
  for (size_t k = 0; k < axes.size(); ++k) {
    int64_t onnx_axis = axes[k];
    if (onnx_axis < 0) onnx_axis += rank;
    GGONNX_ASSERT(onnx_axis >= 0 && onnx_axis < rank,
                  "Slice compile: axis out of range");
    const int ggml_axis = static_cast<int>(rank - 1 - onnx_axis);
    attrs.raw_start[ggml_axis] = (*starts)[k];
    attrs.raw_end[ggml_axis] = (*ends)[k];
    attrs.raw_step[ggml_axis] = steps[k];
    attrs.sliced[ggml_axis] = true;
  }

  compiled_node->attrs = attrs;
}

EmitResult EmitSliceNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::SliceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Slice node missing attributes");
  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Slice node missing GGML data input");
  if (std::getenv("GGONNX_TRACE_SLICE") != nullptr) {
    fprintf(stderr, "[slice] node=%s data ne=[%lld,%lld,%lld,%lld]\n",
            node.name.c_str(),
            (long long)data->ne[0], (long long)data->ne[1],
            (long long)data->ne[2], (long long)data->ne[3]);
  }

  // ggml_view_*d forces nb[0]=type_size, so the source's innermost stride
  // must already be the element size; otherwise the view would silently
  // re-interpret the buffer. This is the same loophole that bit unary ops
  // (see memory note) — guard explicitly.
  if (data->nb[0] != ggml_type_size(data->type)) {
    data = ggml_cont(ctx, data);
  }

  // Resolve per-axis start/length/step against the runtime tensor shape.
  // Axes that the ONNX spec didn't enumerate pass through unsliced (start=0,
  // length=ne, step=1). For sliced axes, apply ONNX clamping rules using
  // data->ne[axis] as the length.
  int64_t out_ne[GGML_MAX_DIMS];
  int64_t starts_g[GGML_MAX_DIMS];
  int64_t steps_g[GGML_MAX_DIMS];
  for (int g = 0; g < GGML_MAX_DIMS; ++g) {
    if (attrs->sliced[g]) {
      const int64_t dim = data->ne[g];
      const int64_t step = attrs->raw_step[g];
      GGONNX_ASSERT(step > 0,
                    "Slice emit: support predicate must reject non-positive step");
      int64_t s = attrs->raw_start[g];
      int64_t e = attrs->raw_end[g];
      if (s < 0) s += dim;
      if (e < 0) e += dim;
      if (s < 0) s = 0; else if (s > dim) s = dim;
      if (e < 0) e = 0; else if (e > dim) e = dim;
      const int64_t span = e > s ? e - s : 0;
      const int64_t length = span > 0 ? (span + step - 1) / step : 0;
      starts_g[g] = s;
      out_ne[g] = length;
      steps_g[g] = step;
    } else {
      starts_g[g] = 0;
      out_ne[g] = data->ne[g];
      steps_g[g] = 1;
    }
  }

  GGONNX_ASSERT(steps_g[0] == 1,
                "Slice emit: step on ggml axis 0 should have been rejected by support predicate");

  const size_t offset_bytes =
      static_cast<size_t>(starts_g[0]) * data->nb[0] +
      static_cast<size_t>(starts_g[1]) * data->nb[1] +
      static_cast<size_t>(starts_g[2]) * data->nb[2] +
      static_cast<size_t>(starts_g[3]) * data->nb[3];

  ggml_tensor* view = ggml_view_4d(ctx, data,
                                   out_ne[0], out_ne[1], out_ne[2], out_ne[3],
                                   data->nb[1] * static_cast<size_t>(steps_g[1]),
                                   data->nb[2] * static_cast<size_t>(steps_g[2]),
                                   data->nb[3] * static_cast<size_t>(steps_g[3]),
                                   offset_bytes);
  // Zero-element view: skip the pack. ggml_cont on a zero-sized tensor trips
  // gallocr's buffer==NULL assert; the view itself is trivially contiguous
  // when there's nothing to copy.
  const bool zero_length = out_ne[0] == 0 || out_ne[1] == 0 || out_ne[2] == 0 || out_ne[3] == 0;
  if (zero_length) return EmitOutputs{view};
  // Materialize because downstream kernels (and the graph-output copy path)
  // require contiguous memory.
  return EmitOutputs{ggml_cont(ctx, view)};
}

// ONNX BatchNormalization (inference mode): 5 inputs, 1 output.
// Y = scale * (X - mean) / sqrt(var + eps) + bias, broadcast over channel axis.
// Supports ONNX rank 2..4. ONNX channel is always axis 1, which in ggml's
// reversed layout is ggml axis (rank-2). ORT usually folds BN into the
// preceding Conv bias, so this mostly matters for BN outside Conv (e.g. the
// trailing BN on arcface's flattened feature vector, or language models).
SupportResult IsSupportedBatchNormNode(Ort::ConstNode node, const MetaAnalysis* /*meta*/) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr) return false;
  }
  if (outputs[0] == nullptr) return false;
  if (readNodeAttribute<int64_t>(node, "training_mode").value_or(0) != 0) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() < 2 || x.dims.size() > 4) return false;
  if (x.dims.size() != y.dims.size()) return false;
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;

  for (int i = 1; i < 5; ++i) {
    const TensorMetadata p = getTensorMetadata(inputs[i]);
    if (p.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (p.dims.size() != 1) return false;
  }
  return true;
}

void CompileBatchNormAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  compiled_node->attrs = NodeDesc::BatchNormAttrs{
      .epsilon = readNodeAttribute<float>(node, "epsilon").value_or(1e-5f),
      .onnx_rank = static_cast<int>(x.dims.size()),
  };
}

EmitResult EmitBatchNormNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 5 && node.outputs.size() == 1,
                "compiled BatchNormalization node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::BatchNormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled BatchNormalization node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  ggml_tensor* bias = values[node.inputs[2]];
  ggml_tensor* mean = values[node.inputs[3]];
  ggml_tensor* var = values[node.inputs[4]];
  GGONNX_NOT_NULL(x, "compiled BatchNormalization node missing GGML X input");
  GGONNX_NOT_NULL(scale, "compiled BatchNormalization node missing scale input");
  GGONNX_NOT_NULL(bias, "compiled BatchNormalization node missing bias input");
  GGONNX_NOT_NULL(mean, "compiled BatchNormalization node missing mean input");
  GGONNX_NOT_NULL(var, "compiled BatchNormalization node missing var input");

  // Place C at ggml axis (rank-2), 1s elsewhere. For rank 4 NCHW: ne[2]=C; for
  // rank 3 [N,C,L]: ne[1]=C; for rank 2 [N,C]: ne[0]=C. ggml tensors always
  // carry 4 ne slots, so a 4D reshape works for all ranks.
  const int channel_axis = attrs->onnx_rank - 2;
  GGONNX_ASSERT(channel_axis >= 0 && channel_axis < GGML_MAX_DIMS,
                "BatchNormalization has unexpected onnx_rank");
  const int64_t c = x->ne[channel_axis];
  int64_t ne[4] = {1, 1, 1, 1};
  ne[channel_axis] = c;
  ggml_tensor* scale_4d = ggml_reshape_4d(ctx, scale, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* mean_4d = ggml_reshape_4d(ctx, mean, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* var_4d = ggml_reshape_4d(ctx, var, ne[0], ne[1], ne[2], ne[3]);

  // std = sqrt(var + eps); scale_bias(a, s, b) = s*a + b, so we can fold the
  // scalar epsilon add into the same op without a separate eps-filled tensor.
  ggml_tensor* std_tensor = ggml_sqrt(ctx, ggml_scale_bias(ctx, var_4d, 1.0f, attrs->epsilon));
  ggml_tensor* centered = ggml_sub(ctx, x, mean_4d);
  ggml_tensor* normed = ggml_div(ctx, centered, std_tensor);
  ggml_tensor* out = ggml_add(ctx, ggml_mul(ctx, normed, scale_4d), bias_4d);
  return EmitOutputs{out};
}

// ONNX Split (opset 2/11/13): produces N outputs by slicing `data` along `axis`.
// We handle the case where the split sizes are known at compile time — either
// via the `split` attribute (opset <13), the `split` input (opset 13+), or by
// equal division into `outputs.size()` chunks (the default).
SupportResult IsSupportedSplitNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.empty() ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr) return false;
  const TensorMetadata data = getTensorMetadata(inputs[0]);
  if (data.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
  if (!rankSupportedByGGML(data) || data.dims.empty()) return false;
  const auto resolved_in = (meta != nullptr) ? ResolveShape(inputs[0], *meta) : std::nullopt;
  if (!resolved_in.has_value()) return false;
  if (resolved_in->size() != data.dims.size()) return false;
  if (!rankSupportedByGGML({data.element_type, *resolved_in})) return false;

  const int64_t rank = static_cast<int64_t>(resolved_in->size());
  int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return false;
  if ((*resolved_in)[static_cast<size_t>(axis)] < 0) return false;

  // Split sizes: prefer the `split` input (opset 13+), then the attribute
  // (opset 2/11), then equal division.
  std::vector<int64_t> split_sizes;
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto s = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (!s) return false;
    split_sizes = *s;
  } else if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "split")) {
    split_sizes = *attr;
  } else {
    const int64_t n = static_cast<int64_t>(outputs.size());
    if (n <= 0 || (*resolved_in)[static_cast<size_t>(axis)] % n != 0) return false;
    split_sizes.assign(n, (*resolved_in)[static_cast<size_t>(axis)] / n);
  }
  if (split_sizes.size() != outputs.size()) return false;
  int64_t total = 0;
  for (int64_t s : split_sizes) {
    if (s < 0) return false;
    total += s;
  }
  if (total != (*resolved_in)[static_cast<size_t>(axis)]) return false;

  for (size_t i = 0; i < outputs.size(); ++i) {
    Ort::ConstValueInfo out = outputs[i];
    if (out == nullptr) return false;
    const TensorMetadata out_meta = getTensorMetadata(out);
    if (out_meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    const auto resolved_out = (meta != nullptr) ? ResolveShape(out, *meta) : std::nullopt;
    if (!resolved_out.has_value()) return false;
    if (resolved_out->size() != resolved_in->size()) return false;
    if (!rankSupportedByGGML({out_meta.element_type, *resolved_out})) return false;
    auto expected = *resolved_in;
    expected[static_cast<size_t>(axis)] = split_sizes[i];
    if (*resolved_out != expected) return false;
  }
  return true;
}

void CompileSplitAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(meta != nullptr, "Split compile requires meta analysis");
  const auto resolved_in = ResolveShape(inputs[0], *meta);
  GGONNX_ASSERT(resolved_in.has_value(),
                "Split compile: input shape must be resolvable (support predicate invariant)");
  const int64_t rank = static_cast<int64_t>(resolved_in->size());
  int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  if (axis < 0) axis += rank;

  NodeDesc::SplitAttrs attrs;
  attrs.ggml_axis = static_cast<int>(rank - 1 - axis);
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto s = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    GGONNX_ASSERT(s.has_value(), "Split compile: split input must be constant int64");
    attrs.lengths = *s;
  } else if (const auto a = readNodeAttribute<std::vector<int64_t>>(node, "split")) {
    attrs.lengths = *a;
  } else {
    const int64_t n = static_cast<int64_t>(outputs.size());
    attrs.lengths.assign(n, (*resolved_in)[static_cast<size_t>(axis)] / n);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitSplitNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::SplitAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Split node missing attributes");
  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Split node missing GGML data input");

  const int axis = attrs->ggml_axis;
  EmitOutputs out;
  out.reserve(attrs->lengths.size());
  int64_t offset_elems = 0;
  int64_t ne[GGML_MAX_DIMS];
  for (int i = 0; i < GGML_MAX_DIMS; ++i) ne[i] = data->ne[i];
  for (int64_t length : attrs->lengths) {
    ne[axis] = length;
    const size_t offset_bytes = static_cast<size_t>(offset_elems) * data->nb[axis];
    ggml_tensor* view = ggml_view_4d(ctx, data,
                                     ne[0], ne[1], ne[2], ne[3],
                                     data->nb[1], data->nb[2], data->nb[3],
                                     offset_bytes);
    out.push_back(ggml_cont(ctx, view));
    offset_elems += length;
  }
  return out;
}

// Derive reduction axes from a shape comparison when the axes input exists but
// isn't a compile-time constant. Only unambiguous when `keepdims=1`: every
// reduced axis in the output has dim 1, so the differing positions name the
// reduced axes. Returns nullopt if the input/output shapes aren't fully
// static or the comparison would be ambiguous (keepdims=0, or an output dim
// doesn't collapse to 1 where it should).
static std::optional<std::vector<int64_t>> DeriveReduceAxesFromShape(
    const std::vector<int64_t>& in_dims,
    const std::vector<int64_t>& out_dims,
    bool keepdims) {
  if (!keepdims) return std::nullopt;
  if (in_dims.size() != out_dims.size()) return std::nullopt;
  if (!shapeIsFullyStatic(in_dims) || !shapeIsFullyStatic(out_dims)) return std::nullopt;
  std::vector<int64_t> axes;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] == out_dims[i]) continue;
    // A reduced axis must collapse to 1 when keepdims=1.
    if (out_dims[i] != 1) return std::nullopt;
    axes.push_back(static_cast<int64_t>(i));
  }
  return axes;
}

// Resolve reduction axes from node attributes / inputs / static shapes.
// Shared between IsSupported and Compile; returns a normalized-but-unsorted
// list in ONNX axis coordinates (negative values not yet resolved).
static std::optional<std::vector<int64_t>> ResolveReduceAxes(
    Ort::ConstNode node,
    const TensorMetadata& x,
    const TensorMetadata& y,
    bool keepdims,
    const ConstantValueMap* constants) {
  const auto inputs = node.GetInputs();
  const int64_t rank = static_cast<int64_t>(x.dims.size());
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto v = readConstantInputArray<int64_t>(
        node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, constants);
    if (v.has_value()) return *v;
    // Fall back to shape-derived axes when the axes input is dynamic but the
    // input/output shapes are both static and keepdims=1.
    return DeriveReduceAxesFromShape(x.dims, y.dims, keepdims);
  }
  if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "axes")) {
    return *attr;
  }
  // Default: reduce all axes.
  std::vector<int64_t> all(rank);
  for (int64_t i = 0; i < rank; ++i) all[i] = i;
  return all;
}

// ONNX ReduceMean (opset 11/13/18). We support the common case of reducing a
// contiguous suffix of the ONNX dims — e.g. axes=[2,3] on an [N,C,H,W] input —
// because that's what ggml_mean (which reduces only axis 0) can express after
// collapsing the trailing dims.
SupportResult IsSupportedReduceMeanNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  SUPPORT_CHECK(!inputs.empty() && inputs.size() <= 2 && outputs.size() == 1 &&
                    node.GetImplicitInputs().size() == 0,
                "ReduceMean: wrong input/output count");
  SUPPORT_CHECK(inputs[0] != nullptr && outputs[0] != nullptr, "ReduceMean: null input or output");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  SUPPORT_CHECK(x.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    y.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "ReduceMean: non-float type");
  SUPPORT_CHECK(rankSupportedByGGML(x) && rankSupportedByGGML(y),
                "ReduceMean: rank exceeds GGML_MAX_DIMS");

  const int64_t rank = static_cast<int64_t>(x.dims.size());
  SUPPORT_CHECK(rank > 0, "ReduceMean: scalar input");

  const bool keepdims = readNodeAttribute<int64_t>(node, "keepdims").value_or(1) != 0;
  const auto resolved = ResolveReduceAxes(node, x, y, keepdims, constants);
  SUPPORT_CHECK(resolved.has_value(),
                "ReduceMean: axes input is not a compile-time constant and "
                "cannot be derived from static output shape");
  std::vector<int64_t> axes = *resolved;
  SUPPORT_CHECK(!axes.empty(), "ReduceMean: empty axes");

  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t a : axes) {
    if (a < 0) a += rank;
    SUPPORT_CHECK(a >= 0 && a < rank, "ReduceMean: axis out of range");
    normalized.push_back(a);
  }
  std::sort(normalized.begin(), normalized.end());
  for (size_t i = 1; i < normalized.size(); ++i) {
    SUPPORT_CHECK(normalized[i] != normalized[i - 1], "ReduceMean: duplicate axes");
  }
  // Require a contiguous block (not necessarily trailing) so the permute
  // approach collapses them in one pass.
  for (size_t i = 1; i < normalized.size(); ++i) {
    SUPPORT_CHECK(normalized[i] == normalized[i - 1] + 1,
                  "ReduceMean: axes must be a contiguous block (non-contiguous at " +
                      std::to_string(normalized[i]) + ")");
  }
  SUPPORT_CHECK(static_cast<int64_t>(normalized.size()) + rank <= GGML_MAX_DIMS * 2,
                "ReduceMean: too many axes for GGML");
  return support_ok();
}

void CompileReduceMeanAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  const int64_t rank = static_cast<int64_t>(x.dims.size());

  const bool keepdims = readNodeAttribute<int64_t>(node, "keepdims").value_or(1) != 0;
  const auto resolved = ResolveReduceAxes(node, x, y, keepdims, constants);
  GGONNX_ASSERT(resolved.has_value(), "ReduceMean compile: axes could not be resolved");
  std::vector<int64_t> axes = *resolved;

  NodeDesc::ReduceAttrs attrs;
  attrs.trailing_count = static_cast<int>(axes.size());
  attrs.keepdims = keepdims;

  // Normalize axes and compute the GGML permutation that moves the reduction
  // GGML axes to positions [0, k-1] so the existing "leading GGML axes"
  // mean logic applies. For trailing ONNX axes this is the identity.
  std::vector<int64_t> onnx_axes = axes;
  for (int64_t& a : onnx_axes) { if (a < 0) a += rank; }
  std::sort(onnx_axes.begin(), onnx_axes.end());

  // In GGML (reversed dims) the ONNX axis i maps to GGML axis (rank-1-i).
  // Collect the GGML positions that need to be reduced, sorted ascending.
  std::vector<int> ggml_reduce_axes;
  for (int64_t a : onnx_axes) ggml_reduce_axes.push_back(static_cast<int>(rank - 1 - a));
  std::sort(ggml_reduce_axes.begin(), ggml_reduce_axes.end());

  // Build perm: put reduction axes first, then the rest in original order.
  std::array<int, GGML_MAX_DIMS> perm{0, 1, 2, 3};
  std::array<int, GGML_MAX_DIMS> inv_perm{0, 1, 2, 3};
  const int k = static_cast<int>(ggml_reduce_axes.size());
  std::vector<int> non_reduce;
  for (int i = 0; i < static_cast<int>(rank); ++i) {
    if (std::find(ggml_reduce_axes.begin(), ggml_reduce_axes.end(), i) == ggml_reduce_axes.end()) {
      non_reduce.push_back(i);
    }
  }
  for (int i = 0; i < k; ++i) perm[i] = ggml_reduce_axes[i];
  for (int i = 0; i < static_cast<int>(non_reduce.size()); ++i) perm[k + i] = non_reduce[i];
  // Fill unused GGML dims (rank < GGML_MAX_DIMS) as identity.
  for (int i = static_cast<int>(rank); i < GGML_MAX_DIMS; ++i) perm[i] = i;
  // Inverse permutation: inv_perm[perm[i]] = i.
  for (int i = 0; i < GGML_MAX_DIMS; ++i) inv_perm[perm[i]] = i;

  attrs.perm = perm;
  attrs.inv_perm = inv_perm;
  compiled_node->attrs = attrs;
}

// Shared reduce emit: applies permute → collapse → reduce_fn → reshape → inv_permute.
static EmitResult EmitReduceCore(ggml_context* ctx,
                                 const NodeDesc& node,
                                 const std::vector<ggml_tensor*>& values,
                                 ggml_tensor* (*reduce_fn)(ggml_context*, ggml_tensor*)) {
  const auto* attrs = std::get_if<NodeDesc::ReduceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Reduce node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Reduce node missing GGML input");

  const int k = attrs->trailing_count;

  // Step 1: permute so the reduction axes are at GGML positions [0, k-1].
  // Skip permute when perm is already the identity (trailing-axes case).
  // attrs->perm uses the "source index" convention (out[i] <- in[perm[i]]),
  // but ggml_permute takes the "destination index" convention (in[i] -> out[axisi]).
  // These are inverses, so pass inv_perm to the forward permute here.
  const auto& fwd = attrs->inv_perm;
  const bool is_identity_perm =
      (fwd[0] == 0 && fwd[1] == 1 && fwd[2] == 2 && fwd[3] == 3);
  ggml_tensor* xp =
      is_identity_perm ? x : ggml_permute(ctx, x, fwd[0], fwd[1], fwd[2], fwd[3]);

  // Step 2: collapse the k leading GGML axes into one, then reduce.
  int64_t collapsed = 1;
  for (int i = 0; i < k; ++i) collapsed *= xp->ne[i];
  int64_t keep[GGML_MAX_DIMS] = {1, 1, 1, 1};
  keep[0] = collapsed;
  for (int i = k; i < GGML_MAX_DIMS; ++i) keep[i - k + 1] = xp->ne[i];

  ggml_tensor* src = ggml_is_contiguous(xp) ? xp : ggml_cont(ctx, xp);
  ggml_tensor* flat = ggml_reshape_4d(ctx, src, keep[0], keep[1], keep[2], keep[3]);
  ggml_tensor* reduced = reduce_fn(ctx, flat);  // ne[0] becomes 1

  if (!attrs->keepdims) {
    // Drop the leading size-1 axis by repacking the kept axes.
    int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < GGML_MAX_DIMS - 1; ++i) out_ne[i] = reduced->ne[i + 1];
    return EmitOutputs{
        ggml_reshape_4d(ctx, reduced, out_ne[0], out_ne[1], out_ne[2], out_ne[3])};
  }

  // keepdims: expand the single collapsed axis back to k separate size-1 axes,
  // then permute back to the original GGML axis order.
  int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = k; i < GGML_MAX_DIMS; ++i) out_ne[i] = reduced->ne[i - k + 1];
  ggml_tensor* expanded =
      ggml_reshape_4d(ctx, reduced, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);

  // Apply inverse permutation to restore the original axis ordering.
  // Skip when perm was identity (trailing-axes case, no reorder needed).
  // Mirror of the forward step: pass attrs->perm here (the ggml "destination"
  // form of the inverse permutation).
  const auto& inv = attrs->perm;
  const bool is_identity_inv =
      (inv[0] == 0 && inv[1] == 1 && inv[2] == 2 && inv[3] == 3);
  return EmitOutputs{is_identity_inv
                         ? expanded
                         : ggml_permute(ctx, expanded, inv[0], inv[1], inv[2], inv[3])};
}

EmitResult EmitReduceMeanNode(ggml_context* ctx,
                              const NodeDesc& node,
                              const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  return EmitReduceCore(ctx, node, values, ggml_mean);
}

// ONNX ReduceSum: identical structure to ReduceMean; swap ggml_mean -> ggml_sum_rows.
EmitResult EmitReduceSumNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  return EmitReduceCore(ctx, node, values, ggml_sum_rows);
}

// Reduce-max along ggml axis 0: ggml has no max_rows, but max-pool 1D with
// kernel = stride = ne[0] collapses axis 0 to a single element — exactly a
// per-row max. Other axes pass through untouched.
static ggml_tensor* ggonnx_max_rows(ggml_context* ctx, ggml_tensor* a) {
  const int k = static_cast<int>(a->ne[0]);
  GGONNX_ASSERT(k > 0, "max-rows pool requires positive axis-0 size");
  return ggml_pool_1d(ctx, a, GGML_OP_POOL_MAX, k, k, 0);
}

// ReduceMin(x) = -ReduceMax(-x). Same pool_1d trick on the negated input.
static ggml_tensor* ggonnx_min_rows(ggml_context* ctx, ggml_tensor* a) {
  ggml_tensor* neg = ggml_neg(ctx, a);
  ggml_tensor* max_neg = ggonnx_max_rows(ctx, neg);
  return ggml_neg(ctx, max_neg);
}

EmitResult EmitReduceMaxNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  return EmitReduceCore(ctx, node, values, ggonnx_max_rows);
}

EmitResult EmitReduceMinNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  return EmitReduceCore(ctx, node, values, ggonnx_min_rows);
}

// ONNX DepthToSpace: [N, C, H, W] -> [N, C/b^2, H*b, W*b]. Two modes:
//   DCR (default) reshapes C as [b, b, C/b^2] (row-block outer, col-block middle).
//   CRD            reshapes C as [C/b^2, b, b] (channel outer, then row, col).
// A faithful lowering needs a 5D intermediate (ggml max is 4D), so we split it
// into two 4D permute+reshape passes — one expands H, one expands W — by
// factoring the C axis into two at a time. With N == 1 the batch axis is free
// and both passes fit in 4D directly. For N > 1 we first pay an NCHW -> CNHW
// permute to fold N into the height axis, run the N == 1 shuffle on the fat
// height, then unfold and swap N/Cout back. The N == 1 fast path is kept
// separate — waifu2x / image SR hit it almost exclusively and we don't want
// to pay the extra conts there.
//
// KNOWN INEFFICIENCY: the N == 1 path emits two ggml_cont copies (one per
// pass, each materializing a permuted view before the following reshape can
// collapse two axes). Within 4D ggml ops this is the floor — the output index
// 'wb*b + cb' can't be encoded as a constant-stride axis over the input, so
// no single strided view can feed a lone cont. The N > 1 path stacks another
// two conts on top of that (input N<->C and output N<->Cout). A custom kernel
// (ggml_custom_4d with hand-rolled stride math) could collapse everything to
// a single pass but we're staying pure-ggml here; revisit if a profile pins a
// hot model on this op.

// Emit the 2-pass shuffle for a tensor with ne=[W, H, C, 1]; returns
// ne=[W*b, H*b, C/(b*b), 1]. Requires `src` to be contiguous.
static ggml_tensor* EmitDepthToSpaceShuffleBatch1(ggml_context* ctx,
                                                  ggml_tensor* src,
                                                  int b,
                                                  bool crd) {
  const int64_t W = src->ne[0];
  const int64_t H = src->ne[1];
  const int64_t C = src->ne[2];
  const int64_t Cout = C / (b * b);

  if (!crd) {
    // DCR: channel c = r*b*Cout + c_blk*Cout + cc.
    // Pass 1 — expand H. Factor C as (b_row outer, b_col*Cout inner).
    // ggml ne=[W, H, b_col*Cout, b_row] -> permute (b_row between H and W-inner) ->
    // ne=[W, b_row, H, b_col*Cout], then collapse axes 1,2 into H*b.
    ggml_tensor* v1 = ggml_reshape_4d(ctx, src, W, H, b * Cout, b);
    ggml_tensor* p1 = ggml_cont(ctx, ggml_permute(ctx, v1, 0, 2, 3, 1));
    ggml_tensor* r1 = ggml_reshape_4d(ctx, p1, W, H * b, b * Cout, 1);

    // Pass 2 — expand W. Factor b_col*Cout as (b_col outer, Cout inner).
    // ggml ne=[W, H*b, Cout, b_col] -> permute (b_col innermost) ->
    // ne=[b_col, W, H*b, Cout], then collapse axes 0,1 into W*b.
    ggml_tensor* v2 = ggml_reshape_4d(ctx, r1, W, H * b, Cout, b);
    ggml_tensor* p2 = ggml_cont(ctx, ggml_permute(ctx, v2, 1, 2, 3, 0));
    return ggml_reshape_4d(ctx, p2, W * b, H * b, Cout, 1);
  }

  // CRD: channel c = cc*b*b + r*b + c_blk.
  // Pass 1 — expand W. Factor C as (Cout*b_row outer, b_col inner).
  // ggml ne=[W, H, b_col, Cout*b_row] -> permute (b_col innermost) ->
  // ne=[b_col, W, H, Cout*b_row], then collapse axes 0,1 into W*b.
  ggml_tensor* v1 = ggml_reshape_4d(ctx, src, W, H, b, Cout * b);
  ggml_tensor* p1 = ggml_cont(ctx, ggml_permute(ctx, v1, 1, 2, 0, 3));
  ggml_tensor* r1 = ggml_reshape_4d(ctx, p1, W * b, H, Cout * b, 1);

  // Pass 2 — expand H. Factor Cout*b_row as (Cout outer, b_row inner).
  // ggml ne=[W*b, H, b_row, Cout] -> permute (b_row between H and W) ->
  // ne=[W*b, b_row, H, Cout], then collapse axes 1,2 into H*b.
  ggml_tensor* v2 = ggml_reshape_4d(ctx, r1, W * b, H, b, Cout);
  ggml_tensor* p2 = ggml_cont(ctx, ggml_permute(ctx, v2, 0, 2, 1, 3));
  return ggml_reshape_4d(ctx, p2, W * b, H * b, Cout, 1);
}

SupportResult IsSupportedDepthToSpaceNode(Ort::ConstNode node, const MetaAnalysis* meta) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (in.dims.size() != 4 || out.dims.size() != 4) return false;
  const auto in_shape = (meta != nullptr) ? ResolveShape(inputs[0], *meta) : std::nullopt;
  const auto out_shape = (meta != nullptr) ? ResolveShape(outputs[0], *meta) : std::nullopt;
  if (!in_shape.has_value() || !out_shape.has_value()) return false;

  const auto blocksize = readNodeAttribute<int64_t>(node, "blocksize");
  if (!blocksize.has_value() || *blocksize < 1) return false;
  const int64_t b = *blocksize;
  if ((*in_shape)[1] % (b * b) != 0) return false;

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("DCR");
  if (mode != "DCR" && mode != "CRD") return false;

  return true;
}

void CompileDepthToSpaceAttributes(Ort::ConstNode node, NodeDesc* compiled_node, const MetaAnalysis* /*meta*/) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::DepthToSpaceAttrs attrs;
  attrs.blocksize = static_cast<int>(*readNodeAttribute<int64_t>(node, "blocksize"));
  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("DCR");
  attrs.crd = (mode == "CRD");
  compiled_node->attrs = attrs;
}

EmitResult EmitDepthToSpaceNode(ggml_context* ctx,
                                const NodeDesc& node,
                                const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::DepthToSpaceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled DepthToSpace node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled DepthToSpace node missing GGML input");

  const int b = attrs->blocksize;
  const int64_t W = x->ne[0];
  const int64_t H = x->ne[1];
  const int64_t C = x->ne[2];
  const int64_t N = x->ne[3];
  GGONNX_ASSERT(C % (b * b) == 0, "DepthToSpace: C not divisible by blocksize^2");
  const int64_t Cout = C / (b * b);

  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);

  if (N == 1) {
    return EmitOutputs{EmitDepthToSpaceShuffleBatch1(ctx, src, b, attrs->crd)};
  }

  // N > 1: permute ONNX NCHW -> CNHW so N sits next to H in memory, then collapse
  // (N, H) into one axis and run the N==1 shuffle on the fat height. In ggml C is
  // at ne[2] and N at ne[3], so the permute swaps axes 2 and 3.
  ggml_tensor* cnhw = ggml_cont(ctx, ggml_permute(ctx, src, 0, 1, 3, 2));
  // ne=[W, H, N, C] contiguous -> ne=[W, N*H, C, 1]. N and H are adjacent with H
  // inner (stride W) and N outer (stride W*H), so the collapsed axis indexes as
  // n*H + h, which is exactly what the shuffle will later invert.
  ggml_tensor* folded = ggml_reshape_4d(ctx, cnhw, W, N * H, C, 1);

  ggml_tensor* shuffled = EmitDepthToSpaceShuffleBatch1(ctx, folded, b, attrs->crd);
  // shuffled ne=[W*b, N*H*b, Cout, 1]. The expanded height encodes n*H*b + h*b + r
  // = n*(H*b) + (h*b + r), so splitting it as (H*b inner, N outer) is a pure view.
  // The resulting ne=[W*b, H*b, N, Cout] means ONNX [Cout, N, H*b, W*b] — one more
  // permute (swap N <-> Cout) lands us at the expected ONNX [N, Cout, H*b, W*b].
  ggml_tensor* split = ggml_reshape_4d(ctx, shuffled, W * b, H * b, N, Cout);
  return EmitOutputs{ggml_cont(ctx, ggml_permute(ctx, split, 0, 1, 3, 2))};
}

// Synthetic op: general fused Reshape(4D->XD)->Transpose->Reshape(XD->4D)
// where X > GGML_MAX_DIMS. Consecutive ONNX axes that travel together through
// the transpose are merged into single GGML axes (stored in grouped_ggml_dims),
// and the inter-group permutation is a plain rank-4 ggml_permute. This handles
// any such triple that coalesces to ≤4 groups — including Swin window-shuffle
// (rank-6, perm=[0,1,3,2,4,5]) and ShuffleNet channel-shuffle (rank-5,
// perm=[0,2,1,3,4]) as special cases.
EmitResult EmitGenericShuffleNode(ggml_context* ctx,
                                  const NodeDesc& node,
                                  const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::GenericShuffleAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled __GenericShuffle node missing attributes");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled __GenericShuffle node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled __GenericShuffle node missing GGML input");

  const auto& gd = attrs->grouped_ggml_dims;
  const auto& p = attrs->ggml_perm;
  for (int64_t d : gd) GGONNX_ASSERT(d > 0, "__GenericShuffle grouped dims must be positive");

  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  ggml_tensor* grouped = ggml_reshape_4d(ctx, src, gd[0], gd[1], gd[2], gd[3]);
  ggml_tensor* permuted = ggml_permute(ctx, grouped, p[0], p[1], p[2], p[3]);
  ggml_tensor* packed = ggml_cont(ctx, permuted);

  const std::array<int64_t, GGML_MAX_DIMS> target =
      ToPaddedGGMLDims(attrs->output_onnx_dims);
  ggml_tensor* out =
      ggml_reshape_4d(ctx, packed, target[0], target[1], target[2], target[3]);
  return EmitOutputs{out};
}

// Synthetic op: fused Reshape([B, M*M, 3C]->[B, M*M, 3, H, D]) ->
// Transpose(perm=[2,0,3,1,4]) -> Split(axis=0) -> 3x Squeeze(axes=[0]).
// The rank-5 intermediates never materialize: we reinterpret the packed input
// as rank-4 ne=[D, 3H, M*M, B], take Q/K/V as strided views at offsets
// {0, H*D, 2*H*D} scalars into that buffer, and swap the heads/tokens axes
// to land at the canonical attention layout ne=[D, M*M, H, B]. The final cont
// matches what a downstream ggml_mul_mat would already pay for.
EmitResult EmitQKVSplitNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::QKVSplitAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled __QKVSplit node missing attributes");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 3,
                "compiled __QKVSplit node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled __QKVSplit node missing GGML input");

  const int64_t H = attrs->num_heads;
  const int64_t D = attrs->num_heads > 0 ? attrs->head_dim : 0;
  const int64_t T = attrs->num_tokens;
  const int64_t B = attrs->num_batch;
  GGONNX_ASSERT(H > 0 && D > 0 && T > 0 && B > 0,
                "QKVSplit dims must be positive");

  // Collapse the input to a contiguous rank-4 view [D, 3H, T, B]. The input is
  // produced by a Gemm/MatMul so it's normally contiguous; the cont guards
  // against a non-standard producer.
  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  ggml_tensor* packed = ggml_reshape_4d(ctx, src, D, 3 * H, T, B);

  const size_t type_size = ggml_type_size(packed->type);
  const size_t nb1 = static_cast<size_t>(packed->nb[1]);
  const size_t nb2 = static_cast<size_t>(packed->nb[2]);
  const size_t nb3 = static_cast<size_t>(packed->nb[3]);
  // Q/K/V slice offsets along the packed `3H` axis.
  const std::array<size_t, 3> offsets{
      0,
      static_cast<size_t>(H) * D * type_size,
      2 * static_cast<size_t>(H) * D * type_size,
  };

  EmitOutputs out;
  out.reserve(3);
  for (size_t i = 0; i < 3; ++i) {
    // View of shape [D, H, T, B] starting at the i-th QKV chunk.
    ggml_tensor* view = ggml_view_4d(ctx, packed, D, H, T, B, nb1, nb2, nb3, offsets[i]);
    // Swap heads <-> tokens so the output is [D, T, H, B].
    ggml_tensor* permuted = ggml_permute(ctx, view, 0, 2, 1, 3);
    out.push_back(ggml_cont(ctx, permuted));
  }
  return out;
}

SupportResult get_node_support(Ort::ConstNode node, const MetaAnalysis* meta) {
  const ConstantValueMap* constants = (meta != nullptr) ? &meta->constants : nullptr;
  try {
    if (node == nullptr) {
      return support_error("node metadata must not be null");
    }

    const Ort::ConstGraph graph = node.GetGraph();
    if (graph.GetParentNode() != nullptr) {
      for (Ort::ConstValueInfo output : node.GetOutputs()) {
        if (output == nullptr) continue;
        const auto meta = try_get_tensor_metadata(output);
        if (!meta.has_value()) {
          auto result = support_error("subgraph output '" + std::string(output.GetName()) +
                                     "' is not a usable tensor: " + meta.error());
          trace_support_decision(node, result);
          return result;
        }
        if (meta.value().element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            meta.value().dims.empty()) {
          auto result = support_error("subgraph scalar float outputs are not supported");
          trace_support_decision(node, result);
          return result;
        }
      }
    }

    auto has_zero_dim = [](Ort::ConstValueInfo vi) -> SupportResult {
      if (vi == nullptr) return support_ok();
      const auto meta = try_get_tensor_metadata(vi);
      if (!meta.has_value()) return support_ok();
      for (int64_t dim : meta.value().dims) {
        if (dim == 0) {
          return support_error("zero-sized tensors are not supported");
        }
      }
      return support_ok();
    };

    for (Ort::ConstValueInfo input : node.GetInputs()) {
      SupportResult result = has_zero_dim(input);
      if (!result.has_value()) {
        trace_support_decision(node, result);
        return result;
      }
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      SupportResult result = has_zero_dim(output);
      if (!result.has_value()) {
        trace_support_decision(node, result);
        return result;
      }
    }

    const OpDefinition* op = FindOpDefinition(node.GetDomain(), node.GetOperatorType());
    if (op == nullptr || op->support == nullptr) {
      auto result = support_error("operator is not registered in GGONNX");
      trace_support_decision(node, result);
      return result;
    }

    SupportResult result = normalize_support_result(node, op->support(node, meta));
    trace_support_decision(node, result);
    return result;
  } catch (const Ort::Exception& ex) {
    SupportResult result = support_error(std::string("ORT exception in support check: ") + ex.what());
    trace_support_decision(node, result);
    return result;
  } catch (const std::exception& ex) {
    SupportResult result = support_error(std::string("support check threw: ") + ex.what());
    trace_support_decision(node, result);
    return result;
  }
}

const OpDefinition* FindOpDefinition(std::string_view domain, std::string_view op_type) {

  struct PairHash {
    size_t operator()(const std::pair<std::string_view, std::string_view>& p) const {
      return std::hash<std::string_view>{}(p.first) ^ std::hash<std::string_view>{}(p.second);
    }
  };

  static const std::unordered_map<std::pair<std::string_view, std::string_view>, const OpDefinition, PairHash> ops_table = {
      {{"", "Add"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Sub"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Mul"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Div"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Max"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Min"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Mod"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "GRU"}, {IsSupportedGRUNode, CompileGRUAttributes, EmitGRUNode}},
      {{"", "LSTM"}, {IsSupportedLSTMNode, CompileLSTMAttributes, EmitLSTMNode}},
      {{"", "Relu"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sigmoid"},  {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Tanh"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Neg"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Abs"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sqrt"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Exp"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Log"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Erf"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Reciprocal"}, {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Softplus"}, {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Elu"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Pow"},      {IsSupportedPowNode, nullptr, EmitPowNode}},
      {{"", "CumSum"},   {IsSupportedCumSumNode, nullptr, EmitCumSumNode}},
      {{"", "Range"},    {IsSupportedRangeNode, CompileRangeAttributes, EmitRangeNode}},
      {{"", "LeakyRelu"}, {IsSupportedLeakyReluNode, CompileLeakyReluAttributes, EmitLeakyReluNode}},
      {{"", "PRelu"},     {IsSupportedPReluNode, nullptr, EmitPReluNode}},
      {{"", "Clip"},      {IsSupportedClipNode, CompileClipAttributes, EmitClipNode}},
      {{"", "Softmax"},  {IsSupportedSoftmaxNode, CompileSoftmaxAttributes, EmitSoftmaxNode}},
      {{"", "MatMul"},   {IsSupportedMatMulNode, CompileMatMulAttributes, EmitMatMulNode, MatMulConstantLayout}},
      {{"", "Conv"},     {IsSupportedConvNode, CompileConvAttributes, EmitConvNode, nullptr}},
      {{"", "ConvTranspose"}, {IsSupportedConvTransposeNode, CompileConvTransposeAttributes, EmitConvTransposeNode}},
      {{"", "Expand"},   {IsSupportedExpandNode, CompileExpandAttributes, EmitExpandNode}},
      {{"", "Gemm"},     {IsSupportedGemmNode, CompileGemmAttributes, EmitGemmNode, GemmConstantLayout}},
      {{"", "Reshape"},  {IsSupportedReshapeNode, CompileReshapeAttributes, EmitReshapeNode}},
      {{"", "Flatten"},  {IsSupportedFlattenNode, CompileFlattenAttributes, EmitReshapeNode}},
      {{"", "Squeeze"},   {IsSupportedSqueezeNode, CompileSqueezeAttributes, EmitSqueezeNode}},
      {{"", "Unsqueeze"}, {IsSupportedSqueezeNode, CompileSqueezeAttributes, EmitSqueezeNode}},
      {{"", "MaxPool"},       {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "AveragePool"},   {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "GlobalMaxPool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
      {{"", "GlobalAveragePool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
      {{"", "Pad"}, {IsSupportedPadNode, CompilePadAttributes, EmitPadNode}},
      {{"", "InstanceNormalization"},
       {IsSupportedInstanceNormalizationNode, CompileInstanceNormalizationAttributes, EmitInstanceNormalizationNode}},
      {{"", "LayerNormalization"},
       {IsSupportedLayerNormalizationNode, CompileLayerNormalizationAttributes, EmitLayerNormalizationNode}},
      {{"", "RMSNormalization"},
       {IsSupportedRMSNormalizationNode, CompileRMSNormalizationAttributes, EmitRMSNormalizationNode}},
      {{"", "Dropout"}, {IsSupportedDropoutNode, nullptr, EmitDropoutNode}},
      {{"", "Upsample"}, {IsSupportedUpsampleNode, CompileUpsampleAttributes, EmitUpsampleNode}},
      {{"", "Resize"}, {IsSupportedUpsampleNode, CompileUpsampleAttributes, EmitUpsampleNode}},
      {{"", "Identity"}, {IsSupportedIdentityNode, nullptr, EmitIdentityNode}},
      {{"", "Cast"}, {IsSupportedCastNode, CompileCastAttributes, EmitCastNode}},
      {{"", "Transpose"}, {IsSupportedTransposeNode, CompileTransposeAttributes, EmitTransposeNode}},
      {{"", "Concat"}, {IsSupportedConcatNode, CompileConcatAttributes, EmitConcatNode}},
      {{"", "Slice"}, {IsSupportedSliceNode, CompileSliceAttributes, EmitSliceNode}},
      {{"", "Split"}, {IsSupportedSplitNode, CompileSplitAttributes, EmitSplitNode}},
      {{"", "ReduceMean"},
       {IsSupportedReduceMeanNode, CompileReduceMeanAttributes, EmitReduceMeanNode}},
      {{"", "ReduceSum"},
       {IsSupportedReduceMeanNode, CompileReduceMeanAttributes, EmitReduceSumNode}},
      {{"", "ReduceMax"},
       {IsSupportedReduceMeanNode, CompileReduceMeanAttributes, EmitReduceMaxNode}},
      // ReduceMin disabled: EmitReduceMinNode works in isolation (conformance +
      // standalone tests pass) but tiny-yolov3 regresses heavily when enabled.
      // Investigation needed — suspected interaction with Vulkan scheduler or
      // neg/pool_1d/neg chain on a specific shape class the tests don't cover.
      // {{"", "ReduceMin"},
      //  {IsSupportedReduceMeanNode, CompileReduceMeanAttributes, EmitReduceMinNode}},
      {{"", "BatchNormalization"},
       {IsSupportedBatchNormNode, CompileBatchNormAttributes, EmitBatchNormNode}},
      {{"", "DepthToSpace"},
       {IsSupportedDepthToSpaceNode, CompileDepthToSpaceAttributes, EmitDepthToSpaceNode}},
      // Synthetic op injected by FusionPlan. `support` and `compile_attrs` are
      // null because the fusion machinery builds the NodeDesc directly and the
      // per-node support check is never called on __WindowShuffle.
      {{"", "__GenericShuffle"}, {nullptr, nullptr, EmitGenericShuffleNode}},
      {{"", "__QKVSplit"}, {nullptr, nullptr, EmitQKVSplitNode}},
  };

  auto it = ops_table.find({domain, op_type});
  if (it != ops_table.end()) {
    return &it->second;
  }
  return nullptr;
}
