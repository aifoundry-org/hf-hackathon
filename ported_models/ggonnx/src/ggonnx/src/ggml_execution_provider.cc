#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"
#include "meta_eval.hpp"
#include "ops.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace {

constexpr const char* kRegistrationName = "GGONNX";
constexpr const char* kEpName = "GGMLExecutionProvider";
constexpr const char* kVendorName = "nekko";
constexpr const char* kVersion = "0.0.1";

std::atomic<uint64_t> g_debug_graph_build_count{0};
std::atomic<uint64_t> g_debug_last_graph_node_count{0};
std::atomic<uint64_t> g_debug_last_graph_output_count{0};

using ggonnx::ort_internal::GetOrtApi;
using ggonnx::ort_internal::GetOrtEpApi;
using ggonnx::ort_internal::InitializeOrtApi;
using ggonnx::ort_internal::IsOrtApiInitialized;
using ggonnx::ort_internal::MakeStatus;
using ggonnx::ort_internal::THROW_ON_ERROR;
using ggonnx::ort_internal::WrapStatus;

struct GGMLFactory {
  OrtEpFactory iface{};
  std::string registered_name{kRegistrationName};
};

// Backend selection resolved from EP provider options at session create time.
// Shared (by pointer) between the EP instance and every CompiledPartition it
// produces, so that the partitions outlive the EP object safely.
struct BackendSelection {
  // Ordered list of ggml devices we'll actually execute on. Index 0 is the
  // "primary" backend — constant buffers land on its buffer type, and it's
  // used directly (no scheduler) when the list has exactly one entry.
  std::vector<ggml_backend_dev_t> devices;
  // Human-readable names, parallel to `devices`, for logging / compatibility
  // info strings (so EP-context caches don't get shared across device picks).
  std::vector<std::string> device_names;
  bool enable_cpu_fallback{true};
  // When true, all MatMul outputs are tagged with GGML_PREC_F32 so backends
  // don't silently accumulate in fp16. Default false — matches ggml's usual
  // behavior where fp16-capable GPU backends pick the faster fp16-accumulation
  // kernels. Set via `ep.ggonnx.matmul_precision=f32`.
  bool force_matmul_f32{false};
  int n_threads{0};
};

struct GGMLEp {
  OrtEp iface{};
  const OrtLogger* logger{};
  std::vector<std::string> selected_devices;
  std::shared_ptr<BackendSelection> selection;
  // Computed once over the FULL graph in EpGetCapability and reused by every
  // CompilePartition call. Re-running AnalyzeCompileTimeConstants on a single
  // partition subgraph gives different (less complete) results because
  // foldable ancestors may have been split into a different partition — their
  // outputs then arrive as opaque boundary inputs and downstream folding
  // breaks. Keeping the full-graph analysis avoids that divergence.
  std::shared_ptr<const MetaAnalysis> capability_meta;
};

struct ValueDesc {
  std::string name;
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
  bool is_graph_input{};
  bool is_graph_output{};
};

struct CompiledPartition {
  std::vector<ValueDesc> values;
  std::vector<size_t> graph_inputs;
  std::vector<size_t> graph_outputs;
  // Kernel-input-index → value_id. Non-float kernel inputs (e.g. Reshape's
  // int64 shape tensor) are represented as kOptionalValueAbsent and ignored
  // at runtime — their semantic role was already consumed at compile time.
  std::vector<size_t> kernel_input_to_value;
  std::vector<NodeDesc> nodes;
  // Persistent ggml context holding pre-materialized constant initializers.
  // Entries in `constants` are indexed by value_id; nullptr means "not a
  // constant". Constants never appear in graph_inputs — their data is snapshotted
  // at compile time, with per-op layout hints (e.g. MatMul B is pre-transposed).
  // backends[0] is the primary backend: constants live on its default buffer
  // type, and when the list has exactly one entry the partition uses the
  // direct ggml_backend_graph_compute path (no scheduler). With >1 backends,
  // `sched` is built over the list and handles per-op backend assignment and
  // cross-backend tensor copies.
  std::vector<ggml_backend_t> backends;
  ggml_backend_sched_t sched{nullptr};
  // Cached from BackendSelection so the emit path (which runs at graph-build
  // time, not compile time) can apply the precision hint without threading
  // the selection all the way through.
  bool force_matmul_f32_prec{false};
  ggml_context* constant_ctx{nullptr};
  ggml_backend_buffer_t constant_buffer{nullptr};
  std::vector<ggml_tensor*> constants;
  std::vector<std::optional<ConstantTensor>> folded_constants;
  // Meta-eval pseudo-constants whose data contains dynamic-shape sentinels
  // bound to runtime input dims. Materialized per-graph in
  // BuildMaterializedGraph (not in the persistent constant_ctx) because their
  // resolved values change with the runtime input shape key.
  std::unordered_map<size_t, ConstantTensor> shape_bound_constants;

  ggml_backend_t primary_backend() const { return backends.empty() ? nullptr : backends.front(); }

  CompiledPartition() = default;
  CompiledPartition(const CompiledPartition&) = delete;
  CompiledPartition& operator=(const CompiledPartition&) = delete;
  CompiledPartition(CompiledPartition&& other) noexcept { *this = std::move(other); }
  CompiledPartition& operator=(CompiledPartition&& other) noexcept {
    if (this == &other) return *this;
    // Destroy in reverse-creation order: sched references backends, buffers
    // reference the backend they were allocated on.
    if (sched != nullptr) ggml_backend_sched_free(sched);
    if (constant_buffer != nullptr) ggml_backend_buffer_free(constant_buffer);
    if (constant_ctx != nullptr) ggml_free(constant_ctx);
    for (ggml_backend_t b : backends) {
      if (b != nullptr) ggml_backend_free(b);
    }
    values = std::move(other.values);
    graph_inputs = std::move(other.graph_inputs);
    graph_outputs = std::move(other.graph_outputs);
    kernel_input_to_value = std::move(other.kernel_input_to_value);
    nodes = std::move(other.nodes);
    backends = std::move(other.backends);
    sched = other.sched;
    constant_ctx = other.constant_ctx;
    constant_buffer = other.constant_buffer;
    constants = std::move(other.constants);
    folded_constants = std::move(other.folded_constants);
    shape_bound_constants = std::move(other.shape_bound_constants);
    other.sched = nullptr;
    other.constant_ctx = nullptr;
    other.constant_buffer = nullptr;
    return *this;
  }
  ~CompiledPartition() {
    if (sched != nullptr) ggml_backend_sched_free(sched);
    if (constant_buffer != nullptr) ggml_backend_buffer_free(constant_buffer);
    if (constant_ctx != nullptr) ggml_free(constant_ctx);
    for (ggml_backend_t b : backends) {
      if (b != nullptr) ggml_backend_free(b);
    }
  }
};

struct ShapeKey {
  std::vector<std::vector<int64_t>> input_dims;

  bool operator==(const ShapeKey& other) const {
    return input_dims == other.input_dims;
  }

  bool operator!=(const ShapeKey& other) const {
    return !(*this == other);
  }
};

struct MaterializedGraph {
  ShapeKey key;
  ggml_context* ctx{};
  ggml_cgraph* graph{};
  ggml_gallocr_t gallocr{};
  std::vector<ggml_tensor*> values;
  std::vector<std::vector<int64_t>> value_dims;
  std::vector<ggml_tensor*> input_tensors;
  std::vector<ggml_tensor*> output_tensors;
};

struct GGMLComputeState {
  const CompiledPartition* partition{nullptr};  // owned by GGMLNodeComputeInfo
  std::unique_ptr<MaterializedGraph> active_graph;
};

struct GGMLNodeComputeInfo {
  OrtNodeComputeInfo iface{};
  CompiledPartition partition;
};

GGMLFactory* AsFactory(OrtEpFactory* factory) {
  return reinterpret_cast<GGMLFactory*>(factory);
}

const GGMLFactory* AsFactory(const OrtEpFactory* factory) {
  return reinterpret_cast<const GGMLFactory*>(factory);
}

GGMLEp* AsEp(OrtEp* ep) {
  return reinterpret_cast<GGMLEp*>(ep);
}

const GGMLEp* AsEp(const OrtEp* ep) {
  return reinterpret_cast<const GGMLEp*>(ep);
}

GGMLNodeComputeInfo* AsNodeComputeInfo(OrtNodeComputeInfo* info) {
  return reinterpret_cast<GGMLNodeComputeInfo*>(info);
}

GGMLComputeState* AsComputeState(void* state) {
  return reinterpret_cast<GGMLComputeState*>(state);
}

std::string DescribeHardwareDevice(const OrtHardwareDevice* device) {
  GGONNX_NOT_NULL(device, "hardware device pointer must not be null");
  const char* vendor_ptr = GGONNX_NOT_NULL(GetOrtApi().HardwareDevice_Vendor(device),
                                          "hardware device vendor must not be null");
  std::string vendor = vendor_ptr;
  const auto device_type = GetOrtApi().HardwareDevice_Type(device);
  const auto vendor_id = GetOrtApi().HardwareDevice_VendorId(device);
  const auto device_id = GetOrtApi().HardwareDevice_DeviceId(device);

  return vendor + ":" + std::to_string(static_cast<int>(device_type)) + ":" +
         std::to_string(vendor_id) + ":" + std::to_string(device_id);
}

// Load every ggml backend the runtime can find (dynamically registered CUDA,
// Metal, Vulkan, SYCL, RPC, …) exactly once, then enumerate what landed in the
// registry so the rest of the EP can reason about available devices.
void EnsureGgmlBackendsLoaded() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    ggml_backend_load_all();
    const size_t dev_count = ggml_backend_dev_count();
    std::fprintf(stderr, "[ggonnx] ggml backends available: %zu device(s)\n", dev_count);
    for (size_t i = 0; i < dev_count; ++i) {
      ggml_backend_dev_t dev = ggml_backend_dev_get(i);
      if (dev == nullptr) continue;
      const char* name = ggml_backend_dev_name(dev);
      const char* desc = ggml_backend_dev_description(dev);
      const auto type = ggml_backend_dev_type(dev);
      const char* type_str = "UNKNOWN";
      switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   type_str = "CPU";   break;
        case GGML_BACKEND_DEVICE_TYPE_GPU:   type_str = "GPU";   break;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  type_str = "IGPU";  break;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: type_str = "ACCEL"; break;
      }
      std::fprintf(stderr, "[ggonnx]   [%zu] %s (%s) — %s\n", i,
                   name != nullptr ? name : "?",
                   type_str,
                   desc != nullptr ? desc : "");
    }
  });
}

// Pick the "best" available ggml device when no explicit selection is given.
// Ranking: GPU > IGPU > ACCEL > CPU.
ggml_backend_dev_t AutoSelectPrimaryDevice() {
  auto rank = [](enum ggml_backend_dev_type t) -> int {
    switch (t) {
      case GGML_BACKEND_DEVICE_TYPE_GPU:   return 3;
      case GGML_BACKEND_DEVICE_TYPE_IGPU:  return 2;
      case GGML_BACKEND_DEVICE_TYPE_ACCEL: return 1;
      case GGML_BACKEND_DEVICE_TYPE_CPU:   return 0;
    }
    return -1;
  };
  ggml_backend_dev_t chosen = nullptr;
  int best_rank = -1;
  const size_t dev_count = ggml_backend_dev_count();
  for (size_t i = 0; i < dev_count; ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (dev == nullptr) continue;
    const int r = rank(ggml_backend_dev_type(dev));
    if (r > best_rank) {
      best_rank = r;
      chosen = dev;
    }
  }
  return chosen;
}

ggml_backend_dev_t GetCpuDevice() {
  return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

// Read a provider option set by the user through SessionOptionsAppendExecutionProvider_V2.
// ORT stores these as session config entries under `ep.<ep_name>.<key>`.
std::optional<std::string> LookupSessionConfig(const OrtSessionOptions* session_options,
                                               const std::string& full_key) {
  int has = 0;
  OrtStatus* status = GetOrtApi().HasSessionConfigEntry(session_options, full_key.c_str(), &has);
  if (status != nullptr) {
    GetOrtApi().ReleaseStatus(status);
    return std::nullopt;
  }
  if (!has) return std::nullopt;
  size_t size = 0;
  status = GetOrtApi().GetSessionConfigEntry(session_options, full_key.c_str(), nullptr, &size);
  if (status != nullptr) {
    GetOrtApi().ReleaseStatus(status);
    return std::nullopt;
  }
  std::string value(size, '\0');
  status = GetOrtApi().GetSessionConfigEntry(session_options, full_key.c_str(), value.data(), &size);
  if (status != nullptr) {
    GetOrtApi().ReleaseStatus(status);
    return std::nullopt;
  }
  // `size` is the string length including the trailing null; strip it.
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

std::optional<std::string> GetEpOption(const OrtSessionOptions* session_options,
                                       const char* key) {
  if (session_options == nullptr || key == nullptr) return std::nullopt;
  // Per ORT plugin-EP convention, options from SessionOptionsAppendExecutionProvider_V2
  // land in session config under "ep.<lowercased_ep_name>.<key>".
  return LookupSessionConfig(session_options, std::string("ep.ggmlexecutionprovider.") + key);
}

std::vector<std::string> SplitCsv(const std::string& in) {
  std::vector<std::string> out;
  std::string token;
  std::istringstream iss(in);
  while (std::getline(iss, token, ',')) {
    // trim whitespace
    const auto first = token.find_first_not_of(" \t");
    const auto last = token.find_last_not_of(" \t");
    if (first == std::string::npos) continue;
    out.emplace_back(token.substr(first, last - first + 1));
  }
  return out;
}

bool ParseBoolOption(const std::string& value, bool default_value) {
  if (value.empty()) return default_value;
  if (value == "1" || value == "true" || value == "True" || value == "TRUE" || value == "on") return true;
  if (value == "0" || value == "false" || value == "False" || value == "FALSE" || value == "off") return false;
  return default_value;
}

// Turn user-provided options into a concrete BackendSelection. Unknown device
// names are a hard error (typos surface loudly instead of silently downgrading).
BackendSelection ResolveBackendSelection(const OrtSessionOptions* session_options) {
  EnsureGgmlBackendsLoaded();

  BackendSelection selection;
  const auto device_list_opt = GetEpOption(session_options, "ggml_device_list");
  const auto fallback_opt = GetEpOption(session_options, "enable_cpu_fallback");
  const auto precision_opt = GetEpOption(session_options, "matmul_precision");
  const auto threads_opt = GetEpOption(session_options, "n_threads");
  selection.enable_cpu_fallback = ParseBoolOption(fallback_opt.value_or(""), /*default=*/true);
  if (threads_opt.has_value() && !threads_opt->empty()) {
    selection.n_threads = std::stoi(*threads_opt);
  } else {
    // Default to half of available hardware threads.
    const unsigned int hw = std::thread::hardware_concurrency();
    selection.n_threads = std::max(1u, hw / 2);
  }
  if (precision_opt.has_value()) {
    const std::string& p = *precision_opt;
    if (p == "f32" || p == "fp32" || p == "float32") {
      selection.force_matmul_f32 = true;
    } else if (!p.empty() && p != "default") {
      throw std::runtime_error("ep.ggonnx.matmul_precision: unknown value '" + p +
                               "' (expected 'default' or 'f32')");
    }
  }

  if (device_list_opt.has_value() && !device_list_opt->empty()) {
    for (const std::string& name : SplitCsv(*device_list_opt)) {
      ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
      if (dev == nullptr) {
        throw std::runtime_error("ep.ggonnx.ggml_device_list: unknown ggml device name '" + name + "'");
      }
      selection.devices.push_back(dev);
      selection.device_names.push_back(name);
    }
  } else {
    ggml_backend_dev_t primary = AutoSelectPrimaryDevice();
    if (primary != nullptr) {
      selection.devices.push_back(primary);
      const char* n = ggml_backend_dev_name(primary);
      selection.device_names.emplace_back(n != nullptr ? n : "");
    }
  }

  // Append CPU for fallback if requested and not already present.
  if (selection.enable_cpu_fallback) {
    ggml_backend_dev_t cpu = GetCpuDevice();
    const bool already_have_cpu = cpu != nullptr &&
        std::find(selection.devices.begin(), selection.devices.end(), cpu) != selection.devices.end();
    if (cpu != nullptr && !already_have_cpu) {
      selection.devices.push_back(cpu);
      const char* n = ggml_backend_dev_name(cpu);
      selection.device_names.emplace_back(n != nullptr ? n : "CPU");
    }
  }

  if (selection.devices.empty()) {
    // Last-resort: something must run. Force CPU.
    ggml_backend_dev_t cpu = GetCpuDevice();
    if (cpu != nullptr) {
      selection.devices.push_back(cpu);
      selection.device_names.emplace_back("CPU");
    }
  }
  return selection;
}

bool IsSupportedHardwareDevice(const OrtHardwareDevice* device) {
  GGONNX_NOT_NULL(device, "hardware device pointer must not be null");
  // Start with host execution only. This keeps the first milestone aligned with
  // differential debugging against ORT CPU before device-specific GGML backends.
  return GetOrtApi().HardwareDevice_Type(device) == OrtHardwareDeviceType_CPU;
}

std::vector<int64_t> inferBroadcastOutputDims(const std::vector<int64_t>& lhs_dims,
                                              const std::vector<int64_t>& rhs_dims) {
  const size_t output_rank = std::max(lhs_dims.size(), rhs_dims.size());
  std::vector<int64_t> output_dims(output_rank, 1);

  for (size_t i = 0; i < output_rank; ++i) {
    const size_t lhs_index = output_rank - i;
    const int64_t lhs_dim = i < lhs_dims.size() ? lhs_dims[lhs_dims.size() - 1 - i] : 1;
    const int64_t rhs_dim = i < rhs_dims.size() ? rhs_dims[rhs_dims.size() - 1 - i] : 1;
    if (lhs_dim == rhs_dim || lhs_dim == 1) {
      output_dims[lhs_index - 1] = rhs_dim;
      continue;
    }
    if (rhs_dim == 1) {
      output_dims[lhs_index - 1] = lhs_dim;
      continue;
    }
    throw std::runtime_error("runtime binary op shapes are not ONNX-broadcastable: " +
                             FormatDims(lhs_dims) + " and " + FormatDims(rhs_dims));
  }

  return output_dims;
}

SupportResult isNodeSupported(Ort::ConstNode node, const MetaAnalysis& meta) {
  return get_node_support(node, &meta);
}


size_t elementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (const int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "tensor element count requested for dynamic or invalid shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

size_t EstimatePartitionTensorCount(const CompiledPartition& partition) {
  size_t count = partition.values.size() + partition.nodes.size();
  return std::max<size_t>(count, 8);
}

size_t EstimateGraphMetadataBytes(const CompiledPartition& partition) {
  // Metadata-only graph context. Tensor payloads are allocated via GGML backend
  // buffers/gallocr, not from the ggml_context arena.
  const size_t tensor_count = partition.values.size() + partition.nodes.size() * 4 +
                              partition.graph_inputs.size() + partition.graph_outputs.size() + 16;
  return tensor_count * ggml_tensor_overhead() +
         ggml_graph_overhead() +
         64 * 1024;
}

// Byte count for a constant value given its ONNX dims (float32 only).
size_t ConstantByteSize(const std::vector<int64_t>& dims) {
  return elementCount(dims) * sizeof(float);
}

// Produce layout-transformed float bytes for a constant in a host-side buffer
// so the caller can upload them to a backend via ggml_backend_tensor_set. For
// AS_IS, bytes are identical to ONNX (ggml col-major reinterpretation already
// flips the dim order). For MATMUL_WEIGHT_TRANSPOSED, we physically transpose
// the last two ONNX dims, so the resulting ggml tensor has ne[0]=K and is
// ready to feed as the first arg of ggml_mul_mat.
void MaterializeConstantData(const float* src,
                             const std::vector<int64_t>& onnx_dims,
                             ConstantLayout layout,
                             float* dst) {
  if (layout == ConstantLayout::AS_IS) {
    std::memcpy(dst, src, ConstantByteSize(onnx_dims));
    return;
  }
  GGONNX_ASSERT(onnx_dims.size() >= 2,
                "MATMUL_WEIGHT_TRANSPOSED layout requires rank >= 2");
  // Transpose the last two ONNX dims. For ONNX shape [..., K, N], produce
  // data in ONNX shape [..., N, K], which under ggml's reversed interpretation
  // becomes ne=[K, N, ...] — exactly what ggml_mul_mat wants.
  const int64_t K = onnx_dims[onnx_dims.size() - 2];
  const int64_t N = onnx_dims[onnx_dims.size() - 1];
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < onnx_dims.size(); ++i) batch *= onnx_dims[i];
  // Optimized tiled transpose for better cache efficiency
  constexpr int64_t TILE = 32;
  for (int64_t b = 0; b < batch; ++b) {
    const float* src_mat = src + b * K * N;
    float* dst_mat = dst + b * K * N;
    for (int64_t k0 = 0; k0 < K; k0 += TILE) {
      for (int64_t n0 = 0; n0 < N; n0 += TILE) {
        int64_t k_end = std::min(k0 + TILE, K);
        int64_t n_end = std::min(n0 + TILE, N);
        for (int64_t n = n0; n < n_end; ++n) {
          float* dst_row = dst_mat + n * K;
          for (int64_t k = k0; k < k_end; ++k) {
            dst_row[k] = src_mat[k * N + n];
          }
        }
      }
    }
  }
}

CompiledPartition CompilePartition(const OrtGraph* graph, const BackendSelection& selection,
                                   const MetaAnalysis* shared_meta) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  const Ort::ConstGraph ort_graph{graph};
  // Prefer the capability-time analysis (computed over the full graph) when
  // available. Re-running AnalyzeCompileTimeConstants on the partition
  // subgraph alone produces stricter (incomplete) results — any folded
  // ancestor that landed in a different partition appears here as an opaque
  // boundary input, breaking downstream folding (e.g. Reshape's shape input
  // computed by a Shape→Gather chain split across partitions). Keys are
  // value/output names which are preserved through ORT's partitioning.
  MetaAnalysis meta_analysis = (shared_meta != nullptr)
      ? *shared_meta
      : AnalyzeCompileTimeConstants(graph);
  CompiledPartition partition;

  GGONNX_ASSERT(!selection.devices.empty(), "BackendSelection must contain at least one device");
  partition.force_matmul_f32_prec = selection.force_matmul_f32;
  partition.backends.reserve(selection.devices.size());
  for (size_t i = 0; i < selection.devices.size(); ++i) {
    ggml_backend_dev_t dev = selection.devices[i];
    ggml_backend_t b = ggml_backend_dev_init(dev, /*params=*/nullptr);
    GGONNX_NOT_NULL(b, std::string("ggml_backend_dev_init failed for device '") +
                            (ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "?") + "'");
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && selection.n_threads > 0) {
      ggml_backend_cpu_set_n_threads(b, selection.n_threads);
    }
    partition.backends.push_back(b);
  }
  std::unordered_map<std::string, size_t> value_ids;
  ConstantValueMap constants_by_name = meta_analysis.constants;
  // value_id -> chosen layout. Populated during node walking; first writer wins,
  // later uses that disagree force AS_IS and the op falls back to runtime transpose.
  std::unordered_map<size_t, ConstantLayout> constant_layout_by_id;
  auto record_constant_use = [&](size_t value_id, ConstantLayout layout) {
    auto [it, inserted] = constant_layout_by_id.emplace(value_id, layout);
    if (!inserted && it->second != layout) {
      it->second = ConstantLayout::AS_IS;
    }
  };

  auto ensure_value = [&](Ort::ConstValueInfo value_info) -> size_t {
    const std::string name = value_info.GetName();
    TensorMetadata metadata = getTensorMetadata(value_info);
    // Fusion-reduced constants: ORT exposes the original rank but the fusion
    // only uses the rank-reduced view. Substitute the reduced dims here so
    // the rank assert passes and downstream bookkeeping agrees.
    const auto override_it = meta_analysis.fusions.constant_override_dims.find(name);
    if (override_it != meta_analysis.fusions.constant_override_dims.end()) {
      metadata.dims = override_it->second;
    }
    if (OnnxTypeToGGML(metadata.element_type).has_value()) {
      GGONNX_ASSERT(rankSupportedByGGML(metadata),
                    "compiled partition requires tensor rank <= " + std::to_string(GGML_MAX_DIMS) +
                        ", got " + std::to_string(metadata.dims.size()) + " for value '" + name + "'");
    }
    const auto it = value_ids.find(name);
    if (it != value_ids.end()) {
      partition.values[it->second].element_type = metadata.element_type;
      // Nested If subgraphs sometimes omit output shape metadata entirely. Do
      // not overwrite a known shape with "unknown" ([]); the materialized GGML
      // graph will fill any still-unknown outputs from the emitted tensor.
      if (!metadata.dims.empty() || partition.values[it->second].dims.empty()) {
        partition.values[it->second].dims = metadata.dims;
      }
      return it->second;
    }

    const size_t id = partition.values.size();
    value_ids.emplace(name, id);
    ValueDesc value;
    value.name = name;
    value.element_type = metadata.element_type;
    value.dims = metadata.dims;
    partition.values.push_back(std::move(value));
    return id;
  };

  const auto graph_inputs = ort_graph.GetInputs();
  for (Ort::ConstValueInfo input : graph_inputs) {
    GGONNX_ASSERT(input != nullptr, "graph input metadata must not be null");
    // Non-float graph inputs (e.g. Reshape's int64 shape initializer lifted by
    // ORT into the subgraph boundary) are consumed at compile time via
    // attributes; they are not ggml runtime values.
    const TensorMetadata meta = getTensorMetadata(input);
    if (!OnnxTypeToGGML(meta.element_type).has_value()) {
      partition.kernel_input_to_value.push_back(kOptionalValueAbsent);
      continue;
    }
    // Constant initializers: materialize once at compile time. ORT still passes
    // them as kernel inputs so we reserve a kernel slot but ignore the value at
    // runtime — the compile-time copy in partition.constants is authoritative.
    if (constants_by_name.count(std::string(input.GetName())) > 0) {
      ensure_value(input);  // reserve value_id so node walking can reference it
      partition.kernel_input_to_value.push_back(kOptionalValueAbsent);
      continue;
    }
    const size_t id = ensure_value(input);
    partition.values[id].is_graph_input = true;
    partition.graph_inputs.push_back(id);
    partition.kernel_input_to_value.push_back(id);
  }

  const auto graph_outputs = ort_graph.GetOutputs();
  for (Ort::ConstValueInfo output : graph_outputs) {
    GGONNX_ASSERT(output != nullptr, "graph output metadata must not be null");
    const size_t id = ensure_value(output);
    partition.values[id].is_graph_output = true;
    partition.graph_outputs.push_back(id);
  }

  // Helper to locate graph-input value_info for a given name — needed when a
  // fusion anchor's wired-in input is a graph input (not produced by a node).
  auto find_input_value_info = [&](const std::string& name) -> Ort::ConstValueInfo {
    for (Ort::ConstValueInfo v : ort_graph.GetInputs()) {
      if (v != nullptr && std::string(v.GetName()) == name) return v;
    }
    return Ort::ConstValueInfo{nullptr};
  };
  auto find_value_info_for = [&](const std::string& name) -> Ort::ConstValueInfo {
    for (Ort::ConstNode n : ort_graph.GetNodes()) {
      for (Ort::ConstValueInfo v : n.GetOutputs()) {
        if (v != nullptr && std::string(v.GetName()) == name) return v;
      }
      for (Ort::ConstValueInfo v : n.GetInputs()) {
        if (v != nullptr && std::string(v.GetName()) == name) return v;
      }
    }
    return find_input_value_info(name);
  };

  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    GGONNX_ASSERT(node != nullptr, "graph node must not be null");
    const std::string node_key = NodeKey(node);
    if (meta_analysis.folded_nodes.count(node_key) > 0) {
      for (Ort::ConstValueInfo output : node.GetOutputs()) {
        if (output != nullptr) {
          ensure_value(output);
        }
      }
      continue;
    }
    // Fusion-consumed nodes (the two Reshapes bracketing a window-shuffle
    // Transpose) are dropped entirely — their rank-6 outputs would trip the
    // ensure_value rank check, and they're never referenced by any retained
    // node because the synthetic __WindowShuffle wires directly to the leading
    // Reshape's input and the trailing Reshape's output.
    if (meta_analysis.fusions.consumed_nodes.count(node_key) > 0) {
      continue;
    }
    // Fusion anchor (the Transpose): emit a __GenericShuffle NodeDesc that
    // computes the Reshape->Transpose->Reshape triple within rank-4 by merging
    // co-moving ONNX axes into single GGML dims.
    if (meta_analysis.fusions.generic_shuffle_anchors.count(node_key) > 0) {
      const auto& attrs = meta_analysis.fusions.generic_shuffle_anchors.at(node_key);
      const auto& io = meta_analysis.fusions.anchor_io.at(node_key);
      NodeDesc compiled_node;
      compiled_node.op_type = "__GenericShuffle";
      compiled_node.domain = "";
      compiled_node.name = io.anchor_node_name;
      compiled_node.attrs = attrs;
      GGONNX_ASSERT(io.input_values.size() == 1 && io.output_values.size() == 1,
                    "generic-shuffle fusion must have one input and one output");
      Ort::ConstValueInfo in_vi = find_value_info_for(io.input_values[0]);
      Ort::ConstValueInfo out_vi = find_value_info_for(io.output_values[0]);
      GGONNX_ASSERT(in_vi != nullptr && out_vi != nullptr,
                    "generic-shuffle fusion input/output value_info not found");
      compiled_node.inputs.push_back(ensure_value(in_vi));
      compiled_node.outputs.push_back(ensure_value(out_vi));
      partition.nodes.push_back(std::move(compiled_node));
      continue;
    }
    if (meta_analysis.fusions.window_mask_add_anchors.count(node_key) > 0) {
      const auto& io = meta_analysis.fusions.anchor_io.at(node_key);
      GGONNX_ASSERT(io.input_values.size() == 2 && io.output_values.size() == 1,
                    "window-mask-add fusion must have two inputs and one output");
      Ort::ConstValueInfo x_vi = find_value_info_for(io.input_values[0]);
      Ort::ConstValueInfo mask_vi = find_value_info_for(io.input_values[1]);
      Ort::ConstValueInfo out_vi = find_value_info_for(io.output_values[0]);
      GGONNX_ASSERT(x_vi != nullptr && mask_vi != nullptr && out_vi != nullptr,
                    "window-mask-add fusion input/output value_info not found");
      NodeDesc compiled_node;
      compiled_node.op_type = "Add";
      compiled_node.domain = "";
      compiled_node.name = io.anchor_node_name;
      const size_t x_id = ensure_value(x_vi);
      const size_t mask_id = ensure_value(mask_vi);
      const size_t out_id = ensure_value(out_vi);
      compiled_node.inputs.push_back(x_id);
      compiled_node.inputs.push_back(mask_id);
      compiled_node.outputs.push_back(out_id);
      // Mask is always a compile-time constant for this fusion.
      record_constant_use(mask_id, ConstantLayout::AS_IS);
      partition.nodes.push_back(std::move(compiled_node));
      continue;
    }
    if (meta_analysis.fusions.qkv_split_anchors.count(node_key) > 0) {
      const auto& attrs = meta_analysis.fusions.qkv_split_anchors.at(node_key);
      const auto& io = meta_analysis.fusions.anchor_io.at(node_key);
      NodeDesc compiled_node;
      compiled_node.op_type = "__QKVSplit";
      compiled_node.domain = "";
      compiled_node.name = io.anchor_node_name;
      compiled_node.attrs = attrs;
      GGONNX_ASSERT(io.input_values.size() == 1 && io.output_values.size() == 3,
                    "qkv-split fusion must have one input and three outputs");
      Ort::ConstValueInfo in_vi = find_value_info_for(io.input_values[0]);
      GGONNX_ASSERT(in_vi != nullptr, "qkv-split fusion input value_info not found");
      compiled_node.inputs.push_back(ensure_value(in_vi));
      for (const std::string& name : io.output_values) {
        Ort::ConstValueInfo out_vi = find_value_info_for(name);
        GGONNX_ASSERT(out_vi != nullptr, "qkv-split fusion output value_info not found");
        compiled_node.outputs.push_back(ensure_value(out_vi));
      }
      partition.nodes.push_back(std::move(compiled_node));
      continue;
    }
    const SupportResult support = isNodeSupported(node, meta_analysis);
    if (!support.has_value()) {
      throw std::runtime_error("GGONNX Compile received an unsupported node '" +
                               std::string(node.GetOperatorType()) + "' (" + node.GetName() +
                               "): " + support.error());
    }

    const auto node_inputs = node.GetInputs();
    const auto node_outputs = node.GetOutputs();

    NodeDesc compiled_node;
    compiled_node.op_type = node.GetOperatorType();
    compiled_node.domain = node.GetDomain();
    compiled_node.name = node.GetName();
    const OpDefinition* op = FindOpDefinition(compiled_node.domain, compiled_node.op_type);
    GGONNX_ASSERT(op != nullptr, "compiled partition could not locate supported op definition");
    if (op->compile_attrs != nullptr) {
      op->compile_attrs(node, &compiled_node, &meta_analysis);
    }
    if (compiled_node.op_type == "MatMul") {
      compiled_node.attrs = NodeDesc::MatMulAttrs{.force_f32 = selection.force_matmul_f32};
    }

    // Fold an absorbed zero-Pad into this Conv's padding attributes.
    const FusionPlan::AbsorbedPad* absorbed_pad = nullptr;
    const FusionPlan::FoldedBatchNorm* folded_bn = nullptr;
    if (compiled_node.op_type == "Conv") {
      auto apit = meta_analysis.fusions.absorbed_pads.find(node_key);
      if (apit != meta_analysis.fusions.absorbed_pads.end()) {
        absorbed_pad = &apit->second;
        if (auto* ca = std::get_if<NodeDesc::Conv2DAttrs>(&compiled_node.attrs)) {
          ca->p0 += absorbed_pad->p0;
          ca->p1 += absorbed_pad->p1;
        }
      }
      auto fbit = meta_analysis.fusions.folded_batchnorms.find(node_key);
      if (fbit != meta_analysis.fusions.folded_batchnorms.end()) {
        folded_bn = &fbit->second;
      }
    }

    const size_t compiled_input_count =
        (compiled_node.op_type == "Conv" && folded_bn != nullptr) ? 3 : node_inputs.size();
    for (size_t input_idx = 0; input_idx < compiled_input_count; ++input_idx) {
      Ort::ConstValueInfo input{nullptr};
      if (compiled_node.op_type == "Conv" && folded_bn != nullptr && input_idx <= 2) {
        if (input_idx == 0) {
          input = node_inputs[0];
        } else if (input_idx == 1) {
          input = find_value_info_for(folded_bn->weight_input_name);
        } else {
          input = find_value_info_for(folded_bn->bias_input_name);
        }
      } else if (input_idx < node_inputs.size()) {
        input = node_inputs[input_idx];
      }
      // For an absorbed-Pad Conv, bypass the Pad output and wire the Pad's
      // upstream data input directly so no padded tensor is materialised.
      if (input_idx == 0 && absorbed_pad != nullptr) {
        input = find_value_info_for(absorbed_pad->data_input_name);
        GGONNX_ASSERT(input != nullptr,
                      "absorbed Pad data input value_info not found in graph");
      }
      if (input == nullptr) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      // Inputs with no GGML representation (e.g. Reshape's int64 shape tensor)
      // are consumed at compile time via attributes and have no ggml value.
      const TensorMetadata meta = getTensorMetadata(input);
      if (!OnnxTypeToGGML(meta.element_type).has_value()) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      const size_t id = ensure_value(input);
      compiled_node.inputs.push_back(id);
      {
        auto cit = constants_by_name.find(std::string(input.GetName()));
        if (cit != constants_by_name.end()) {
          if (cit->second.dim_bindings.empty()) {
            // Fully-static constant: persistent materialization.
            const ConstantLayout layout = (op->constant_layout != nullptr)
                ? op->constant_layout(compiled_node, input_idx)
                : ConstantLayout::AS_IS;
            record_constant_use(id, layout);
          } else {
            // Sentinel-bearing pseudo-constant: resolved per-graph from
            // runtime input metadata in BuildMaterializedGraph.
            partition.shape_bound_constants.emplace(id, cit->second);
          }
        }
      }
    }
    if (compiled_node.op_type == "Conv" && folded_bn != nullptr) {
      Ort::ConstValueInfo output = find_value_info_for(folded_bn->output_value_name);
      GGONNX_ASSERT(output != nullptr, "folded BatchNormalization output value_info not found");
      compiled_node.outputs.push_back(ensure_value(output));
    } else {
      for (Ort::ConstValueInfo output : node_outputs) {
        if (output == nullptr) {
          compiled_node.outputs.push_back(kOptionalValueAbsent);
          continue;
        }
        compiled_node.outputs.push_back(ensure_value(output));
      }
    }

    partition.nodes.push_back(std::move(compiled_node));
  }

  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain at least one graph output");
  std::unordered_set<size_t> used_value_ids;
  for (const NodeDesc& node : partition.nodes) {
    for (size_t input_id : node.inputs) {
      if (input_id != kOptionalValueAbsent) {
        used_value_ids.insert(input_id);
      }
    }
  }
  std::vector<size_t> live_graph_inputs;
  live_graph_inputs.reserve(partition.graph_inputs.size());
  for (size_t& value_id : partition.kernel_input_to_value) {
    if (value_id == kOptionalValueAbsent) continue;
    if (used_value_ids.count(value_id) == 0) {
      value_id = kOptionalValueAbsent;
      continue;
    }
    live_graph_inputs.push_back(value_id);
  }
  partition.graph_inputs = std::move(live_graph_inputs);

  partition.folded_constants.resize(partition.values.size());
  for (size_t id = 0; id < partition.values.size(); ++id) {
    auto it = constants_by_name.find(partition.values[id].name);
    if (it != constants_by_name.end()) {
      partition.folded_constants[id] = it->second;
    }
  }

  // Materialize detected constants into a persistent ggml context. Sized
  // tightly: sum of constant data bytes + per-tensor overhead + slack.
  if (!constant_layout_by_id.empty()) {
    size_t constant_bytes = 0;
    for (const auto& [id, layout] : constant_layout_by_id) {
      auto it = constants_by_name.find(partition.values[id].name);
      GGONNX_ASSERT(it != constants_by_name.end(), "constant not found during byte-size estimation");
      constant_bytes += it->second.data.size();
    }
    const size_t mem_size =
        constant_bytes +
        (constant_layout_by_id.size() + 4) * ggml_tensor_overhead() +
        4 * 1024;

    ggml_init_params p{};
    p.mem_size = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    partition.constant_ctx = ggml_init(p);
    GGONNX_NOT_NULL(partition.constant_ctx, "ggml_init failed for constants");

    partition.constants.assign(partition.values.size(), nullptr);
    for (const auto& [id, layout] : constant_layout_by_id) {
      const ValueDesc& value = partition.values[id];
      auto it = constants_by_name.find(value.name);
      GGONNX_ASSERT(it != constants_by_name.end(), "constant tensor lost during compile");
      const auto gtype = OnnxTypeToGGML(it->second.element_type);
      GGONNX_ASSERT(gtype.has_value(), "GGML constant has no GGML type representation");

      // Prefer ORT metadata dims (which carry fusion overrides such as
      // rank-reduced mask shapes) over the raw constant dims. Fall back to
      // the actual constant dims only when the ORT metadata is dynamic (e.g.
      // meta-eval-folded INT64 shape tensors whose ORT shape is still [-1]).
      std::vector<int64_t> tensor_dims =
          shapeIsFullyStatic(value.dims) ? value.dims : it->second.dims;
      if (layout == ConstantLayout::MATMUL_WEIGHT_TRANSPOSED && tensor_dims.size() >= 2) {
        std::swap(tensor_dims[tensor_dims.size() - 2], tensor_dims[tensor_dims.size() - 1]);
      }
      // Scalar ONNX constants (rank 0) are represented in ggml as a rank-1 size-1 tensor.
      std::vector<int64_t> ggml_src_dims = tensor_dims.empty() ? std::vector<int64_t>{1} : tensor_dims;
      const std::array<int64_t, GGML_MAX_DIMS> ggml_dims = ToGGMLDims(ggml_src_dims);
      ggml_tensor* t = ggml_new_tensor(partition.constant_ctx, *gtype,
                                       static_cast<int>(ggml_src_dims.size()),
                                       ggml_dims.data());
      GGONNX_NOT_NULL(t, "ggml_new_tensor failed for constant");
      partition.constants[id] = t;
    }

    partition.constant_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(partition.constant_ctx,
                                                 ggml_backend_get_default_buffer_type(partition.primary_backend()));
    GGONNX_NOT_NULL(partition.constant_buffer, "ggml_backend_alloc_ctx_tensors_from_buft failed for constants");
    ggml_backend_buffer_set_usage(partition.constant_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto& [id, layout] : constant_layout_by_id) {
      ggml_tensor* t = partition.constants[id];
      GGONNX_NOT_NULL(t, "constant tensor missing backend allocation");
      const ValueDesc& value = partition.values[id];
      auto it = constants_by_name.find(value.name);
      GGONNX_ASSERT(it != constants_by_name.end(), "constant tensor lost during backend upload");
      GGONNX_ASSERT(OnnxTypeToGGML(it->second.element_type).has_value(),
                    "GGML constant has no GGML type representation");

      GGONNX_ASSERT(t->buffer != nullptr,
                    "constant tensor '" + value.name + "' has no backend buffer after allocation");
      if (it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::vector<float> materialized(elementCount(value.dims));
        MaterializeConstantData(reinterpret_cast<const float*>(it->second.data.data()),
                                value.dims, layout, materialized.data());
        ggml_backend_tensor_set(t, materialized.data(), 0, materialized.size() * sizeof(float));
      } else {
        // Non-F32 constants: upload raw bytes as-is (no transposition support for now).
        ggml_backend_tensor_set(t, it->second.data.data(), 0, it->second.data.size());
      }
    }
  }

  // With >1 backends we need a scheduler to route per-op and to copy tensors
  // across backends on demand. Single-backend partitions take the lighter
  // direct path via ggml_backend_graph_compute.
  if (partition.backends.size() > 1) {
    const int n_backends = static_cast<int>(partition.backends.size());
    partition.sched = ggml_backend_sched_new(partition.backends.data(),
                                             /*bufts=*/nullptr,
                                             n_backends,
                                             GGML_DEFAULT_GRAPH_SIZE,
                                             /*parallel=*/false,
                                             /*op_offload=*/true);
    GGONNX_NOT_NULL(partition.sched, "ggml_backend_sched_new failed");
  }

  return partition;
}

ShapeKey MakeShapeKey(const std::vector<TensorMetadata>& input_metadata) {
  ShapeKey key;
  key.input_dims.reserve(input_metadata.size());
  for (const TensorMetadata& meta : input_metadata) {
    GGONNX_ASSERT(shapeIsFullyStatic(meta.dims),
                  "runtime input shapes must be concrete before GGML execution");
    key.input_dims.push_back(meta.dims);
  }
  return key;
}

ggml_tensor* CreateTensorForOnnxShape(ggml_context* ctx, const std::vector<int64_t>& dims,
                                      ONNXTensorElementDataType element_type) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto gtype = OnnxTypeToGGML(element_type);
  GGONNX_ASSERT(gtype.has_value(), "unsupported ONNX element type for GGML tensor");
  if (dims.empty()) {
    return ggml_new_tensor_1d(ctx, *gtype, 1);
  }

  const std::array<int64_t, GGML_MAX_DIMS> ggml_dims = ToGGMLDims(dims);
  return ggml_new_tensor(ctx, *gtype, static_cast<int>(dims.size()), ggml_dims.data());
}

void CopyInputDataToTensor(const OrtValue* input_value,
                           const std::vector<int64_t>& expected_dims,
                           const std::string& tensor_name,
                           ggml_tensor* tensor) {
  GGONNX_NOT_NULL(input_value, "graph input value must not be null");
  GGONNX_NOT_NULL(tensor, "cached GGML input tensor must not be null");
  const TensorMetadata meta = getTensorMetadata(Ort::ConstValue{input_value});
  if (!OnnxTypeToGGML(meta.element_type).has_value()) {
    throw std::runtime_error("GGONNX graph input has no GGML type representation");
  }

  const void* input_data = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(input_value, &input_data));
  GGONNX_NOT_NULL(input_data, "ORT returned null tensor data for graph input");

  GGONNX_ASSERT(meta.dims == expected_dims,
                "runtime input shape mismatch for tensor '" + tensor_name + "': expected " +
                    FormatDims(expected_dims) + ", got " + FormatDims(meta.dims));
  const size_t element_count = elementCount(meta.dims);
  const size_t bytes = element_count * ggml_element_size(tensor);
  AssertShapeMatchesGGML(meta.dims, tensor, tensor_name);
  ggml_backend_tensor_set(tensor, input_data, 0, bytes);
}

EmitResult EmitNode(ggml_context* ctx,
                    const NodeDesc& node,
                    const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const OpDefinition* op = FindOpDefinition(node.domain, node.op_type);
  GGONNX_ASSERT(op != nullptr && op->emit != nullptr,
                "compiled partition contains op without GGML emitter");
  return op->emit(ctx, node, values);
}

void DestroyMaterializedGraph(std::unique_ptr<MaterializedGraph>& graph) {
  if (graph != nullptr) {
    if (graph->gallocr != nullptr) {
      ggml_gallocr_free(graph->gallocr);
    }
    if (graph->ctx != nullptr) {
      ggml_free(graph->ctx);
    }
  }
  graph.reset();
}

std::unique_ptr<MaterializedGraph> BuildMaterializedGraph(const CompiledPartition& partition,
                                                          ShapeKey key,
                                                          const std::vector<TensorMetadata>& input_metadata) {
  const size_t mem_size =
      EstimatePartitionTensorCount(partition) * ggml_tensor_overhead() +
      ggml_graph_overhead() +
      EstimateGraphMetadataBytes(partition);

  ggml_init_params params{};
  params.mem_size = mem_size;
  params.mem_buffer = nullptr;
  params.no_alloc = true;

  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) {
    throw std::runtime_error("ggml_init failed");
  }

  try {
    auto graph_state = std::make_unique<MaterializedGraph>();
    graph_state->key = std::move(key);
    graph_state->ctx = ctx;
    graph_state->values.resize(partition.values.size(), nullptr);
    graph_state->value_dims.resize(partition.values.size());
    graph_state->input_tensors.resize(partition.graph_inputs.size(), nullptr);
    graph_state->output_tensors.resize(partition.graph_outputs.size(), nullptr);

    // Preload compile-time constants into the value table. value_dims keeps
    // the declared (logical) ONNX shape even if the constant was physically
    // pre-transposed — emits that use these tensors inspect ne[] directly.
    for (size_t id = 0; id < partition.constants.size(); ++id) {
      ggml_tensor* c = partition.constants[id];
      if (c == nullptr) continue;
      graph_state->values[id] = c;
      graph_state->value_dims[id] = partition.values[id].dims;
    }

    // Map subgraph-input names to their input_metadata index so shape-bound
    // constants can resolve "dim A of input B" bindings against runtime shapes.
    std::unordered_map<std::string, size_t> input_idx_by_name;
    input_idx_by_name.reserve(partition.graph_inputs.size());
    for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
      const size_t value_id = partition.graph_inputs[i];
      input_idx_by_name.emplace(partition.values[value_id].name, i);
    }

    for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
      const size_t value_id = partition.graph_inputs[i];
      const TensorMetadata& meta = input_metadata[i];
      const ValueDesc& value = partition.values[value_id];
      GGONNX_ASSERT(OnnxTypeToGGML(meta.element_type).has_value(),
                    "GGONNX runtime input has no GGML type representation");
      GGONNX_ASSERT(broadcastSupportedByGGML(value.dims, meta.dims),
                    "runtime input shape mismatch for tensor '" + value.name + "': declared " +
                        FormatDims(value.dims) + ", got " + FormatDims(meta.dims));
      GGONNX_ASSERT(shapeIsFullyStatic(meta.dims),
                    "runtime input shapes must be concrete for tensor '" + value.name + "'");

      ggml_tensor* input_tensor = CreateTensorForOnnxShape(ctx, meta.dims, meta.element_type);
      GGONNX_NOT_NULL(input_tensor, "failed to allocate cached GGML input tensor");
      ggml_set_input(input_tensor);
      AssertShapeMatchesGGML(meta.dims, input_tensor, partition.values[value_id].name);
      graph_state->values[value_id] = input_tensor;
      graph_state->value_dims[value_id] = meta.dims;
      graph_state->input_tensors[i] = input_tensor;
    }

    // Create tensors for shape-bound pseudo-constants (sentinel-bearing folds).
    // We resolve their data below, after gallocr allocates the backend buffers.
    std::vector<std::pair<ggml_tensor*, const ConstantTensor*>> shape_bound_uploads;
    shape_bound_uploads.reserve(partition.shape_bound_constants.size());
    for (const auto& [vid, constant] : partition.shape_bound_constants) {
      std::vector<int64_t> dims = constant.dims.empty() ? std::vector<int64_t>{1} : constant.dims;
      ggml_tensor* t = CreateTensorForOnnxShape(ctx, dims, constant.element_type);
      GGONNX_NOT_NULL(t, "failed to allocate shape-bound constant tensor");
      ggml_set_input(t);  // make gallocr allocate a backend buffer for us
      graph_state->values[vid] = t;
      graph_state->value_dims[vid] = constant.dims;
      shape_bound_uploads.push_back({t, &constant});
    }

    for (const NodeDesc& node : partition.nodes) {
      EmitResult emitted_outputs = EmitNode(ctx, node, graph_state->values);
      GGONNX_ASSERT(emitted_outputs.has_value(),
                    "EmitNode() returned unsupported for a compiled node");

      size_t emitted_index = 0;
      for (size_t output_id : node.outputs) {
        if (output_id == kOptionalValueAbsent) {
          continue;
        }
        GGONNX_ASSERT(emitted_index < emitted_outputs->size(), "compiled node emitted too few outputs");
        GGONNX_ASSERT(output_id < graph_state->values.size(), "compiled node output index out of range");
        GGONNX_ASSERT(graph_state->values[output_id] == nullptr,
                      "compiled node output slot already populated");
        ggml_tensor* node_output = (*emitted_outputs)[emitted_index++];
        const std::vector<int64_t> emitted_dims = ToOnnxDims(node_output);
        const std::vector<int64_t>& declared_dims = partition.values[output_id].dims;
        GGONNX_ASSERT(broadcastSupportedByGGML(declared_dims, emitted_dims),
                      "emitted output shape mismatch for tensor '" + partition.values[output_id].name +
                          "': declared " + FormatDims(declared_dims) + ", got " +
                          FormatDims(emitted_dims));
        graph_state->values[output_id] = node_output;
        // Preserve the declared ONNX rank even when some dims are dynamic: GGML
        // strips trailing unit dims (and only carries 4), so a declared shape
        // like (1, S, V) would otherwise collapse to (S, V) when taken straight
        // from the ggml tensor. Merge right-aligned — declared wins when
        // concrete, emitted fills in dynamic slots.
        if (declared_dims.empty() && !emitted_dims.empty()) {
          // ORT may omit tensor shape metadata for nested-subgraph values even
          // though the emitted GGML tensor has a concrete non-scalar shape.
          graph_state->value_dims[output_id] = emitted_dims;
        } else if (shapeIsFullyStatic(declared_dims)) {
          graph_state->value_dims[output_id] = declared_dims;
        } else {
          std::vector<int64_t> merged(declared_dims.size(), 1);
          const size_t off = declared_dims.size() >= emitted_dims.size()
                                 ? declared_dims.size() - emitted_dims.size()
                                 : 0;
          for (size_t i = 0; i < merged.size(); ++i) {
            if (declared_dims[i] >= 0) {
              merged[i] = declared_dims[i];
            } else if (i >= off) {
              merged[i] = emitted_dims[i - off];
            }
          }
          graph_state->value_dims[output_id] = std::move(merged);
        }
      }
      GGONNX_ASSERT(emitted_index == emitted_outputs->size(), "compiled node emitted too many outputs");
    }

    bool has_runtime_output = false;
    for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
      const size_t output_id = partition.graph_outputs[i];
      GGONNX_ASSERT(output_id < graph_state->values.size(), "graph output index out of range");
      ggml_tensor* output = graph_state->values[output_id];
      if (output == nullptr) {
        GGONNX_ASSERT(partition.folded_constants.size() > output_id &&
                          partition.folded_constants[output_id].has_value(),
                      "compiled partition output was not materialized: " +
                          partition.values[output_id].name);
        graph_state->value_dims[output_id] = partition.values[output_id].dims;
        continue;
      }
      if (!has_runtime_output) {
        graph_state->graph = ggml_new_graph(ctx);
        GGONNX_NOT_NULL(graph_state->graph, "ggml_new_graph failed");
        has_runtime_output = true;
      }
      if (!ggml_is_contiguous(output)) {
        output = ggml_cont(ctx, output);
        graph_state->values[output_id] = output;
      }
      ggml_set_output(output);
      ggml_build_forward_expand(graph_state->graph, output);
      graph_state->output_tensors[i] = output;
    }

    if (has_runtime_output) {
      if (partition.sched != nullptr) {
        // Sched path: reset prior assignments (if any) and allocate compute
        // buffers across all backends for this graph shape.
        ggml_backend_sched_reset(partition.sched);
        GGONNX_ASSERT(ggml_backend_sched_alloc_graph(partition.sched, graph_state->graph),
                      "ggml_backend_sched_alloc_graph failed");
      } else {
        graph_state->gallocr =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(partition.primary_backend()));
        GGONNX_NOT_NULL(graph_state->gallocr, "ggml_gallocr_new failed");
        GGONNX_ASSERT(ggml_gallocr_alloc_graph(graph_state->gallocr, graph_state->graph),
                      "ggml_gallocr_alloc_graph failed");
      }
    }

    // Input tensors consumed only as compile-time attributes (e.g. Reshape's
    // shape input) are not wired into any GGML op by their emitter, so they
    // never enter the cgraph and gallocr never allocates them. Null them out so
    // the data-copy loop skips them rather than crashing on a null buffer.
    for (ggml_tensor*& t : graph_state->input_tensors) {
      if (t != nullptr && t->buffer == nullptr) {
        t = nullptr;
      }
    }

    // Resolve and upload shape-bound constants: substitute each bound element's
    // sentinel with the actual dim from runtime input_metadata, then upload the
    // patched bytes to the backend buffer that gallocr just allocated.
    for (const auto& [t, pc] : shape_bound_uploads) {
      if (t->buffer == nullptr) continue;  // not consumed by any op → safe to skip
      std::vector<uint8_t> bytes = pc->data;
      for (size_t i = 0; i < pc->dim_bindings.size(); ++i) {
        const auto& binding = pc->dim_bindings[i];
        if (!binding.has_value()) continue;
        auto nit = input_idx_by_name.find(binding->source_name);
        GGONNX_ASSERT(nit != input_idx_by_name.end(),
                      "shape-bound constant '" + partition.values[static_cast<size_t>(0)].name +
                          "' references unknown subgraph input '" + binding->source_name + "'");
        const std::vector<int64_t>& src_dims = input_metadata[nit->second].dims;
        GGONNX_ASSERT(binding->axis >= 0 &&
                          static_cast<size_t>(binding->axis) < src_dims.size(),
                      "shape-bound constant axis out of range for '" + binding->source_name + "'");
        const int64_t resolved = src_dims[static_cast<size_t>(binding->axis)];
        if (pc->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          std::memcpy(bytes.data() + i * sizeof(int64_t), &resolved, sizeof(int64_t));
        } else if (pc->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          const int32_t v32 = static_cast<int32_t>(resolved);
          std::memcpy(bytes.data() + i * sizeof(int32_t), &v32, sizeof(int32_t));
        } else {
          GGONNX_ABORT("shape-bound constant has non-integer element type");
        }
      }
      ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }

    g_debug_graph_build_count.fetch_add(1, std::memory_order_relaxed);
    if (graph_state->graph != nullptr) {
      g_debug_last_graph_node_count.store(static_cast<uint64_t>(ggml_graph_n_nodes(graph_state->graph)),
                                          std::memory_order_relaxed);
      uint64_t output_count = 0;
      for (ggml_tensor* t : graph_state->output_tensors) {
        if (t != nullptr) ++output_count;
      }
      g_debug_last_graph_output_count.store(output_count, std::memory_order_relaxed);
    } else {
      g_debug_last_graph_node_count.store(0, std::memory_order_relaxed);
      g_debug_last_graph_output_count.store(0, std::memory_order_relaxed);
    }
    return graph_state;
  } catch (...) {
    ggml_free(ctx);
    throw;
  }
}

void ExecutePartitionWithGGML(GGMLComputeState& state, OrtKernelContext* kernel_context) {
  GGONNX_NOT_NULL(state.partition, "compute state missing compiled partition");
  const CompiledPartition& partition = *state.partition;
  GGONNX_NOT_NULL(kernel_context, "kernel context must not be null");
  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain graph outputs");
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetInputCount(kernel_context, &num_inputs));
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetOutputCount(kernel_context, &num_outputs));
  if (num_inputs != partition.kernel_input_to_value.size() ||
      num_outputs != partition.graph_outputs.size()) {
    throw std::runtime_error("kernel IO count does not match compiled partition");
  }

  std::vector<const OrtValue*> float_input_values;
  std::vector<TensorMetadata> input_metadata;
  float_input_values.reserve(partition.graph_inputs.size());
  input_metadata.reserve(partition.graph_inputs.size());
  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValue* value = nullptr;
    THROW_ON_ERROR(GetOrtApi().KernelContext_GetInput(kernel_context, i, &value));
    if (value == nullptr) {
      throw std::runtime_error("compiled partition received null input");
    }

    int is_tensor = 0;
    THROW_ON_ERROR(GetOrtApi().IsTensor(value, &is_tensor));
    if (!is_tensor) {
      throw std::runtime_error("compiled partition input is not a tensor");
    }

    if (partition.kernel_input_to_value[i] == kOptionalValueAbsent) {
      continue;  // compile-time-only input, ignore at runtime
    }
    float_input_values.push_back(value);
    input_metadata.push_back(getTensorMetadata(Ort::ConstValue{value}));
  }

  const ShapeKey shape_key = MakeShapeKey(input_metadata);
  if (state.active_graph == nullptr || state.active_graph->key != shape_key) {
    DestroyMaterializedGraph(state.active_graph);
    state.active_graph = BuildMaterializedGraph(partition, shape_key, input_metadata);
  }

  GGONNX_NOT_NULL(state.active_graph.get(), "active GGML graph must not be null");
  for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
    if (state.active_graph->input_tensors[i] == nullptr) continue;
    const size_t input_id = partition.graph_inputs[i];
    CopyInputDataToTensor(float_input_values[i],
                          state.active_graph->value_dims[input_id],
                          partition.values[input_id].name,
                          state.active_graph->input_tensors[i]);
  }

  if (state.active_graph->graph != nullptr) {
    const ggml_status status = (partition.sched != nullptr)
        ? ggml_backend_sched_graph_compute(partition.sched, state.active_graph->graph)
        : ggml_backend_graph_compute(partition.primary_backend(), state.active_graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("ggml graph compute failed");
    }
  }

  for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
    const size_t output_id = partition.graph_outputs[i];
    const std::vector<int64_t>& output_dims = state.active_graph->value_dims[output_id];
    OrtValue* output_value = nullptr;
    THROW_ON_ERROR(GetOrtApi().KernelContext_GetOutput(
        kernel_context, i, output_dims.data(), output_dims.size(), &output_value));
    if (output_value == nullptr) {
      throw std::runtime_error("failed to allocate ORT output");
    }

    void* output_data = nullptr;
    THROW_ON_ERROR(GetOrtApi().GetTensorMutableData(output_value, &output_data));
    GGONNX_NOT_NULL(output_data, "ORT returned null output buffer");

    if (partition.folded_constants.size() > output_id && partition.folded_constants[output_id].has_value()) {
      const ConstantTensor& constant = *partition.folded_constants[output_id];
      const size_t bytes = constant.data.size();
      if (bytes > 0) {
        std::memcpy(output_data, constant.data.data(), bytes);
      }
      continue;
    }

    ggml_tensor* output_tensor = state.active_graph->output_tensors[i];
    GGONNX_NOT_NULL(output_tensor, "missing cached GGML output tensor");
    AssertShapeMatchesGGML(output_dims, output_tensor, partition.values[output_id].name);
    GGONNX_ASSERT(ggml_is_contiguous(output_tensor),
                  "GGML output tensor must be contiguous before copy for '" +
                      partition.values[output_id].name + "'");

    const size_t bytes = ggml_nbytes(output_tensor);
    GGONNX_ASSERT(bytes == elementCount(output_dims) * ggml_element_size(output_tensor),
                  "GGML output byte size mismatch for '" + partition.values[output_id].name + "'");
    ggml_backend_tensor_get(output_tensor, output_data, 0, bytes);
  }
}

OrtStatus* NodeComputeInfoCreateState(OrtNodeComputeInfo* this_ptr,
                                      OrtNodeComputeContext* /*compute_context*/,
                                      void** compute_state) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(this_ptr, "node compute info must not be null");
    GGONNX_NOT_NULL(compute_state, "compute_state output must not be null");
    auto* info = AsNodeComputeInfo(this_ptr);
    auto state = std::make_unique<GGMLComputeState>();
    state->partition = &info->partition;
    *compute_state = state.release();
  });
}

OrtStatus* NodeComputeInfoCompute(OrtNodeComputeInfo* /*this_ptr*/,
                                  void* compute_state,
                                  OrtKernelContext* kernel_context) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(compute_state, "compute state must not be null");
    auto* state = AsComputeState(compute_state);
    ExecutePartitionWithGGML(*state, kernel_context);
  });
}

void NodeComputeInfoReleaseState(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state) noexcept {
  if (compute_state == nullptr) {
    return;
  }
  auto* state = AsComputeState(compute_state);
  DestroyMaterializedGraph(state->active_graph);
  delete state;
}

std::unique_ptr<GGMLNodeComputeInfo> BuildNodeComputeInfo(CompiledPartition partition) {
  auto info = std::make_unique<GGMLNodeComputeInfo>();
  info->iface.ort_version_supported = ORT_API_VERSION;
  info->iface.CreateState = NodeComputeInfoCreateState;
  info->iface.Compute = NodeComputeInfoCompute;
  info->iface.ReleaseState = NodeComputeInfoReleaseState;
  info->partition = std::move(partition);
  return info;
}

const char* FactoryGetName(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpName;
}

const char* FactoryGetVendor(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kVendorName;
}

uint32_t FactoryGetVendorId(const OrtEpFactory* /*this_ptr*/) noexcept {
  return 0;
}

const char* FactoryGetVersion(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kVersion;
}

OrtStatus* FactoryGetSupportedDevices(OrtEpFactory* this_ptr,
                                      const OrtHardwareDevice* const* devices,
                                      size_t num_devices,
                                      OrtEpDevice** ep_devices,
                                      size_t max_ep_devices,
                                      size_t* num_ep_devices) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(this_ptr, "factory must not be null");
    GGONNX_NOT_NULL(ep_devices, "ep_devices output array must not be null");
    GGONNX_NOT_NULL(num_ep_devices, "num_ep_devices output must not be null");
    auto* factory = AsFactory(this_ptr);
    size_t count = 0;

    for (size_t i = 0; i < num_devices; ++i) {
      const OrtHardwareDevice* device = devices[i];
      if (!IsSupportedHardwareDevice(device)) {
        continue;
      }
      if (count >= max_ep_devices) {
        throw std::runtime_error("ORT provided too few EP device slots");
      }

      THROW_ON_ERROR(GetOrtEpApi().CreateEpDevice(&factory->iface, device, nullptr, nullptr, &ep_devices[count]));
      ++count;
    }

    *num_ep_devices = count;
  });
}

OrtStatus* EpGetCapability(OrtEp* this_ptr,
                           const OrtGraph* graph,
                           OrtEpGraphSupportInfo* graph_support_info) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(graph_support_info, "graph support info must not be null");
    const Ort::ConstGraph ort_graph{graph};
    auto shared_meta = std::make_shared<MetaAnalysis>(AnalyzeCompileTimeConstants(graph));
    if (this_ptr != nullptr) {
      AsEp(this_ptr)->capability_meta = shared_meta;
    }
    const MetaAnalysis& meta_analysis = *shared_meta;

    // Partition by walking nodes in topological order (how ORT yields them) and
    // sealing the current fuse group whenever we hit an unsupported node. Any
    // directed path between two supported nodes must pass through every node
    // that lies strictly between them in topological order, so an unsupported
    // node on such a path is guaranteed to split them into different groups
    // Because ONNX wants that and will complain if not in that way

    // Build the set of value names that are "live at this graph level" — either
    // consumed as a direct or implicit input by some node in this graph, or
    // named as a graph output. A node whose outputs are only consumed inside a
    // deeper nested subgraph (via implicit capture of the inner If/Loop/Scan
    // body) is NOT visible here: ORT's fused-node output computation only looks
    // at direct consumption at the current level, so including such a node in
    // a fuse group produces a partition with zero graph outputs and Compile
    // rejects it. Filter those nodes out of our capability.
    std::unordered_set<std::string> live_names;
    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      for (Ort::ConstValueInfo v : node.GetInputs()) {
        if (v != nullptr) live_names.insert(std::string(v.GetName()));
      }
      for (Ort::ConstValueInfo v : node.GetImplicitInputs()) {
        if (v != nullptr) live_names.insert(std::string(v.GetName()));
      }
    }
    for (Ort::ConstValueInfo v : ort_graph.GetOutputs()) {
      if (v != nullptr) live_names.insert(std::string(v.GetName()));
    }
    auto has_visible_output = [&](Ort::ConstNode node) {
      for (Ort::ConstValueInfo v : node.GetOutputs()) {
        if (v == nullptr) continue;
        if (live_names.count(std::string(v.GetName())) > 0) return true;
      }
      return false;
    };

    std::vector<const OrtNode*> current_group;
    bool current_group_has_runtime_node = false;
    auto flush_group = [&] {
      if (current_group.empty()) return;
      if (!current_group_has_runtime_node) {
        current_group.clear();
        current_group_has_runtime_node = false;
        return;
      }
      THROW_ON_ERROR(GetOrtEpApi().EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, current_group.data(), current_group.size(), nullptr));
      current_group.clear();
      current_group_has_runtime_node = false;
    };

    const bool debug_cap = std::getenv("GGONNX_DEBUG_CAPABILITY") != nullptr;
    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      const std::string key = NodeKey(node);
      const bool is_fusion_anchor =
          meta_analysis.fusions.generic_shuffle_anchors.count(key) > 0 ||
          meta_analysis.fusions.qkv_split_anchors.count(key) > 0 ||
          meta_analysis.fusions.window_mask_add_anchors.count(key) > 0;
      const bool is_fusion_consumed = meta_analysis.fusions.consumed_nodes.count(key) > 0;
      const SupportResult sup = isNodeSupported(node, meta_analysis);
      const bool supported = (sup.has_value() || is_fusion_anchor) && has_visible_output(node);
      if (debug_cap && !supported && !is_fusion_consumed &&
          meta_analysis.folded_nodes.count(key) == 0) {
        std::fprintf(stderr, "[ggonnx-cap] reject %s (%s): %s\n",
                     std::string(node.GetOperatorType()).c_str(),
                     std::string(node.GetName()).c_str(),
                     sup.has_value() ? "no visible output" : sup.error().c_str());
      }
      if (supported || meta_analysis.folded_nodes.count(key) > 0 || is_fusion_consumed) {
        current_group.push_back(node);
        current_group_has_runtime_node = current_group_has_runtime_node || supported;
      } else {
        flush_group();
      }
    }
    flush_group();
  });
}

OrtStatus* EpCompile(OrtEp* this_ptr,
                     const OrtGraph** graphs,
                     const OrtNode** /*fused_nodes*/,
                     size_t count,
                     OrtNodeComputeInfo** node_compute_infos,
                     OrtNode** /*ep_context_nodes*/) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(this_ptr, "ep must not be null");
    GGONNX_NOT_NULL(graphs, "graphs array must not be null");
    GGONNX_NOT_NULL(node_compute_infos, "node_compute_infos output array must not be null");
    const GGMLEp* ep = AsEp(this_ptr);
    GGONNX_NOT_NULL(ep->selection.get(), "ep selection must be populated before Compile");
    const BackendSelection& selection = *ep->selection;
    const MetaAnalysis* shared_meta = ep->capability_meta.get();
    for (size_t i = 0; i < count; ++i) {
      GGONNX_NOT_NULL(graphs[i], "graph entry must not be null");
      auto compute_info = BuildNodeComputeInfo(CompilePartition(graphs[i], selection, shared_meta));
      node_compute_infos[i] = &compute_info.release()->iface;
    }
  });
}

void EpReleaseNodeComputeInfos(OrtEp* /*this_ptr*/,
                               OrtNodeComputeInfo** node_compute_infos,
                               size_t num_node_compute_infos) noexcept {
  if (node_compute_infos == nullptr) {
    return;
  }
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    if (node_compute_infos[i] == nullptr) {
      continue;
    }
    delete AsNodeComputeInfo(node_compute_infos[i]);
  }
}

OrtStatus* EpGetPreferredDataLayout(OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept {
  if (preferred_data_layout == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "preferred_data_layout output must not be null");
  }
  *preferred_data_layout = OrtEpDataLayout_NCHW;
  return nullptr;
}

OrtStatus* EpOnRunStart(OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/) noexcept {
  return nullptr;
}

OrtStatus* EpOnRunEnd(OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/, bool /*sync_stream*/) noexcept {
  return nullptr;
}

OrtStatus* EpCreateAllocator(OrtEp* /*this_ptr*/,
                             const OrtMemoryInfo* /*memory_info*/,
                             OrtAllocator** allocator) noexcept {
  if (allocator == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "allocator output must not be null");
  }
  *allocator = nullptr;
  return nullptr;
}

OrtStatus* EpCreateSyncStreamForDevice(OrtEp* /*this_ptr*/,
                                       const OrtMemoryDevice* /*memory_device*/,
                                       OrtSyncStreamImpl** stream) noexcept {
  if (stream == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "stream output must not be null");
  }
  *stream = nullptr;
  return nullptr;
}

const char* EpGetCompiledModelCompatibilityInfo(OrtEp* /*this_ptr*/, const OrtGraph* /*graph*/) noexcept {
  return "ggonnx-scaffold";
}

OrtStatus* EpGetKernelRegistry(OrtEp* /*this_ptr*/, const OrtKernelRegistry** kernel_registry) noexcept {
  if (kernel_registry == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "kernel_registry output must not be null");
  }
  *kernel_registry = nullptr;
  return nullptr;
}

OrtStatus* EpIsConcurrentRunSupported(OrtEp* /*this_ptr*/, bool* is_supported) noexcept {
  if (is_supported == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "is_supported output must not be null");
  }
  *is_supported = false;
  return nullptr;
}

const char* EpGetName(const OrtEp* /*this_ptr*/) noexcept {
  return kEpName;
}

OrtStatus* FactoryCreateEp(OrtEpFactory* /*this_ptr*/,
                           const OrtHardwareDevice* const* devices,
                           const OrtKeyValuePairs* const* ep_metadata_pairs,
                           size_t num_devices,
                           const OrtSessionOptions* session_options,
                           const OrtLogger* logger,
                           OrtEp** ep) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(devices, "devices array must not be null");
    GGONNX_NOT_NULL(ep, "ep output must not be null");
    (void)ep_metadata_pairs;
    auto impl = std::make_unique<GGMLEp>();
    impl->selection = std::make_shared<BackendSelection>(ResolveBackendSelection(session_options));
    // Log the resolved device list once per session create so users can see
    // what ep.ggonnx.ggml_device_list / enable_cpu_fallback actually produced.
    std::fprintf(stderr, "[ggonnx] session backend selection (cpu_fallback=%s, matmul_precision=%s, n_threads=%d):\n",
                 impl->selection->enable_cpu_fallback ? "on" : "off",
                 impl->selection->force_matmul_f32 ? "f32" : "default",
                 impl->selection->n_threads);
    for (size_t i = 0; i < impl->selection->devices.size(); ++i) {
      const char* n = ggml_backend_dev_name(impl->selection->devices[i]);
      std::fprintf(stderr, "[ggonnx]   [%zu] %s\n", i, n != nullptr ? n : "?");
    }
    impl->iface.ort_version_supported = ORT_API_VERSION;
    impl->iface.GetName = EpGetName;
    impl->iface.GetCapability = EpGetCapability;
    impl->iface.Compile = EpCompile;
    impl->iface.ReleaseNodeComputeInfos = EpReleaseNodeComputeInfos;
    impl->iface.GetPreferredDataLayout = EpGetPreferredDataLayout;
    impl->iface.ShouldConvertDataLayoutForOp = nullptr;
    impl->iface.SetDynamicOptions = nullptr;
    impl->iface.OnRunStart = EpOnRunStart;
    impl->iface.OnRunEnd = EpOnRunEnd;
    impl->iface.CreateAllocator = EpCreateAllocator;
    impl->iface.CreateSyncStreamForDevice = EpCreateSyncStreamForDevice;
    impl->iface.GetCompiledModelCompatibilityInfo = EpGetCompiledModelCompatibilityInfo;
    impl->iface.GetKernelRegistry = EpGetKernelRegistry;
    impl->iface.IsConcurrentRunSupported = EpIsConcurrentRunSupported;
    impl->logger = logger;

    for (size_t i = 0; i < num_devices; ++i) {
      impl->selected_devices.push_back(DescribeHardwareDevice(devices[i]));
    }

    *ep = &impl.release()->iface;
  });
}

void FactoryReleaseEp(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  if (ep == nullptr) {
    return;
  }
  delete AsEp(ep);
}

OrtStatus* FactoryValidateCompiledModelCompatibilityInfo(OrtEpFactory* /*this_ptr*/,
                                                         const OrtHardwareDevice* const* /*devices*/,
                                                         size_t /*num_devices*/,
                                                         const char* compatibility_info,
                                                         OrtCompiledModelCompatibility* compatibility) noexcept {
  if (compatibility_info == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "compatibility_info must not be null");
  }
  if (compatibility == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "compatibility output must not be null");
  }
  *compatibility = std::strcmp(compatibility_info, "ggonnx-scaffold") == 0
                       ? OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL
                       : OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  return nullptr;
}

OrtStatus* FactoryCreateAllocator(OrtEpFactory* /*this_ptr*/,
                                  const OrtMemoryInfo* /*memory_info*/,
                                  const OrtKeyValuePairs* /*allocator_options*/,
                                  OrtAllocator** allocator) noexcept {
  if (allocator == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "allocator output must not be null");
  }
  *allocator = nullptr;
  return nullptr;
}

void FactoryReleaseAllocator(OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {}

OrtStatus* FactoryCreateDataTransfer(OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  if (data_transfer == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "data_transfer output must not be null");
  }
  *data_transfer = nullptr;
  return nullptr;
}

bool FactoryIsStreamAware(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

OrtStatus* FactoryCreateSyncStreamForDevice(OrtEpFactory* /*this_ptr*/,
                                            const OrtMemoryDevice* /*memory_device*/,
                                            const OrtKeyValuePairs* /*stream_options*/,
                                            OrtSyncStreamImpl** stream) noexcept {
  if (stream == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "stream output must not be null");
  }
  *stream = nullptr;
  return nullptr;
}

OrtStatus* FactoryGetHardwareDeviceIncompatibilityDetails(OrtEpFactory* /*this_ptr*/,
                                                          const OrtHardwareDevice* /*hw*/,
                                                          OrtDeviceEpIncompatibilityDetails* /*details*/) noexcept {
  return nullptr;
}

OrtStatus* FactoryCreateExternalResourceImporterForDevice(
    OrtEpFactory* /*this_ptr*/,
    const OrtEpDevice* /*ep_device*/,
    OrtExternalResourceImporterImpl** out_importer) noexcept {
  if (out_importer == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "external resource importer output must not be null");
  }
  *out_importer = nullptr;
  return MakeStatus(ORT_NOT_IMPLEMENTED, "external resource import is not supported");
}

OrtStatus* FactoryGetNumCustomOpDomains(OrtEpFactory* /*this_ptr*/, size_t* num_domains) noexcept {
  if (num_domains == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "num_domains output must not be null");
  }
  *num_domains = 0;
  return nullptr;
}

OrtStatus* FactoryGetCustomOpDomains(OrtEpFactory* /*this_ptr*/,
                                     OrtCustomOpDomain** /*domains*/,
                                     size_t num_domains) noexcept {
  if (num_domains != 0) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "GGONNX does not provide custom op domains");
  }
  return nullptr;
}

std::unique_ptr<GGMLFactory> BuildFactory(const char* registered_name, const OrtApiBase* ort_api_base) {
  InitializeOrtApi(ort_api_base);

  auto factory = std::make_unique<GGMLFactory>();
  factory->registered_name = registered_name != nullptr ? registered_name : kRegistrationName;
  factory->iface.ort_version_supported = ORT_API_VERSION;
  factory->iface.GetName = FactoryGetName;
  factory->iface.GetVendor = FactoryGetVendor;
  factory->iface.GetSupportedDevices = FactoryGetSupportedDevices;
  factory->iface.CreateEp = FactoryCreateEp;
  factory->iface.ReleaseEp = FactoryReleaseEp;
  factory->iface.GetVendorId = FactoryGetVendorId;
  factory->iface.GetVersion = FactoryGetVersion;
  factory->iface.ValidateCompiledModelCompatibilityInfo = FactoryValidateCompiledModelCompatibilityInfo;
  factory->iface.CreateAllocator = FactoryCreateAllocator;
  factory->iface.ReleaseAllocator = FactoryReleaseAllocator;
  factory->iface.CreateDataTransfer = FactoryCreateDataTransfer;
  factory->iface.IsStreamAware = FactoryIsStreamAware;
  factory->iface.CreateSyncStreamForDevice = FactoryCreateSyncStreamForDevice;
  factory->iface.GetHardwareDeviceIncompatibilityDetails = FactoryGetHardwareDeviceIncompatibilityDetails;
  factory->iface.CreateExternalResourceImporterForDevice = FactoryCreateExternalResourceImporterForDevice;
  factory->iface.GetNumCustomOpDomains = FactoryGetNumCustomOpDomains;
  factory->iface.GetCustomOpDomains = FactoryGetCustomOpDomains;
  return factory;
}

}  // namespace

extern "C" OrtStatus* CreateEpFactories(const char* registered_name,
                                        const OrtApiBase* ort_api_base,
                                        const OrtLogger* /*default_logger*/,
                                        OrtEpFactory** factories,
                                        size_t max_factories,
                                        size_t* num_factories) noexcept {
  if (ort_api_base != nullptr && !IsOrtApiInitialized()) {
    InitializeOrtApi(ort_api_base);
  }

  return WrapStatus([&] {
    if (max_factories == 0) {
      throw std::runtime_error("max_factories must be greater than zero");
    }
    GGONNX_NOT_NULL(factories, "factories output array must not be null");
    GGONNX_NOT_NULL(num_factories, "num_factories output must not be null");
    auto factory = BuildFactory(registered_name, ort_api_base);
    factories[0] = &factory.release()->iface;
    *num_factories = 1;
  });
}

extern "C" OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) noexcept {
  if (factory == nullptr) {
    return nullptr;
  }
  delete AsFactory(factory);
  return nullptr;
}

extern "C" uint64_t GGONNX_DebugGetGraphBuildCount() noexcept {
  return g_debug_graph_build_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t GGONNX_DebugGetLastGraphNodeCount() noexcept {
  return g_debug_last_graph_node_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t GGONNX_DebugGetLastGraphOutputCount() noexcept {
  return g_debug_last_graph_output_count.load(std::memory_order_relaxed);
}

extern "C" void GGONNX_DebugResetGraphBuildCount() noexcept {
  g_debug_graph_build_count.store(0, std::memory_order_relaxed);
  g_debug_last_graph_node_count.store(0, std::memory_order_relaxed);
  g_debug_last_graph_output_count.store(0, std::memory_order_relaxed);
}
