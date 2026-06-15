#pragma once

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "ops.hpp"

// Sentinel stored in folded INT64/INT32 tensors to mark a dimension that is
// dynamic (unknown at compile time). Such "constants" must never be used as
// compile-time GGML values — treat them as runtime inputs instead.
inline constexpr int64_t kUnknownShapeDimSentinel =
    static_cast<int64_t>(std::numeric_limits<int32_t>::min());

// Returns true if a folded constant tensor contains the dynamic sentinel.
bool ConstantContainsSentinel(const ConstantTensor& tensor);

struct MetaAnalysis {
  ConstantValueMap constants;
  // Per-tensor fully-static shape, populated by propagation past ORT's
  // shape-inference gaps. Seeded from ORT-declared static shapes and from
  // folded-constant dims, then extended by per-op shape rules that can
  // derive output shapes from folded constants (e.g. Pad, Reshape). Entries
  // are guaranteed to be fully static (all dims >= 0).
  std::unordered_map<std::string, std::vector<int64_t>> inferred_shapes;
  std::unordered_set<std::string> folded_nodes;
  FusionPlan fusions;
};

MetaAnalysis AnalyzeCompileTimeConstants(const OrtGraph* graph);
std::string NodeKey(Ort::ConstNode node);

// Returns the fully-static dims for a tensor, consulting (1) ORT's declared
// shape when all dims are >= 0, then (2) the inferred_shapes map in `meta`.
// Returns nullopt if no fully-static shape is available.
std::optional<std::vector<int64_t>> ResolveShape(Ort::ConstValueInfo vi,
                                                 const MetaAnalysis& meta);
