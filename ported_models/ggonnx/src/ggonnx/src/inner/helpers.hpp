#pragma once

#include <onnxruntime/onnxruntime_c_api.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include <ggml.h>

[[noreturn]] static void GGONNX_ABORT(const char* message) noexcept {
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  std::abort();
}

static void GGONNX_ASSERT(bool condition, const std::string& message) {
  if (condition) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename T>
T* GGONNX_NOT_NULL(T* ptr, const std::string& message) {
  if (ptr == nullptr) {
    throw std::runtime_error(message);
  }
  return ptr;
}

template <typename T>
const T* GGONNX_NOT_NULL(const T* ptr, const std::string& message) {
  if (ptr == nullptr) {
    throw std::runtime_error(message);
  }
  return ptr;
}

static std::string FormatDims(const std::vector<int64_t>& dims) {
  std::string result = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += std::to_string(dims[i]);
  }
  result += "]";
  return result;
}


inline std::array<int64_t, GGML_MAX_DIMS> ToGGMLDims(const std::vector<int64_t>& onnx_dims) {
  GGONNX_ASSERT(onnx_dims.size() <= GGML_MAX_DIMS,
                "ONNX tensor rank exceeds GGML maximum rank of " + std::to_string(GGML_MAX_DIMS));
  std::array<int64_t, GGML_MAX_DIMS> ggml_dims{};
  for (size_t i = 0; i < onnx_dims.size(); ++i) {
    ggml_dims[i] = onnx_dims[onnx_dims.size() - 1 - i];
  }
  return ggml_dims;
}

inline std::array<int64_t, GGML_MAX_DIMS> ToPaddedGGMLDims(const std::vector<int64_t>& onnx_dims) {
  GGONNX_ASSERT(onnx_dims.size() <= GGML_MAX_DIMS,
                "ONNX tensor rank exceeds GGML maximum rank of " + std::to_string(GGML_MAX_DIMS));
  std::array<int64_t, GGML_MAX_DIMS> ggml_dims;
  ggml_dims.fill(1);
  for (size_t i = 0; i < onnx_dims.size(); ++i) {
    ggml_dims[i] = onnx_dims[onnx_dims.size() - 1 - i];
  }
  return ggml_dims;
}

inline std::vector<int64_t> ToOnnxDims(const ggml_tensor* tensor) {
  GGONNX_NOT_NULL(tensor, "ggml tensor must not be null");
  if (ggml_is_scalar(tensor)) {
    return {};
  }

  const int rank = ggml_n_dims(tensor);
  GGONNX_ASSERT(rank >= 1 && rank <= GGML_MAX_DIMS, "ggml tensor has invalid rank");

  std::vector<int64_t> onnx_dims;
  onnx_dims.reserve(static_cast<size_t>(rank));
  for (int i = rank - 1; i >= 0; --i) {
    onnx_dims.push_back(tensor->ne[i]);
  }
  return onnx_dims;
}
