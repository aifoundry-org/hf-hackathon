#pragma once

#include <onnxruntime/onnxruntime_c_api.h>

#include <exception>
#include <string>

namespace ggonnx::ort_internal {

void InitializeOrtApi(const OrtApiBase* ort_api_base);
bool IsOrtApiInitialized() noexcept;

const OrtApi& GetOrtApi();
const OrtEpApi& GetOrtEpApi();

OrtStatus* MakeStatus(OrtErrorCode code, const std::string& message) noexcept;

template <typename Fn>
OrtStatus* WrapStatus(Fn&& fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const std::exception& ex) {
    return MakeStatus(ORT_RUNTIME_EXCEPTION, ex.what());
  } catch (...) {
    return MakeStatus(ORT_RUNTIME_EXCEPTION, "unknown exception");
  }
}

void THROW_ON_ERROR(OrtStatus* status);

}  // namespace ggonnx::ort_internal
