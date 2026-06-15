#include "inner/ort_api_helpers.hpp"

#include "inner/helpers.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <stdexcept>

namespace ggonnx::ort_internal {
namespace {

const OrtApi* g_ort_api = nullptr;

}  // namespace

void InitializeOrtApi(const OrtApiBase* ort_api_base) {
  GGONNX_NOT_NULL(ort_api_base, "OrtApiBase must not be null");
  g_ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  GGONNX_NOT_NULL(g_ort_api, "failed to get ORT API");
  Ort::InitApi(g_ort_api);
  GGONNX_NOT_NULL(g_ort_api->GetEpApi(), "failed to initialize ORT EP API");
}

bool IsOrtApiInitialized() noexcept {
  return g_ort_api != nullptr;
}

const OrtApi& GetOrtApi() {
  return *GGONNX_NOT_NULL(g_ort_api, "ORT API has not been initialized");
}

const OrtEpApi& GetOrtEpApi() {
  const OrtEpApi* ep_api = GGONNX_NOT_NULL(GetOrtApi().GetEpApi(), "ORT EP API has not been initialized");
  return *ep_api;
}

OrtStatus* MakeStatus(OrtErrorCode code, const std::string& message) noexcept {
  try {
    return GetOrtApi().CreateStatus(code, message.c_str());
  } catch (const std::exception& ex) {
    GGONNX_ABORT(ex.what());
  } catch (...) {
    GGONNX_ABORT("failed to create ORT status");
  }
}

void THROW_ON_ERROR(OrtStatus* status) {
  if (status == nullptr) {
    return;
  }

  std::string message = GetOrtApi().GetErrorMessage(status);
  GetOrtApi().ReleaseStatus(status);
  throw std::runtime_error(message);
}

}  // namespace ggonnx::ort_internal
