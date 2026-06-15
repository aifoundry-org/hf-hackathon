#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime/cpu_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace {

constexpr const char* kRegistrationName = "GGONNX";
constexpr const char* kEpName = "GGMLExecutionProvider";

std::filesystem::path DefaultEpPath(const std::filesystem::path& argv0) {
  return std::filesystem::absolute(argv0).parent_path() / "libggonnx_ep.so";
}

std::vector<Ort::ConstEpDevice> SelectDevices(Ort::Env& env) {
  std::vector<Ort::ConstEpDevice> selected;
  for (const auto& device : env.GetEpDevices()) {
    if (std::string(device.EpName()) == kEpName) {
      selected.push_back(device);
    }
  }
  return selected;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ggonnx_runner /absolute/path/to/model.onnx\n";
    return 1;
  }

  try {
    Ort::InitApi();
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ggonnx"};

    const char* env_path = std::getenv("GGONNX_EP_PATH");
    const std::filesystem::path ep_path = env_path != nullptr ? env_path : DefaultEpPath(argv[0]);
    env.RegisterExecutionProviderLibrary(kRegistrationName, ep_path.string());

    auto devices = SelectDevices(env);
    std::cout << "Registered GGONNX library: " << ep_path << "\n";
    std::cout << "Discovered GGML EP devices: " << devices.size() << "\n";

    Ort::SessionOptions session_options;
    if (!devices.empty()) {
      session_options.AppendExecutionProvider_V2(env, devices, std::unordered_map<std::string, std::string>{});
    }
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));

    Ort::Session session{env, argv[1], session_options};
    std::cout << "Session created for model: " << argv[1] << "\n";
    std::cout << "Input count: " << session.GetInputCount() << "\n";
    std::cout << "Output count: " << session.GetOutputCount() << "\n";
  } catch (const Ort::Exception& ex) {
    std::cerr << "ORT error: " << ex.what() << "\n";
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 3;
  }

  return 0;
}
