cmake_minimum_required(VERSION 3.22)

project(ggonnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

option(GGONNX_BUILD_TOOLS "Build local debug tools" ON)

set(LLAMA_CPP_ROOT "/home/marty/Documents/llama.cpp" CACHE PATH "Path to llama.cpp checkout")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_ROOT}/build" CACHE PATH "Path to llama.cpp build directory")

find_package(onnxruntime CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GGML_INCLUDE_DIR "${LLAMA_CPP_ROOT}/ggml/include")
set(GGML_LIBRARY "${LLAMA_CPP_BUILD_DIR}/bin/libggml.so")
set(GGML_BASE_LIBRARY "${LLAMA_CPP_BUILD_DIR}/bin/libggml-base.so")
set(GGML_CPU_LIBRARY "${LLAMA_CPP_BUILD_DIR}/bin/libggml-cpu.so")

foreach(path_var GGML_INCLUDE_DIR GGML_LIBRARY GGML_BASE_LIBRARY GGML_CPU_LIBRARY)
  if(NOT EXISTS "${${path_var}}")
    message(FATAL_ERROR "${path_var} points to a missing path: ${${path_var}}")
  endif()
endforeach()

add_library(ggonnx_ep SHARED
  src/inner/ort_api_helpers.cc
  src/ggml_execution_provider.cc
  src/meta_eval.cpp
  src/ops.cpp
)

target_include_directories(ggonnx_ep
  PRIVATE
    ${GGML_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(ggonnx_ep
  PRIVATE
    onnxruntime::onnxruntime
    ${GGML_LIBRARY}
    ${GGML_BASE_LIBRARY}
    ${GGML_CPU_LIBRARY}
)

target_compile_definitions(ggonnx_ep PRIVATE ORT_API_MANUAL_INIT)

set_target_properties(ggonnx_ep PROPERTIES
  OUTPUT_NAME ggonnx_ep
  BUILD_RPATH "/usr/lib;${LLAMA_CPP_BUILD_DIR}/bin"
)

if(GGONNX_BUILD_TOOLS)
  add_executable(ggonnx_runner
    tools/ggonnx_runner.cc
  )

  target_link_libraries(ggonnx_runner PRIVATE onnxruntime::onnxruntime)
  target_compile_definitions(ggonnx_runner PRIVATE ORT_API_MANUAL_INIT)
  set_target_properties(ggonnx_runner PROPERTIES
    BUILD_RPATH "/usr/lib;${CMAKE_CURRENT_BINARY_DIR};${LLAMA_CPP_BUILD_DIR}/bin"
  )
endif()

add_test(
  NAME ggonnx_tests
  COMMAND
    ${CMAKE_COMMAND}
    -E
    env
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    ${Python3_EXECUTABLE}
    -m
    pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    -q
    -m
    "not integration"
    --ep-library
    $<TARGET_FILE:ggonnx_ep>
)

add_test(
  NAME ggonnx_integration_tests
  COMMAND
    ${CMAKE_COMMAND}
    -E
    env
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    ${Python3_EXECUTABLE}
    -m
    pytest
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_integration.py
    -q
    --ep-library
    $<TARGET_FILE:ggonnx_ep>
)
