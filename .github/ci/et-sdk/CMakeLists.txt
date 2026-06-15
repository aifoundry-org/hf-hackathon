# Minimal et-platform SDK for hf-hackathon CI (launcher + sys-emu only).
# ET_PLATFORM_SRC must point at a cloned aifoundry-org/et-platform tree.

cmake_minimum_required(VERSION 3.21)
project(et-sdk-ci LANGUAGES CXX C)

# et-platform-src is a symlink to the cloned tree (created by install_et_sdk.sh).
set(ET_PLATFORM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/et-platform-src")
if(NOT IS_DIRECTORY "${ET_PLATFORM_SRC_DIR}/common-sw")
  message(FATAL_ERROR "Missing et-platform-src symlink; run .github/ci/scripts/install_et_sdk.sh")
endif()

set(ET_PLATFORM_PATH "${CMAKE_INSTALL_PREFIX}")
set(TOOLCHAIN_DIR "${CMAKE_INSTALL_PREFIX}")
set(STAGING_DIR "${CMAKE_INSTALL_PREFIX}")
set(PROJECT_BASE_DIR "et-platform-src")

list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_SRC_DIR}/cmake" "${ET_PLATFORM_SRC_DIR}/cmake-modules")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake")
include("${ET_PLATFORM_SRC_DIR}/cmake/ProjectFunctions.cmake")

set(EXTERNAL_CACHE_FILE "${CMAKE_BINARY_DIR}/external_cache.cmake")
file(WRITE "${EXTERNAL_CACHE_FILE}"
    "set(CMAKE_MODULE_PATH \"${CMAKE_MODULE_PATH}\" CACHE PATH \"\" FORCE)\n"
    "set(CMAKE_PREFIX_PATH \"${CMAKE_PREFIX_PATH}\" CACHE PATH \"\" FORCE)\n"
    "set(CMAKE_CXX_FLAGS \"-Wno-error=unused-result\" CACHE STRING \"\" FORCE)\n"
    "set(CMAKE_C_FLAGS \"-Wno-error=unused-result\" CACHE STRING \"\" FORCE)\n"
)

ThirdParty(g3log https://github.com/KjellKod/g3log v1.1-408-g38dbddc "-DUSE_DYNAMIC_LOGGING_LEVELS=ON -DADD_G3LOG_UNIT_TEST=OFF -DINSTALL_G3LOG=ON")
ThirdParty(cereal https://github.com/USCiLab/cereal.git v1.3.2 "-DJUST_INSTALL_CEREAL=ON")
ThirdParty(easy_profiler https://github.com/yse/easy_profiler.git v2.1.0-66-gcc0e154 "-DEASY_PROFILER_NO_GUI=ON")

HostProject(et-driver "")
HostProject(common-sw "" g3log)
HostProject(device-api "" common-sw)
HostProject(erbium-hal "")
HostProject(sw-sysemu "" g3log common-sw et-driver erbium-hal)
HostProject(devicelayer "" common-sw sw-sysemu)
HostProject(esperanto-tools-libs "-DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF" devicelayer device-api g3log cereal easy_profiler)
