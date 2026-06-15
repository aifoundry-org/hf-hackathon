cmake_minimum_required(VERSION 3.21)

project(erbium_soc1sim_argbuf_dynmem VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# runtimeConfig.cmake uses find_dependency(libcap); upstream ships Findlibcap.cmake via pkg-config.
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../et-sdk/et-platform-src/cmake")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../et-sdk/et-platform-src/cmake")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/lib/cmake")

find_package(deviceLayer REQUIRED)
find_package(runtime REQUIRED)
find_package(hostUtils REQUIRED)
find_package(sw-sysemu REQUIRED)

add_executable(erbium_soc1sim_argbuf_dynmem erbium_soc1sim_argbuf_dynmem.cpp)

target_link_libraries(erbium_soc1sim_argbuf_dynmem
    PRIVATE
        deviceLayer::deviceLayer
        runtime::etrt
        hostUtils::logging
        sw-sysemu::sw-sysemu
)

target_compile_definitions(erbium_soc1sim_argbuf_dynmem
    PRIVATE
        ET_PLATFORM_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
)
