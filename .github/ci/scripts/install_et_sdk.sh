#!/usr/bin/env bash
# Build minimal et-platform SDK (sys-emu + launcher deps) into ET_INSTALL.

set -euo pipefail

source "$(dirname "$0")/env.sh"

sdk_marker="${ET_INSTALL}/lib/cmake/sw-sysemu/sw-sysemuConfig.cmake"
runtime_marker="${ET_INSTALL}/lib/cmake/runtime/runtimeConfig.cmake"
if [[ -f "${sdk_marker}" && -f "${runtime_marker}" ]]; then
  echo "ET platform SDK already installed at ${ET_INSTALL}"
  "$(dirname "$0")/install_firmware.sh"
  exit 0
fi

if [[ ! -x "${ET_INSTALL}/bin/riscv64-unknown-elf-gcc" ]]; then
  echo "error: RISC-V toolchain required first (.github/ci/scripts/install_toolchain.sh)" >&2
  exit 1
fi

platform_url="${ET_PLATFORM_GIT_URL:-https://github.com/${ET_PLATFORM_REPO}.git}"
ET_PLATFORM_SRC="$(realpath -m "${ET_PLATFORM_SRC}")"
mkdir -p "$(dirname "${ET_PLATFORM_SRC}")"

if [[ ! -d "${ET_PLATFORM_SRC}/.git" ]]; then
  git clone --depth 1 --branch "${ET_PLATFORM_REF}" "${platform_url}" "${ET_PLATFORM_SRC}" || {
    echo "warn: branch ${ET_PLATFORM_REF} missing, cloning default branch" >&2
    git clone --depth 1 "${platform_url}" "${ET_PLATFORM_SRC}"
  }
fi

et_build="${BUILD_ROOT}/et-sdk"
mkdir -p "${et_build}"
ln -sfn "${ET_PLATFORM_SRC}" "${REPO_ROOT}/.github/ci/et-sdk/et-platform-src"

echo "Building minimal et-platform SDK into ${ET_INSTALL} ..."
cmake -S "${REPO_ROOT}/.github/ci/et-sdk" -B "${et_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTOOLCHAIN_DIR="${ET_INSTALL}" \
  -DCMAKE_INSTALL_PREFIX="${ET_INSTALL}"

cmake --build "${et_build}" -j"$(nproc)"
cmake --install "${et_build}" 2>/dev/null || true

# ExternalProject installs into STAGING_DIR; ensure markers exist.
if [[ ! -f "${sdk_marker}" ]]; then
  echo "error: SDK install failed; missing ${sdk_marker}" >&2
  exit 1
fi
if [[ ! -f "${runtime_marker}" ]]; then
  echo "error: SDK install failed; missing ${runtime_marker}" >&2
  exit 1
fi

"$(dirname "$0")/install_firmware.sh"

echo "ET platform SDK installed."
