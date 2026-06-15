#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

if [[ -x "${LAUNCHER}" ]]; then
  echo "Launcher already built: ${LAUNCHER}"
  exit 0
fi

"$(dirname "$0")/install_et_sdk.sh"
"$(dirname "$0")/install_firmware.sh"

et_platform_src="${ET_PLATFORM_SRC}"
if [[ ! -f "${et_platform_src}/cmake/Findlibcap.cmake" ]]; then
  et_platform_src="${REPO_ROOT}/.github/ci/et-sdk/et-platform-src"
fi
if [[ ! -f "${et_platform_src}/cmake/Findlibcap.cmake" ]]; then
  echo "error: et-platform source missing (run install_et_sdk.sh)" >&2
  exit 1
fi

cmake -S "${REPO_ROOT}/.github/ci/launcher" \
  -B "${BUILD_ROOT}/erbium_soc1sim_argbuf" \
  -DCMAKE_PREFIX_PATH="${ET_INSTALL}" \
  -DCMAKE_MODULE_PATH="${et_platform_src}/cmake" \
  -DCMAKE_INSTALL_PREFIX="${ET_INSTALL}"

cmake --build "${BUILD_ROOT}/erbium_soc1sim_argbuf" -j"$(nproc)"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "error: launcher not found at ${LAUNCHER}" >&2
  exit 1
fi

"${LAUNCHER}" --help | head -3
