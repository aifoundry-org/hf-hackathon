#!/usr/bin/env bash
# Resolve erbium-soc1sim build paths for upstream et-platform (master) or legacy layouts.
# Source after env.sh; sets ERBIUM_LD, ERBIUM_LAYOUT, ERBIUM_GCC_INCLUDES.

set -euo pipefail

resolve_et_platform_paths() {
  local src="${ET_PLATFORM_SRC:?ET_PLATFORM_SRC not set}"

  # Upstream aifoundry-org/et-platform (gp-sdk layout).
  if [[ -f "${src}/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" ]]; then
    export ERBIUM_LD="${src}/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld"
    export ERBIUM_LAYOUT="${src}/gp-sdk/device/sdk/lib/erbium-soc1sim/layout.c"
    export ERBIUM_GCC_INCLUDES=(
      "-I${src}/et-common-libs/include"
      "-I${src}/et-common-libs/include/erbium-soc1sim"
      "-I${src}/erbium-hal/include"
      "-I${src}/hal/platform/erbium/include"
      "-I${src}/hal/platform/etsoc/include"
    )
    export ERBIUM_GCC_INCLUDE_FLAGS="${ERBIUM_GCC_INCLUDES[*]}"
    echo "Using upstream gp-sdk erbium-soc1sim paths"
    return 0
  fi

  # Legacy soc1sim-poc style layout.
  if [[ -f "${src}/et-common-libs/share/erbium-soc1sim/erbium.ld" ]]; then
    export ERBIUM_LD="${src}/et-common-libs/share/erbium-soc1sim/erbium.ld"
    export ERBIUM_LAYOUT="${src}/erbium-examples/runtime/erbium-soc1sim/layout.c"
    export ERBIUM_GCC_INCLUDES=(
      "-I${src}/et-common-libs/build-headers/erbium-soc1sim-staged-include"
      "-I${src}/hal/platform/erbium/include"
      "-I${src}/hal/platform/etsoc/include"
      "-I${src}/et-common-libs/include"
    )
    export ERBIUM_GCC_INCLUDE_FLAGS="${ERBIUM_GCC_INCLUDES[*]}"
    echo "Using legacy erbium-soc1sim staged-include paths"
    return 0
  fi

  echo "error: no erbium-soc1sim linker script found under ${src}" >&2
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  source "$(dirname "$0")/env.sh"
  resolve_et_platform_paths
  echo "ERBIUM_LD=${ERBIUM_LD}"
  echo "ERBIUM_LAYOUT=${ERBIUM_LAYOUT}"
fi
