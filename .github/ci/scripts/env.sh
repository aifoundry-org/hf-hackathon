#!/usr/bin/env bash
# Shared CI / local benchmark environment. Source from other scripts:
#   source "$(dirname "$0")/env.sh"

set -euo pipefail

_ci_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_ci_env_dir}/../../.." && pwd)"

export REPO_ROOT
export WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.ci-work}"
export ET_PLATFORM_SRC="${ET_PLATFORM_SRC:-${WORK_ROOT}/et-platform}"
export ET_INSTALL="${ET_INSTALL:-${WORK_ROOT}/et}"
export AMP_ROOT="${AMP_ROOT:-${REPO_ROOT}/local-artifacts/erbium_amp_probe}"
export BUILD_ROOT="${BUILD_ROOT:-${WORK_ROOT}/build}"
export BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-${WORK_ROOT}/benchmark-output}"
export BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-${REPO_ROOT}/.github/ci/benchmark_config.json}"

export ET_PLATFORM_REPO="${ET_PLATFORM_REPO:-aifoundry-org/et-platform}"
export ET_PLATFORM_REF="${ET_PLATFORM_REF:-master}"
export TOOLCHAIN_ARCH="${TOOLCHAIN_ARCH:-rv64imfc}"
export TOOLCHAIN_ABI="${TOOLCHAIN_ABI:-lp64f}"
export TOOLCHAIN_DISTRO="${TOOLCHAIN_DISTRO:-ubuntu}"
export TOOLCHAIN_DISTRO_VERSION="${TOOLCHAIN_DISTRO_VERSION:-24.04}"
export RISCV_TOOLCHAIN_REPO="${RISCV_TOOLCHAIN_REPO:-aifoundry-org/riscv-gnu-toolchain}"

export LAUNCHER="${LAUNCHER:-${BUILD_ROOT}/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem}"

# Sys-emu timeouts (seconds). Override for slow hosts.
export SYSEMU_TIMEOUT="${SYSEMU_TIMEOUT:-3600}"
export SYSEMU_LAUNCHER_TIMEOUT="${SYSEMU_LAUNCHER_TIMEOUT:-1800}"

mkdir -p "$WORK_ROOT" "$BUILD_ROOT" "$BENCHMARK_OUTPUT" "$AMP_ROOT"
