#!/usr/bin/env bash
# Run the repository's real ET compiler in its supported Ubuntu 24.04 image.
#
# This mirrors .github/workflows/benchmark-board.yml. It is intentionally not
# a generic RISC-V compiler fallback: the executable inside the container must
# be the ET toolchain and the platform tree is mounted read-only.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
workspace="${ET_DOCKER_WORKSPACE:-$repo_root}"
real_et_root="${ET_DOCKER_REAL_ET_ROOT:-/opt/et}"
platform_src="${ET_DOCKER_PLATFORM_SRC:-${ET_PLATFORM_SRC:-}}"
image="${ET_DOCKER_IMAGE:-et-gcc:24.04}"
compiler="$real_et_root/bin/riscv64-unknown-elf-gcc"

[[ -d "$workspace"
   && ( "$PWD" == "$workspace" || "$PWD" == "$workspace/"* ) ]] || {
  echo "error: current directory must be inside ET_DOCKER_WORKSPACE=$workspace" >&2
  exit 2
}
[[ -x "$compiler" ]] || {
  echo "error: real ET compiler is not executable: $compiler" >&2
  exit 2
}
[[ -n "$platform_src" && -d "$platform_src" ]] || {
  echo "error: ET platform source tree not found: $platform_src" >&2
  exit 2
}
command -v docker >/dev/null || {
  echo "error: Docker is required for the supported Ubuntu 24.04 ET compiler path" >&2
  exit 2
}

exec docker run --rm \
  -v "$workspace:$workspace" \
  -v "$real_et_root:$real_et_root:ro" \
  -v "$platform_src:$platform_src:ro" \
  -w "$PWD" \
  "$image" \
  "$compiler" "$@"
