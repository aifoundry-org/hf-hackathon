#!/usr/bin/env bash
# Install the RISC-V cross toolchain from aifoundry-org/riscv-gnu-toolchain releases.
# Does not build from source.

set -euo pipefail

source "$(dirname "$0")/env.sh"

_toolchain_ok() {
  local gcc="${1}/bin/riscv64-unknown-elf-gcc"
  [[ -x "${gcc}" ]] || return 1
  "${gcc}" --version >/dev/null 2>&1 || return 1
  echo 'csrr x0, flb' | "${gcc}" -c -x assembler - -o /dev/null 2>/dev/null
}

for _tc_root in "${ET_INSTALL}" "${HOME}/et" /opt/et /opt/temp "${REPO_ROOT}/.ci-work/et"; do
  if _toolchain_ok "${_tc_root}"; then
    export ET_INSTALL="${_tc_root}"
    export PATH="${ET_INSTALL}/bin:${PATH}"
    echo "RISC-V toolchain already present at ${ET_INSTALL}"
    "${ET_INSTALL}/bin/riscv64-unknown-elf-gcc" --version | head -1
    exit 0
  fi
done

repo="${RISCV_TOOLCHAIN_REPO:-aifoundry-org/riscv-gnu-toolchain}"
mkdir -p "${ET_INSTALL}"

# Try requested distro version, then common release targets (24.04 is current upstream default).
versions=()
if [[ -n "${TOOLCHAIN_DISTRO_VERSION:-}" ]]; then
  versions+=("${TOOLCHAIN_DISTRO_VERSION}")
fi
for v in 24.04 22.04; do
  [[ " ${versions[*]} " == *" ${v} "* ]] || versions+=("${v}")
done

_release_asset_url() {
  local artifact="$1"
  curl -fsSL "https://api.github.com/repos/${repo}/releases/latest" | python3 -c '
import json, sys
name = sys.argv[1]
data = json.load(sys.stdin)
for asset in data.get("assets", []):
    if asset.get("name") == name:
        print(asset.get("browser_download_url", ""))
        break
' "${artifact}"
}

for ver in "${versions[@]}"; do
  artifact="riscv64-elf-${TOOLCHAIN_DISTRO}-${ver}-gcc-stripped.tar.xz"
  url="$(_release_asset_url "${artifact}")"
  if [[ -n "${url}" && "${url}" != "null" ]]; then
    echo "Downloading ${artifact} from ${repo} releases"
    curl -fsSL "${url}" | tar --strip-components=1 -xJ -C "${ET_INSTALL}"
    export PATH="${ET_INSTALL}/bin:${PATH}"
    if _toolchain_ok "${ET_INSTALL}"; then
      "${ET_INSTALL}/bin/riscv64-unknown-elf-gcc" --version | head -1
      exit 0
    fi
    echo "warn: ${artifact} installed but failed erbium CSR probe; trying next" >&2
  fi
  echo "Release asset not found: ${artifact}" >&2
done

echo "error: no matching riscv-gnu-toolchain release tarball for ${TOOLCHAIN_DISTRO}" >&2
exit 1
