#!/usr/bin/env bash
# Install sys-emu firmware ELFs expected under ${ET_INSTALL}/lib/esperanto-fw/.

set -euo pipefail

source "$(dirname "$0")/env.sh"

fw_root="${ET_INSTALL}/lib/esperanto-fw"
marker="${fw_root}/BootromTrampolineToBL2/BootromTrampolineToBL2.elf"

required=(
  "${fw_root}/BootromTrampolineToBL2/BootromTrampolineToBL2.elf"
  "${fw_root}/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf"
  "${fw_root}/MachineMinion/MachineMinion.elf"
  "${fw_root}/MasterMinion/MasterMinion.elf"
  "${fw_root}/WorkerMinion/WorkerMinion.elf"
)

firmware_ready() {
  local f
  for f in "${required[@]}"; do
    [[ -f "$f" ]] || return 1
  done
  return 0
}

if firmware_ready; then
  echo "Sys-emu firmware already installed under ${fw_root}"
  exit 0
fi

install_tree() {
  local src="$1"
  mkdir -p "${fw_root}"
  cp -a "${src}/." "${fw_root}/"
}

if [[ -n "${ET_FIRMWARE_DIR:-}" && -d "${ET_FIRMWARE_DIR}" ]]; then
  install_tree "${ET_FIRMWARE_DIR}"
elif [[ -d "${REPO_ROOT}/.github/ci/firmware/esperanto-fw" ]]; then
  install_tree "${REPO_ROOT}/.github/ci/firmware/esperanto-fw"
elif [[ -n "${ET_FIRMWARE_URL:-}" ]]; then
  tmp="$(mktemp -d)"
  auth_args=()
  if [[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]]; then
    auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN:-${GH_TOKEN}}" -H "Accept: application/octet-stream")
  fi
  if curl -fsSL "${auth_args[@]}" -L "${ET_FIRMWARE_URL}" -o "${tmp}/firmware.tar.gz" \
    && tar -xzf "${tmp}/firmware.tar.gz" -C "${tmp}"; then
    if [[ -d "${tmp}/esperanto-fw" ]]; then
      install_tree "${tmp}/esperanto-fw"
    else
      install_tree "${tmp}"
    fi
  else
    echo "error: failed to download ET_FIRMWARE_URL" >&2
    exit 1
  fi
  rm -rf "${tmp}"
else
  echo "error: sys-emu firmware missing; set ET_FIRMWARE_DIR, ET_FIRMWARE_URL, or add .github/ci/firmware/esperanto-fw" >&2
  exit 1
fi

if ! firmware_ready; then
  echo "error: incomplete firmware install under ${fw_root}" >&2
  exit 1
fi

echo "Sys-emu firmware installed under ${fw_root}"
