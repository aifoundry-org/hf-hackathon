#!/usr/bin/env bash
# One-time: copy Esperanto RISC-V toolchain to soc3 (stash mount; root disk is often full).
set -euo pipefail
SRC="${ET_TOOLCHAIN_SRC:-$HOME/et}"
DEST="${SOC3_TOOLCHAIN_DEST:-/opt/et}"
HOST="${SOC3_SSH_ALIAS:-board-host}"
if [[ ! -x "${SRC}/bin/riscv64-unknown-elf-gcc" ]]; then
  echo "error: no toolchain at ${SRC}/bin/riscv64-unknown-elf-gcc" >&2
  exit 1
fi
echo "==> rsync ${SRC} -> ${HOST}:${DEST}"
rsync -az --info=progress2 "${SRC}/bin" "${SRC}/lib" "${HOST}:${DEST}/"
ssh "${HOST}" "${DEST}/bin/riscv64-unknown-elf-gcc --version | head -1"
echo "Set ET_INSTALL=${DEST} on soc3 benchmarks."
