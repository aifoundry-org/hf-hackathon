#!/usr/bin/env bash
# Harden the root-owned GitHub Actions runner so board jobs cannot modify kernel
# controls or invoke host recovery operations. The ET device nodes remain
# available; the restrictions apply to the runner and all of its descendants.
set -euo pipefail

unit="${1:-actions.runner.aifoundry-org-hf-hackathon.aifoundry3-et-soc1.service}"
if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: run this installer as root" >&2
  exit 1
fi
if ! systemctl cat "$unit" >/dev/null 2>&1; then
  echo "error: runner service does not exist: $unit" >&2
  exit 1
fi

dropin="/etc/systemd/system/${unit}.d"
install -d -m 0755 "$dropin"
install -m 0644 /dev/stdin "$dropin/et-board-safety.conf" <<'EOF'
[Service]
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
ProtectClock=yes
ReadOnlyPaths=/sys/bus/pci/devices /sys/devices
IPAddressDeny=10.20.10.117
NoNewPrivileges=yes
LockPersonality=yes
RestrictRealtime=yes
CapabilityBoundingSet=~CAP_SYS_ADMIN CAP_SYS_BOOT CAP_SYS_MODULE CAP_SYS_RAWIO
SystemCallFilter=~@mount @reboot @module
EOF

systemctl daemon-reload
echo "Installed runner safety policy: $dropin/et-board-safety.conf"
echo "The service was not started or restarted."
