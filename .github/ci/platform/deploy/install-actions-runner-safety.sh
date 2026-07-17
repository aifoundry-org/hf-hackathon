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

# The lock and quarantine must be writable by the account that actually runs the
# service. Do not assume the optional `etsoc` worker account exists on a host
# running the root-owned GitHub Actions service.
service_user="$(systemctl show "$unit" -p User --value)"
service_group="$(systemctl show "$unit" -p Group --value)"
service_user="${service_user:-root}"
if [[ -z "$service_group" ]]; then
  service_group="$(id -gn "$service_user")"
fi
if ! id "$service_user" >/dev/null 2>&1 || ! getent group "$service_group" >/dev/null; then
  echo "error: invalid runner service identity: $service_user:$service_group" >&2
  exit 1
fi

state_dir="${ET_BOARD_STATE_DIR:-/var/lib/et-soc1-ci}"
board_lock="${BOARD_LOCK:-$state_dir/board.lock}"
install -d -o "$service_user" -g "$service_group" -m 0770 "$state_dir"
touch "$board_lock"
chown "$service_user:$service_group" "$board_lock"
chmod 0660 "$board_lock"

dropin="/etc/systemd/system/${unit}.d"
install -d -m 0755 "$dropin"
install -m 0644 /dev/stdin "$dropin/et-board-safety.conf" <<'EOF'
[Service]
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
ProtectClock=yes
ReadOnlyPaths=/sys/bus/pci/devices /sys/devices /opt/et /opt/et-platform
IPAddressDeny=10.20.10.117
NoNewPrivileges=yes
LockPersonality=yes
RestrictRealtime=yes
CapabilityBoundingSet=~CAP_SYS_ADMIN CAP_SYS_BOOT CAP_SYS_MODULE CAP_SYS_RAWIO
SystemCallFilter=~@mount @reboot @module
EOF

systemctl daemon-reload
echo "Installed runner safety policy: $dropin/et-board-safety.conf"
echo "Installed board state and lock: $state_dir ($service_user:$service_group), $board_lock"
echo "The service was not started or restarted."
