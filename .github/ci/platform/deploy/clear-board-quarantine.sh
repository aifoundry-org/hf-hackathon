#!/usr/bin/env bash
# Clear the persistent ET-SoC1 CI quarantine after a maintainer-verified
# external power cycle. This script never performs recovery operations itself.
set -euo pipefail

if [[ "${1:-}" != "--after-external-power-cycle" || "$#" -ne 1 ]]; then
  echo "usage: $0 --after-external-power-cycle" >&2
  exit 2
fi

quarantine="${ET_BOARD_QUARANTINE_FILE:-/var/lib/et-soc1-ci/quarantine}"
if [[ ! -f "$quarantine" ]]; then
  echo "ET-SoC1 is not quarantined: $quarantine does not exist"
  exit 0
fi

old_boot_id="$(sed -n 's/^boot_id=//p' "$quarantine" | head -1)"
new_boot_id="$(cat /proc/sys/kernel/random/boot_id)"
if [[ -z "$old_boot_id" || "$old_boot_id" == unknown ]]; then
  echo "error: quarantine has no auditable boot ID; refusing to clear it" >&2
  exit 1
fi
if [[ "$old_boot_id" == "$new_boot_id" ]]; then
  echo "error: host boot ID did not change; an external power cycle was not verified" >&2
  exit 1
fi

error_pattern='ET [0-9a-fA-F:.]+: Error Event Detected|OPS Kernel Launch|CM Runtime|MM2CMLaunch|KernelLaunch Failed|Execution error|illegal instruction|Couldn.t dispatch event:'
if ! dmesg >/dev/null 2>&1; then
  echo "error: cannot inspect the kernel log; refusing to clear quarantine" >&2
  exit 1
fi
if dmesg | grep -Eq "$error_pattern"; then
  echo "error: the new boot already contains ET runtime or firmware errors" >&2
  dmesg | grep -E "$error_pattern" | tail -40 >&2
  exit 1
fi

archive="${quarantine}.cleared.$(date -u +%Y%m%dT%H%M%SZ)"
mv "$quarantine" "$archive"
echo "ET-SoC1 quarantine cleared after verified boot change: $old_boot_id -> $new_boot_id"
echo "Audit record: $archive"
