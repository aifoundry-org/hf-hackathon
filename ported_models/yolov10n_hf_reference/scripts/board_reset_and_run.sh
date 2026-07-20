#!/usr/bin/env bash
# Reset ET-SoC1 using the same sysfs control as soc3-benchmark.sh, then exec.
set -euo pipefail

[[ $# -gt 0 ]] || {
  echo "error: board_reset_and_run.sh requires a launcher command" >&2
  exit 2
}

reset_path=""
shopt -s nullglob
for candidate in \
  /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate \
  /sys/bus/pci/devices/*/soc_reset/reinitiate; do
  if [[ -w "$candidate" ]]; then
    reset_path="$candidate"
    break
  fi
done
shopt -u nullglob

[[ -n "$reset_path" ]] || {
  echo "error: no writable ET-SoC1 reset control was found" >&2
  exit 2
}
echo "Resetting ET-SoC1 via $reset_path"
printf '1\n' > "$reset_path"
sleep 2
exec "$@"
