#!/usr/bin/env bash
# Put ET-SoC1 into the board-specific, validated CI operating point.
#
# aifoundry3's ET-SoC1 becomes unreliable when firmware DVFS raises the minion
# clock above its 600 MHz minimum.  Frequencies outside the provisioned VMIN
# LUT (600/700/800 MHz) are also unsafe: firmware repeatedly reports an invalid
# operating point and floods the completion queue with error events.  Keep the
# board at the supported 600/400 MHz point by setting a zero-watt software TDP
# ceiling.  At the minimum clock, this leaves active thermal/power management
# enabled but prevents a throttle-up transition.
#
# Keep the default scoped to aifoundry3 so other boards retain their
# provisioned policy. Operators can explicitly set both clock variables and,
# optionally, ET_BOARD_TDP_W for another host.
set -euo pipefail

host="${ET_BOARD_HOSTNAME:-$(hostname -s)}"
target_minion="${ET_MINION_FREQUENCY_MHZ:-}"
target_noc="${ET_NOC_FREQUENCY_MHZ:-}"
target_tdp="${ET_BOARD_TDP_W:-}"

if [[ -z "$target_minion" && -z "$target_noc" && -z "$target_tdp" ]] \
  && { [[ "$host" == aifoundry3* ]] || [[ "$host" == esperanto-soc3* ]]; }; then
  target_minion=600
  target_noc=400
  target_tdp=0
fi

if [[ -z "$target_minion" && -z "$target_noc" && -z "$target_tdp" ]]; then
  echo "ET board clock guard: no clock policy configured for $host"
  exit 0
fi
if [[ -z "$target_minion" || -z "$target_noc" ]]; then
  echo "error: set both ET_MINION_FREQUENCY_MHZ and ET_NOC_FREQUENCY_MHZ" >&2
  exit 2
fi
if [[ ! "$target_minion" =~ ^[0-9]+$ || ! "$target_noc" =~ ^[0-9]+$ ]]; then
  echo "error: ET board clock targets must be integer MHz values" >&2
  exit 2
fi
if [[ -n "$target_tdp" ]] && {
  [[ ! "$target_tdp" =~ ^[0-9]+$ ]] || (( target_tdp > 255 ));
}; then
  echo "error: ET_BOARD_TDP_W must be an integer from 0 through 255" >&2
  exit 2
fi

service="${ET_DEV_MNGT_SERVICE:-${ET_INSTALL:-/opt/et}/bin/dev_mngt_service}"
retries="${ET_BOARD_CLOCK_RETRIES:-5}"
retry_delay="${ET_BOARD_CLOCK_RETRY_DELAY_S:-2}"
command_timeout="${ET_BOARD_CLOCK_COMMAND_TIMEOUT_S:-20}"
marker="${ET_BOARD_CLOCK_MARKER:-/run/et-board-clock-guard.ok}"
boot_id="${ET_BOARD_BOOT_ID:-$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)}"
device_path="${ET_BOARD_DEVICE_PATH:-/dev/et0_mgmt}"
marker_tdp="${target_tdp:--}"
expected_marker="$boot_id $target_minion $target_noc $marker_tdp"

# The iBoot outlet removes power from the ET card without rebooting Linux, so a
# host-boot marker alone cannot prove the card still has the configured state.
# Always read the real device state under the shared board lock.  The marker is
# retained as auditable evidence for score provenance, never as a cache bypass.

if [[ ! -x "$service" ]]; then
  echo "error: ET device management service is not executable: $service" >&2
  exit 1
fi
if [[ ! "$retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: ET_BOARD_CLOCK_RETRIES must be a positive integer" >&2
  exit 2
fi

# Benchmark launchers prepend host bundles to LD_LIBRARY_PATH.  Those bundles
# are valid for the launcher but may contain a newer libg3log/libstdc++ than
# this host can load into dev_mngt_service.  The management binary is installed
# and validated against the host libraries, so invoke it with those overrides
# removed while preserving every unrelated environment variable.
service_cmd=(env -u LD_LIBRARY_PATH -u LIBRARY_PATH "$service")

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

read_state() {
  : >"$log"
  timeout --kill-after=2s "$command_timeout" \
    "${service_cmd[@]}" -m DM_CMD_GET_ASIC_FREQUENCIES >>"$log" 2>&1
  if [[ -n "$target_tdp" ]]; then
    timeout --kill-after=2s "$command_timeout" \
      "${service_cmd[@]}" -m DM_CMD_GET_MODULE_STATIC_TDP_LEVEL >>"$log" 2>&1
  fi
}

state_value() {
  local label="$1"
  sed -n "s/.*ASIC Frequency ${label}: \\([0-9][0-9]*\\) Mhz.*/\\1/p" "$log" | tail -1
}

tdp_value() {
  sed -n 's/.*TDP Level Output: \([0-9][0-9]*\).*/\1/p' "$log" | tail -1
}

state_matches() {
  local current_minion current_noc current_tdp
  current_minion="$(state_value "Minion Shire")"
  current_noc="$(state_value "NOC")"
  current_tdp="$(tdp_value)"
  [[ "$current_minion" == "$target_minion" ]] \
    && [[ "$current_noc" == "$target_noc" ]] \
    && { [[ -z "$target_tdp" ]] || [[ "$current_tdp" == "$target_tdp" ]]; }
}

mark_verified() {
  local marker_tmp="${marker}.tmp.$$"
  mkdir -p "$(dirname "$marker")"
  printf '%s\n' "$expected_marker" > "$marker_tmp"
  chmod 0644 "$marker_tmp"
  mv -f "$marker_tmp" "$marker"
}

for attempt in $(seq 1 "$retries"); do
  if [[ ! -e "$device_path" ]]; then
    echo "ET board clock guard: waiting for $device_path (attempt $attempt/$retries)"
    if [[ "$attempt" -lt "$retries" ]]; then
      sleep "$retry_delay"
    fi
    continue
  fi

  if read_state && state_matches; then
    mark_verified
    echo "ET board clock guard: verified minion=${target_minion}MHz noc=${target_noc}MHz tdp=${marker_tdp}W on $host"
    exit 0
  fi

  echo "ET board clock guard: setting minion=${target_minion}MHz noc=${target_noc}MHz tdp=${marker_tdp}W on $host (attempt $attempt/$retries)"
  configured=1
  if [[ -n "$target_tdp" ]] && ! timeout --kill-after=2s "$command_timeout" \
    "${service_cmd[@]}" -m DM_CMD_SET_MODULE_STATIC_TDP_LEVEL -l "$target_tdp" >"$log" 2>&1; then
    configured=0
  fi
  # Set the supported clock after the TDP ceiling so firmware cannot race the
  # preflight by raising the minion operating point between the two requests.
  if (( configured )) && ! timeout --kill-after=2s "$command_timeout" \
    "${service_cmd[@]}" -m DM_CMD_SET_FREQUENCY -f "${target_minion},${target_noc}" >"$log" 2>&1; then
    configured=0
  fi
  if (( configured )) && read_state && state_matches; then
    mark_verified
    echo "ET board clock guard: verified minion=${target_minion}MHz noc=${target_noc}MHz tdp=${marker_tdp}W on $host"
    exit 0
  fi

  if [[ "$attempt" -lt "$retries" ]]; then
    sleep "$retry_delay"
  fi
done

echo "error: ET board clock guard could not verify minion=${target_minion}MHz noc=${target_noc}MHz tdp=${marker_tdp}W on $host" >&2
tail -80 "$log" >&2 || true
exit 1
