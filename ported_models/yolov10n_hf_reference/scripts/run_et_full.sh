#!/usr/bin/env bash
# Launch the complete scalar reference graph without leaderboard registration.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"

device="sys_emu"
full_dir="$repo_root/local-artifacts/yolov10n_hf_reference/full_graph/deterministic"
elf=""
launcher="${LAUNCHER:-}"
output_dir=""
outer_timeout=""
launcher_timeout=""
lock_timeout=600
shire=0

usage() {
  cat <<'EOF'
Usage: run_et_full.sh --elf FILE --launcher FILE --output-dir DIR [options]

Options:
  --full-dir DIR           Schema-v2 full package. Default: deterministic.
  --device NAME            sys_emu or soc1sim (real PCIe). Default: sys_emu.
  --outer-timeout SEC      Bounded outer timeout. Defaults: 43200/2400.
  --launcher-timeout SEC   Bounded launcher timeout. Defaults: 43140/2340.
  --lock-timeout SEC       Board-lock wait for soc1sim. Default: 600.
  --shire INDEX            Shire index. Default: 0.

This is a correctness-evidence runner. It never invokes the model leaderboard.
EOF
}

need_value() {
  [[ -n "${2:-}" ]] || {
    echo "error: $1 requires a value" >&2
    exit 2
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) need_value "$1" "${2:-}"; device="$2"; shift 2 ;;
    --full-dir) need_value "$1" "${2:-}"; full_dir="$2"; shift 2 ;;
    --elf) need_value "$1" "${2:-}"; elf="$2"; shift 2 ;;
    --launcher) need_value "$1" "${2:-}"; launcher="$2"; shift 2 ;;
    --output-dir) need_value "$1" "${2:-}"; output_dir="$2"; shift 2 ;;
    --outer-timeout) need_value "$1" "${2:-}"; outer_timeout="$2"; shift 2 ;;
    --launcher-timeout) need_value "$1" "${2:-}"; launcher_timeout="$2"; shift 2 ;;
    --lock-timeout) need_value "$1" "${2:-}"; lock_timeout="$2"; shift 2 ;;
    --shire) need_value "$1" "${2:-}"; shire="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$device" == "sys_emu" || "$device" == "soc1sim" ]] || {
  echo "error: --device must be sys_emu or soc1sim" >&2
  exit 2
}
if [[ -z "$outer_timeout" ]]; then
  [[ "$device" == "sys_emu" ]] && outer_timeout=43200 || outer_timeout=2400
fi
if [[ -z "$launcher_timeout" ]]; then
  [[ "$device" == "sys_emu" ]] && launcher_timeout=43140 || launcher_timeout=2340
fi
for value in "$outer_timeout" "$launcher_timeout" "$lock_timeout" "$shire"; do
  case "$value" in
    ''|*[!0-9]*)
      echo "error: timeout/shire values must be integers" >&2
      exit 2
      ;;
  esac
done
[[ "$outer_timeout" -gt 0 && "$launcher_timeout" -gt 0 \
   && "$lock_timeout" -gt 0 ]] || {
  echo "error: timeout values must be greater than zero" >&2
  exit 2
}
[[ "$launcher_timeout" -lt "$outer_timeout" ]] || {
  echo "error: launcher timeout must be less than outer timeout" >&2
  exit 2
}
[[ -n "$elf" && -n "$launcher" && -n "$output_dir" ]] || {
  usage >&2
  exit 2
}

"$port_root/scripts/verify_full_package.sh" "$full_dir"
"$port_root/scripts/verify_full_elf.sh" "$full_dir" "$elf"
"$port_root/scripts/run_et_slice.sh" \
  --device "$device" \
  --slice-dir "$full_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$output_dir" \
  --outer-timeout "$outer_timeout" \
  --launcher-timeout "$launcher_timeout" \
  --lock-timeout "$lock_timeout" \
  --shire "$shire"

echo "ET_FULL_LAUNCH PASS selector=N000:N307 device=$device output=$output_dir correctness=unvalidated next=validate_et_full.sh leaderboard=not_registered"
