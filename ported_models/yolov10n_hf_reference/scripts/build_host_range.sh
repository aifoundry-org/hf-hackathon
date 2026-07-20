#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
range_dir="${1:?usage: build_host_range.sh RANGE_DIR [OUTPUT]}"
output="${2:-$range_dir/host_range_runner}"
cc="${CC:-cc}"

test -f "$range_dir/slice_manifest.h" || {
  echo "error: run tools/capture_range.py first; missing $range_dir/slice_manifest.h" >&2
  exit 2
}
test -f "$range_dir/slice_manifest.json" || {
  echo "error: run tools/capture_range.py first; missing $range_dir/slice_manifest.json" >&2
  exit 2
}

"$cc" \
  -std=c11 -O1 -fno-fast-math -ffp-contract=off \
  -Wall -Wextra -Werror \
  -I"$port_root/src" -I"$range_dir" \
  "$port_root/src/ref_runtime.c" \
  "$port_root/src/host_range_runner.c" \
  -o "$output"

echo "HOST_RANGE_BUILD PASS output=$output"
