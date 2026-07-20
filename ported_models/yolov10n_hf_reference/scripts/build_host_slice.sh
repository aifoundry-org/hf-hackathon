#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slice_dir="${1:?usage: build_host_slice.sh SLICE_DIR [OUTPUT]}"
output="${2:-$slice_dir/host_slice_runner}"
cc="${CC:-cc}"

test -f "$slice_dir/slice_manifest.h" || {
  echo "error: run tools/capture_slice.py first; missing $slice_dir/slice_manifest.h" >&2
  exit 2
}

"$cc" \
  -std=c11 -O1 -fno-fast-math -ffp-contract=off \
  -Wall -Wextra -Werror \
  -I"$port_root/src" -I"$slice_dir" \
  "$port_root/src/ref_runtime.c" \
  "$port_root/src/host_slice_runner.c" \
  -o "$output"

echo "HOST_BUILD PASS output=$output"
