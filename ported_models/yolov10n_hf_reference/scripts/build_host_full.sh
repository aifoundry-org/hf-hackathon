#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
full_dir="${1:-$repo_root/local-artifacts/yolov10n_hf_reference/full_graph/deterministic}"
output="${2:-$full_dir/host_full_runner}"
cc="${CC:-cc}"

test -f "$full_dir/slice_manifest.h" || {
  echo "error: run tools/generate_full_graph.py first; missing $full_dir/slice_manifest.h" >&2
  exit 2
}
test -f "$full_dir/slice_manifest.json" || {
  echo "error: run tools/generate_full_graph.py first; missing $full_dir/slice_manifest.json" >&2
  exit 2
}

"$cc" \
  -std=c11 -O1 -fno-fast-math -ffp-contract=off \
  -Wall -Wextra -Werror \
  -I"$port_root/src" -I"$full_dir" \
  "$port_root/src/ref_runtime.c" \
  "$port_root/src/host_full_runner.c" \
  -o "$output"

echo "HOST_FULL_BUILD PASS output=$output"
