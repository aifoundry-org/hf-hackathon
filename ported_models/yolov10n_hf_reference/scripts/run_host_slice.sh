#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
slice="${1:-n263_n265_dw_silu}"
slice_dir="${2:-$repo_root/local-artifacts/yolov10n_hf_reference/slices/$slice}"
python="${YOLOV10N_HOST_PYTHON:-$repo_root/local-artifacts/yolov10n_hf_reference/venv/bin/python}"
dump="$slice_dir/host_dump.bin"
report="$slice_dir/host_compare.json"

"$port_root/scripts/build_host_slice.sh" "$slice_dir"
"$slice_dir/host_slice_runner" \
  "$slice_dir/inputs.bin" \
  "$slice_dir/weights.bin" \
  "$dump"
"$python" "$port_root/tools/compare_slice.py" \
  "$slice_dir" "$dump" --json "$report"
