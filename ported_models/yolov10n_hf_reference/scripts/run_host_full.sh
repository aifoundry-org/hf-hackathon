#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
full_dir="${1:-$repo_root/local-artifacts/yolov10n_hf_reference/full_graph/deterministic}"
python="${YOLOV10N_HOST_PYTHON:-$repo_root/local-artifacts/yolov10n_hf_reference/venv/bin/python}"
model="${YOLOV10N_MODEL:-$repo_root/local-artifacts/yolov10n_hf_reference/model.onnx}"
dump="$full_dir/host_full_dump.bin"
report="$full_dir/host_full_compare.json"

"$port_root/scripts/build_host_full.sh" "$full_dir"
runner_status=0
"$full_dir/host_full_runner" \
  "$full_dir/inputs.bin" \
  "$full_dir/weights.bin" \
  "$dump" || runner_status=$?
compare_status=0
"$python" "$port_root/tools/compare_full.py" \
  "$full_dir" "$dump" \
  --model "$model" \
  --json "$report" || compare_status=$?

if (( runner_status != 0 )); then
  echo "HOST_FULL_RUN FAIL runner_status=$runner_status report=$report" >&2
  exit "$runner_status"
fi
if (( compare_status != 0 )); then
  echo "HOST_FULL_RUN FAIL compare_status=$compare_status report=$report" >&2
  exit "$compare_status"
fi
echo "HOST_FULL_RUN PASS report=$report"
