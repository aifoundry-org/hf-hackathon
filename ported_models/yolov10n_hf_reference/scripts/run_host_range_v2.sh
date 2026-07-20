#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
range_dir="${1:?usage: run_host_range_v2.sh RANGE_DIR [DUMP] [REPORT]}"
python="${YOLOV10N_HOST_PYTHON:-$repo_root/local-artifacts/yolov10n_hf_reference/venv/bin/python}"
model="${YOLOV10N_MODEL:-$repo_root/local-artifacts/yolov10n_hf_reference/model.onnx}"
dump="${2:-$range_dir/host_range_v2_dump.bin}"
report="${3:-$range_dir/host_range_v2_compare.json}"

"$port_root/scripts/build_host_range_v2.sh" "$range_dir"
runner_status=0
"$range_dir/host_range_v2_runner" \
  "$range_dir/inputs.bin" \
  "$range_dir/weights.bin" \
  "$dump" || runner_status=$?
compare_status=0
"$python" "$port_root/tools/compare_range_v2.py" \
  "$range_dir" "$dump" \
  --model "$model" \
  --json "$report" || compare_status=$?

if (( runner_status != 0 )); then
  echo "HOST_RANGE_V2_RUN FAIL runner_status=$runner_status report=$report" >&2
  exit "$runner_status"
fi
if (( compare_status != 0 )); then
  echo "HOST_RANGE_V2_RUN FAIL compare_status=$compare_status report=$report" >&2
  exit "$compare_status"
fi
echo "HOST_RANGE_V2_RUN PASS report=$report"
