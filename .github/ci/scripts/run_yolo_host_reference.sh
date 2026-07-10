#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

output="${1:-${BENCHMARK_OUTPUT}/yolo-host-reference.json}"
contract="${REPO_ROOT}/.github/ci/reference/yolo.json"
cache="${WORK_ROOT}/yolo-host-reference"
checkpoint="${cache}/yolov10n.pt"
venv="${cache}/venv"
mkdir -p "$cache" "$(dirname "$output")"

read -r checkpoint_url checkpoint_sha ultralytics_version < <(
  python3 - "$contract" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1]))
source = contract["model"]["source"]
runtime = contract["model"]["host_runtime"]
print(source["url"], source["sha256"], runtime["ultralytics"])
PY
)

if [[ ! -f "$checkpoint" ]] || ! printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum -c - >/dev/null 2>&1; then
  tmp_checkpoint="${checkpoint}.tmp"
  rm -f "$tmp_checkpoint"
  echo "Downloading pinned YOLO host reference from Hugging Face..."
  curl --fail --location --retry 3 --retry-delay 2 "$checkpoint_url" -o "$tmp_checkpoint"
  printf '%s  %s\n' "$checkpoint_sha" "$tmp_checkpoint" | sha256sum -c -
  mv "$tmp_checkpoint" "$checkpoint"
fi

reference_python="python3"
if ! python3 - "$ultralytics_version" <<'PY' >/dev/null 2>&1
import sys
import torch
import ultralytics

raise SystemExit(0 if ultralytics.__version__ == sys.argv[1] else 1)
PY
then
  reference_python="${venv}/bin/python"
  if ! "$reference_python" - "$ultralytics_version" <<'PY' >/dev/null 2>&1
import sys
import torch
import ultralytics

raise SystemExit(0 if ultralytics.__version__ == sys.argv[1] else 1)
PY
  then
    rm -rf "$venv"
    python3 -m venv --system-site-packages "$venv"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$reference_python" -m pip install \
      "ultralytics==${ultralytics_version}"
  fi
fi

export YOLO_CONFIG_DIR="${cache}/ultralytics-config"
mkdir -p "$YOLO_CONFIG_DIR"
"$reference_python" "${REPO_ROOT}/ported_models/yolo/tools/host_reference.py" \
  --contract "$contract" \
  --checkpoint "$checkpoint" \
  --output "$output"
