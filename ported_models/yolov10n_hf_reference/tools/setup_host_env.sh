#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
venv="${YOLOV10N_HOST_VENV:-$repo_root/local-artifacts/yolov10n_hf_reference/venv}"

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --disable-pip-version-check \
  -r "$port_root/requirements-host.txt"
"$venv/bin/python" - <<'PY'
import numpy
import onnx
import onnxruntime

print(
    "HOST_ENV PASS "
    f"numpy={numpy.__version__} "
    f"onnx={onnx.__version__} "
    f"onnxruntime={onnxruntime.__version__}"
)
PY
