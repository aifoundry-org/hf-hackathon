#!/usr/bin/env bash
set -euo pipefail

ref="${1:?usage: trusted_yolo_input_hash.sh GIT_REF}"

git ls-tree -r "$ref" -- \
  .github/ci/benchmark_config.json \
  .github/ci/reference/yolo.json \
  .github/ci/scripts \
  .github/ci/platform/deploy \
  .github/workflows/benchmark-board.yml \
  scripts/run_sysemu_model_ports.sh \
  ported_models/yolo/src \
  ported_models/yolo/assets/yolo \
  ported_models/yolo/manifests \
  data/yolo.json \
  | sha256sum \
  | awk '{print $1}'
