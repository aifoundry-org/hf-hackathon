#!/usr/bin/env bash
set -euo pipefail

ref="${1:-HEAD}"
paths=(
  .github/ci/reference/model_ports_track.json
  .github/ci/scripts/model_port_claim.py
  .github/ci/scripts/model_port_credit.py
  .github/ci/scripts/prepare_trusted_model_port_tree.py
  .github/ci/scripts/render_model_port_standings.py
  .github/ci/scripts/leaderboard_gate.py
  .github/ci/scripts/benchmark_config_helpers.py
  .github/ci/scripts/install_toolchain.sh
  .github/ci/scripts/run_model_benchmark.sh
  .github/ci/scripts/score_results.py
  .github/ci/launcher/CMakeLists.txt
  .github/ci/launcher/erbium_soc1sim_argbuf_dynmem.cpp
  .github/ci/platform/deploy/soc3-benchmark.sh
  .github/workflows/trusted-model-port-pr.yml
  .github/ci/benchmark_config.json
  data/model-port-identities.json
  data/model-port-credits.json
)

for path in "${paths[@]}"; do
  printf '%s\0' "$path"
  git show "$ref:$path" 2>/dev/null || true
  printf '\0'
done | sha256sum | awk '{print $1}'
