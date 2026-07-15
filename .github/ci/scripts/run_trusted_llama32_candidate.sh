#!/usr/bin/env bash
set -euo pipefail

base="${1:?usage: run_trusted_llama32_candidate.sh BASE HEAD OUTPUT_DIR}"
head="${2:?usage: run_trusted_llama32_candidate.sh BASE HEAD OUTPUT_DIR}"
output_dir="${3:?usage: run_trusted_llama32_candidate.sh BASE HEAD OUTPUT_DIR}"

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
contract="$repo_root/.github/ci/reference/llama32_1b.json"
track_policy="$repo_root/.github/ci/reference/llama32_1b_track.json"
config="$repo_root/.github/ci/benchmark_config.json"
runtime_path="ported_models/llama_cpp_et/src/llama.cpp-et"
work="$(mktemp -d)"
original_config="$work/benchmark_config.json"
metadata="$work/candidate-metadata.json"
baseline_config="$work/baseline-config.json"
candidate_config="$work/candidate-config.json"
main_runtime_sha="$(git -C "$repo_root" ls-tree HEAD "$runtime_path" | awk '{print $3}')"
dest="${SOC3_DEST:-/root/et-jobs-deploy}"

mkdir -p "$output_dir/baseline" "$output_dir/candidate"
cp "$config" "$original_config"

restore() {
  cp "$original_config" "$config"
  if [[ -d "$repo_root/$runtime_path/.git" || -f "$repo_root/$runtime_path/.git" ]]; then
    git -C "$repo_root/$runtime_path" checkout --detach "$main_runtime_sha" >/dev/null 2>&1 || true
  fi
  rm -rf "$work"
}
trap restore EXIT

python3 "$repo_root/.github/ci/scripts/prepare_trusted_llama32_candidate.py" \
  --base "$base" \
  --head "$head" \
  --baseline-config "$baseline_config" \
  --output-config "$candidate_config" \
  --metadata "$metadata"
cp "$metadata" "$output_dir/candidate-metadata.json"

if [[ "$(python3 -c 'import json,sys; print(1 if json.load(open(sys.argv[1]))["targeted"] else 0)' "$metadata")" != "1" ]]; then
  echo "PR does not change the trusted Llama 3.2 1B implementation or candidate manifest." >&2
  exit 3
fi

source "$repo_root/.github/ci/platform/deploy/soc3-ssh.sh"
pull_outputs() {
  local target="$1"
  mkdir -p "$target"
  if soc3_is_local; then
    cp -a "$dest/benchmark-output/." "$target/"
  else
    local rsh host
    rsh="$(soc3_rsync_rsh)"
    host="$(soc3_rsync_host)"
    rsync -az -e "$rsh" "$host:$dest/benchmark-output/" "$target/"
  fi
}

echo "==> Trusted Llama baseline: $main_runtime_sha"
cp "$baseline_config" "$config"
MODELS=llama32_1b \
LLAMA_CPP_ET_SOURCE_REVISION="$main_runtime_sha" \
TRUSTED_LLAMA_BUILD_KEY="$main_runtime_sha" \
SOC3_FAIL_ON_MODEL_FAILURE=0 \
  "$repo_root/.github/ci/platform/deploy/soc3-benchmark.sh"
pull_outputs "$output_dir/baseline"

if [[ "$(jq -r '.passed // false' "$output_dir/baseline/score-llama32_1b.json")" != "true" ]]; then
  cat > "$output_dir/verdict.md" <<'EOF'
## Trusted Llama 3.2 1B Gate

Result: infrastructure error. The current-main paired baseline did not pass, so
the candidate was not evaluated and no merge verdict was produced.
EOF
  cat "$output_dir/verdict.md" >&2
  exit 2
fi

candidate_runtime_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_revision"])' "$metadata")"
candidate_runtime_url="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_url"])' "$metadata")"
regression_models="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["regression_models"]))' "$metadata")"
evaluation_mode="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mode"])' "$metadata")"

echo "==> Trusted Llama candidate runtime: $candidate_runtime_url@$candidate_runtime_sha"
git -C "$repo_root/$runtime_path" remote remove trusted-candidate >/dev/null 2>&1 || true
git -C "$repo_root/$runtime_path" remote add trusted-candidate "$candidate_runtime_url"
git -C "$repo_root/$runtime_path" fetch --no-tags --depth 1 trusted-candidate "$candidate_runtime_sha"
git -C "$repo_root/$runtime_path" checkout --detach "$candidate_runtime_sha"
cp "$candidate_config" "$config"

baseline_cpu_ppl="$dest/local-artifacts/frameworks/llama.cpp-et/build-$main_runtime_sha/bin/llama-perplexity"
models="llama32_1b${regression_models:+ $regression_models}"
MODELS="$models" \
LLAMA_CPP_ET_SOURCE_REVISION="$candidate_runtime_sha" \
TRUSTED_LLAMA_BUILD_KEY="$candidate_runtime_sha" \
TRUSTED_LLAMA_CPU_PPL_BIN="$baseline_cpu_ppl" \
SOC3_FAIL_ON_MODEL_FAILURE=0 \
  "$repo_root/.github/ci/platform/deploy/soc3-benchmark.sh"
pull_outputs "$output_dir/candidate"

python3 "$repo_root/.github/ci/scripts/trusted_llama32_gate.py" \
  --contract "$contract" \
  --track-policy "$track_policy" \
  --mode "$evaluation_mode" \
  --baseline-score "$output_dir/baseline/score-llama32_1b.json" \
  --candidate-scores-dir "$output_dir/candidate" \
  --regression-models "$regression_models" \
  --participant "${LEADERBOARD_TEAM:?LEADERBOARD_TEAM is required}" \
  --head-sha "$head" \
  --candidate-metadata "$output_dir/candidate-metadata.json" \
  --output "$output_dir/verdict.md" \
  --result-output "$output_dir/track-result.json"
