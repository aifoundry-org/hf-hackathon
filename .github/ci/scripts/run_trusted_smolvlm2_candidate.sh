#!/usr/bin/env bash
set -euo pipefail

base="${1:?usage: run_trusted_smolvlm2_candidate.sh BASE HEAD OUTPUT_DIR}"
head="${2:?usage: run_trusted_smolvlm2_candidate.sh BASE HEAD OUTPUT_DIR}"
output_dir="${3:?usage: run_trusted_smolvlm2_candidate.sh BASE HEAD OUTPUT_DIR}"

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
contract="$repo_root/.github/ci/reference/smolvlm2_500m_video.json"
config="$repo_root/.github/ci/benchmark_config.json"
runtime_path="ported_models/llama_cpp_et/src/llama.cpp-et"
work="$(mktemp -d)"
original_config="$work/benchmark_config.json"
baseline_config="$work/baseline-config.json"
candidate_config="$work/candidate-config.json"
metadata="$work/candidate-metadata.json"
main_runtime_sha="$(git -C "$repo_root" ls-tree HEAD "$runtime_path" | awk '{print $3}')"
candidate_build_key="$(printf '%s' trusted-smolvlm2-candidate-build | sha1sum | awk '{print $1}')"
dest="${SOC3_DEST:-/root/et-jobs-deploy}"

mkdir -p "$output_dir/baseline-before" "$output_dir/candidate" "$output_dir/baseline-after"
cp "$config" "$original_config"

restore() {
  cp "$original_config" "$config"
  if [[ -e "$repo_root/$runtime_path/.git" ]]; then
    git -C "$repo_root/$runtime_path" checkout --detach "$main_runtime_sha" >/dev/null 2>&1 || true
  fi
  rm -rf "$work"
}
trap restore EXIT

python3 "$repo_root/.github/ci/scripts/prepare_trusted_smolvlm2_candidate.py" \
  --base "$base" \
  --head "$head" \
  --baseline-config "$baseline_config" \
  --output-config "$candidate_config" \
  --metadata "$metadata"
cp "$metadata" "$output_dir/candidate-metadata.json"

if [[ "$(jq -r 'if .targeted then 1 else 0 end' "$metadata")" != "1" ]]; then
  echo "PR does not change the SmolVLM2 runtime." >&2
  exit 3
fi

source "$repo_root/.github/ci/platform/deploy/soc3-ssh.sh"
pull_outputs() {
  local target="$1"
  mkdir -p "$target"
  if soc3_is_local; then
    cp -a "$dest/benchmark-output/." "$target/"
  else
    rsync -az -e "$(soc3_rsync_rsh)" \
      "$(soc3_rsync_host):$dest/benchmark-output/" "$target/"
  fi
}

run_board() {
  local run_config="$1" runtime_sha="$2" build_key="$3" target="$4" skip_ppl="$5" skip_host="$6"
  cp "$run_config" "$config"
  env -u SMOLVLM2_500M_VIDEO_MODEL_PATH \
    -u SMOLVLM2_500M_VIDEO_MMPROJ_PATH \
    MODELS=smolvlm2_500m_video \
    LLAMA_CPP_ET_SOURCE_REVISION="$runtime_sha" \
    TRUSTED_LLAMA_BUILD_KEY="$build_key" \
    TRUSTED_LLAMA_REUSE_BUILD="$([[ "$build_key" == "$candidate_build_key" ]] && echo 1 || echo 0)" \
    TRUSTED_SMOLVLM2_CPU_BUILD_KEY="$main_runtime_sha" \
    SOC3_SKIP_BOARD_SMOKE=1 \
    SMOLVLM2_SKIP_PPL="$skip_ppl" \
    SMOLVLM2_SKIP_HOST_REFERENCE="$skip_host" \
    SOC3_FAIL_ON_MODEL_FAILURE=0 \
    "$repo_root/.github/ci/platform/deploy/soc3-benchmark.sh"
  pull_outputs "$target"
}

echo "==> Trusted SmolVLM2 main before: $main_runtime_sha"
run_board "$baseline_config" "$main_runtime_sha" "$main_runtime_sha" "$output_dir/baseline-before" 1 1

candidate_runtime_sha="$(jq -r .runtime_revision "$metadata")"
candidate_runtime_url="$(jq -r .runtime_url "$metadata")"
echo "==> Trusted SmolVLM2 candidate: $candidate_runtime_url@$candidate_runtime_sha"
git -C "$repo_root/$runtime_path" remote remove trusted-candidate >/dev/null 2>&1 || true
git -C "$repo_root/$runtime_path" remote add trusted-candidate "$candidate_runtime_url"
git -C "$repo_root/$runtime_path" fetch --no-tags --depth 1 trusted-candidate "$candidate_runtime_sha"
if ! python3 "$repo_root/.github/ci/scripts/trusted_smolvlm2_runtime_scope.py" \
  --repo "$repo_root/$runtime_path" \
  --base "$main_runtime_sha" \
  --head "$candidate_runtime_sha" \
  --output "$output_dir/runtime-changes.json"; then
  echo "error: candidate runtime changes paths outside the tracked SmolVLM2 implementation surface" >&2
  exit 1
fi
git -C "$repo_root/$runtime_path" checkout --detach "$candidate_runtime_sha"
run_board "$candidate_config" "$candidate_runtime_sha" "$candidate_build_key" "$output_dir/candidate" 0 1

echo "==> Trusted SmolVLM2 main after: $main_runtime_sha"
git -C "$repo_root/$runtime_path" checkout --detach "$main_runtime_sha"
run_board "$baseline_config" "$main_runtime_sha" "$main_runtime_sha" "$output_dir/baseline-after" 1 1

python3 "$repo_root/.github/ci/scripts/trusted_smolvlm2_gate.py" \
  --contract "$contract" \
  --baseline-before "$output_dir/baseline-before/score-smolvlm2_500m_video.json" \
  --candidate "$output_dir/candidate/score-smolvlm2_500m_video.json" \
  --baseline-after "$output_dir/baseline-after/score-smolvlm2_500m_video.json" \
  --output "$output_dir/verdict.md"
