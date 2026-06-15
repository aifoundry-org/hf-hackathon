#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

model="${1:-all}"

while IFS= read -r bench_dir; do
  [[ -n "$bench_dir" ]] && mkdir -p "${AMP_ROOT}/${bench_dir}"
done < <(
  python3 - "$BENCHMARK_CONFIG" "$REPO_ROOT" <<'PY'
import sys

cfg_path, repo = sys.argv[1:3]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config

cfg = load_config(cfg_path)
for model in cfg.get("models", {}).values():
    if model.get("bench_dir"):
        print(model["bench_dir"])
PY
)
mkdir -p "${AMP_ROOT}/dncnn3-bench"

# 2 MiB zero blob used by all smoke kernels.
if [[ ! -f "${AMP_ROOT}/dncnn3-bench/zero2m.bin" ]]; then
  python3 -c "open('${AMP_ROOT}/dncnn3-bench/zero2m.bin','wb').write(b'\\0' * (2 * 1024 * 1024))"
fi
ln -sf dncnn3-bench/zero2m.bin "${AMP_ROOT}/zero2m.bin"

copy_if_exists() {
  local src="$1" dst="$2"
  if [[ -f "$src" && ! -e "$dst" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}

configured_asset_paths() {
  python3 - "$BENCHMARK_CONFIG" "$REPO_ROOT" <<'PY'
import sys

cfg_path, repo = sys.argv[1:3]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config

cfg = load_config(cfg_path)
seen = set()
for model in cfg.get("models", {}).values():
    for load in model.get("file_loads", []):
        paths = load.get("paths") or [load.get("path")]
        for path in paths:
            if path and path not in seen:
                seen.add(path)
                print(path)
PY
}

copy_configured_assets() {
  local src_root="$1"
  local rel
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    copy_if_exists "${src_root}/${rel}" "${AMP_ROOT}/${rel}"
  done < <(configured_asset_paths)
}

# Optional local bundle (e.g. zephyr local-artifacts path).
if [[ -n "${BENCHMARK_ASSETS_DIR:-}" && -d "${BENCHMARK_ASSETS_DIR}" ]]; then
  copy_configured_assets "${BENCHMARK_ASSETS_DIR}"
fi

# Optional tarball URL (set in GitHub Actions secrets). Uses GITHUB_TOKEN for private repos.
if [[ -n "${BENCHMARK_ASSETS_URL:-}" ]]; then
  tmp="$(mktemp -d)"
  auth_args=()
  if [[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]]; then
    auth_args=(
      -H "Authorization: Bearer ${GITHUB_TOKEN:-${GH_TOKEN}}"
      -H "Accept: application/octet-stream"
    )
  fi
  if curl -fsSL "${auth_args[@]}" -L "${BENCHMARK_ASSETS_URL}" -o "${tmp}/assets.tar.gz" \
    && tar -xzf "${tmp}/assets.tar.gz" -C "${tmp}"; then
    copy_configured_assets "${tmp}"
  else
    echo "warn: failed to download/extract BENCHMARK_ASSETS_URL" >&2
  fi
  rm -rf "${tmp}"
fi

export DNCNN_INPUTS_READY=0
if [[ (-f "${AMP_ROOT}/dncnn3_input.bin" || -f "${AMP_ROOT}/dncnn3-bench/dncnn3_input.bin") && \
      (-f "${AMP_ROOT}/dncnn3_weights.bin" || -f "${AMP_ROOT}/dncnn3-bench/dncnn3_weights.bin") ]]; then
  export DNCNN_INPUTS_READY=1
fi

if [[ "$model" == "dncnn" || "$model" == "all" ]]; then
  if [[ "$DNCNN_INPUTS_READY" -eq 0 ]]; then
    echo "warn: dncnn3_input.bin / dncnn3_weights.bin missing; dncnn benchmark will be skipped." >&2
    echo "Set BENCHMARK_ASSETS_DIR or BENCHMARK_ASSETS_URL to enable dncnn CI." >&2
  fi
fi

echo "${DNCNN_INPUTS_READY}" > "${BENCHMARK_OUTPUT}/.dncnn_inputs_ready"
echo "zero2m.bin ready under ${AMP_ROOT}/dncnn3-bench"
echo "DNCNN_INPUTS_READY=${DNCNN_INPUTS_READY}"
