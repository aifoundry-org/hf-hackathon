#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

read -r SYSEMU_TIMEOUT SYSEMU_LAUNCHER_TIMEOUT BENCHMARK_DEVICE < <(
  python3 -c "
import os, sys
sys.path.insert(0, '${REPO_ROOT}/.github/ci/scripts')
from benchmark_config_helpers import load_config, sysemu_timeouts, benchmark_device
cfg = load_config('${BENCHMARK_CONFIG}')
outer, launcher = sysemu_timeouts(cfg)
print(outer, launcher, benchmark_device(cfg))
"
)
export SYSEMU_TIMEOUT SYSEMU_LAUNCHER_TIMEOUT BENCHMARK_DEVICE

model="${1:?model required}"

python3 - "$model" "$BENCHMARK_CONFIG" "$REPO_ROOT" <<'PY'
import sys

model, cfg_path, repo = sys.argv[1:4]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import board_mode, load_config, parse_model_selection

try:
    target = "board" if board_mode() else "sysemu"
    parse_model_selection(model, load_config(cfg_path), target=target)
except ValueError as exc:
    raise SystemExit(f"error: {exc}")
PY

write_score() {
  local status="$1" note="$2"
  local run_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-local}/actions/runs/${GITHUB_RUN_ID:-0}"
  python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
    --model "$model" \
    --output "${BENCHMARK_OUTPUT}/score-${model}.json" \
    --sha "${GITHUB_SHA:-local}" \
    --ref "${GITHUB_REF:-local}" \
    --actor "${GITHUB_ACTOR:-local}" \
    --run-url "$run_url" \
    --status "$status" \
    --note "$note"
}

runner="$(python3 - "$model" "$BENCHMARK_CONFIG" "$REPO_ROOT" <<'PY'
import sys

model, cfg_path, repo = sys.argv[1:4]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config, model_runner

print(model_runner(load_config(cfg_path), model))
PY
)"

if [[ "$runner" == "llama_server" ]]; then
  if [[ "$BENCHMARK_DEVICE" != "soc1sim" ]]; then
    write_score skipped "${model} framework benchmark requires the ET-SoC1 board runner."
    exit 0
  fi

  run_dir="${BENCHMARK_OUTPUT}/llama-${model}"
  score_out="${BENCHMARK_OUTPUT}/score-${model}.json"
  python3 "${REPO_ROOT}/.github/ci/scripts/run_llama_server_benchmark.py" \
    --model "$model" \
    --results-dir "$run_dir" \
    --output "$score_out"
  exit 0
fi

if [[ "$runner" != "elf" ]]; then
  write_score fail "unknown benchmark runner '${runner}' for ${model}"
  exit 0
fi

"$(dirname "$0")/prepare_benchmark_inputs.sh" "$model"
DNCNN_INPUTS_READY="$(cat "${BENCHMARK_OUTPUT}/.dncnn_inputs_ready" 2>/dev/null || echo 0)"

requires_dncnn="$(python3 -c "import json; c=json.load(open('$BENCHMARK_CONFIG')); print(int(c['models']['$model'].get('requires_dncnn_inputs', False)))")"
if [[ "$model" == "dncnn" && "$requires_dncnn" -eq 1 && "$DNCNN_INPUTS_READY" -eq 0 ]]; then
  python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
    --model "$model" \
    --status skipped \
    --note "DnCNN input blobs missing. Set BENCHMARK_ASSETS_URL or BENCHMARK_ASSETS_DIR." \
    --output "${BENCHMARK_OUTPUT}/score-${model}.json"
  exit 0
fi

bench_dir="$(python3 -c "import json; print(json.load(open('$BENCHMARK_CONFIG'))['models']['$model']['bench_dir'])")"
variant="$(python3 -c "import json; print(json.load(open('$BENCHMARK_CONFIG'))['models']['$model']['canonical_variant'])")"
prebuilt_elf="${AMP_ROOT}/${bench_dir}/${variant}.elf"
prebuilt_available=0
if [[ -f "${prebuilt_elf}" && ("${SOC3_PREBUILT:-}" == "1" || "${SOC3_REUSE_ELF:-}" == "1") ]]; then
  prebuilt_available=1
fi

if [[ "$BENCHMARK_DEVICE" == "soc1sim" ]]; then
  echo "Board benchmark: device=soc1sim (real PCIe)"
  export ET_INSTALL="${ET_INSTALL:-/opt/et}"
  export ET_PLATFORM="${ET_PLATFORM:-${ET_INSTALL}}"
  if [[ -z "${ET_PLATFORM_SRC:-}" ]]; then
    for candidate in "${HOME}/et-platform" "${HOME}/et" "/root/et-platform"; do
      if [[ -f "${candidate}/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" ]]; then
        export ET_PLATFORM_SRC="${candidate}"
        break
      fi
    done
  fi
  platform_src_ready=0
  [[ -n "${ET_PLATFORM_SRC:-}" && -f "${ET_PLATFORM_SRC}/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" ]] && platform_src_ready=1
  if [[ -n "${LAUNCHER:-}" && -x "${LAUNCHER}" ]]; then
    echo "Using LAUNCHER=${LAUNCHER}"
  elif [[ -x "${ET_INSTALL}/bin/erbium_soc1sim_argbuf_dynmem" ]]; then
    export LAUNCHER="${ET_INSTALL}/bin/erbium_soc1sim_argbuf_dynmem"
  else
    if [[ "$platform_src_ready" -ne 1 ]]; then
      write_score fail "ET_PLATFORM_SRC missing on board host (need et-platform tree for launcher build)"
      exit 0
    fi
    if ! "$(dirname "$0")/build_launcher.sh"; then
      write_score fail "erbium_soc1sim_argbuf launcher build failed on board host"
      exit 0
    fi
  fi
  if [[ "$prebuilt_available" -ne 1 && "$platform_src_ready" -ne 1 ]]; then
    write_score fail "ET_PLATFORM_SRC missing on board host (need et-platform tree for ELF build)"
    exit 0
  fi
else
  if ! "$(dirname "$0")/install_toolchain.sh"; then
    write_score fail "RISC-V toolchain download failed (aifoundry-org/riscv-gnu-toolchain release)"
    exit 0
  fi
  if ! "$(dirname "$0")/install_et_sdk.sh"; then
    write_score fail "et-platform sys-emu SDK build failed (see CI log)"
    exit 0
  fi
  if ! "$(dirname "$0")/build_launcher.sh"; then
    write_score fail "erbium_soc1sim_argbuf launcher build failed"
    exit 0
  fi
fi

if [[ -f "${prebuilt_elf}" && "${SOC3_PREBUILT:-}" == "1" ]]; then
  echo "Using prebuilt ELF: ${prebuilt_elf}"
elif [[ -f "${prebuilt_elf}" && "${SOC3_REUSE_ELF:-}" == "1" ]]; then
  echo "Using existing ELF on board: ${prebuilt_elf}"
else
  # RISC-V gcc for ELF build (/opt/et on board hosts, or downloaded toolchain).
  if ! "$(dirname "$0")/install_toolchain.sh"; then
    write_score fail "RISC-V toolchain not available (ET_INSTALL or download)"
    exit 0
  fi
  if ! "$(dirname "$0")/build_leaderboard_elf.sh" "$model"; then
    write_score fail "ELF build failed for ${model}"
    exit 0
  fi
fi

run_dir="${BENCHMARK_OUTPUT}/sysemu-${model}"
mkdir -p "$run_dir"

score_out="${BENCHMARK_OUTPUT}/score-${model}.json"
run_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-local}/actions/runs/${GITHUB_RUN_ID:-0}"
score_common=(
  --model "$model"
  --output "$score_out"
  --sha "${GITHUB_SHA:-local}"
  --ref "${GITHUB_REF:-local}"
  --actor "${GITHUB_ACTOR:-local}"
  --run-url "$run_url"
)

cd "${REPO_ROOT}"
runner_log="$run_dir/runner.log"
set +e
bash scripts/run_sysemu_model_ports.sh \
  --launcher "${LAUNCHER}" \
  --suite smoke \
  --model "$model" \
  --variant "$variant" \
  --output-dir "$run_dir" \
  --timeout "${SYSEMU_TIMEOUT}" \
  --launcher-timeout "${SYSEMU_LAUNCHER_TIMEOUT}" \
  --device "${BENCHMARK_DEVICE}" \
  --no-keep-going >"$runner_log" 2>&1
runner_rc=$?
set -e
cat "$runner_log"

if [[ "$runner_rc" -ne 0 ]]; then
  if [[ ! -f "$run_dir/results.tsv" ]]; then
    write_score fail "benchmark runner exited rc=${runner_rc} before results.tsv; see sysemu-${model}/runner.log"
    exit 0
  fi
  python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
    "${score_common[@]}" \
    --results-dir "$run_dir" 2>/dev/null || \
  python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
    "${score_common[@]}" \
    --status fail \
    --note "benchmark runner exited rc=${runner_rc}; see sysemu-${model}/runner.log"
  exit 0
fi

if ! python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
  "${score_common[@]}" \
  --results-dir "$run_dir"; then
  if [[ ! -f "$score_out" ]]; then
    python3 "${REPO_ROOT}/.github/ci/scripts/score_results.py" \
      "${score_common[@]}" \
      --status fail \
      --note "scoring failed"
  fi
fi
