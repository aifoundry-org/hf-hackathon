#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

if [[ -z "${ET_PLATFORM_SRC:-}" || ! -d "${ET_PLATFORM_SRC}" ]]; then
  echo "Set ET_PLATFORM_SRC to an et-platform tree with erbium-soc1sim support." >&2
  echo "Example: export ET_PLATFORM_SRC=\$HOME/et-platform-vidas" >&2
  exit 1
fi

if [[ ! -x "${ET_INSTALL:-}/bin/riscv64-unknown-elf-gcc" ]]; then
  if [[ -x "${HOME}/et/bin/riscv64-unknown-elf-gcc" ]]; then
    export ET_INSTALL="${HOME}/et"
  else
    echo "Set ET_INSTALL to the RISC-V ET toolchain prefix, or run install_toolchain.sh first." >&2
    exit 1
  fi
fi

# Reuse an existing launcher when present (faster local iteration).
if [[ -x "${LAUNCHER_OVERRIDE:-}" ]]; then
  export LAUNCHER="${LAUNCHER_OVERRIDE}"
elif [[ -x "${HOME}/zephyr/local-artifacts/erbium_amp_probe/whisper-real/host-dynmem/build/erbium_soc1sim_argbuf_dynmem" ]]; then
  export LAUNCHER="${HOME}/zephyr/local-artifacts/erbium_amp_probe/whisper-real/host-dynmem/build/erbium_soc1sim_argbuf_dynmem"
fi

if [[ -n "${BENCHMARK_ASSETS_DIR:-}" ]]; then
  export BENCHMARK_ASSETS_DIR
elif [[ -d "${HOME}/zephyr/local-artifacts/erbium_amp_probe" ]]; then
  export BENCHMARK_ASSETS_DIR="${HOME}/zephyr/local-artifacts/erbium_amp_probe"
fi

models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then
  mapfile -t models < <(python3 "$(dirname "$0")/benchmark_config_helpers.py" --target sysemu --format lines)
fi

chmod +x "$(dirname "$0")"/*.sh

for model in "${models[@]}"; do
  echo "=== benchmark ${model} ==="
  "$(dirname "$0")/run_model_benchmark.sh" "$model" || true
done

python3 "$(dirname "$0")/format_pr_comment.py" \
  --scores-dir "${BENCHMARK_OUTPUT}" \
  --output "${BENCHMARK_OUTPUT}/pr-comment.md" \
  --target sysemu \
  --sha "local" \
  --ref "local"

echo "Wrote ${BENCHMARK_OUTPUT}/pr-comment.md"
