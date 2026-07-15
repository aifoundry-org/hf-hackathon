#!/usr/bin/env bash
# Run leaderboard smoke benchmarks on board-host (real silicon, soc1sim).
#
# Usage:
#   .github/ci/platform/deploy/install-soc3-ssh-key.sh   # once, in your terminal
#   MODELS=yolo .github/ci/platform/deploy/soc3-benchmark.sh
#   USE_TAILSCALE_SSH=0 .github/ci/platform/deploy/soc3-benchmark.sh  # force OpenSSH
#
set -euo pipefail

DEST="${SOC3_DEST:-/root/et-jobs-deploy}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
BENCHMARK_ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT:-${ROOT}/local-artifacts/model-port-benchmarks}"

et_platform_src_complete() {
  local src="$1"
  [[ -f "$src/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" \
    && -f "$src/gp-sdk/device/sdk/lib/erbium-soc1sim/layout.c" \
    && -f "$src/et-common-libs/include/erbium/isa/atomic.h" \
    && -f "$src/et-common-libs/include/erbium-soc1sim/isa/syscall.h" ]]
}

if [[ -z "${LLAMA_CPP_ET_SOURCE_REVISION:-}" ]]; then
  _llama_source="${ROOT}/ported_models/llama_cpp_et/src/llama.cpp-et"
  if [[ -d "${_llama_source}" ]]; then
    LLAMA_CPP_ET_SOURCE_REVISION="$(git -C "${_llama_source}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -z "${LLAMA_CPP_ET_SOURCE_REVISION:-}" ]]; then
    LLAMA_CPP_ET_SOURCE_REVISION="$(git -C "${ROOT}" ls-tree HEAD ported_models/llama_cpp_et/src/llama.cpp-et 2>/dev/null | awk '{print $3}')"
  fi
  export LLAMA_CPP_ET_SOURCE_REVISION
fi
MODELS="${MODELS:-}"
MODELS="$(python3 "${ROOT}/.github/ci/scripts/benchmark_config_helpers.py" --target board --models "$MODELS" --format space)"
# shellcheck source=soc3-ssh.sh
source "$(dirname "$0")/soc3-ssh.sh"
SOC3_HOST="${SOC3_HOST:-root@board-host}"
export SOC3_HOST

TRANSPORT="$(soc3_resolve_transport)"
SSH_CMD=()
if [[ "$TRANSPORT" != "local" ]]; then
  read -r -a SSH_CMD <<<"$(soc3_ssh_cmd)"
fi

# A failed local prebuild must not leave an older run's score eligible for
# collection. Clear the board-host output before any operation that can fail.
if [[ "$TRANSPORT" == "local" ]]; then
  rm -rf "${DEST}/benchmark-output"
  mkdir -p "${DEST}/benchmark-output"
else
  output_dir="$(printf "%q" "${DEST}/benchmark-output")"
  "${SSH_CMD[@]}" "rm -rf ${output_dir} && mkdir -p ${output_dir}"
fi

if [[ -x "${SOC3_BUILD_ET:-$HOME/et}/bin/riscv64-unknown-elf-gcc" ]]; then
  _etp="${ET_PLATFORM_SRC:-}"
  if [[ -n "$_etp" ]] && ! et_platform_src_complete "$_etp"; then
    echo "WARN: local ET_PLATFORM_SRC=$_etp is incomplete; searching for a usable et-platform tree" >&2
    _etp=""
  fi
  if [[ -z "$_etp" ]]; then
    for candidate in \
      "$HOME/et-platform" \
      "$HOME/.ci-work/et-platform" \
      /opt/et-platform \
      /root/et-platform \
      "${ROOT}/.ci-work/et-platform"; do
      if et_platform_src_complete "$candidate"; then
        _etp="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$_etp" ]]; then
    echo "WARN: skipping local ELF prebuild because no complete et-platform source tree was found" >&2
  else
    echo "==> Pre-building ELFs locally (${SOC3_BUILD_ET:-$HOME/et}; ET_PLATFORM_SRC=$_etp)"
    mkdir -p "${BENCHMARK_ARTIFACT_ROOT}"
    ELF_MODELS="$(python3 - "$ROOT" "$MODELS" <<'PY'
import sys
repo, raw_models = sys.argv[1:3]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config, model_runner, parse_model_selection
cfg = load_config()
models = parse_model_selection(raw_models, cfg, target="board")
print(" ".join(model for model in models if model_runner(cfg, model) == "elf"))
PY
)"
    for _m in ${ELF_MODELS}; do
      REPO_ROOT="${ROOT}" BOARD_BENCHMARK=1 BENCHMARK_DEVICE=soc1sim \
        ET_INSTALL="${SOC3_BUILD_ET:-$HOME/et}" ET_PLATFORM_SRC="${_etp}" \
        BENCHMARK_ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT}" \
        bash "${ROOT}/.github/ci/scripts/build_leaderboard_elf.sh" "${_m}"
    done
    export SOC3_PREBUILT=1
  fi
fi

echo "==> soc3 transport: ${TRANSPORT} (set USE_TAILSCALE_SSH=0 to force OpenSSH)"

RSYNC_HOST="${SOC3_HOST}"
if [[ "$TRANSPORT" == "local" ]]; then
  RSYNC_HOST="local"
elif [[ "$TRANSPORT" == "openssh" ]]; then
  RSYNC_HOST="$(soc3_rsync_host)"
fi

echo "==> Syncing repo to ${RSYNC_HOST}${DEST}"
"${ROOT}/.github/ci/platform/deploy/rsync-slim.sh" "${RSYNC_HOST%/}" "$DEST"
if [[ "${SOC3_PREBUILT:-}" == "1" ]]; then
  echo "==> Syncing prebuilt ELFs"
  if [[ "$TRANSPORT" == "local" ]]; then
    mkdir -p "${DEST}/local-artifacts/model-port-benchmarks"
    rsync -az \
      "${BENCHMARK_ARTIFACT_ROOT}/" \
      "${DEST}/local-artifacts/model-port-benchmarks/"
  else
    rsync -az -e "$(soc3_rsync_rsh)" \
      "${BENCHMARK_ARTIFACT_ROOT}/" \
      "${RSYNC_HOST%/}:${DEST}/local-artifacts/model-port-benchmarks/"
  fi
fi

REMOTE_ENV=()
if [[ -n "${BENCHMARK_ASSETS_DIR:-}" ]]; then
  REMOTE_ENV+=("BENCHMARK_ASSETS_DIR=$(printf "%q" "${BENCHMARK_ASSETS_DIR}")")
fi
if [[ -n "${BENCHMARK_ASSETS_URL:-}" ]]; then
  REMOTE_ENV+=("BENCHMARK_ASSETS_URL=$(printf "%q" "${BENCHMARK_ASSETS_URL}")")
fi
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  REMOTE_ENV+=("GITHUB_TOKEN=$(printf "%q" "${GITHUB_TOKEN}")")
fi
if [[ -n "${GH_TOKEN:-}" ]]; then
  REMOTE_ENV+=("GH_TOKEN=$(printf "%q" "${GH_TOKEN}")")
fi
if [[ -n "${GITHUB_SHA:-}" ]]; then
  REMOTE_ENV+=("GITHUB_SHA=$(printf "%q" "${GITHUB_SHA}")")
fi
if [[ -n "${GITHUB_REF:-}" ]]; then
  REMOTE_ENV+=("GITHUB_REF=$(printf "%q" "${GITHUB_REF}")")
fi
if [[ -n "${GITHUB_ACTOR:-}" ]]; then
  REMOTE_ENV+=("GITHUB_ACTOR=$(printf "%q" "${GITHUB_ACTOR}")")
fi
if [[ -n "${LEADERBOARD_TEAM:-}" ]]; then
  REMOTE_ENV+=("LEADERBOARD_TEAM=$(printf "%q" "${LEADERBOARD_TEAM}")")
fi
if [[ -n "${GITHUB_SERVER_URL:-}" ]]; then
  REMOTE_ENV+=("GITHUB_SERVER_URL=$(printf "%q" "${GITHUB_SERVER_URL}")")
fi
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  REMOTE_ENV+=("GITHUB_REPOSITORY=$(printf "%q" "${GITHUB_REPOSITORY}")")
fi
if [[ -n "${GITHUB_RUN_ID:-}" ]]; then
  REMOTE_ENV+=("GITHUB_RUN_ID=$(printf "%q" "${GITHUB_RUN_ID}")")
fi
for name in \
  BENCHMARK_SHA \
  BENCHMARK_REF \
  BENCHMARK_RUN_URL \
  TRUSTED_BOARD_LAUNCHER_TIMEOUT_CAP \
  TRUSTED_BOARD_OUTER_TIMEOUT_CAP; do
  if [[ -n "${!name:-}" ]]; then
    REMOTE_ENV+=("${name}=$(printf "%q" "${!name}")")
  fi
done
if [[ -n "${SOC3_PREBUILT:-}" ]]; then
  REMOTE_ENV+=("SOC3_PREBUILT=$(printf "%q" "${SOC3_PREBUILT}")")
fi
if [[ -n "${SOC3_FAIL_ON_MODEL_FAILURE:-}" ]]; then
  REMOTE_ENV+=("SOC3_FAIL_ON_MODEL_FAILURE=$(printf "%q" "${SOC3_FAIL_ON_MODEL_FAILURE}")")
fi
if [[ -n "${SOC3_SKIP_BOARD_SMOKE:-}" ]]; then
  REMOTE_ENV+=("SOC3_SKIP_BOARD_SMOKE=$(printf "%q" "${SOC3_SKIP_BOARD_SMOKE}")")
fi
if [[ -n "${SMOLVLM2_SKIP_PPL:-}" ]]; then
  REMOTE_ENV+=("SMOLVLM2_SKIP_PPL=$(printf "%q" "${SMOLVLM2_SKIP_PPL}")")
fi
if [[ -n "${SMOLVLM2_SKIP_HOST_REFERENCE:-}" ]]; then
  REMOTE_ENV+=("SMOLVLM2_SKIP_HOST_REFERENCE=$(printf "%q" "${SMOLVLM2_SKIP_HOST_REFERENCE}")")
fi
if [[ -n "${LLAMA_CPP_ET_SOURCE_REVISION:-}" ]]; then
  REMOTE_ENV+=("LLAMA_CPP_ET_SOURCE_REVISION=$(printf "%q" "${LLAMA_CPP_ET_SOURCE_REVISION}")")
fi
if [[ -n "${TRUSTED_LLAMA_BUILD_KEY:-}" ]]; then
  REMOTE_ENV+=("TRUSTED_LLAMA_BUILD_KEY=$(printf "%q" "${TRUSTED_LLAMA_BUILD_KEY}")")
fi
if [[ -n "${TRUSTED_LLAMA_REUSE_BUILD:-}" ]]; then
  REMOTE_ENV+=("TRUSTED_LLAMA_REUSE_BUILD=$(printf "%q" "${TRUSTED_LLAMA_REUSE_BUILD}")")
fi
if [[ -n "${TRUSTED_LLAMA_CPU_PPL_BIN:-}" ]]; then
  REMOTE_ENV+=("TRUSTED_LLAMA_CPU_PPL_BIN=$(printf "%q" "${TRUSTED_LLAMA_CPU_PPL_BIN}")")
fi
if [[ -n "${TRUSTED_SMOLVLM2_CPU_BUILD_KEY:-}" ]]; then
  REMOTE_ENV+=("TRUSTED_SMOLVLM2_CPU_BUILD_KEY=$(printf "%q" "${TRUSTED_SMOLVLM2_CPU_BUILD_KEY}")")
fi
if [[ -n "${LAUNCHER:-}" ]]; then
  REMOTE_ENV+=("LAUNCHER=$(printf "%q" "${LAUNCHER}")")
fi
if [[ -n "${ET_LIB_PATH:-}" ]]; then
  REMOTE_ENV+=("ET_LIB_PATH=$(printf "%q" "${ET_LIB_PATH}")")
fi
if [[ -n "${SOC3_REMOTE_ET_INSTALL:-}" ]]; then
  REMOTE_ENV+=("ET_INSTALL=$(printf "%q" "${SOC3_REMOTE_ET_INSTALL}")")
fi
if [[ -n "${SOC3_REMOTE_ET_PLATFORM:-}" ]]; then
  REMOTE_ENV+=("ET_PLATFORM=$(printf "%q" "${SOC3_REMOTE_ET_PLATFORM}")")
fi
if [[ -n "${SOC3_REMOTE_ET_PLATFORM_SRC:-}" ]]; then
  REMOTE_ENV+=("ET_PLATFORM_SRC=$(printf "%q" "${SOC3_REMOTE_ET_PLATFORM_SRC}")")
fi
# Framework source/build/binary env vars (LLAMA_CPP_ET_SOURCE,
# LFM25_LLAMA_SOURCE, etc.) are intentionally NOT forwarded: framework source
# comes from the committed submodule, not an operator-pointed path. Forwarded
# vars below are external blob locations (GGUFs, datasets) and operational
# runtime paths only.
for name in \
  LLAMA_CPP_ET_TTS \
  LLAMA_CPP_ET_LD_LIBRARY_PATH \
  LLAMA32_1B_MODEL_PATH \
  LLAMA32_3B_MODEL_PATH \
  LLAMA31_8B_MODEL_PATH \
  GEMMA3N_E2B_MODEL_PATH \
  TINYLLAMA11B_MODEL_PATH \
  RWKV7_15B_MODEL_PATH \
  QWEN3_06B_MODEL_PATH \
  QWEN3_17B_MODEL_PATH \
  QWEN3_4B_MODEL_PATH \
  SMOLLM2_135M_MODEL_PATH \
  SMOLLM2_360M_MODEL_PATH \
  SMOLLM2_17B_MODEL_PATH \
  SMOLVLM2_500M_VIDEO_MODEL_PATH \
  SMOLVLM2_500M_VIDEO_MMPROJ_PATH \
  SMOLVLM2_COCO_CAT_PATH \
  SMOLVLM2_COCO_GIRAFFES_PATH \
  LFM25_MODEL_PATH \
  LFM25_LD_LIBRARY_PATH \
  WIKITEXT_RAW_PATH; do
  if [[ -n "${!name:-}" ]]; then
    REMOTE_ENV+=("${name}=$(printf "%q" "${!name}")")
  fi
done

echo "==> Running board benchmarks via ${TRANSPORT}"
REMOTE_SCRIPT="$(mktemp)"
cat > "$REMOTE_SCRIPT" <<'REMOTE'
set -euo pipefail
DEST="$1"
shift
MODELS="$*"
cd "$DEST"
export REPO_ROOT="$DEST"
export BENCHMARK_DEVICE=soc1sim
export BOARD_BENCHMARK=1
export ET_INSTALL="${ET_INSTALL:-/opt/et}"
export ET_PLATFORM="${ET_PLATFORM:-/opt/et}"
export TOOLCHAIN_DISTRO_VERSION="${TOOLCHAIN_DISTRO_VERSION:-22.04}"
export PATH="${ET_INSTALL}/bin:/opt/et/bin:/opt/riscv/bin:${PATH}"
export ET_LIB_PATH="${ET_LIB_PATH:-/opt/et/lib:/opt/et/host}"
export LD_LIBRARY_PATH="${ET_LIB_PATH}:${LD_LIBRARY_PATH:-}"
export BOARD_LOCK="${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}"
export SOC3_PREBUILT="${SOC3_PREBUILT:-0}"
export WORK_ROOT="$DEST/.ci-work"
export BENCHMARK_OUTPUT="$DEST/benchmark-output"
export BENCHMARK_ARTIFACT_ROOT="$DEST/local-artifacts/model-port-benchmarks"
export AMP_ROOT="$BENCHMARK_ARTIFACT_ROOT"
_llama_build_key="${TRUSTED_LLAMA_BUILD_KEY:-${LLAMA_CPP_ET_SOURCE_REVISION:-}}"
if [[ -n "$_llama_build_key" ]]; then
  if [[ ! "$_llama_build_key" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: llama.cpp build key must be a full git commit SHA" >&2
    exit 2
  fi
  export LLAMA_CPP_ET_WORKDIR="$DEST/local-artifacts/frameworks/llama.cpp-et/build-$_llama_build_key"
  export LLAMA_CPP_ET_SERVER="$LLAMA_CPP_ET_WORKDIR/bin/llama-server"
  export LLAMA_CPP_ET_PERPLEXITY="$LLAMA_CPP_ET_WORKDIR/bin/llama-perplexity"
fi
if [[ -n "${TRUSTED_SMOLVLM2_CPU_BUILD_KEY:-}" ]]; then
  if [[ ! "$TRUSTED_SMOLVLM2_CPU_BUILD_KEY" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: trusted SmolVLM2 CPU build key must be a full git commit SHA" >&2
    exit 2
  fi
  _smol_cpu_dir="$DEST/local-artifacts/frameworks/llama.cpp-et/build-$TRUSTED_SMOLVLM2_CPU_BUILD_KEY/bin"
  export TRUSTED_SMOLVLM2_CPU_SERVER="$_smol_cpu_dir/llama-server"
  export TRUSTED_SMOLVLM2_CPU_PERPLEXITY="$_smol_cpu_dir/llama-perplexity"
fi
export PATH="$ET_INSTALL/bin:$PATH"
mkdir -p "$WORK_ROOT" "$BENCHMARK_OUTPUT" "$AMP_ROOT" "$(dirname "$BOARD_LOCK")"
rm -f "$BENCHMARK_OUTPUT"/score-*.json
touch "$BOARD_LOCK" 2>/dev/null || true

# Framework builds (-DGGML_ET=ON) use CMake file(CONFIGURE), which requires cmake
# 3.18+. The board host's system cmake may be older (fails with "file does not
# recognize sub-command CONFIGURE"); bootstrap a modern cmake onto PATH if so.
CMAKE_BIN_DIR="$(bash .github/ci/scripts/ensure_cmake.sh || true)"
if [[ -n "${CMAKE_BIN_DIR:-}" ]]; then
  export PATH="$CMAKE_BIN_DIR:$PATH"
fi
echo "cmake: $(command -v cmake) ($(cmake --version 2>/dev/null | head -1))"

if [[ -f "$ET_INSTALL/lib/cmake/runtime/runtimeTargets.cmake" && ! -e "$ET_INSTALL/bin/server" ]]; then
  echo "Creating missing runtime::server placeholder at $ET_INSTALL/bin/server for CMake import validation"
  if mkdir -p "$ET_INSTALL/bin"; then
    cat > "$ET_INSTALL/bin/server" <<'EOF'
#!/usr/bin/env bash
echo "runtime::server placeholder: this board host does not install the server executable" >&2
exit 127
EOF
    chmod +x "$ET_INSTALL/bin/server"
  else
    echo "WARN: could not create $ET_INSTALL/bin/server placeholder" >&2
  fi
fi

# The ggml-et kernel build needs the ET device headers at
# $ET_INSTALL/include/esperanto (e.g. esperanto/et-trace/encoder.h). A complete SDK
# install provides these. Diagnose the layout, and clean up a stale workaround
# symlink that lacks the real headers so a proper install is used instead of being
# shadowed. We intentionally do NOT fabricate a stub: missing device headers mean
# the /opt/et install is incomplete and must be completed on the board host.
echo "ET SDK include layout ($ET_INSTALL/include):"
ls -la "$ET_INSTALL/include" 2>/dev/null || echo "  (no $ET_INSTALL/include)"
esp="$ET_INSTALL/include/esperanto"
if [[ -e "$esp/et-trace/encoder.h" ]]; then
  echo "ET device headers present at $esp"
else
  if [[ -L "$esp" ]]; then
    echo "Removing stale ET esperanto symlink without device headers -> $(readlink "$esp")"
    rm -f "$esp"
  fi
  echo "WARN: ET device headers missing at $esp (need esperanto/et-trace/encoder.h). The /opt/et SDK install on this board host is incomplete; the ggml-et framework build will fail until the device headers are installed." >&2
fi

et_platform_src_complete() {
  local src="$1"
  [[ -f "$src/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" \
    && -f "$src/gp-sdk/device/sdk/lib/erbium-soc1sim/layout.c" \
    && -f "$src/et-common-libs/include/erbium/isa/atomic.h" \
    && -f "$src/et-common-libs/include/erbium-soc1sim/isa/syscall.h" ]]
}

# Resolve the complete platform source before creating Docker compiler wrappers:
# the wrapper must bind-mount the same absolute source path used by the build.
if [[ -n "${ET_PLATFORM_SRC:-}" ]] && ! et_platform_src_complete "$ET_PLATFORM_SRC"; then
  echo "WARN: ET_PLATFORM_SRC=$ET_PLATFORM_SRC is incomplete; searching for a usable et-platform tree" >&2
  unset ET_PLATFORM_SRC
fi
if [[ -z "${ET_PLATFORM_SRC:-}" ]]; then
  for candidate in "$HOME/et-platform" "$HOME/.ci-work/et-platform" "$HOME/et" /opt/et-platform /root/et-platform "$DEST/et-platform"; do
    if et_platform_src_complete "$candidate"; then
      export ET_PLATFORM_SRC="$candidate"
      break
    fi
  done
fi
if [[ -z "${ET_PLATFORM_SRC:-}" ]]; then
  echo "Cloning a complete shallow et-platform tree..."
  rm -rf -- "$DEST/et-platform"
  git clone --depth 1 https://github.com/aifoundry-org/et-platform.git "$DEST/et-platform"
  export ET_PLATFORM_SRC="$DEST/et-platform"
  if ! et_platform_src_complete "$ET_PLATFORM_SRC"; then
    echo "error: cloned et-platform tree is incomplete" >&2
    exit 1
  fi
fi

# The ET SDK's RISC-V gcc (/opt/et/bin) may need a newer glibc than this host has
# (fails with "GLIBC_2.38 not found"). The ggml-et framework build resolves its
# toolchain from $ENV{ET_PLATFORM} -> ${ET_PLATFORM}/bin/riscv64-unknown-elf-*. If
# the real gcc can't run here, build a shadow ET_PLATFORM whose bin/* are Docker
# wrappers (run the real tool inside et-gcc:24.04, new glibc) and lib/include
# symlink to the real SDK, then point the framework build at it.
ET_GCC_REAL="$ET_INSTALL/bin/riscv64-unknown-elf-gcc"
if [[ -x "$ET_GCC_REAL" ]] && ! "$ET_GCC_REAL" --version >/dev/null 2>&1; then
  echo "WARN: $ET_GCC_REAL cannot run on this host (likely glibc too old); wrapping via docker" >&2
  if command -v docker >/dev/null 2>&1; then
    if ! docker image inspect et-gcc:24.04 >/dev/null 2>&1; then
      _imgd="$(mktemp -d)"
      # lld: ggml-et device kernels link with -fuse-ld=lld and /opt/et ships none.
      printf 'FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y --no-install-recommends libmpc3 lld && rm -rf /var/lib/apt/lists/*\n' > "$_imgd/Dockerfile"
      docker build -t et-gcc:24.04 "$_imgd" || echo "WARN: could not build et-gcc:24.04" >&2
      rm -rf "$_imgd"
    fi
    SHADOW="$WORK_ROOT/et-docker-platform"
    rm -rf "$SHADOW"; mkdir -p "$SHADOW/bin"
    # Mirror the whole SDK so its exported CMake configs resolve relative to the
    # shadow (the import-existence checks compute paths from ET_PLATFORM): symlink
    # every top-level entry except bin, and symlink every bin entry, then override
    # only the riscv64-unknown-elf-* tools with docker wrappers.
    for entry in "$ET_INSTALL"/*; do
      b="$(basename "$entry")"
      [[ "$b" == bin ]] && continue
      ln -s "$entry" "$SHADOW/$b"
    done
    for b in "$ET_INSTALL"/bin/*; do
      ln -s "$b" "$SHADOW/bin/$(basename "$b")"
    done
    _plat_src="${ET_PLATFORM_SRC:-/opt/et-platform}"
    for real in "$ET_INSTALL"/bin/riscv64-unknown-elf-*; do
      [[ -e "$real" ]] || continue
      name="$(basename "$real")"
      rm -f "$SHADOW/bin/$name"
      cat > "$SHADOW/bin/$name" <<WRAPEOF
#!/usr/bin/env bash
exec docker run --rm -v "$DEST:$DEST" -v "$ET_INSTALL:$ET_INSTALL:ro" -v "$_plat_src:$_plat_src:ro" -w "\$PWD" et-gcc:24.04 "$ET_INSTALL/bin/$name" "\$@"
WRAPEOF
      chmod +x "$SHADOW/bin/$name"
    done
    # Both variables must point at the mirrored SDK. install_toolchain.sh runs
    # as a child process, so changing ET_INSTALL there would not reach the
    # subsequent ELF build; leaving ET_INSTALL at /opt/et would bypass these
    # wrappers and execute the glibc-incompatible compiler directly.
    export ET_PLATFORM="$SHADOW"
    export ET_INSTALL="$SHADOW"
    # Drop a stale framework CMake cache that pinned the old (broken) toolchain path.
    _fwcache="$DEST/local-artifacts/frameworks/llama.cpp-et/build-et/CMakeCache.txt"
    if [[ -f "$_fwcache" ]] && ! grep -q "$SHADOW" "$_fwcache" 2>/dev/null; then
      rm -rf "$DEST/local-artifacts/frameworks/llama.cpp-et/build-et"
    fi
    echo "RISC-V toolchain wrapped via docker; ET_PLATFORM=$SHADOW ET_INSTALL=$SHADOW"
    "$SHADOW/bin/riscv64-unknown-elf-gcc" --version 2>/dev/null | head -1 || echo "WARN: wrapped gcc still failing" >&2
  else
    echo "WARN: docker unavailable; cannot wrap RISC-V toolchain for the framework build" >&2
  fi
fi

# Build and prefer the main-owned launcher unless the operator explicitly
# supplied one. Host-installed launchers can be stale and some older versions
# return zero after an ET stream error. The input marker keeps warm CI runs fast
# while ensuring a source change rebuilds the cached binary.
if [[ -z "${LAUNCHER:-}" ]]; then
  export LAUNCHER="$WORK_ROOT/build/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem"
  _launcher_marker="${LAUNCHER}.inputs.sha256"
  _launcher_inputs_sha="$({
    sha256sum .github/ci/launcher/CMakeLists.txt
    sha256sum .github/ci/launcher/erbium_soc1sim_argbuf_dynmem.cpp
  } | sha256sum | awk '{print $1}')"
  if [[ ! -x "$LAUNCHER" || "$(cat "$_launcher_marker" 2>/dev/null || true)" != "$_launcher_inputs_sha" ]]; then
    rm -f "$LAUNCHER" "$_launcher_marker"
    bash .github/ci/scripts/build_launcher.sh
    printf '%s\n' "$_launcher_inputs_sha" > "$_launcher_marker"
  fi
fi

# An explicitly provisioned launcher may be a symlink to a self-contained host
# bundle. Prefer the libraries built with that launcher over unrelated /opt/et
# copies; mixing the two produces a runtime ABI symbol failure before execution.
_launcher_real="$(readlink -f "$LAUNCHER" 2>/dev/null || true)"
_launcher_lib_dir="$(dirname "${_launcher_real:-$LAUNCHER}")"
if [[ -f "$_launcher_lib_dir/libetrt.so" \
  && -f "$_launcher_lib_dir/libdeviceLayer.so" \
  && -f "$_launcher_lib_dir/libsw-sysemu.so" ]]; then
  export ET_LIB_PATH="$_launcher_lib_dir:$ET_LIB_PATH"
  export LD_LIBRARY_PATH="$_launcher_lib_dir:$LD_LIBRARY_PATH"
fi

reset_board() {
  local reset
  for reset in /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate \
    /sys/bus/pci/devices/*/soc_reset/reinitiate; do
    if [[ -w "$reset" ]]; then
      echo "Resetting ET-SoC1 via $reset"
      echo 1 > "$reset" || true
      sleep 2
      return
    fi
  done
}

board_smoke() {
  local smoke_elf="${BOARD_SMOKE_ELF:-/opt/et/kernels/histogram.erbium-soc1sim.elf}"
  local smoke_dir="$BENCHMARK_OUTPUT/board-smoke"
  local smoke_log="$smoke_dir/run.log"
  local smoke_dump="$smoke_dir/dump.bin"

  if [[ ! -f "$smoke_elf" ]]; then
    echo "WARN: board smoke ELF missing at $smoke_elf" >&2
    return 0
  fi

  mkdir -p "$smoke_dir"
  rm -f "$smoke_log" "$smoke_dump"
  reset_board
  echo ""
  echo "========== board smoke: $(basename "$smoke_elf") =========="
  if flock -x -w 60 "$BOARD_LOCK" \
    timeout --kill-after=10s "${BOARD_SMOKE_TIMEOUT:-180}" \
    "$LAUNCHER" \
      --device soc1sim \
      --elf-load "$smoke_elf" \
      --shire 0 \
      --dump_after "$smoke_dump" \
      --timeout "${BOARD_SMOKE_LAUNCHER_TIMEOUT:-120}" \
      --mem_size 16777216 \
      --dump_size 8192 >"$smoke_log" 2>&1; then
    tail -40 "$smoke_log"
    return 0
  fi

  echo "WARN: board smoke failed; tailing $smoke_log" >&2
  tail -120 "$smoke_log" >&2 || true
  return 1
}

if [[ ! -x "$LAUNCHER" ]]; then
  echo "Launcher missing at $LAUNCHER; building committed launcher..."
  if ! bash .github/ci/scripts/build_launcher.sh; then
    echo "error: launcher build failed at $LAUNCHER" >&2
    exit 1
  fi
fi
if [[ ! -e /dev/et0_mgmt ]]; then
  echo "error: /dev/et0_mgmt missing — not a board host?" >&2
  exit 1
fi
echo "Host: $(hostname) device=$(stat -c '%a %U:%G' /dev/et0_mgmt) launcher=$LAUNCHER"

chmod +x .github/ci/scripts/*.sh scripts/*.sh 2>/dev/null || true
FAIL=0
if [[ "${SOC3_SKIP_BOARD_SMOKE:-0}" != "1" ]] && ! board_smoke; then
  FAIL=1
fi
for model in $MODELS; do
  echo ""
  echo "========== benchmark: $model =========="
  reset_board
  if ! bash .github/ci/scripts/run_model_benchmark.sh "$model"; then
    echo "WARN: $model benchmark script returned non-zero" >&2
    FAIL=1
  fi
  if [[ -f "$BENCHMARK_OUTPUT/score-${model}.json" ]]; then
    score_file="$BENCHMARK_OUTPUT/score-${model}.json"
    cat "$score_file"
    if ! python3 - "$score_file" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    score = json.load(f)
sys.exit(0 if score.get("passed") else 1)
PY
    then
      echo "WARN: $model benchmark score did not pass" >&2
      FAIL=1
    fi
  else
    echo "missing score-${model}.json" >&2
    FAIL=1
  fi
done

echo ""
echo "Scores under $BENCHMARK_OUTPUT"
ls -la "$BENCHMARK_OUTPUT"/*.json 2>/dev/null || true
if [[ "$FAIL" -ne 0 && "${SOC3_FAIL_ON_MODEL_FAILURE:-1}" != "1" ]]; then
  echo "Model failures were recorded in score artifacts; leaving board infrastructure job green."
  exit 0
fi
exit "$FAIL"
REMOTE

set +e
if [[ "$TRANSPORT" == "local" ]]; then
  _remote_cmd="${REMOTE_ENV[*]} bash $(printf "%q" "$REMOTE_SCRIPT") $(printf "%q" "$DEST") $(printf "%q" "$MODELS")"
  bash -lc "$_remote_cmd"
  RUN_STATUS=$?
else
  "${SSH_CMD[@]}" "${REMOTE_ENV[*]} bash -s" -- "$DEST" "$MODELS" < "$REMOTE_SCRIPT"
  RUN_STATUS=$?
fi
set -e
rm -f "$REMOTE_SCRIPT"

echo "==> Done. Pull scores with:"
if [[ "$TRANSPORT" == "local" ]]; then
  echo "  cp -a ${DEST}/benchmark-output/. ./.ci-work/soc3-benchmark-output/"
else
  echo "  rsync -az -e ssh board-host:${DEST}/benchmark-output/ ./.ci-work/soc3-benchmark-output/"
fi
exit "$RUN_STATUS"
