#!/usr/bin/env bash
# Install ET job API + board worker on an Esperanto board host (e.g. board-host).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PLATFORM="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${1:-$PLATFORM/deploy/config.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Create $ENV_FILE from config.env.example first" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

sudo mkdir -p "$JOBS_DATA_DIR" "${JOBS_DATA_DIR%/lib/*}/logs" 2>/dev/null || mkdir -p "$JOBS_DATA_DIR"
sudo chown -R "$(whoami):" "$JOBS_DATA_DIR" 2>/dev/null || true

python3 -m pip install --user -r "$PLATFORM/requirements.txt"

# Build launcher if missing (needs CMake >= 3.21; pip install cmake on older hosts)
if [[ ! -x "${LAUNCHER:-}" ]]; then
  echo "Building launcher..."
  export ET_INSTALL="${ET_INSTALL:-/opt/et}"
  CMAKE="${CMAKE:-cmake}"
  if ! "$CMAKE" --version 2>/dev/null | awk '/version/{if ($3+0 < 3.21) exit 1}'; then
    python3 -m pip install --user -q 'cmake>=3.21' 2>/dev/null || true
    CMAKE="${HOME}/.local/bin/cmake"
  fi
  if [[ -d "$ROOT/.github/ci/launcher" ]]; then
    "$CMAKE" -S "$ROOT/.github/ci/launcher" -B "$ROOT/.ci-work/build/erbium_soc1sim_argbuf" \
      -DCMAKE_INSTALL_PREFIX="$ET_INSTALL" \
      -DET_PLATFORM_INSTALL_PREFIX="$ET_INSTALL"
    "$CMAKE" --build "$ROOT/.ci-work/build/erbium_soc1sim_argbuf" -j"$(nproc)"
    export LAUNCHER="$ROOT/.ci-work/build/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem"
  fi
fi

# Dedicated board worker user (recommended; run worker as etsoc after re-login)
if [[ -e /dev/et0_mgmt ]]; then
  sudo groupadd -f etsoc 2>/dev/null || true
  if ! id etsoc &>/dev/null; then
    sudo useradd -r -g etsoc -s /usr/sbin/nologin -d /var/lib/etsoc-worker etsoc 2>/dev/null || true
  fi
  sudo usermod -aG etsoc etsoc 2>/dev/null || true
  sudo chown root:etsoc /dev/et0_mgmt /dev/et0_ops 2>/dev/null || true
  sudo chmod 660 /dev/et0_mgmt /dev/et0_ops 2>/dev/null || \
    echo "warn: could not tighten /dev/et0_* (run as root)"
  sudo chown -R etsoc:etsoc "$JOBS_DATA_DIR" 2>/dev/null || true
  echo "Run board worker as: sudo -u etsoc env HOST_ID=... python3 -m et_jobs worker --pool board"
fi

bash "$ROOT/.github/ci/scripts/prepare_board_lock.sh" "$BOARD_LOCK"

echo "Install done. Start:"
echo "  source $ENV_FILE"
echo "  cd $PLATFORM && PYTHONPATH=$PLATFORM python3 -m et_jobs api &"
echo "  PYTHONPATH=$PLATFORM python3 -m et_jobs worker --pool board"
