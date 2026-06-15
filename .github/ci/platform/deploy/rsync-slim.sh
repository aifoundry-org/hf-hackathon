#!/usr/bin/env bash
# Minimal deploy bundle: ships the committed repo (including framework
# submodules) to the board. Excludes local-artifacts (built on the board) and
# .git (the board doesn't need git metadata, just the source files).
set -euo pipefail
HOST="${1:?usage: rsync-slim.sh user@host}"
DEST="${2:-~/et-jobs-deploy}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
# shellcheck source=soc3-ssh.sh
source "$(dirname "$0")/soc3-ssh.sh" 2>/dev/null || true
_ex=(
  --exclude '.git'
  --exclude 'local-artifacts'
  --exclude '.ci-work/et-platform'
  --exclude '.ci-work/benchmark-output'
  --exclude '.pytest_cache'
  --exclude '__pycache__'
  --exclude 'ci/platform/tests'
  --exclude 'ci/platform/deploy/config.local.env'
  --exclude 'ci/platform/deploy/config.env'
  --exclude '*.pyc'
)
_rsh=()
if [[ "$HOST" == "local" || "$HOST" == "localhost" || "$HOST" == "127.0.0.1" || "$HOST" == "root@localhost" || "$HOST" == "root@127.0.0.1" ]]; then
  :
elif [[ "${HOST}" == *"${SOC3_SSH_ALIAS:-board-host}"* || "${HOST}" == "${SOC3_HOST:-root@board-host}" ]]; then
  _rsh=(-e "$(soc3_rsync_rsh)")
  HOST="$(soc3_rsync_host)"
elif command -v tailscale >/dev/null 2>&1 && [[ "${USE_TAILSCALE_SSH:-}" == "1" ]]; then
  _rsh=(-e "tailscale ssh")
fi
_rsync() { rsync -az "${_rsh[@]}" "${_ex[@]}" "$@"; }
if [[ "$HOST" == "local" || "$HOST" == "localhost" || "$HOST" == "127.0.0.1" || "$HOST" == "root@localhost" || "$HOST" == "root@127.0.0.1" ]]; then
  mkdir -p "$DEST/.github" "$DEST/scripts" "$DEST/ported_models"
  _rsync "$ROOT/.github/" "$DEST/.github/"
  _rsync "$ROOT/scripts/run_sysemu_model_ports.sh" "$DEST/scripts/"
  _rsync "$ROOT/ported_models/" "$DEST/ported_models/"
  echo "Synced to local:$DEST"
else
  _rsync "$ROOT/.github/" "$HOST:$DEST/.github/"
  _rsync "$ROOT/scripts/run_sysemu_model_ports.sh" "$HOST:$DEST/scripts/"
  _rsync "$ROOT/ported_models/" "$HOST:$DEST/ported_models/"
  echo "Synced to $HOST:$DEST"
fi
