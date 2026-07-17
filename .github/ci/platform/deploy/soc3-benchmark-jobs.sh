#!/usr/bin/env bash
# Full job-API flow on board-host: API + board worker + benchmark jobs (soc1sim).
set -euo pipefail

DEST="${SOC3_DEST:-/root/et-jobs-deploy}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
MODELS="${MODELS:-}"
MODELS="$(python3 "${ROOT}/.github/ci/scripts/benchmark_config_helpers.py" --target board --models "$MODELS" --format space)"
# shellcheck source=soc3-ssh.sh
source "$(dirname "$0")/soc3-ssh.sh"
export SOC3_HOST="${SOC3_HOST:-root@board-host}"
read -r -a SSH_CMD <<<"$(soc3_ssh_cmd)"
RSYNC_HOST="$(soc3_rsync_host)"
RSYNC_HOST="${RSYNC_HOST%/}"

_rsh=(-e "$(soc3_rsync_rsh)")
"${ROOT}/.github/ci/platform/deploy/rsync-slim.sh" "${RSYNC_HOST}" "$DEST"
rsync -az "${_rsh[@]}" --exclude '.git' "$ROOT/ported_models/" "${RSYNC_HOST}:${DEST}/ported_models/"
rsync -az "${_rsh[@]}" "$ROOT/.github/ci/support/" "${RSYNC_HOST}:${DEST}/.github/ci/support/"

"${SSH_CMD[@]}" "bash -s" -- "$DEST" "$MODELS" <<'REMOTE'
set -euo pipefail
DEST="$1"
shift
MODELS="$*"
PLATFORM="$DEST/.github/ci/platform"
ENV="$PLATFORM/deploy/config.env"

if [[ ! -f "$ENV" ]]; then
  cp "$PLATFORM/deploy/config.env.example" "$ENV"
  sed -i "s|JOBS_REPO_ROOT=.*|JOBS_REPO_ROOT=$DEST|" "$ENV"
  sed -i "s|HOST_ID=.*|HOST_ID=board-host|" "$ENV"
  tok="$(openssl rand -hex 24)"
  sed -i "s|generate-a-long-random-token|$tok|g" "$ENV"
fi
# shellcheck disable=SC1090
source "$ENV"
export PYTHONPATH="$PLATFORM"
export ET_JOBS_DRY_RUN=0
export KERNEL_TIMEOUT=600
export ET_INSTALL="${ET_INSTALL:-/opt/et}"
export ET_PLATFORM="${ET_PLATFORM:-$ET_INSTALL}"
export LAUNCHER="${LAUNCHER:-/opt/et/bin/erbium_soc1sim_argbuf_dynmem}"
export ET_LIB_PATH="${ET_LIB_PATH:-/opt/et/host:/opt/et/lib}"
export BOARD_LOCK="${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}"
export SOC_RESET_SYSFS="${SOC_RESET_SYSFS:-/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate}"
export SIM_ONLY_PUBLIC=1

for candidate in "$HOME/et-platform" "$HOME/et" /root/et-platform "$DEST/et-platform"; do
  [[ -f "${candidate}/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld" ]] && export ET_PLATFORM_SRC="$candidate" && break
done
[[ -n "${ET_PLATFORM_SRC:-}" ]] || { git clone --depth 1 https://github.com/aifoundry-org/et-platform.git "$DEST/et-platform"; export ET_PLATFORM_SRC="$DEST/et-platform"; }

python3 -m pip install --user -q -r "$PLATFORM/requirements.txt" 2>/dev/null || true
chmod +x "$DEST/.github/ci/scripts/"*.sh "$DEST/scripts/"*.sh 2>/dev/null || true
bash "$DEST/.github/ci/scripts/prepare_board_lock.sh" "$BOARD_LOCK"

mkdir -p "$JOBS_DATA_DIR/run"
for p in "$JOBS_DATA_DIR/run"/*.pid; do kill "$(cat "$p")" 2>/dev/null || true; done
rm -f "$JOBS_DATA_DIR/run"/*.pid

nohup env PYTHONPATH="$PLATFORM" JOBS_BIND="$JOBS_BIND" JOBS_API_URL="$JOBS_API_URL" \
  JOBS_DATA_DIR="$JOBS_DATA_DIR" WORKER_TOKENS="$WORKER_TOKENS" SUBMITTER_TOKEN="$SUBMITTER_TOKEN" \
  OPERATOR_TOKEN="$OPERATOR_TOKEN" HOST_ID="$HOST_ID" JOBS_REPO_ROOT="$JOBS_REPO_ROOT" \
  python3 -m et_jobs api >"$JOBS_DATA_DIR/run/api.log" 2>&1 &
echo $! >"$JOBS_DATA_DIR/run/api.pid"
sleep 2
curl -sf "$JOBS_API_URL/health" >/dev/null

nohup env PYTHONPATH="$PLATFORM" JOBS_API_URL="$JOBS_API_URL" JOBS_DATA_DIR="$JOBS_DATA_DIR" \
  WORKER_TOKENS="$WORKER_TOKENS" HOST_ID="$HOST_ID" JOBS_REPO_ROOT="$JOBS_REPO_ROOT" \
  LAUNCHER="$LAUNCHER" ET_INSTALL="$ET_INSTALL" ET_PLATFORM="$ET_PLATFORM" ET_PLATFORM_SRC="$ET_PLATFORM_SRC" \
  ET_LIB_PATH="$ET_LIB_PATH" BOARD_LOCK="$BOARD_LOCK" SOC_RESET_SYSFS="$SOC_RESET_SYSFS" \
  ET_JOBS_DRY_RUN=0 KERNEL_TIMEOUT=600 \
  python3 -m et_jobs worker --pool board >"$JOBS_DATA_DIR/run/board.log" 2>&1 &
echo $! >"$JOBS_DATA_DIR/run/board.pid"

FAIL=0
for model in $MODELS; do
  job="$(curl -sf -X POST "$JOBS_API_URL/v1/jobs" \
    -H "Authorization: Bearer $SUBMITTER_TOKEN" \
    -H 'Content-Type: application/json' \
    -d "{\"job_type\":\"benchmark\",\"model\":\"$model\",\"want_board\":true,\"submitter\":\"soc3-bench\"}")"
  jid="$(echo "$job" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
  echo "job $model id=$jid"
  curl -sf -X POST "$JOBS_API_URL/v1/internal/operator/advance-sim" \
    -H "Authorization: Bearer $OPERATOR_TOKEN" \
    -H 'Content-Type: application/json' \
    -d "{\"job_id\":\"$jid\",\"want_board\":true}" >/dev/null || true

  for i in $(seq 1 120); do
    st="$(curl -sf "$JOBS_API_URL/v1/jobs/$jid" -H "Authorization: Bearer $SUBMITTER_TOKEN")"
    status="$(echo "$st" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
    echo "  [$model][$i] status=$status"
    if [[ "$status" == "completed" || "$status" == "failed" ]]; then
      echo "$st" | python3 -m json.tool
      curl -sf "$JOBS_API_URL/v1/jobs/$jid/logs/board" -H "Authorization: Bearer $SUBMITTER_TOKEN" | \
        python3 -c 'import sys,json; t=json.load(sys.stdin)["tail"]; print(t[-800:])'
      [[ "$status" == "completed" ]] || FAIL=1
      break
    fi
    sleep 5
  done
done

[[ $FAIL -eq 0 ]] && echo "SOC3 BENCHMARK JOBS PASSED" || { echo "SOC3 BENCHMARK JOBS FAILED" >&2; exit 1; }
REMOTE
