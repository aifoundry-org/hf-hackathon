#!/usr/bin/env bash
# Start ET Job API + workers locally and optionally run automated tests.
set -euo pipefail

PLATFORM="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$PLATFORM/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PLATFORM/deploy/config.local.env}"

MODE=test
FG=0
SIM_ONLY=0
FULL_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) MODE=test; shift ;;
    --test-sim-only) MODE=test; SIM_ONLY=1; shift ;;
    --full) FULL_RUN=1; shift ;;
    --fg) FG=1; shift ;;
    *) echo "usage: $0 [--test|--test-sim-only|--full] [--fg]" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$PLATFORM/deploy/config.local.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE — review and re-run."
  exit 0
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ "$FULL_RUN" == 1 ]]; then
  export ET_JOBS_DRY_RUN=0
  export KERNEL_TIMEOUT="${KERNEL_TIMEOUT:-300}"
fi

mkdir -p "$JOBS_DATA_DIR"
export PYTHONPATH="$PLATFORM"

if ! python3 -c "import fastapi" 2>/dev/null; then
  python3 -m pip install --user -q -r "$PLATFORM/requirements.txt"
fi

PID_DIR="$JOBS_DATA_DIR/run"
mkdir -p "$PID_DIR"
stop_all() {
  for f in "$PID_DIR"/*.pid; do
    [[ -f "$f" ]] || continue
    kill "$(cat "$f")" 2>/dev/null || true
  done
  rm -f "$PID_DIR"/*.pid
}
trap stop_all EXIT

stop_all
echo "Starting API on $JOBS_BIND (dry_run=$ET_JOBS_DRY_RUN)"
python3 -m et_jobs api >"$PID_DIR/api.log" 2>&1 &
echo $! >"$PID_DIR/api.pid"
sleep 2
curl -sf "$JOBS_API_URL/health" >/dev/null

export HOST_ID="${HOST_ID:-local-pc}"
python3 -m et_jobs worker --pool sim >"$PID_DIR/sim.log" 2>&1 &
echo $! >"$PID_DIR/sim.pid"

if [[ -e /dev/et0_mgmt ]]; then
  python3 -m et_jobs worker --pool board >"$PID_DIR/board.log" 2>&1 &
  echo $! >"$PID_DIR/board.pid"
elif [[ "${ET_JOBS_DRY_RUN:-0}" == "1" ]]; then
  python3 -m et_jobs worker --pool board >"$PID_DIR/board.log" 2>&1 &
  echo $! >"$PID_DIR/board.pid"
fi

echo "API docs: $JOBS_API_URL/docs"
echo "Logs: $PID_DIR/{api,sim,board}.log"

if [[ "$MODE" == "test" ]]; then
  echo "Running automated e2e..."
  AUTH=()
  [[ -n "${SUBMITTER_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $SUBMITTER_TOKEN")
  want_board=true
  [[ "$SIM_ONLY" == 1 || ! -e /dev/et0_mgmt ]] && want_board=false
  [[ "${ET_JOBS_DRY_RUN:-0}" == "1" && "$SIM_ONLY" != 1 ]] && want_board=true
  job="$(curl -sf -X POST "$JOBS_API_URL/v1/jobs" \
    "${AUTH[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"job_type\":\"smoke_histogram\",\"want_board\":$want_board,\"submitter\":\"run-local-test\"}")"
  jid="$(echo "$job" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")"
  echo "job_id=$jid want_board=$want_board dry_run=$ET_JOBS_DRY_RUN"
  max_wait=60
  [[ "${ET_JOBS_DRY_RUN:-0}" == "0" ]] && max_wait=600
  for i in $(seq 1 "$max_wait"); do
    st="$(curl -sf "$JOBS_API_URL/v1/jobs/$jid" "${AUTH[@]}")"
    status="$(echo "$st" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")"
    echo "  [$i] status=$status"
    if [[ "$status" == "completed" || "$status" == "failed" ]]; then
      echo "$st" | python3 -m json.tool
      log="$(curl -sf "$JOBS_API_URL/v1/jobs/$jid/logs/sim" "${AUTH[@]}")"
      echo "$log" | python3 -c "import sys,json; t=json.load(sys.stdin)['tail']; assert 'Kernel wait seconds' in t, t; print('PASS: sim log ok')"
      if [[ "$status" != "completed" ]]; then
        echo "FAIL: expected completed" >&2
        exit 1
      fi
      echo "ALL TESTS PASSED"
      exit 0
    fi
    sleep 1
  done
  echo "TIMEOUT" >&2
  tail -20 "$PID_DIR/sim.log" >&2 || true
  exit 1
fi

if [[ "$FG" == 1 ]]; then
  echo "Running in foreground; Ctrl-C stops all."
  wait
else
  echo "Stack running. Stop with: kill \$(cat $PID_DIR/*.pid)"
  echo "Submit: curl -X POST $JOBS_API_URL/v1/jobs -H 'Authorization: Bearer \$SUBMITTER_TOKEN' -H 'Content-Type: application/json' -d '{\"job_type\":\"smoke_histogram\",\"submitter\":\"dev\"}'"
fi
