#!/usr/bin/env bash
# Security verification for ET Job API (starts temporary instance if needed).
set -euo pipefail

PLATFORM="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$PLATFORM/deploy/config.local.env}"
# shellcheck disable=SC1090
source "$ENV_FILE"
export PYTHONPATH="$PLATFORM"
export ET_JOBS_PRODUCTION=0

PID_DIR="$JOBS_DATA_DIR/run"
mkdir -p "$PID_DIR"
STARTED=0
stop_api() {
  [[ "$STARTED" == 1 ]] && kill "$(cat "$PID_DIR/api.pid" 2>/dev/null)" 2>/dev/null || true
}

if ! curl -sf "$JOBS_API_URL/health" >/dev/null 2>&1; then
  STARTED=1
  export ET_JOBS_DRY_RUN=1
  python3 -m et_jobs api >"$PID_DIR/api-sec.log" 2>&1 &
  echo $! >"$PID_DIR/api.pid"
  sleep 2
fi
trap stop_api EXIT

FAIL=0
http_code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
AUTH_SUB=(-H "Authorization: Bearer $SUBMITTER_TOKEN")
AUTH_OP=(-H "Authorization: Bearer ${OPERATOR_TOKEN:-}")

assert_code() {
  local name="$1" expect="$2"
  shift 2
  got="$(http_code "$@")"
  if [[ "$got" == "$expect" ]]; then echo "PASS: $name ($got)"; else echo "FAIL: $name expected $expect got $got" >&2; FAIL=1; fi
}

assert_code "claim without token" 401 -X POST "$JOBS_API_URL/v1/internal/claim" \
  -H "Content-Type: application/json" -d '{"pool":"sim","host_id":"x"}'
assert_code "claim bad worker token" 403 -X POST "$JOBS_API_URL/v1/internal/claim" \
  -H "Authorization: Bearer bad" -H "Content-Type: application/json" -d '{"pool":"sim","host_id":"x"}'
assert_code "list jobs without auth" 401 "$JOBS_API_URL/v1/jobs"
assert_code "list jobs with submitter" 200 "${AUTH_SUB[@]}" "$JOBS_API_URL/v1/jobs"

if [[ -n "${SUBMITTER_TOKEN:-}" ]]; then
  assert_code "submit without token" 401 -X POST "$JOBS_API_URL/v1/jobs" \
    -H "Content-Type: application/json" \
    -d '{"job_type":"smoke_histogram","submitter":"x","want_board":false}'
fi

if [[ -n "${OPERATOR_TOKEN:-}" ]]; then
  assert_code "operator advance without token" 401 -X POST "$JOBS_API_URL/v1/internal/operator/advance-sim" \
    -H "Content-Type: application/json" -d '{"job_id":"00000000-0000-0000-0000-000000000001","want_board":true}'
fi

up="$(http_code -X POST "$JOBS_API_URL/v1/upload")"
[[ "$up" == "404" || "$up" == "405" ]] && echo "PASS: no upload endpoint ($up)" || { echo "FAIL: upload $up" >&2; FAIL=1; }

[[ "$JOBS_BIND" == 127.0.0.1:* ]] && echo "PASS: API localhost bind" || echo "WARN: bind=$JOBS_BIND"

# Per-host worker token: board token must not claim as sim-host
if [[ -n "${WORKER_TOKENS:-}" ]]; then
  board_tok="${WORKER_TOKENS#*:}"
  board_tok="${board_tok%%,*}"
  code="$(http_code -X POST "$JOBS_API_URL/v1/internal/claim" \
    -H "Authorization: Bearer $board_tok" \
    -H "Content-Type: application/json" \
    -d '{"pool":"sim","host_id":"sim-host"}')"
  [[ "$code" == "403" ]] && echo "PASS: board token rejected for sim-host" || { echo "FAIL: cross-host token got $code" >&2; FAIL=1; }
fi

echo "Running pytest security suite..."
if python3 -m pytest "$PLATFORM/tests/test_security.py" -q --tb=short 2>&1; then
  echo "PASS: pytest security tests"
else
  echo "FAIL: pytest" >&2
  FAIL=1
fi

[[ $FAIL -eq 0 ]] && echo "=== SECURITY CHECKS PASSED ===" || exit 1
