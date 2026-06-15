#!/usr/bin/env bash
# Full verification: security checks, local sim (real launcher), and board host.
set -euo pipefail

PLATFORM="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$PLATFORM/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PLATFORM/deploy/config.local.env}"
SOC3="${SOC3_HOST:-root@board-host}"
SOC3_DEST="${SOC3_DEST:-/root/et-jobs-deploy}"
FAIL=0

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }

# shellcheck disable=SC1090
source "$ENV_FILE"
export PYTHONPATH="$PLATFORM"
export KERNEL_TIMEOUT="${KERNEL_TIMEOUT:-300}"

echo "=== Security checks (local API) ==="
stop_api() { kill "$(cat "$PID_DIR/api.pid" 2>/dev/null)" 2>/dev/null || true; }
PID_DIR="$JOBS_DATA_DIR/run"
mkdir -p "$PID_DIR"
stop_api
export ET_JOBS_DRY_RUN=1
export JOBS_BIND=127.0.0.1:18080
export JOBS_API_URL=http://127.0.0.1:18080
python3 -m et_jobs api >"$PID_DIR/api-sec.log" 2>&1 &
echo $! >"$PID_DIR/api.pid"
sleep 2

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$JOBS_API_URL/v1/internal/claim" \
  -H 'Content-Type: application/json' -d '{"pool":"sim","host_id":"evil"}')"
[[ "$code" == "401" || "$code" == "403" ]] && pass "claim without token rejected ($code)" || fail "claim without token should 401/403 got $code"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$JOBS_API_URL/v1/internal/claim" \
  -H "Authorization: Bearer wrong" \
  -H 'Content-Type: application/json' -d '{"pool":"sim","host_id":"evil"}')"
[[ "$code" == "403" ]] && pass "claim with bad token rejected" || fail "bad worker token got $code"

if [[ -n "${SUBMITTER_TOKEN:-}" ]]; then
  code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$JOBS_API_URL/v1/jobs" \
    -H 'Content-Type: application/json' \
    -d '{"job_type":"smoke_histogram","submitter":"anon"}')"
  [[ "$code" == "401" ]] && pass "submit without submitter token rejected" || fail "missing submitter token got $code"
fi

[[ "$JOBS_BIND" == 127.0.0.1:* ]] && pass "API bound to localhost only" || fail "API bind $JOBS_BIND (should be 127.0.0.1 for dev)"

stop_api

echo ""
echo "=== Local full sim (sys_emu, repo ELF) ==="
export ET_JOBS_DRY_RUN=0
"$PLATFORM/deploy/run-local.sh" --full --test-sim-only || fail "local full sim"

echo ""
echo "=== Board host ($SOC3) ==="
"$PLATFORM/deploy/rsync-slim.sh" "$SOC3" "$SOC3_DEST"
ssh "$SOC3" "bash -s" <<REMOTE
set -euo pipefail
DEST="$SOC3_DEST"
PLATFORM="\$DEST/.github/ci/platform"
ENV="\$PLATFORM/deploy/config.env"
if [[ ! -f "\$ENV" ]]; then
  cp "\$PLATFORM/deploy/config.env.example" "\$ENV"
  sed -i "s|JOBS_REPO_ROOT=.*|JOBS_REPO_ROOT=\$DEST|" "\$ENV"
  sed -i "s|LAUNCHER=.*|LAUNCHER=\$DEST/.ci-work/build/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem|" "\$ENV"
  sed -i "s|generate-a-long-random-token|\$(openssl rand -hex 24)|" "\$ENV"
  echo "WORKER_TOKEN=\$(grep WORKER_TOKEN \$ENV | cut -d= -f2)" > "\$DEST/.worker_token"
fi
source "\$ENV"
export PYTHONPATH="\$PLATFORM"
export ET_JOBS_DRY_RUN=0
export KERNEL_TIMEOUT=300
export ET_PLATFORM_SRC="\${ET_PLATFORM_SRC:-\$HOME/et-platform}"
export ET_INSTALL=/opt/et

"\$PLATFORM/deploy/install-board-host.sh" "\$ENV"

# Security: device perms
perms=\$(stat -c '%a %U:%G' /dev/et0_mgmt)
[[ "\$perms" == "660 root:etsoc" ]] && echo "PASS: /dev/et0_mgmt permissions" || echo "WARN: /dev/et0_mgmt is \$perms (want 660 root:etsoc)"

mkdir -p "\$JOBS_DATA_DIR/run"
for p in "\$JOBS_DATA_DIR/run"/*.pid; do kill "\$(cat "\$p")" 2>/dev/null || true; done
rm -f "\$JOBS_DATA_DIR/run"/*.pid

python3 -m et_jobs api >"\$JOBS_DATA_DIR/run/api.log" 2>&1 &
echo \$! >"\$JOBS_DATA_DIR/run/api.pid"
sleep 2
curl -sf "\$JOBS_API_URL/health" >/dev/null

python3 -m et_jobs worker --pool sim >"\$JOBS_DATA_DIR/run/sim.log" 2>&1 &
echo \$! >"\$JOBS_DATA_DIR/run/sim.pid"
python3 -m et_jobs worker --pool board >"\$JOBS_DATA_DIR/run/board.log" 2>&1 &
echo \$! >"\$JOBS_DATA_DIR/run/board.pid"

job="\$(curl -sf -X POST "\$JOBS_API_URL/v1/jobs" -H 'Content-Type: application/json' \
  -d '{"job_type":"smoke_histogram","want_board":true,"submitter":"verify-all"}')"
jid="\$(echo "\$job" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
echo "board job_id=\$jid"

for i in \$(seq 1 120); do
  st="\$(curl -sf "\$JOBS_API_URL/v1/jobs/\$jid")"
  status="\$(echo "\$st" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
  echo "  [\$i] status=\$status"
  if [[ "\$status" == "completed" || "\$status" == "failed" ]]; then
    echo "\$st" | python3 -m json.tool
    log="\$(curl -sf "\$JOBS_API_URL/v1/jobs/\$jid/logs/board")"
    echo "\$log" | python3 -c 'import sys,json; t=json.load(sys.stdin)["tail"]; assert "Kernel wait seconds" in t, t[:500]; print("PASS: board log has kernel wait")'
    [[ "\$status" == "completed" ]] || exit 1
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT board job" >&2
tail -30 "\$JOBS_DATA_DIR/run/board.log" >&2
exit 1
REMOTE

board_ok=$?
[[ $board_ok -eq 0 ]] && pass "board e2e on soc3" || fail "board e2e on soc3"

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "=== ALL VERIFY PASSED ==="
else
  echo "=== VERIFY HAD FAILURES ===" >&2
  exit 1
fi
