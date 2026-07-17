#!/usr/bin/env bash
# Full sim+board e2e on board-host (real PCIe, histogram kernel).
set -euo pipefail
SOC3="${SOC3_HOST:-root@board-host}"
DEST="${SOC3_DEST:-/root/et-jobs-deploy}"

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
rsync -az "$ROOT/.github/ci/platform/" "$SOC3:$DEST/.github/ci/platform/"
rsync -az "$ROOT/.github/ci/scripts/board_lock.py" "$ROOT/.github/ci/scripts/prepare_board_lock.sh" \
  "$SOC3:$DEST/.github/ci/scripts/"

ssh "$SOC3" "bash -s" <<REMOTE
set -euo pipefail
DEST="$DEST"
PLATFORM="\$DEST/.github/ci/platform"
ENV="\$PLATFORM/deploy/config.env"
cat > "\$ENV" <<EOF
JOBS_BIND=127.0.0.1:8080
JOBS_API_URL=http://127.0.0.1:8080
JOBS_DATA_DIR=/root/et-jobs-data
WORKER_TOKENS=board-host:\$(openssl rand -hex 32)
OPERATOR_TOKEN=\$(openssl rand -hex 32)
SUBMITTER_TOKEN=\$(openssl rand -hex 32)
HOST_ID=board-host
JOBS_REPO_ROOT=\$DEST
ET_INSTALL=/opt/et
ET_PLATFORM=/opt/et
ET_LIB_PATH=/opt/et/host:/opt/et/lib
LAUNCHER=/opt/et/bin/erbium_soc1sim_argbuf_dynmem
SMOKE_ELF=/opt/et/kernels/histogram.erbium-soc1sim.elf
BOARD_LOCK=/var/lock/etsoc-shire0.lock
SOC_RESET_SYSFS=/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate
SIM_ONLY_PUBLIC=0
EOF

export ET_JOBS_DRY_RUN=0
export KERNEL_TIMEOUT=120
source "\$ENV"
export PYTHONPATH="\$PLATFORM"

python3 -m pip install --user -q -r "\$PLATFORM/requirements.txt"

# Device perms (already 660 on this host)
groupadd -f etsoc 2>/dev/null || true
chown root:etsoc /dev/et0_mgmt /dev/et0_ops 2>/dev/null || true
chmod 660 /dev/et0_mgmt /dev/et0_ops 2>/dev/null || true
stat -c '%a %U:%G' /dev/et0_mgmt
bash "\$DEST/.github/ci/scripts/prepare_board_lock.sh" "\$BOARD_LOCK"

mkdir -p "\$JOBS_DATA_DIR/run"
shopt -s nullglob
for p in "\$JOBS_DATA_DIR/run"/*.pid; do kill "\$(cat "\$p")" 2>/dev/null || true; done
rm -f "\$JOBS_DATA_DIR/run"/*.pid
shopt -u nullglob

nohup env PYTHONPATH="\$PLATFORM" JOBS_BIND="\$JOBS_BIND" JOBS_API_URL="\$JOBS_API_URL" \
  JOBS_DATA_DIR="\$JOBS_DATA_DIR" WORKER_TOKENS="\$WORKER_TOKENS" SUBMITTER_TOKEN="\$SUBMITTER_TOKEN" \
  OPERATOR_TOKEN="\$OPERATOR_TOKEN" HOST_ID="\$HOST_ID" \
  JOBS_REPO_ROOT="\$JOBS_REPO_ROOT" LAUNCHER="\$LAUNCHER" ET_INSTALL="\$ET_INSTALL" \
  ET_LIB_PATH="\$ET_LIB_PATH" ET_PLATFORM="\$ET_PLATFORM" SMOKE_ELF="\$SMOKE_ELF" \
  SIM_ONLY_PUBLIC="\$SIM_ONLY_PUBLIC" ET_JOBS_DRY_RUN=0 KERNEL_TIMEOUT=120 \
  python3 -m et_jobs api >"\$JOBS_DATA_DIR/run/api.log" 2>&1 &
echo \$! >"\$JOBS_DATA_DIR/run/api.pid"
sleep 2
curl -sf "\$JOBS_API_URL/health" >/dev/null

# Board hosts often lack sys_emu; sim runs on VPS. Gate job to board stage for this test.
nohup env PYTHONPATH="\$PLATFORM" JOBS_API_URL="\$JOBS_API_URL" JOBS_DATA_DIR="\$JOBS_DATA_DIR" \
  WORKER_TOKENS="\$WORKER_TOKENS" OPERATOR_TOKEN="\$OPERATOR_TOKEN" SUBMITTER_TOKEN="\$SUBMITTER_TOKEN" \
  HOST_ID="\$HOST_ID" JOBS_REPO_ROOT="\$JOBS_REPO_ROOT" \
  LAUNCHER="\$LAUNCHER" ET_LIB_PATH="\$ET_LIB_PATH" ET_INSTALL="\$ET_INSTALL" ET_PLATFORM="\$ET_PLATFORM" \
  SMOKE_ELF="\$SMOKE_ELF" BOARD_LOCK="\$BOARD_LOCK" SOC_RESET_SYSFS="\$SOC_RESET_SYSFS" \
  ET_JOBS_DRY_RUN=0 KERNEL_TIMEOUT=120 \
  python3 -m et_jobs worker --pool board >"\$JOBS_DATA_DIR/run/board.log" 2>&1 &
echo \$! >"\$JOBS_DATA_DIR/run/board.pid"

job="\$(curl -sf -X POST "\$JOBS_API_URL/v1/jobs" \
  -H "Authorization: Bearer \$SUBMITTER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"job_type":"smoke_histogram","want_board":true,"submitter":"soc3-e2e"}')"
jid="\$(echo "\$job" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
echo "job_id=\$jid (board-only gate: sim_passed)"
curl -sf -X POST "\$JOBS_API_URL/v1/internal/operator/advance-sim" \
  -H "Authorization: Bearer \$OPERATOR_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"job_id\":\"\$jid\",\"want_board\":true}" >/dev/null

for i in \$(seq 1 60); do
  st="\$(curl -sf "\$JOBS_API_URL/v1/jobs/\$jid" -H "Authorization: Bearer \$SUBMITTER_TOKEN")"
  status="\$(echo "\$st" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
  echo "  [\$i] status=\$status"
  if [[ "\$status" == "completed" || "\$status" == "failed" ]]; then
    echo "\$st" | python3 -m json.tool
    curl -sf "\$JOBS_API_URL/v1/jobs/\$jid/logs/board" -H "Authorization: Bearer \$SUBMITTER_TOKEN" | python3 -c \
      "import sys,json; t=json.load(sys.stdin)['tail']; assert 'Kernel wait seconds' in t, t[-400:]"
    echo "PASS: board log (real PCIe / soc1sim)"
    [[ "\$status" == "completed" ]] || exit 1
    echo "SOC3 E2E PASSED"
    exit 0
  fi
  sleep 3
done
echo "TIMEOUT" >&2
tail -40 "\$JOBS_DATA_DIR/run/board.log" >&2
exit 1
REMOTE
