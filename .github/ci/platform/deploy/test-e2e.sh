#!/usr/bin/env bash
# End-to-end test against running API (smoke_histogram + board).
set -euo pipefail

API="${JOBS_API_URL:-http://127.0.0.1:8080}"
AUTH=()
[[ -n "${SUBMITTER_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $SUBMITTER_TOKEN")

echo "API=$API"

job="$(curl -sf -X POST "$API/v1/jobs" \
  "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"job_type":"smoke_histogram","want_board":true,"submitter":"e2e-test"}')"
jid="$(echo "$job" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")"
echo "created job $jid"

for i in $(seq 1 120); do
  st="$(curl -sf "$API/v1/jobs/$jid" "${AUTH[@]}")"
  status="$(echo "$st" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")"
  stage="$(echo "$st" | python3 -c "import sys,json; print(json.load(sys.stdin)['stage'])")"
  echo "[$i] status=$status stage=$stage"
  if [[ "$status" == "completed" || "$status" == "failed" ]]; then
    echo "$st" | python3 -m json.tool
    exit 0
  fi
  sleep 5
done
echo "timeout waiting for job" >&2
exit 1
