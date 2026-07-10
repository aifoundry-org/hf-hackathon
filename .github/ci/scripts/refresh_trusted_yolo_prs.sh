#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
before="${1:?usage: refresh_trusted_yolo_prs.sh BEFORE_REF [AFTER_REF] [pending-only|rerun]}"
after="${2:-HEAD}"
mode="${3:-rerun}"
zero_sha="0000000000000000000000000000000000000000"

if [[ "$mode" != "pending-only" && "$mode" != "rerun" ]]; then
  echo "usage: refresh_trusted_yolo_prs.sh BEFORE_REF [AFTER_REF] [pending-only|rerun]" >&2
  exit 2
fi

if [[ "$before" == "$zero_sha" ]]; then
  changed=1
else
  before_hash="$(.github/ci/scripts/trusted_yolo_input_hash.sh "$before")"
  after_hash="$(.github/ci/scripts/trusted_yolo_input_hash.sh "$after")"
  [[ "$before_hash" != "$after_hash" ]] && changed=1 || changed=0
fi

if [[ "$changed" != "1" ]]; then
  echo "Trusted YOLO inputs did not change; no PR refresh is needed."
  exit 0
fi

refreshed=0
while IFS=$'\t' read -r pr head_sha; do
  [[ -n "$pr" && -n "$head_sha" ]] || continue
  if ! gh api --paginate "repos/${repo}/pulls/${pr}/files?per_page=100" \
    --jq '.[].filename' \
    | python3 -c '
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".github/ci/scripts").resolve()))
from prepare_trusted_yolo_tree import is_allowed_path

raise SystemExit(0 if any(is_allowed_path(line.strip()) for line in sys.stdin if line.strip()) else 1)
'; then
    continue
  fi

  gh api --method POST "repos/${repo}/statuses/${head_sha}" \
    -f state=pending \
    -f context=trusted-yolo/main-gate \
    -f description='Main changed; queued against the latest trusted YOLO harness.' \
    -f target_url="${GITHUB_SERVER_URL}/${repo}/pull/${pr}" \
    >/dev/null
  if [[ "$mode" == "pending-only" ]]; then
    echo "Marked trusted YOLO status pending for PR #${pr}."
    refreshed=$((refreshed + 1))
    continue
  fi

  run_record="$(gh api --paginate \
    "repos/${repo}/actions/workflows/trusted-yolo-pr.yml/runs?event=pull_request_target&per_page=100" \
    --jq ".workflow_runs[] | select(.display_title == \"Trusted YOLO PR #${pr} (${head_sha})\") | [.id, .status, .created_at] | @tsv" \
    | sort -t $'\t' -k3,3 \
    | tail -n 1)"
  if [[ -z "$run_record" ]]; then
    echo "No trusted YOLO run exists yet for PR #${pr}; its next PR update will create one." >&2
    continue
  fi

  IFS=$'\t' read -r run_id run_status _ <<< "$run_record"
  if [[ "$run_status" != "completed" ]]; then
    gh api --method POST "repos/${repo}/actions/runs/${run_id}/cancel" >/dev/null || true
    for _ in {1..30}; do
      run_status="$(gh api "repos/${repo}/actions/runs/${run_id}" --jq .status)"
      [[ "$run_status" == "completed" ]] && break
      sleep 2
    done
  fi
  if [[ "$run_status" != "completed" ]]; then
    echo "Trusted YOLO run ${run_id} for PR #${pr} did not stop in time; its freshness check will reject stale inputs." >&2
    continue
  fi

  gh api --method POST "repos/${repo}/actions/runs/${run_id}/rerun" >/dev/null
  echo "Re-ran trusted YOLO Actions run ${run_id} for PR #${pr}."
  refreshed=$((refreshed + 1))
done < <(gh pr list --repo "$repo" --state open --limit 1000 \
  --json number,headRefOid --jq '.[] | [.number, .headRefOid] | @tsv')

echo "Processed ${refreshed} open YOLO PR(s) in ${mode} mode."
