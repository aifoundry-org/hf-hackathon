#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
before="${1:?usage: refresh_model_port_prs.sh BEFORE_REF [AFTER_REF] [pending-only|rerun]}"
after="${2:-HEAD}"
mode="${3:-rerun}"

if [[ "$mode" != pending-only && "$mode" != rerun ]]; then
  echo "mode must be pending-only or rerun" >&2
  exit 2
fi
before_hash="$(.github/ci/scripts/model_port_input_hash.sh "$before")"
after_hash="$(.github/ci/scripts/model_port_input_hash.sh "$after")"
if [[ "$before_hash" == "$after_hash" ]]; then
  echo "Trusted model-port inputs did not change; no refresh is needed."
  exit 0
fi

count=0
while IFS=$'\t' read -r pr head_sha; do
  [[ -n "$pr" && -n "$head_sha" ]] || continue
  files="$(gh api --paginate "repos/${repo}/pulls/${pr}/files?per_page=100" --jq '.[].filename')"
  if ! grep -qE '^ported_models/submissions/model_ports/[a-z0-9][a-z0-9_-]+\.json$' <<< "$files"; then
    continue
  fi
  gh api --method POST "repos/${repo}/statuses/${head_sha}" \
    -f state=pending \
    -f context=trusted-track/model-port-credit \
    -f description='Main policy, identities, ledger, or trusted runner changed; re-evaluation required.' \
    -f target_url="${GITHUB_SERVER_URL}/${repo}/pull/${pr}" >/dev/null
  if [[ "$mode" == rerun ]]; then
    gh workflow run trusted-model-port-pr.yml --repo "$repo" --ref main -f pr_number="$pr"
    echo "Dispatched trusted model-port evaluation for PR #$pr."
  else
    echo "Marked trusted model-port status pending for PR #$pr."
  fi
  count=$((count + 1))
done < <(gh pr list --repo "$repo" --state open --limit 1000 --json number,headRefOid --jq '.[] | [.number, .headRefOid] | @tsv')

echo "Processed $count open model-port claim PR(s)."
