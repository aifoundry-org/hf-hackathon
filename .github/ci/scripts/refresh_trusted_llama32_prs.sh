#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
before="${1:?usage: refresh_trusted_llama32_prs.sh BEFORE_REF [AFTER_REF] [pending-only|rerun]}"
after="${2:-HEAD}"
mode="${3:-rerun}"

if [[ "$mode" != "pending-only" && "$mode" != "rerun" ]]; then
  echo "mode must be pending-only or rerun" >&2
  exit 2
fi

before_hash="$(.github/ci/scripts/trusted_llama32_input_hash.sh "$before")"
after_hash="$(.github/ci/scripts/trusted_llama32_input_hash.sh "$after")"
if [[ "$before_hash" == "$after_hash" ]]; then
  echo "Trusted Llama inputs did not change; no refresh is needed."
  exit 0
fi

count=0
while IFS=$'\t' read -r pr head_sha; do
  [[ -n "$pr" && -n "$head_sha" ]] || continue
  files="$(gh api --paginate "repos/${repo}/pulls/${pr}/files?per_page=100" --jq '.[].filename')"
  if ! grep -qxE '\.gitmodules|ported_models/llama_cpp_et/src/llama\.cpp-et|ported_models/llama_cpp_et/submissions/llama32_1b(\.track)?\.json' <<< "$files"; then
    continue
  fi
  context=trusted-model/llama32_1b
  description='Main changed; trusted Llama evaluation must run again.'
  if ! grep -qxE 'ported_models/llama_cpp_et/submissions/llama32_1b(\.track)?\.json' <<< "$files" \
    && grep -qxE '\.gitmodules|ported_models/llama_cpp_et/src/llama\.cpp-et' <<< "$files"; then
    context=trusted-regression/llama32_1b
    description='Main changed; shared-runtime regression evaluation must run again.'
  fi
  gh api --method POST "repos/${repo}/statuses/${head_sha}" \
    -f state=pending \
    -f context="$context" \
    -f description="$description" \
    -f target_url="${GITHUB_SERVER_URL}/${repo}/pull/${pr}" >/dev/null
  if [[ "$context" == "trusted-regression/llama32_1b" ]]; then
    gh api --method POST "repos/${repo}/statuses/${head_sha}" \
      -f state=success \
      -f context=trusted-model/llama32_1b \
      -f description='No explicit Llama track claim; regression is reported separately.' \
      -f target_url="${GITHUB_SERVER_URL}/${repo}/pull/${pr}" >/dev/null
  fi
  if [[ "$mode" == "rerun" ]]; then
    gh workflow run trusted-llama32-pr.yml --repo "$repo" --ref main -f pr_number="$pr"
    echo "Dispatched trusted Llama evaluation for PR #$pr."
  else
    echo "Marked trusted Llama status pending for PR #$pr."
  fi
  count=$((count + 1))
done < <(gh pr list --repo "$repo" --state open --limit 1000 --json number,headRefOid --jq '.[] | [.number, .headRefOid] | @tsv')

echo "Processed $count open trusted Llama PR(s)."
