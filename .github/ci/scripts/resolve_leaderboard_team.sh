#!/usr/bin/env bash
set -euo pipefail

sha="${1:-${GITHUB_SHA:-HEAD}}"
team="${GITHUB_ACTOR:-local}"

if [[ "${GITHUB_EVENT_NAME:-}" == "push" && "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
  author_sha="$sha"
  parents="$(git show -s --format=%P "$sha" 2>/dev/null || true)"
  read -r -a parent_shas <<< "$parents"
  if (( ${#parent_shas[@]} >= 2 )); then
    author_sha="${parent_shas[1]}"
  fi

  login=""
  api_token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [[ -n "${GITHUB_REPOSITORY:-}" && -n "$api_token" ]] && command -v gh >/dev/null 2>&1; then
    # Prefer the merged PR owner. The merge button operator and Git author are
    # not stable participant identities and may be maintainers or display names.
    login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/commits/${sha}/pulls" \
      --jq 'map(select(.merged_at != null and .base.ref == "main")) | sort_by(.merged_at) | last | .user.login // empty' \
      2>/dev/null || true)"
    if [[ -z "$login" ]]; then
      subject="$(git show -s --format=%s "$sha" 2>/dev/null || true)"
      if [[ "$subject" =~ ^Merge\ pull\ request\ \#([0-9]+)\  ]]; then
        pr_number="${BASH_REMATCH[1]}"
        login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/pulls/${pr_number}" \
          --jq 'select(.merged_at != null and .base.ref == "main") | .user.login // empty' \
          2>/dev/null || true)"
      fi
    fi
    if [[ -z "$login" ]]; then
      login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/commits/${author_sha}" \
        --jq '.author.login // empty' 2>/dev/null || true)"
    fi
  fi

  if [[ -n "$login" ]]; then
    team="$login"
  else
    author="$(git show -s --format=%an "$author_sha" 2>/dev/null || true)"
    if [[ -n "$author" ]]; then
      team="$author"
    fi
  fi
fi

printf '%s\n' "$team"
