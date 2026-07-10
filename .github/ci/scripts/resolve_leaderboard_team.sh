#!/usr/bin/env bash
set -euo pipefail

sha="${1:-${GITHUB_SHA:-HEAD}}"
team="${GITHUB_ACTOR:-local}"

if [[ "${GITHUB_EVENT_NAME:-}" == "push" && "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
  login=""
  api_token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [[ -n "${GITHUB_REPOSITORY:-}" && -n "$api_token" ]] && command -v gh >/dev/null 2>&1; then
    # GitHub records the maintainer who clicked Merge as the merge commit's
    # author. Prefer the merged PR submitter so leaderboard ownership follows
    # the contribution rather than the merger.
    login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/commits/${sha}/pulls" \
      --jq "map(select(.merged_at != null and .merge_commit_sha == \"${sha}\"))[0].user.login // empty" \
      2>/dev/null || true)"
    if [[ -z "$login" ]]; then
      subject="$(git show -s --format=%s "$sha" 2>/dev/null || true)"
      if [[ "$subject" =~ ^Merge\ pull\ request\ \#([0-9]+)\  ]]; then
        pr_number="${BASH_REMATCH[1]}"
        login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/pulls/${pr_number}" \
          --jq "select(.merged_at != null and .merge_commit_sha == \"${sha}\") | .user.login // empty" \
          2>/dev/null || true)"
      fi
    fi
    if [[ -z "$login" ]]; then
      login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/commits/${sha}" \
        --jq '.author.login // empty' 2>/dev/null || true)"
    fi
  fi

  if [[ -n "$login" ]]; then
    team="$login"
  else
    author="$(git show -s --format=%an "$sha" 2>/dev/null || true)"
    if [[ -n "$author" ]]; then
      team="$author"
    fi
  fi
fi

printf '%s\n' "$team"
