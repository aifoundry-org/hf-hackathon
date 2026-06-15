#!/usr/bin/env bash
# Local preflight for the parts of CI that run on GitHub-hosted runners:
# selector, PR-comment formatter, config/workflow validity. It does NOT run the
# board hardware benchmark (that needs the ET-SoC1 board), only the logic that
# can break a GitHub-hosted job. Wired as a git pre-push hook; also runnable by
# hand: .github/ci/scripts/ci_preflight.sh
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
fail=0
note() { printf '  %s\n' "$*"; }
step() { printf '==> %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }

step "Python syntax (.github/ci/scripts/*.py)"
if ! python3 -m py_compile .github/ci/scripts/*.py; then
  bad "python compile errors"
fi

step "Bash syntax (CI shell scripts)"
while IFS= read -r -d '' sh; do
  bash -n "$sh" || bad "bash -n: $sh"
done < <(find .github/ci -name '*.sh' -print0)

step "Workflow YAML parses"
for yml in .github/workflows/*.yml; do
  python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$yml" || bad "invalid YAML: $yml"
done

step "JSON config validity (benchmark_config + per-model + artifacts)"
python3 - <<'PY' || fail=1
import json, sys
from pathlib import Path
paths = [Path(".github/ci/benchmark_config.json")]
cfg = json.load(open(paths[0]))
for m in cfg.get("models", {}).values():
    if "config" in m:
        paths.append(Path(m["config"]))
paths += list(Path("ported_models").glob("*/artifacts.json"))
bad = False
for p in paths:
    try:
        json.load(open(p))
    except Exception as exc:
        print(f"FAIL: invalid JSON {p}: {exc}", file=sys.stderr); bad = True
sys.exit(1 if bad else 0)
PY

step "Config loads/expands for board + sysemu targets"
python3 .github/ci/scripts/benchmark_config_helpers.py --target board --models all --format space >/dev/null \
  || bad "board model list failed to resolve"
python3 .github/ci/scripts/benchmark_config_helpers.py --target sysemu --models all --format space >/dev/null \
  || bad "sysemu model list failed to resolve"

step "PR-comment formatter renders"
tmp="$(mktemp -d)"
python3 .github/ci/scripts/format_pr_comment.py --scores-dir "$tmp" --output "$tmp/c.md" \
  --target board --models "" --unregistered "smoketest" --sha x --ref y >/dev/null \
  || bad "format_pr_comment.py failed"
rm -rf "$tmp"

step "Selector runs against merge-base (if available)"
base="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base main HEAD 2>/dev/null || true)"
if [[ -n "$base" ]]; then
  python3 .github/ci/scripts/changed_benchmark_models.py --target board --base "$base" --head HEAD \
    --format space --unregistered-out "$(mktemp)" >/dev/null \
    || bad "changed_benchmark_models.py failed"
else
  note "no merge-base found; skipping selector run"
fi

if [[ "$fail" -ne 0 ]]; then
  printf '\npreflight FAILED — fix the above before pushing to CI.\n' >&2
  exit 1
fi
printf '\npreflight OK\n'
