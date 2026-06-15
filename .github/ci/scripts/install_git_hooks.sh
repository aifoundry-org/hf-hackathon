#!/usr/bin/env bash
# Install repo git hooks into .git/hooks (not version-controlled). Run once after
# cloning: .github/ci/scripts/install_git_hooks.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
hook="$ROOT/.git/hooks/pre-push"
cat > "$hook" <<'EOF'
#!/usr/bin/env bash
# Runs CI preflight before any push so script/config/workflow breakage is caught
# locally instead of on GitHub. Bypass (rarely) with `git push --no-verify`.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
exec "$ROOT/.github/ci/scripts/ci_preflight.sh"
EOF
chmod +x "$hook"
echo "Installed pre-push hook -> $hook"
