#!/usr/bin/env bash
# Install an OpenSSH public key on the ET board host (run once in your terminal).
#
# If the host only has Tailscale SSH, complete the browser check when prompted,
# then this script appends your key for passwordless plain ssh afterward (if sshd accepts keys).
#
# Usage:
#   .github/ci/platform/deploy/install-soc3-ssh-key.sh
#
set -euo pipefail

KEY="${SOC3_SSH_KEY:-$HOME/.ssh/id_et_board}"
PUB="${KEY}.pub"
SOC3="${SOC3_HOST:-root@board-host}"
SOC3_ALIAS="${SOC3_SSH_ALIAS:-board-host}"

if [[ ! -f "$PUB" ]]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "hf-hackathon-et-board"
fi

echo "Public key:"
cat "$PUB"
echo ""

install_via() {
  local runner=("$@")
  "${runner[@]}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  cat "$PUB" | "${runner[@]}" "grep -qF \"$(<"$PUB")\" ~/.ssh/authorized_keys 2>/dev/null || cat >> ~/.ssh/authorized_keys"
}

echo "==> Installing key via tailscale ssh (interactive — approve browser link if asked)"
if command -v tailscale >/dev/null 2>&1; then
  install_via tailscale ssh "$SOC3"
else
  install_via ssh "$SOC3"
fi

echo "==> Testing plain OpenSSH (BatchMode)"
if ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" "$SOC3_ALIAS" hostname 2>/dev/null; then
  echo "SUCCESS: use Host $SOC3_ALIAS and USE_TAILSCALE_SSH=0 for deploy scripts"
elif ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" "$SOC3" hostname 2>/dev/null; then
  echo "SUCCESS: use SOC3_HOST=$SOC3 USE_TAILSCALE_SSH=0"
else
  echo "WARN: key install ran but BatchMode ssh still failed." >&2
  echo "  The board may be Tailscale-SSH-only (no OpenSSH pubkey auth)." >&2
  echo "  Ask tailnet admin for ACL action:accept, or enable sshd on soc3." >&2
  exit 1
fi
