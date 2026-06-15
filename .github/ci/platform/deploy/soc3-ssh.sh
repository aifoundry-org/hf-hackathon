#!/usr/bin/env bash
# Resolve how to reach an ET board host (plain OpenSSH vs tailscale ssh).
# shellcheck disable=SC2034
set -euo pipefail

SOC3_SSH_HOST="${SOC3_HOST:-root@board-host}"
SOC3_SSH_ALIAS="${SOC3_SSH_ALIAS:-board-host}"
SOC3_KEY="${SOC3_SSH_KEY:-$HOME/.ssh/id_et_board}"

soc3_is_local() {
  if [[ "${SOC3_LOCAL:-}" == "1" ]]; then
    return 0
  fi
  case "${SOC3_SSH_HOST}" in
    local|localhost|127.0.0.1|root@localhost|root@127.0.0.1)
      return 0
      ;;
  esac
  return 1
}

soc3_openssh_probe() {
  ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$SOC3_KEY" \
    "${SOC3_SSH_ALIAS}" "true" 2>/dev/null || \
  ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$SOC3_KEY" \
    "${SOC3_SSH_HOST}" "true" 2>/dev/null
}

soc3_resolve_transport() {
  if soc3_is_local; then
    echo local
    return 0
  fi
  if [[ "${USE_TAILSCALE_SSH:-}" == "1" ]]; then
    echo tailscale
    return 0
  fi
  if [[ "${USE_TAILSCALE_SSH:-}" == "0" ]]; then
    echo openssh
    return 0
  fi
  if soc3_openssh_probe; then
    echo openssh
    return 0
  fi
  if command -v tailscale >/dev/null 2>&1; then
    echo tailscale
    return 0
  fi
  echo openssh
}

soc3_ssh_cmd() {
  local transport
  transport="$(soc3_resolve_transport)"
  if [[ "$transport" == "local" ]]; then
    echo bash -lc
  elif [[ "$transport" == "tailscale" ]]; then
    if [[ "$SOC3_SSH_HOST" == *@* ]]; then
      echo tailscale ssh "${SOC3_SSH_HOST}"
    else
      echo tailscale ssh "root@${SOC3_SSH_HOST}"
    fi
  else
    if soc3_openssh_probe 2>/dev/null; then
      echo ssh "${SOC3_SSH_ALIAS}"
    else
      echo ssh -i "$SOC3_KEY" "${SOC3_SSH_HOST}"
    fi
  fi
}

soc3_rsync_rsh() {
  local transport
  transport="$(soc3_resolve_transport)"
  if [[ "$transport" == "local" ]]; then
    echo ""
  elif [[ "$transport" == "tailscale" ]]; then
    echo "tailscale ssh"
  else
    echo "ssh"
  fi
}

soc3_rsync_host() {
  local transport
  transport="$(soc3_resolve_transport)"
  if [[ "$transport" == "local" ]]; then
    echo "local"
  elif [[ "$transport" == "tailscale" ]]; then
    echo "${SOC3_SSH_HOST}"
  else
    echo "${SOC3_SSH_ALIAS}"
  fi
}
