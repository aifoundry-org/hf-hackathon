#!/usr/bin/env bash
# Provision the canonical ET-SoC1 lock before a board worker starts.
set -euo pipefail

LOCK_PATH="${1:-${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}}"
LOCK_DIR="$(dirname "$LOCK_PATH")"
LOCK_GROUP="${BOARD_LOCK_GROUP:-etsoc}"

if [[ -L "$LOCK_PATH" ]]; then
  echo "error: refusing symlink board lock: $LOCK_PATH" >&2
  exit 1
fi

as_root=()
if [[ "$EUID" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    as_root=(sudo -n)
  elif [[ -f "$LOCK_PATH" && -r "$LOCK_PATH" && -w "$LOCK_PATH" ]]; then
    python3 - "$LOCK_PATH" <<'PY'
import os
import sys

fd = os.open(sys.argv[1], os.O_RDWR)
os.close(fd)
PY
    echo "Board lock ready: $LOCK_PATH"
    exit 0
  else
    echo "error: root or passwordless sudo is required to provision $LOCK_PATH" >&2
    exit 1
  fi
fi

if ! getent group "$LOCK_GROUP" >/dev/null 2>&1; then
  LOCK_GROUP=root
fi

"${as_root[@]}" mkdir -p "$LOCK_DIR"
if [[ ! -e "$LOCK_PATH" ]]; then
  "${as_root[@]}" touch "$LOCK_PATH"
fi
if [[ ! -f "$LOCK_PATH" || -L "$LOCK_PATH" ]]; then
  echo "error: board lock must be a regular file: $LOCK_PATH" >&2
  exit 1
fi
"${as_root[@]}" chown "root:$LOCK_GROUP" "$LOCK_PATH"
"${as_root[@]}" chmod 0666 "$LOCK_PATH"

python3 - "$LOCK_PATH" <<'PY'
import os
import sys

fd = os.open(sys.argv[1], os.O_RDWR)
os.close(fd)
PY
echo "Board lock ready: $LOCK_PATH"
