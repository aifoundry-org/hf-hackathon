#!/usr/bin/env bash
# Install the persistent safe operating-point guard on the aifoundry3 board host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
host="$(hostname -s)"
# The physical host keeps its original Esperanto inventory hostname, while
# GitHub Actions and operators refer to it as aifoundry3.
if [[ "$host" != aifoundry3* && "$host" != esperanto-soc3* ]]; then
  echo "error: refusing to install the aifoundry3 clock policy on $host" >&2
  exit 2
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: run this installer as root" >&2
  exit 2
fi
if [[ ! -x /opt/et/bin/dev_mngt_service ]]; then
  echo "error: official ET device management service is missing: /opt/et/bin/dev_mngt_service" >&2
  exit 2
fi

install -m 0755 \
  "$ROOT/.github/ci/scripts/configure_board_clock.sh" \
  /usr/local/sbin/configure-et-board-clock
install -m 0755 \
  "$ROOT/.github/ci/scripts/board_lock.py" \
  /usr/local/sbin/et-board-lock
install -m 0755 \
  "$ROOT/.github/ci/scripts/prepare_board_lock.sh" \
  /usr/local/sbin/prepare-et-board-lock
install -m 0644 \
  "$ROOT/.github/ci/platform/deploy/et-board-clock-guard.service" \
  /etc/systemd/system/et-board-clock-guard.service
# iBoot removes power rather than performing an OS shutdown.  Flush these
# files before enabling the service so a subsequent outlet cycle cannot leave
# a durable zero-length executable through delayed allocation.
sync -f /usr/local/sbin/configure-et-board-clock
sync -f /usr/local/sbin/et-board-lock
sync -f /usr/local/sbin/prepare-et-board-lock
sync -f /etc/systemd/system/et-board-clock-guard.service
systemctl daemon-reload
systemctl enable et-board-clock-guard.service
systemctl restart et-board-clock-guard.service
systemctl --no-pager --full status et-board-clock-guard.service
