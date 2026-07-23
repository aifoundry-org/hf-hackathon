from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("configure_board_clock.sh")


def fake_service(tmp_path: Path) -> tuple[Path, Path, Path]:
    state = tmp_path / "state"
    calls = tmp_path / "calls"
    state.write_text("600 400 65\n")
    calls.write_text("")
    service = tmp_path / "dev_mngt_service"
    service.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
state="${FAKE_CLOCK_STATE:?}"
calls="${FAKE_CLOCK_CALLS:?}"
printf '%s\n' "$*" >> "$calls"
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH-unset}" >> "$calls"
if [[ "$*" == *DM_CMD_SET_FREQUENCY* ]]; then
  value="${*: -1}"
  read -r _ _ tdp < "$state"
  printf '%s %s %s\n' "${value%,*}" "${value#*,}" "$tdp" > "$state"
fi
if [[ "$*" == *DM_CMD_SET_MODULE_STATIC_TDP_LEVEL* ]]; then
  value="${*: -1}"
  read -r minion noc _ < "$state"
  printf '%s %s %s\n' "$minion" "$noc" "$value" > "$state"
fi
read -r minion noc tdp < "$state"
printf 'ASIC Frequency Minion Shire: %s Mhz\n' "$minion"
printf 'ASIC Frequency NOC: %s Mhz\n' "$noc"
printf 'TDP Level Output: %s\n' "$tdp"
"""
    )
    service.chmod(service.stat().st_mode | stat.S_IXUSR)
    return service, state, calls


def run_guard(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    service, _, calls = fake_service(tmp_path)
    env = {
        **os.environ,
        "ET_DEV_MNGT_SERVICE": str(service),
        "FAKE_CLOCK_STATE": str(tmp_path / "state"),
        "FAKE_CLOCK_CALLS": str(calls),
        "ET_BOARD_CLOCK_RETRY_DELAY_S": "0",
        "ET_BOARD_CLOCK_MARKER": str(tmp_path / "clock-marker"),
        "ET_BOARD_BOOT_ID": "test-boot",
        "ET_BOARD_DEVICE_PATH": "/dev/null",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_aifoundry3_defaults_to_supported_fixed_operating_point(tmp_path: Path):
    result = run_guard(tmp_path, ET_BOARD_HOSTNAME="aifoundry3")
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "state").read_text() == "600 400 0\n"
    calls = (tmp_path / "calls").read_text()
    assert "DM_CMD_SET_MODULE_STATIC_TDP_LEVEL -l 0" in calls
    assert "DM_CMD_SET_FREQUENCY -f 600,400" in calls
    assert calls.index("DM_CMD_SET_MODULE_STATIC_TDP_LEVEL") < calls.index("DM_CMD_SET_FREQUENCY")
    assert "verified minion=600MHz noc=400MHz tdp=0W" in result.stdout


def test_physical_esperanto_hostname_uses_aifoundry3_policy(tmp_path: Path):
    result = run_guard(tmp_path, ET_BOARD_HOSTNAME="esperanto-soc3")
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "state").read_text() == "600 400 0\n"


def test_already_safe_clock_is_read_only(tmp_path: Path):
    result = run_guard(
        tmp_path,
        ET_BOARD_HOSTNAME="aifoundry3",
        ET_MINION_FREQUENCY_MHZ="600",
        ET_NOC_FREQUENCY_MHZ="400",
        ET_BOARD_TDP_W="65",
    )
    assert result.returncode == 0, result.stdout
    assert "DM_CMD_SET_FREQUENCY" not in (tmp_path / "calls").read_text()


def test_other_hosts_are_unchanged_without_explicit_policy(tmp_path: Path):
    result = run_guard(tmp_path, ET_BOARD_HOSTNAME="aifoundry2")
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "calls").read_text() == ""
    assert (tmp_path / "state").read_text() == "600 400 65\n"


def test_matching_host_marker_does_not_bypass_real_card_verification(tmp_path: Path):
    (tmp_path / "clock-marker").write_text("test-boot 600 400 0\n")
    result = run_guard(tmp_path, ET_BOARD_HOSTNAME="aifoundry3")
    assert result.returncode == 0, result.stdout
    calls = (tmp_path / "calls").read_text()
    assert "DM_CMD_GET_ASIC_FREQUENCIES" in calls
    assert "DM_CMD_GET_MODULE_STATIC_TDP_LEVEL" in calls
    assert "DM_CMD_SET_MODULE_STATIC_TDP_LEVEL -l 0" in calls
    assert "verified minion=600MHz noc=400MHz tdp=0W" in result.stdout


def test_launcher_library_path_is_not_inherited_by_management_service(tmp_path: Path):
    result = run_guard(
        tmp_path,
        ET_BOARD_HOSTNAME="aifoundry3",
        LD_LIBRARY_PATH="/incompatible/launcher/bundle",
    )
    assert result.returncode == 0, result.stdout
    assert "LD_LIBRARY_PATH=unset" in (tmp_path / "calls").read_text()


def test_partial_policy_fails_closed(tmp_path: Path):
    result = run_guard(
        tmp_path,
        ET_BOARD_HOSTNAME="test-board",
        ET_MINION_FREQUENCY_MHZ="500",
        ET_NOC_FREQUENCY_MHZ="",
    )
    assert result.returncode == 2
    assert "set both" in result.stdout


def test_invalid_tdp_fails_closed(tmp_path: Path):
    result = run_guard(
        tmp_path,
        ET_BOARD_HOSTNAME="test-board",
        ET_MINION_FREQUENCY_MHZ="600",
        ET_NOC_FREQUENCY_MHZ="400",
        ET_BOARD_TDP_W="256",
    )
    assert result.returncode == 2
    assert "integer from 0 through 255" in result.stdout


def test_missing_device_fails_without_calling_management_service(tmp_path: Path):
    result = run_guard(
        tmp_path,
        ET_BOARD_HOSTNAME="aifoundry3",
        ET_BOARD_DEVICE_PATH=str(tmp_path / "missing-device"),
        ET_BOARD_CLOCK_RETRIES="2",
    )
    assert result.returncode == 1
    assert (tmp_path / "calls").read_text() == ""
    assert "waiting for" in result.stdout
