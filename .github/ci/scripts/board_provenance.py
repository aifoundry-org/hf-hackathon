#!/usr/bin/env python3
"""Validate and stamp the physical ET board identity into a score artifact."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


class ProvenanceError(RuntimeError):
    pass


def required_text(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ProvenanceError(f"{name} is required for a board score")
    return value


def required_int(env: dict[str, str], name: str) -> int:
    value = required_text(env, name)
    try:
        return int(value)
    except ValueError as exc:
        raise ProvenanceError(f"{name} must be an integer, got {value!r}") from exc


def board_policy(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot load benchmark config {config_path}: {exc}") from exc
    policy = config.get("board", {}).get("hardware")
    if not isinstance(policy, dict):
        raise ProvenanceError("benchmark config has no board.hardware policy")
    required = {
        "epoch",
        "board_id",
        "hostnames",
        "minion_frequency_mhz",
        "noc_frequency_mhz",
        "tdp_w",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ProvenanceError(f"board.hardware is missing: {', '.join(missing)}")
    return policy


def read_boot_id(path: Path) -> str:
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise ProvenanceError(f"cannot read boot ID from {path}: {exc}") from exc
    if not value:
        raise ProvenanceError(f"empty boot ID in {path}")
    return value


def validated_provenance(
    policy: dict[str, Any],
    env: dict[str, str],
    *,
    hostname: str,
    boot_id: str,
    marker_path: Path,
) -> dict[str, Any]:
    epoch = required_text(env, "ET_BOARD_EPOCH")
    board_id = required_text(env, "ET_BOARD_ID")
    minion = required_int(env, "ET_MINION_FREQUENCY_MHZ")
    noc = required_int(env, "ET_NOC_FREQUENCY_MHZ")
    tdp = required_int(env, "ET_BOARD_TDP_W")

    expected = {
        "epoch": str(policy["epoch"]),
        "board_id": str(policy["board_id"]),
        "minion_frequency_mhz": int(policy["minion_frequency_mhz"]),
        "noc_frequency_mhz": int(policy["noc_frequency_mhz"]),
        "tdp_w": int(policy["tdp_w"]),
    }
    actual = {
        "epoch": epoch,
        "board_id": board_id,
        "minion_frequency_mhz": minion,
        "noc_frequency_mhz": noc,
        "tdp_w": tdp,
    }
    mismatches = [
        f"{key}={actual[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    allowed_hosts = {str(value) for value in policy.get("hostnames", [])}
    if hostname not in allowed_hosts:
        mismatches.append(f"hostname={hostname!r}, expected one of {sorted(allowed_hosts)!r}")

    try:
        marker = marker_path.read_text().strip()
    except OSError as exc:
        raise ProvenanceError(f"cannot read clock guard marker {marker_path}: {exc}") from exc
    expected_marker = f"{boot_id} {minion} {noc} {tdp}"
    if marker != expected_marker:
        mismatches.append(f"clock marker={marker!r}, expected {expected_marker!r}")
    if mismatches:
        raise ProvenanceError("board provenance mismatch: " + "; ".join(mismatches))

    return {
        "schema_version": int(policy.get("schema_version", 1)),
        **actual,
        "hostname": hostname,
        "runner_name": env.get("RUNNER_NAME", "adhoc"),
        "boot_id": boot_id,
        "clock_guard_marker": str(marker_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--marker", type=Path, default=Path(os.environ.get("ET_BOARD_CLOCK_MARKER", "/run/et-board-clock-guard.ok"))
    )
    parser.add_argument("--boot-id-file", type=Path, default=Path("/proc/sys/kernel/random/boot_id"))
    parser.add_argument("--boot-id", default="")
    parser.add_argument("--hostname", default="")
    args = parser.parse_args()

    if not args.score.is_file():
        raise ProvenanceError(f"board benchmark produced no score artifact: {args.score}")
    score = json.loads(args.score.read_text())
    policy = board_policy(args.config)
    hostname = args.hostname or socket.gethostname().split(".", 1)[0]
    boot_id = args.boot_id or read_boot_id(args.boot_id_file)
    hardware = validated_provenance(
        policy,
        dict(os.environ),
        hostname=hostname,
        boot_id=boot_id,
        marker_path=args.marker,
    )
    score["hardware_epoch"] = hardware["epoch"]
    score["hardware"] = hardware
    args.score.write_text(json.dumps(score, indent=2) + "\n")
    print(
        "board provenance: "
        f"epoch={hardware['epoch']} board={hardware['board_id']} "
        f"host={hardware['hostname']} boot={hardware['boot_id']} "
        f"minion={hardware['minion_frequency_mhz']}MHz "
        f"noc={hardware['noc_frequency_mhz']}MHz tdp={hardware['tdp_w']}W"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
