from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("board_provenance.py")
EPOCH = "et-soc1-aifoundry3-600-400-tdp0-v1"
BOOT = "11111111-2222-3333-4444-555555555555"


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "board": {
                    "hardware": {
                        "schema_version": 1,
                        "epoch": EPOCH,
                        "board_id": "aifoundry3",
                        "hostnames": ["esperanto-soc3"],
                        "minion_frequency_mhz": 600,
                        "noc_frequency_mhz": 400,
                        "tdp_w": 0,
                    }
                }
            }
        )
    )
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"model": "llama32_1b", "passed": True}))
    marker = tmp_path / "marker"
    marker.write_text(f"{BOOT} 600 400 0\n")
    return config, score, marker


def run_stamp(
    tmp_path: Path, **overrides: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    config, score, marker = fixture(tmp_path)
    env = {
        **os.environ,
        "ET_BOARD_EPOCH": EPOCH,
        "ET_BOARD_ID": "aifoundry3",
        "ET_MINION_FREQUENCY_MHZ": "600",
        "ET_NOC_FREQUENCY_MHZ": "400",
        "ET_BOARD_TDP_W": "0",
        "RUNNER_NAME": "esperanto-soc3-et-soc1",
        **overrides,
    }
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--score",
            str(score),
            "--config",
            str(config),
            "--marker",
            str(marker),
            "--boot-id",
            BOOT,
            "--hostname",
            "esperanto-soc3",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result, score


def test_stamps_verified_hardware_identity(tmp_path: Path):
    result, score_path = run_stamp(tmp_path)
    assert result.returncode == 0, result.stdout
    score = json.loads(score_path.read_text())
    assert score["hardware_epoch"] == EPOCH
    assert score["hardware"] == {
        "schema_version": 1,
        "epoch": EPOCH,
        "board_id": "aifoundry3",
        "minion_frequency_mhz": 600,
        "noc_frequency_mhz": 400,
        "tdp_w": 0,
        "hostname": "esperanto-soc3",
        "runner_name": "esperanto-soc3-et-soc1",
        "boot_id": BOOT,
        "clock_guard_marker": str(tmp_path / "marker"),
    }


def test_rejects_wrong_epoch(tmp_path: Path):
    result, _ = run_stamp(tmp_path, ET_BOARD_EPOCH="legacy-aifoundry2")
    assert result.returncode != 0
    assert "epoch='legacy-aifoundry2'" in result.stdout


def test_rejects_stale_boot_marker(tmp_path: Path):
    config, score, marker = fixture(tmp_path)
    marker.write_text("old-boot 600 400 0\n")
    env = {
        **os.environ,
        "ET_BOARD_EPOCH": EPOCH,
        "ET_BOARD_ID": "aifoundry3",
        "ET_MINION_FREQUENCY_MHZ": "600",
        "ET_NOC_FREQUENCY_MHZ": "400",
        "ET_BOARD_TDP_W": "0",
    }
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--score",
            str(score),
            "--config",
            str(config),
            "--marker",
            str(marker),
            "--boot-id",
            BOOT,
            "--hostname",
            "esperanto-soc3",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "clock marker='old-boot 600 400 0'" in result.stdout
