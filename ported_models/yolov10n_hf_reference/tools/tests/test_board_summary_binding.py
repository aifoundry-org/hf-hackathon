#!/usr/bin/env python3
"""Negative regression for the compact real-board evidence collector."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
TOOL = PORT_ROOT / "tools/collect_board_summary.py"
FULL_DIR = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/full_graph/"
    "coco_room_000139_full308_v3"
)
RUN_DIR = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/results/full_board/"
    "coco_room_000139_full308_v3"
)


def run(output: Path, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            str(FULL_DIR),
            str(run_dir),
            str(output),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    if (
        not (FULL_DIR / "slice_manifest.json").is_file()
        or not (RUN_DIR / "full_compare.json").is_file()
    ):
        print("BOARD_SUMMARY_BINDING SKIP real-board v3 evidence unavailable")
        return 0

    with tempfile.TemporaryDirectory(prefix="yr_board_summary_") as raw:
        root = Path(raw)
        positive = run(root / "positive.json", RUN_DIR)
        if positive.returncode != 0:
            print(positive.stdout, end="")
            raise SystemExit("positive real-board evidence was rejected")

        tampered_run = root / "tampered_run"
        tampered_run.mkdir()
        for source in RUN_DIR.iterdir():
            if source.name == "full_compare.json" or not source.is_file():
                continue
            (tampered_run / source.name).symlink_to(source)
        comparison = json.loads(
            (RUN_DIR / "full_compare.json").read_text(encoding="utf-8")
        )
        comparison["final_output"]["direct_ort_mismatch_count"] = 1
        (tampered_run / "full_compare.json").write_text(
            json.dumps(comparison, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        negative = run(root / "negative.json", tampered_run)
        if negative.returncode == 0:
            raise SystemExit("tampered direct-output mismatch was accepted")
        if "output0 is not a direct zero-mismatch ORT pass" not in negative.stdout:
            print(negative.stdout, end="")
            raise SystemExit("tamper rejection did not identify output0")

    print(
        "BOARD_SUMMARY_BINDING PASS positive=accepted direct_output_mismatch=rejected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
