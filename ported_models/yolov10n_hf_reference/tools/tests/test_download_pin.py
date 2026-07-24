#!/usr/bin/env python3
"""Negative regression for downloader provenance fields."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
DOWNLOADER = PORT_ROOT / "tools/download_model.py"
ARTIFACTS = PORT_ROOT / "artifacts.json"
MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"


def main() -> int:
    manifest = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    manifest["artifacts"]["yolov10n_onnx"]["source"]["revision"] = "main"
    with tempfile.TemporaryDirectory(prefix="yolov10n-download-pin-") as raw:
        tampered = Path(raw) / "artifacts.json"
        tampered.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(DOWNLOADER),
                "--manifest",
                str(tampered),
                "--output",
                str(MODEL),
                "--verify-only",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: downloader accepted a mutable revision",
                file=sys.stderr,
            )
            return 1
        if "artifact pin mismatch for source.revision" not in completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: downloader failed for an unexpected reason",
                file=sys.stderr,
            )
            return 1
    print("DOWNLOAD_PIN PASS mutable_revision=rejected network_access=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
