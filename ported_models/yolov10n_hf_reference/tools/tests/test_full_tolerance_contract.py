#!/usr/bin/env python3
"""Negative regression: official full validation cannot loosen tolerances."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_PACKAGE = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/full_graph"
    / "deterministic_full308_v3"
)
VERIFIER = PORT_ROOT / "scripts/verify_full_package.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args()


def main() -> int:
    package = parse_args().package.resolve()
    manifest_path = package / "slice_manifest.json"
    if not manifest_path.is_file():
        print("error: full package is incomplete: {}".format(package), file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="yolov10n-full-tolerance-") as raw:
        tampered = Path(raw)
        shutil.copy2(package / "slice_manifest.h", tampered / "slice_manifest.h")
        manifest["tolerances"]["atol"] = 1.0
        (tampered / "slice_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for record in manifest["blobs"].values():
            source = (package / record["path"]).resolve()
            destination = tampered / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(str(source), str(destination))
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, destination)

        completed = subprocess.run(
            [str(VERIFIER), str(tampered)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            print(completed.stdout, end="", file=sys.stderr)
            print("error: verifier accepted atol=1.0", file=sys.stderr)
            return 1
        if "global tolerances differ" not in completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: verifier failed for an unexpected reason",
                file=sys.stderr,
            )
            return 1

    print("FULL_TOLERANCE_CONTRACT PASS loosened_atol=rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
