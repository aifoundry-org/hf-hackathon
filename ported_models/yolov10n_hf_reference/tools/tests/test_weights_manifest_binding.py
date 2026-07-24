#!/usr/bin/env python3
"""Negative regression for packed-initializer offsets and segment binding."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
GENERATOR = PORT_ROOT / "tools/generate_full_graph.py"
MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
WEIGHTS = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/package/weights.bin"
)
WEIGHTS_MANIFEST = PORT_ROOT / "manifests/weights_manifest.json"


def main() -> int:
    manifest = json.loads(
        WEIGHTS_MANIFEST.read_text(encoding="utf-8")
    )
    manifest["initializers"][0]["offset"] = 64
    with tempfile.TemporaryDirectory(
        prefix="yolov10n-weights-manifest-tamper-"
    ) as raw:
        temporary = Path(raw)
        tampered_manifest = temporary / "weights_manifest.json"
        tampered_manifest.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--model",
                str(MODEL),
                "--weights",
                str(WEIGHTS),
                "--weights-manifest",
                str(tampered_manifest),
                "--output-root",
                str(temporary),
                "--name",
                "generated",
                "--execution-manifest",
                str(temporary / "execution.json"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: generator accepted a tampered initializer offset",
                file=sys.stderr,
            )
            return 1
        if "initializer package mapping mismatch" not in completed.stdout:
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: generator failed for an unexpected reason",
                file=sys.stderr,
            )
            return 1
    print(
        "WEIGHTS_MANIFEST_BINDING PASS tampered_offset=rejected "
        "segment_mapping=cross_checked"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
