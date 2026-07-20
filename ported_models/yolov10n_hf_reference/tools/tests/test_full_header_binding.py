#!/usr/bin/env python3
"""Negative regression for full-package generated-header identity binding."""

from __future__ import annotations

import argparse
import errno
import hashlib
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
    header_path = package / "slice_manifest.h"
    if not manifest_path.is_file() or not header_path.is_file():
        print("error: full package is incomplete: {}".format(package), file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="yolov10n-full-header-tamper-") as raw:
        tampered = Path(raw)
        shutil.copy2(manifest_path, tampered / manifest_path.name)
        header = header_path.read_text(encoding="utf-8")
        marker = "static const struct yr_tensor_desc yr_tensors"
        marker_offset = header.find(marker)
        if marker_offset < 0:
            print("error: generated tensor table marker is absent", file=sys.stderr)
            return 2
        suffix = header[marker_offset:]
        old = "{ 1u, 0u,"
        replacement_offset = suffix.find(old)
        if replacement_offset < 0:
            print("error: first tensor descriptor was not recognized", file=sys.stderr)
            return 2
        replacement_offset += marker_offset
        tampered_header = (
            header[:replacement_offset]
            + "{ 1u, 4u,"
            + header[replacement_offset + len(old):]
        )
        (tampered / header_path.name).write_text(
            tampered_header, encoding="utf-8"
        )
        tampered_manifest = json.loads(
            (tampered / manifest_path.name).read_text(encoding="utf-8")
        )
        tampered_bytes = tampered_header.encode("utf-8")
        tampered_manifest["generated"]["header"]["nbytes"] = len(
            tampered_bytes
        )
        tampered_manifest["generated"]["header"]["sha256"] = hashlib.sha256(
            tampered_bytes
        ).hexdigest()
        (tampered / manifest_path.name).write_text(
            json.dumps(tampered_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        for name, record in manifest["blobs"].items():
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
            print(
                "error: verifier accepted a tampered tensor descriptor",
                file=sys.stderr,
            )
            return 1
        if (
            "generated header SHA-256 differs from the pinned topology"
            not in completed.stdout
        ):
            print(completed.stdout, end="", file=sys.stderr)
            print(
                "error: verifier failed for an unexpected reason",
                file=sys.stderr,
            )
            return 1

    print(
        "FULL_HEADER_BINDING PASS tampered_descriptor=rejected "
        "reason=pinned_sha256_mismatch"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
