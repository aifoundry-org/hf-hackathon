#!/usr/bin/env python3
"""Download and verify the pinned YOLOv10n ONNX source artifact.

The file is written atomically into the repository's ignored local-artifacts
cache.  An existing file is never trusted without hashing it first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_MANIFEST = PORT_ROOT / "artifacts.json"
PINNED_SOURCE = {
    "type": "huggingface",
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "url": (
        "https://huggingface.co/onnx-community/yolov10n/resolve/"
        "57657320425ee34056408a57ad9d29c4d4815bd8/"
        "onnx/model.onnx?download=true"
    ),
}
PINNED_SHA256 = (
    "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
)
PINNED_SIZE_BYTES = 9386116
PINNED_LICENSE = "AGPL-3.0"
PINNED_CACHE = "local-artifacts/yolov10n_hf_reference/model.onnx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_from_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as src:
        manifest = json.load(src)
    try:
        artifact = manifest["artifacts"]["yolov10n_onnx"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"error: malformed artifact manifest {path}: {exc}") from exc
    expected = {
        "kind": "model",
        "format": "onnx",
        "dtype": "fp32",
        "size_bytes": PINNED_SIZE_BYTES,
        "sha256": PINNED_SHA256,
        "license": PINNED_LICENSE,
        "local_cache": PINNED_CACHE,
        "source_of_truth": True,
        "export": "none",
    }
    if manifest.get("schema_version") != 1 or not isinstance(artifact, dict):
        raise SystemExit(f"error: malformed artifact manifest {path}")
    for key, expected_value in expected.items():
        if artifact.get(key) != expected_value:
            raise SystemExit(
                "error: artifact pin mismatch for {}: expected={!r} "
                "actual={!r}".format(
                    key, expected_value, artifact.get(key)
                )
            )
    source = artifact.get("source")
    if not isinstance(source, dict):
        raise SystemExit("error: artifact pin source is missing")
    for key, expected_value in PINNED_SOURCE.items():
        if source.get(key) != expected_value:
            raise SystemExit(
                "error: artifact pin mismatch for source.{}: "
                "expected={!r} actual={!r}".format(
                    key, expected_value, source.get(key)
                )
            )
    return artifact


def resolve_cache_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify(path: Path, expected: str, expected_size: int) -> bool:
    if not path.is_file():
        print(f"MISSING path={path}")
        return False
    actual = sha256_file(path)
    actual_size = path.stat().st_size
    status = (
        "PASS"
        if actual == expected and actual_size == expected_size
        else "FAIL"
    )
    print(
        f"CHECKSUM {status} path={path} bytes={actual_size} "
        f"expected_bytes={expected_size} expected={expected} actual={actual}"
    )
    return actual == expected and actual_size == expected_size


def download(
    url: str, destination: Path, expected: str, expected_size: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "hf-hackathon-yolov10n-reference/1"}
        )
        print(f"DOWNLOAD url={url}")
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as dst:
            shutil.copyfileobj(response, dst, length=1024 * 1024)
        actual = sha256_file(tmp)
        actual_size = tmp.stat().st_size
        if actual != expected or actual_size != expected_size:
            raise SystemExit(
                "error: downloaded artifact identity mismatch: "
                f"expected_bytes={expected_size} actual_bytes={actual_size} "
                f"expected_sha256={expected} actual_sha256={actual}"
            )
        os.replace(str(tmp), str(destination))
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-only", action="store_true", help="do not access the network"
    )
    parser.add_argument(
        "--force", action="store_true", help="redownload even if the cache verifies"
    )
    args = parser.parse_args()

    artifact = artifact_from_manifest(args.manifest.resolve())
    expected = str(artifact["sha256"]).lower()
    expected_size = int(artifact["size_bytes"])
    destination = (
        args.output.resolve()
        if args.output
        else resolve_cache_path(str(artifact["local_cache"]))
    )

    if not args.force and verify(destination, expected, expected_size):
        return 0
    if args.verify_only:
        return 1

    download(
        str(artifact["source"]["url"]), destination, expected, expected_size
    )
    return 0 if verify(destination, expected, expected_size) else 1


if __name__ == "__main__":
    sys.exit(main())
