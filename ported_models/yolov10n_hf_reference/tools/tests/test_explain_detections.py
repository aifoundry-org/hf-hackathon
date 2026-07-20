#!/usr/bin/env python3
"""Positive and negative regressions for explain_detections.py."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Optional


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
TOOL = PORT_ROOT / "tools/explain_detections.py"
PACKAGE = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/full_graph"
    / "coco_room_000139_full308_v3"
)
METADATA = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/fixtures"
    / "coco_room_000139/input_fp32.bin.json"
)


def invoke(
    dump: Path,
    report: Optional[Path] = None,
    metadata: Path = METADATA,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(TOOL),
        str(PACKAGE),
        str(dump),
        "--preprocess-metadata",
        str(metadata),
    ]
    if report is not None:
        command.extend(["--json", str(report)])
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def expect_rejected(
    dump: Path, expected_message: str, description: str
) -> None:
    completed = invoke(dump)
    if completed.returncode == 0 or expected_message not in completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
        raise RuntimeError(
            "{} was not rejected for the expected reason".format(description)
        )


def main() -> int:
    dump = PACKAGE / "host_full_dump.bin"
    required = [
        TOOL,
        PACKAGE / "slice_manifest.json",
        dump,
        METADATA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(
            "error: real-image v3 regression artifacts are missing: {}".format(
                ", ".join(missing)
            ),
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="yolov10n-detections-test-") as raw:
        temporary = Path(raw)
        report_path = temporary / "detections.json"
        positive = invoke(dump, report_path)
        if positive.returncode != 0:
            print(positive.stdout, end="", file=sys.stderr)
            return 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        displayed = report["displayed"]
        if (
            not report["pass"]
            or report["records_at_or_above_threshold"] != 10
            or len(displayed) != 10
            or displayed[0]["rank"] != 0
            or displayed[0]["class_id"] != 62
            or abs(displayed[0]["score"] - 0.899284303188324) > 1e-9
            or displayed[0]["box_xyxy_canvas"]
            != [
                4.693794250488281,
                266.9769287109375,
                155.10342407226562,
                375.75567626953125,
            ]
            or displayed[0]["box_xyxy_original"]
            != [
                4.693794250488281,
                186.9769287109375,
                155.10342407226562,
                295.75567626953125,
            ]
        ):
            print(json.dumps(report, indent=2), file=sys.stderr)
            print("error: real-image detection record changed", file=sys.stderr)
            return 1

        original = dump.read_bytes()
        truncated = temporary / "truncated.bin"
        truncated.write_bytes(original[:-1])
        try:
            expect_rejected(truncated, "dump size mismatch", "truncated dump")

            non_full = bytearray(original)
            struct.pack_into("<III", non_full, 5 * 4, 289, 307, 19)
            non_full_path = temporary / "non_full.bin"
            non_full_path.write_bytes(non_full)
            expect_rejected(
                non_full_path,
                "runtime did not execute N000:N307",
                "non-full node range",
            )

            malformed = bytearray(original)
            struct.pack_into("<I", malformed, 0, 0)
            malformed_path = temporary / "bad_magic.bin"
            malformed_path.write_bytes(malformed)
            expect_rejected(
                malformed_path, "YRF1 magic mismatch", "malformed header"
            )

            manifest = json.loads(
                (PACKAGE / "slice_manifest.json").read_text(encoding="utf-8")
            )
            output = next(
                tensor
                for tensor in manifest["tensors"]
                if tensor["name"] == "output0"
            )
            invalid_class = bytearray(original)
            class_offset = (
                int(manifest["result"]["workspace_offset_within_dump"])
                + int(output["offset"])
                + (299 * 6 + 5) * 4
            )
            struct.pack_into("<f", invalid_class, class_offset, 80.0)
            invalid_class_path = temporary / "invalid_class.bin"
            invalid_class_path.write_bytes(invalid_class)
            expect_rejected(
                invalid_class_path,
                "record 299 has invalid class index",
                "invalid undisplayed class ID",
            )

            wrong_transform = json.loads(
                METADATA.read_text(encoding="utf-8")
            )
            wrong_transform["preprocessing"]["placement"] = {
                "row_start": 79,
                "row_end_exclusive": 559,
            }
            wrong_transform_path = temporary / "wrong_transform.json"
            wrong_transform_path.write_text(
                json.dumps(wrong_transform, indent=2) + "\n",
                encoding="utf-8",
            )
            completed = invoke(dump, metadata=wrong_transform_path)
            if (
                completed.returncode == 0
                or "does not describe the captured host transform"
                not in completed.stdout
            ):
                print(completed.stdout, end="", file=sys.stderr)
                raise RuntimeError(
                    "incorrect coordinate transform was not rejected"
                )
        except RuntimeError as error:
            print("error: {}".format(error), file=sys.stderr)
            return 1

    print(
        "EXPLAIN_DETECTIONS PASS real_records=10 top_class=62 "
        "inverse_y_offset=80 malformed_dump=rejected non_full=rejected "
        "invalid_class=rejected wrong_transform=rejected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
