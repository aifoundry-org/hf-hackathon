#!/usr/bin/env python3
"""Create a compact, fail-closed summary of a validated full board run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
EXPECTED_SOURCE = {
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "sha256": "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b",
    "license": "AGPL-3.0",
}
EXPECTED_HEADER_SHA256 = (
    "79be5b751842df025a3612ebb690e283813ea9ac8e373fd1bc44b706ca7a2a7e"
)
EXPECTED_STAGES = (
    ("stem", "N000", "N005"),
    ("backbone", "N006", "N090"),
    ("sppf_psa", "N091", "N128"),
    ("neck", "N129", "N207"),
    ("three_scale_head", "N208", "N270"),
    ("dfl_decode", "N271", "N288"),
    ("topk_selection", "N289", "N307"),
)


class SummaryError(RuntimeError):
    """The run does not constitute strict full-board evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def digest(path: Path) -> Dict[str, Any]:
    require(path.is_file(), "missing {}".format(path))
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return {
        "path": display_path(path),
        "nbytes": path.stat().st_size,
        "sha256": value.hexdigest(),
    }


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path, label: str) -> Dict[str, Any]:
    require(path.is_file(), "missing {} {}".format(label, path))
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "{} is not an object".format(label))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("full_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    full_dir = args.full_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite {}".format(output))

    manifest_path = full_dir / "slice_manifest.json"
    manifest = read_json(manifest_path, "full manifest")
    require(
        manifest.get("schema_version") == 2
        and manifest.get("manifest_kind") == "full_graph_liveness",
        "package is not schema-v2 full graph",
    )
    require(
        all(
            manifest.get("source", {}).get(key) == value
            for key, value in EXPECTED_SOURCE.items()
        ),
        "package source provenance differs from the pinned artifact",
    )
    require(
        manifest.get("selection", {}).get("selector") == "N000:N307"
        and len(manifest.get("nodes", [])) == 308,
        "package does not cover all 308 nodes",
    )
    header_record = manifest.get("generated", {}).get("header", {})
    require(
        header_record.get("nbytes") == 92881
        and header_record.get("sha256") == EXPECTED_HEADER_SHA256,
        "generated header identity is not pinned",
    )
    header_identity = digest(full_dir / "slice_manifest.h")
    require(
        header_identity["nbytes"] == header_record["nbytes"]
        and header_identity["sha256"] == header_record["sha256"],
        "generated header differs from manifest",
    )

    package_blobs: Dict[str, Any] = {}
    for name in ("inputs", "weights", "goldens"):
        record = manifest.get("blobs", {}).get(name, {})
        identity = digest(full_dir / str(record.get("path", "")))
        require(
            identity["nbytes"] == record.get("nbytes")
            and identity["sha256"] == record.get("sha256"),
            "{} blob differs from full manifest".format(name),
        )
        package_blobs[name] = identity
    instrumented_identity = digest(full_dir / "instrumented_full.onnx")
    require(
        instrumented_identity["sha256"]
        == manifest.get("source", {}).get("instrumented_sha256"),
        "instrumented ONNX identity differs from full manifest",
    )

    run_result_path = run_dir / "run_result.json"
    run_result = read_json(run_result_path, "run result")
    require(
        run_result.get("status") == "pass"
        and run_result.get("device") == "soc1sim"
        and run_result.get("hardware") is True
        and run_result.get("return_code") == 0,
        "run result is not a successful hardware launch",
    )
    for field in (
        "launcher_identity_match",
        "completion_log_match",
        "dump_log_match",
        "board_reset_match",
        "dump_size_match",
    ):
        require(run_result.get(field) is True, "{} is not true".format(field))
    require(
        run_result.get("source_sha256") == EXPECTED_SOURCE["sha256"],
        "run source checksum differs",
    )
    selection = run_result.get("selection", {})
    require(
        selection.get("selector") == "N000:N307"
        and selection.get("first_node") == "N000"
        and selection.get("last_node") == "N307",
        "run selection is not N000:N307",
    )
    run_artifact_paths = {
        "launcher": Path(
            str(run_result.get("artifacts", {}).get("launcher", {}).get("path", ""))
        ),
        "elf": run_dir / "slice.elf",
        "build_record": run_dir / "build_record.json",
        "slice_manifest": manifest_path,
        "inputs": full_dir / "inputs.bin",
        "weights": full_dir / "weights.bin",
        "dump": run_dir / "dump.bin",
        "log": run_dir / "run.log",
        "command": run_dir / "command.txt",
        "wrapper_command": run_dir / "wrapper_command.txt",
        "environment": run_dir / "environment.txt",
        "device_evidence": run_dir / "device_evidence.txt",
        "board_lock": run_dir / "board_lock.log",
    }
    for name, path in run_artifact_paths.items():
        stored = run_result.get("artifacts", {}).get(name)
        require(isinstance(stored, dict), "run result lacks {}".format(name))
        current = digest(path)
        require(
            current["nbytes"] == stored.get("bytes")
            and current["sha256"] == stored.get("sha256"),
            "run artifact {} differs from run result".format(name),
        )

    evidence_path = run_dir / "device_evidence.txt"
    evidence = evidence_path.read_text(encoding="utf-8")
    require(
        "backend=soc1sim" in evidence
        and "meaning=real PCIe ET-SoC1 hardware" in evidence
        and "type=character special file" in evidence
        and ("1e0a:eb01" in evidence.lower() or "esperanto" in evidence.lower()),
        "device evidence does not prove real ET-SoC1 hardware",
    )
    log_path = run_dir / "run.log"
    log = log_path.read_text(encoding="utf-8")
    require(
        "Resetting ET-SoC1 via " in log
        and "DevicePcie" in log
        and "PCIe target:                    /dev/et0_ops" in log
        and "Architecture revision:          ETSOC1" in log
        and "Kernel completed successfully" in log,
        "runtime log lacks hardware/reset/completion proof",
    )
    wait_matches = re.findall(
        r"^Kernel wait seconds: ([0-9]+(?:\.[0-9]+)?)$",
        log,
        flags=re.MULTILINE,
    )
    require(len(wait_matches) == 1, "runtime log has no unique kernel wait")
    kernel_wait_seconds = float(wait_matches[0])

    build_path = run_dir / "build_record.json"
    build = read_json(build_path, "build record")
    elf_identity = digest(run_dir / "slice.elf")
    require(
        build.get("elf", {}).get("bytes") == elf_identity["nbytes"]
        and build.get("elf", {}).get("sha256") == elf_identity["sha256"],
        "saved ELF differs from build record",
    )
    compiler = build.get("compiler", {})
    require(
        compiler.get("version") == "riscv64-unknown-elf-gcc (g5115c7e44) 15.2.0"
        and compiler.get("docker_image") == "et-gcc:24.04"
        and compiler.get("docker_image_id")
        == "sha256:6a811b9dcb63231c903d837fd969fbcee64aa2a6e8d685b8e0af3f9d92cfaa67",
        "build does not use the recorded supported ET Docker compiler",
    )
    for label, record in build.get("inputs", {}).items():
        require(isinstance(record, dict), "invalid build input {}".format(label))
        current = digest(Path(str(record.get("path", ""))))
        require(
            current["nbytes"] == record.get("bytes")
            and current["sha256"] == record.get("sha256"),
            "build input {} changed after compilation".format(label),
        )

    comparison_path = run_dir / "full_compare.json"
    comparison = read_json(comparison_path, "full comparison")
    require(
        comparison.get("pass") is True
        and comparison.get("source", {}).get("pass") is True
        and comparison.get("blob_pass") is True
        and comparison.get("generated_header", {}).get("pass") is True
        and comparison.get("header_pass") is True
        and comparison.get("direct_output_required") is True
        and comparison.get("checkpoint_pass") is True,
        "full comparison is not strict PASS",
    )
    checkpoints = comparison.get("checkpoints")
    require(
        isinstance(checkpoints, list)
        and len(checkpoints) == 16
        and all(item.get("pass") is True for item in checkpoints),
        "not all 16 checkpoints pass",
    )
    final = comparison.get("final_output", {})
    selection_validation = final.get("selection_validation", {})
    require(
        final.get("pass") is True
        and final.get("tensor") == "output0"
        and final.get("shape") == [1, 300, 6]
        and final.get("direct_ort_pass") is True
        and final.get("direct_ort_mismatch_count") == 0
        and final.get("unexplained_mismatch_count") == 0
        and selection_validation.get("selected_anchor_overlap") == 300
        and selection_validation.get("final_anchor_class_overlap") == 300,
        "output0 is not a direct zero-mismatch ORT pass",
    )

    pmc_path = run_dir / "pmc_stages.json"
    pmc = read_json(pmc_path, "PMC stages")
    require(
        pmc.get("status") == "PASS" and pmc.get("selector") == "N000:N307",
        "PMC aggregate is not PASS",
    )
    stages = pmc.get("stages")
    require(
        isinstance(stages, list) and len(stages) == len(EXPECTED_STAGES),
        "PMC stage count differs",
    )
    compact_stages: List[Dict[str, Any]] = []
    for item, expected in zip(stages, EXPECTED_STAGES):
        name, first, last = expected
        require(
            item.get("name") == name
            and item.get("first_node") == first
            and item.get("last_node") == last,
            "PMC stage boundary differs for {}".format(name),
        )
        decoded = item.get("decoded", {})
        require(
            decoded.get("status") == "PASS"
            and decoded.get("header", {}).get("status") == "PASS"
            and decoded.get("header", {}).get("active_harts") == 1,
            "PMC stage {} did not decode cleanly".format(name),
        )
        harts = decoded.get("harts")
        require(
            isinstance(harts, list)
            and len(harts) == 1
            and harts[0].get("status") == "PASS",
            "PMC hart record differs for {}".format(name),
        )
        counters = {
            counter["event"]: counter["delta"]
            for counter in harts[0].get("counters", [])
            if counter.get("status") == "PASS"
        }
        required_counters = (
            "minion_cycles",
            "retired_instructions_thread_0",
            "l2_miss_requests",
            "minion_icache_requests",
        )
        require(
            all(key in counters for key in required_counters),
            "PMC stage {} lacks required counters".format(name),
        )
        compact_stages.append(
            {
                "name": name,
                "first_node": first,
                "last_node": last,
                "minion_cycles": counters["minion_cycles"],
                "retired_instructions_thread_0": counters[
                    "retired_instructions_thread_0"
                ],
                "l2_miss_requests": counters["l2_miss_requests"],
                "minion_icache_requests": counters["minion_icache_requests"],
            }
        )

    detections_path = run_dir / "detections.json"
    detections = read_json(detections_path, "detection explanation")
    require(
        detections.get("pass") is True
        and detections.get("records_at_or_above_threshold") == 10
        and len(detections.get("displayed", [])) == 10,
        "real-image detection explanation is not PASS",
    )

    artifact_paths = {
        "full_manifest": manifest_path,
        "generated_header": full_dir / "slice_manifest.h",
        "elf": run_dir / "slice.elf",
        "build_record": build_path,
        "dump": run_dir / "dump.bin",
        "run_log": log_path,
        "run_result": run_result_path,
        "device_evidence": evidence_path,
        "full_compare": comparison_path,
        "pmc_stages": pmc_path,
        "detections": detections_path,
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "kind": "strict_full_graph_real_etsoc1_evidence",
        "source": EXPECTED_SOURCE,
        "selection": {
            "selector": "N000:N307",
            "node_count": 308,
            "output": {"name": "output0", "shape": [1, 300, 6]},
        },
        "device": {
            "launcher_device": "soc1sim",
            "hardware": True,
            "runtime_backend": "DevicePcie",
            "architecture": "ETSOC1",
            "pcie_id": "1e0a:eb01",
            "kernel_wait_seconds": kernel_wait_seconds,
            "launcher_elapsed_seconds": run_result.get("elapsed_seconds"),
        },
        "compiler": compiler,
        "comparison": {
            "checkpoint_count": 16,
            "checkpoint_pass_count": 16,
            "direct_output_required": True,
            "output_mismatch_count": 0,
            "output_unexplained_mismatch_count": 0,
            "output_max_abs": final.get("max_abs"),
            "output_max_rel": final.get("max_rel"),
            "selected_anchor_overlap": 300,
            "final_anchor_class_overlap": 300,
            "actual_output_sha256": final.get("actual_sha256"),
            "reference_output_sha256": final.get("reference_sha256"),
        },
        "real_image": {
            "fixture": "coco_room_000139",
            "input_sha256": package_blobs["inputs"]["sha256"],
            "display_threshold": detections.get("threshold"),
            "records_at_or_above_threshold": 10,
            "top_record": detections["displayed"][0],
            "class_labels": (
                "numeric IDs only; the pinned ONNX does not provide human class names"
            ),
            "postprocessing": ("two-stage in-graph TopK/GatherElements; no NMS node"),
        },
        "pmc": {
            "status": "PASS",
            "scope": pmc.get("scope"),
            "stages": compact_stages,
        },
        "package_blobs": package_blobs,
        "artifacts": {name: digest(path) for name, path in artifact_paths.items()},
    }
    summary["artifacts"]["instrumented_onnx"] = instrumented_identity
    summary["artifacts"]["launcher"] = digest(run_artifact_paths["launcher"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "BOARD_SUMMARY PASS selector=N000:N307 output_mismatches=0 "
        "pmc_stages=7 out={}".format(output)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SummaryError, OSError, ValueError, KeyError) as error:
        print("BOARD_SUMMARY FAIL {}".format(error), file=sys.stderr)
        sys.exit(2)
