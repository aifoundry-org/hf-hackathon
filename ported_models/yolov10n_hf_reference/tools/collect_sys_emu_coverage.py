#!/usr/bin/env python3
"""Fail-closed coverage ledger for the canonical YOLOv10n sys-emu matrix.

The ledger is intentionally independent of the launcher and tensor comparator.
It re-opens their immutable evidence, verifies the hashes that join each step,
and only reports PASS when every planned range is present and the passing
ranges form the exact, gap-free N000:N307 set.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_PLAN = PORT_ROOT / "manifests/sys_emu_coverage_plan.json"
DEFAULT_RANGES_ROOT = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/ranges_v2"
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu"
)
DEFAULT_COMPACT_SUMMARY = PORT_ROOT / "manifests/sys_emu_coverage_summary.json"
EXPECTED_SOURCE = {
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "sha256": "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b",
}
EXPECTED_LICENSE = "AGPL-3.0"
EXPECTED_NODE_COUNT = 308
RUNTIME_FILES = {
    "runtime_c": PORT_ROOT / "src/ref_runtime.c",
    "runtime_h": PORT_ROOT / "src/ref_runtime.h",
    "pmc_h": PORT_ROOT / "src/ref_pmc.h",
    "et_runner_c": PORT_ROOT / "src/et_slice_runner.c",
}


class LedgerError(RuntimeError):
    """The matrix plan or command-line contract is invalid."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def safe_json(path: Path, errors: List[str], label: str) -> Optional[Any]:
    if not path.is_file():
        errors.append("{} missing: {}".format(label, path))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append("{} is not valid JSON: {} ({})".format(label, path, error))
        return None


def check(
    condition: bool,
    errors: List[str],
    message: str,
) -> bool:
    if not condition:
        errors.append(message)
        return False
    return True


def parse_node_id(value: Any) -> Optional[int]:
    if (
        isinstance(value, str)
        and len(value) == 4
        and value.startswith("N")
        and value[1:].isdigit()
    ):
        return int(value[1:])
    return None


def validate_plan(plan_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LedgerError("cannot read matrix plan {}: {}".format(plan_path, error))
    if plan.get("schema_version") != 1:
        raise LedgerError("matrix plan schema_version must be 1")
    if plan.get("plan_kind") != "gap_free_schema_v2_sys_emu_ranges":
        raise LedgerError("unexpected matrix plan kind")
    source = plan.get("source")
    if not isinstance(source, dict) or any(
        source.get(field) != expected for field, expected in EXPECTED_SOURCE.items()
    ):
        raise LedgerError("matrix plan does not pin the expected ONNX artifact")
    ranges = plan.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise LedgerError("matrix plan ranges must be a non-empty list")

    visited: List[int] = []
    names = set()
    for expected_index, record in enumerate(ranges):
        if not isinstance(record, dict):
            raise LedgerError("range {} is not an object".format(expected_index))
        first = record.get("first_node")
        last = record.get("last_node")
        name = record.get("name")
        selector = record.get("selector")
        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(last, int)
            or isinstance(last, bool)
            or first > last
        ):
            raise LedgerError("range {} has invalid bounds".format(expected_index))
        canonical_name = "n{:03d}_n{:03d}".format(first, last)
        canonical_selector = "N{:03d}:N{:03d}".format(first, last)
        if record.get("range_index") != expected_index:
            raise LedgerError("range indexes are not canonical and sequential")
        if name != canonical_name or selector != canonical_selector:
            raise LedgerError(
                "range {} name/selector is not canonical".format(expected_index)
            )
        if record.get("node_count") != last - first + 1:
            raise LedgerError("range {} node count is inconsistent".format(name))
        if name in names:
            raise LedgerError("duplicate range name: {}".format(name))
        names.add(name)
        visited.extend(range(first, last + 1))

    if visited != list(range(EXPECTED_NODE_COUNT)):
        raise LedgerError("matrix plan is not exact gap-free N000:N307 coverage")
    coverage = plan.get("coverage")
    if not isinstance(coverage, dict):
        raise LedgerError("matrix plan coverage summary is missing")
    expected_coverage = {
        "first_node": "N000",
        "last_node": "N307",
        "node_count": EXPECTED_NODE_COUNT,
        "range_count": len(ranges),
        "gap_count": 0,
        "overlap_count": 0,
    }
    if any(
        coverage.get(field) != expected for field, expected in expected_coverage.items()
    ):
        raise LedgerError("matrix plan coverage summary is inconsistent")
    return plan, ranges


def record_matches(
    record: Any,
    actual: Dict[str, Any],
    errors: List[str],
    label: str,
) -> bool:
    expected_bytes = (
        record.get("bytes")
        if isinstance(record, dict) and "bytes" in record
        else record.get("nbytes")
        if isinstance(record, dict)
        else None
    )
    return check(
        isinstance(record, dict)
        and expected_bytes == actual["bytes"]
        and record.get("sha256") == actual["sha256"],
        errors,
        "{} identity does not match its recorded bytes/SHA-256".format(label),
    )


def file_record(
    path: Path,
    errors: List[str],
    label: str,
    expected: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        errors.append("{} missing: {}".format(label, path))
        return None
    actual = identity(path)
    if expected is not None:
        record_matches(expected, actual, errors, label)
    return actual


def package_file(
    package_dir: Path,
    relative: Any,
    errors: List[str],
    label: str,
) -> Optional[Path]:
    if not isinstance(relative, str) or not relative:
        errors.append("{} path is invalid".format(label))
        return None
    path = (package_dir / relative).resolve()
    try:
        path.relative_to(package_dir.resolve())
    except ValueError:
        errors.append("{} escapes its package: {}".format(label, relative))
        return None
    return path


def audit_package(
    planned: Mapping[str, Any],
    package_dir: Path,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    manifest_path = package_dir / "slice_manifest.json"
    manifest = safe_json(manifest_path, errors, "range manifest")
    artifacts: Dict[str, Any] = {}
    if manifest_path.is_file():
        artifacts["manifest"] = identity(manifest_path)
    if not isinstance(manifest, dict):
        return artifacts, None, errors

    check(manifest.get("schema_version") == 2, errors, "manifest schema is not 2")
    check(
        manifest.get("manifest_kind") == "contiguous_node_range",
        errors,
        "manifest kind is not contiguous_node_range",
    )
    source = manifest.get("source")
    check(isinstance(source, dict), errors, "manifest source is missing")
    if isinstance(source, dict):
        for field, expected in EXPECTED_SOURCE.items():
            check(
                source.get(field) == expected,
                errors,
                "manifest source {} is not pinned value".format(field),
            )
        check(
            source.get("license") == EXPECTED_LICENSE,
            errors,
            "manifest source license is not AGPL-3.0",
        )

    selection = manifest.get("selection")
    check(isinstance(selection, dict), errors, "manifest selection is missing")
    if isinstance(selection, dict):
        expected_selection = {
            "selector": planned["selector"],
            "first_node": "N{:03d}".format(planned["first_node"]),
            "last_node": "N{:03d}".format(planned["last_node"]),
            "node_count": planned["node_count"],
        }
        for field, expected in expected_selection.items():
            check(
                selection.get(field) == expected,
                errors,
                "manifest selection {} differs from plan".format(field),
            )

    nodes = manifest.get("nodes")
    check(
        isinstance(nodes, list) and len(nodes) == planned["node_count"],
        errors,
        "manifest node list does not cover the planned range",
    )
    expected_outputs = 0
    if isinstance(nodes, list):
        for local_index, node in enumerate(nodes):
            global_index = planned["first_node"] + local_index
            if not isinstance(node, dict):
                errors.append("manifest node {} is not an object".format(local_index))
                continue
            check(
                node.get("node_id") == "N{:03d}".format(global_index)
                and node.get("index") == global_index
                and node.get("local_index") == local_index,
                errors,
                "manifest node {} ordinal is inconsistent".format(local_index),
            )
            outputs = node.get("outputs")
            if isinstance(outputs, list):
                expected_outputs += len(outputs)
            else:
                errors.append(
                    "manifest node {} outputs are invalid".format(local_index)
                )

    outputs = manifest.get("outputs")
    check(
        isinstance(outputs, list) and len(outputs) == expected_outputs,
        errors,
        "manifest does not retain every selected node output",
    )
    if isinstance(outputs, list):
        output_ids = [
            record.get("output_id") for record in outputs if isinstance(record, dict)
        ]
        check(
            len(output_ids) == len(set(output_ids)) == expected_outputs,
            errors,
            "manifest output IDs are missing or duplicated",
        )

    pmc_stages = manifest.get("pmc_stages")
    check(
        isinstance(pmc_stages, list) and len(pmc_stages) == 1,
        errors,
        "manifest must define exactly one PMC stage",
    )
    if isinstance(pmc_stages, list) and len(pmc_stages) == 1:
        stage = pmc_stages[0]
        check(
            isinstance(stage, dict)
            and stage.get("first_node") == "N{:03d}".format(planned["first_node"])
            and stage.get("last_node") == "N{:03d}".format(planned["last_node"]),
            errors,
            "PMC stage does not wrap exactly the selected range",
        )

    generated = manifest.get("generated")
    if not isinstance(generated, dict) or not isinstance(generated.get("header"), dict):
        errors.append("generated header identity is missing")
    else:
        header_record = generated["header"]
        header_path = package_file(
            package_dir, header_record.get("path"), errors, "generated header"
        )
        if header_path is not None:
            actual = file_record(header_path, errors, "generated header", header_record)
            if actual is not None:
                artifacts["header"] = actual

    blobs = manifest.get("blobs")
    if not isinstance(blobs, dict):
        errors.append("manifest blobs are missing")
    else:
        for name in ("inputs", "weights", "goldens"):
            blob = blobs.get(name)
            if not isinstance(blob, dict):
                errors.append("{} blob record is missing".format(name))
                continue
            path = package_file(
                package_dir, blob.get("path"), errors, "{} blob".format(name)
            )
            if path is not None:
                actual = file_record(path, errors, "{} blob".format(name), blob)
                if actual is not None:
                    artifacts[name] = actual

    if isinstance(source, dict):
        instrumented = package_file(
            package_dir,
            source.get("instrumented_path"),
            errors,
            "instrumented ONNX",
        )
        if instrumented is not None:
            actual = file_record(instrumented, errors, "instrumented ONNX")
            if actual is not None:
                artifacts["instrumented_onnx"] = actual
                check(
                    actual["sha256"] == source.get("instrumented_sha256"),
                    errors,
                    "instrumented ONNX SHA-256 differs from manifest",
                )

    memory = manifest.get("memory_map")
    check(isinstance(memory, dict), errors, "memory map is missing")
    if isinstance(memory, dict) and isinstance(outputs, list):
        check(
            memory.get("workspace_bytes")
            == manifest.get("memory_plan", {}).get("workspace_bytes"),
            errors,
            "workspace bytes differ between memory records",
        )
        check(
            manifest.get("memory_plan", {}).get("checkpoint_count") == len(outputs),
            errors,
            "memory-plan checkpoint count differs from output count",
        )

    artifacts["package_dir"] = str(package_dir.resolve())
    artifacts["expected_output_count"] = expected_outputs
    return artifacts, manifest, errors


def audit_json_identity(
    path: Path,
    errors: List[str],
    label: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    value = safe_json(path, errors, label)
    actual = identity(path) if path.is_file() else None
    if value is not None and not isinstance(value, dict):
        errors.append("{} root must be an object".format(label))
        value = None
    return value, actual


def audit_run(
    planned: Mapping[str, Any],
    package_dir: Path,
    package_manifest: Optional[Mapping[str, Any]],
    package_artifacts: Mapping[str, Any],
    run_dir: Path,
    current_runtime: Mapping[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "artifacts": {},
    }
    if not run_dir.is_dir():
        errors.append("run directory missing: {}".format(run_dir))
        return summary, errors
    if not isinstance(package_manifest, Mapping):
        errors.append("cannot audit run without a valid canonical package")
        return summary, errors

    paths = {
        "run_result": run_dir / "run_result.json",
        "tensor_compare": run_dir / "tensor_compare.json",
        "pmc": run_dir / "pmc.json",
        "elf": run_dir / "slice.elf",
        "build_record": run_dir / "build_record.json",
        "dump": run_dir / "dump.bin",
        "log": run_dir / "run.log",
        "command": run_dir / "command.txt",
        "wrapper_command": run_dir / "wrapper_command.txt",
        "environment": run_dir / "environment.txt",
        "device_evidence": run_dir / "device_evidence.txt",
    }
    for name, path in paths.items():
        actual = file_record(path, errors, name)
        if actual is not None:
            summary["artifacts"][name] = actual

    run_result, _ = audit_json_identity(paths["run_result"], errors, "run result")
    compare, _ = audit_json_identity(
        paths["tensor_compare"], errors, "tensor comparison"
    )
    pmc, _ = audit_json_identity(paths["pmc"], errors, "PMC report")
    build, _ = audit_json_identity(paths["build_record"], errors, "build record")

    if isinstance(run_result, dict):
        required_true = (
            "launcher_identity_match",
            "completion_log_match",
            "dump_log_match",
            "dump_size_match",
            "board_reset_match",
        )
        check(run_result.get("schema_version") == 1, errors, "run schema is not 1")
        check(run_result.get("status") == "pass", errors, "run status is not pass")
        check(
            run_result.get("device") == "sys_emu",
            errors,
            "run device is not sys_emu",
        )
        check(
            run_result.get("hardware") is False,
            errors,
            "sys-emu run is incorrectly marked as hardware",
        )
        check(
            run_result.get("return_code") == 0,
            errors,
            "launcher return code is not zero",
        )
        for field in required_true:
            check(
                run_result.get(field) is True,
                errors,
                "run evidence flag {} is not true".format(field),
            )
        check(
            run_result.get("source_sha256") == EXPECTED_SOURCE["sha256"],
            errors,
            "run source SHA-256 is not pinned",
        )
        selection = run_result.get("selection")
        check(isinstance(selection, dict), errors, "run selection is missing")
        if isinstance(selection, dict):
            for field, expected in {
                "selector": planned["selector"],
                "first_node": "N{:03d}".format(planned["first_node"]),
                "last_node": "N{:03d}".format(planned["last_node"]),
                "node_count": planned["node_count"],
            }.items():
                check(
                    selection.get(field) == expected,
                    errors,
                    "run selection {} differs from plan".format(field),
                )

        run_artifacts = run_result.get("artifacts")
        check(
            isinstance(run_artifacts, dict),
            errors,
            "run artifact identity table is missing",
        )
        if isinstance(run_artifacts, dict):
            local_map = {
                "elf": summary["artifacts"].get("elf"),
                "build_record": summary["artifacts"].get("build_record"),
                "dump": summary["artifacts"].get("dump"),
                "log": summary["artifacts"].get("log"),
                "command": summary["artifacts"].get("command"),
                "wrapper_command": summary["artifacts"].get("wrapper_command"),
                "environment": summary["artifacts"].get("environment"),
                "device_evidence": summary["artifacts"].get("device_evidence"),
                "slice_manifest": package_artifacts.get("manifest"),
                "inputs": package_artifacts.get("inputs"),
                "weights": package_artifacts.get("weights"),
            }
            for name, actual in local_map.items():
                if actual is not None:
                    record_matches(
                        run_artifacts.get(name),
                        actual,
                        errors,
                        "run artifact {}".format(name),
                    )
            check(
                run_artifacts.get("board_lock") is None,
                errors,
                "sys-emu run unexpectedly records a board lock",
            )
            launcher_record = run_artifacts.get("launcher")
            if isinstance(launcher_record, dict):
                launcher_path_value = launcher_record.get("path")
                launcher_path = (
                    Path(launcher_path_value)
                    if isinstance(launcher_path_value, str)
                    else Path("")
                )
                if launcher_path.is_file():
                    launcher_actual = identity(launcher_path)
                    record_matches(launcher_record, launcher_actual, errors, "launcher")
                    summary["artifacts"]["launcher"] = launcher_actual
                else:
                    errors.append(
                        "recorded launcher is unavailable: {}".format(
                            launcher_path_value
                        )
                    )
            else:
                errors.append("run result lacks launcher identity")
        elapsed = run_result.get("elapsed_seconds")
        summary["elapsed_seconds"] = elapsed

    if paths["device_evidence"].is_file():
        evidence = paths["device_evidence"].read_text(
            encoding="utf-8", errors="replace"
        )
        check(
            "backend=sys_emu" in evidence
            and "meaning=software system emulator" in evidence,
            errors,
            "device evidence does not identify the software system emulator",
        )
        check(
            "real PCIe ET-SoC1 hardware" not in evidence,
            errors,
            "sys-emu evidence contains a hardware backend claim",
        )
    if paths["environment"].is_file():
        environment = paths["environment"].read_text(encoding="utf-8", errors="replace")
        check(
            "device=sys_emu" in environment,
            errors,
            "environment evidence does not record device=sys_emu",
        )

    if isinstance(build, dict):
        check(build.get("schema_version") == 1, errors, "build schema is not 1")
        saved_elf = summary["artifacts"].get("elf")
        if saved_elf is not None:
            record_matches(build.get("elf"), saved_elf, errors, "built ELF")
        build_inputs = build.get("inputs")
        check(
            isinstance(build_inputs, dict),
            errors,
            "build input identity table is missing",
        )
        if isinstance(build_inputs, dict):
            header = package_artifacts.get("header")
            if header is not None:
                record_matches(
                    build_inputs.get("slice_manifest_header"),
                    header,
                    errors,
                    "build manifest header",
                )
            for name, actual in current_runtime.items():
                record_matches(
                    build_inputs.get(name),
                    actual,
                    errors,
                    "build input {}".format(name),
                )
            for name in ("linker_script", "layout", "crt"):
                record = build_inputs.get(name)
                if not isinstance(record, dict):
                    errors.append("build input {} record is missing".format(name))
                    continue
                path_value = record.get("path")
                path = Path(path_value) if isinstance(path_value, str) else Path("")
                if not path.is_file():
                    errors.append(
                        "build input {} is unavailable: {}".format(name, path_value)
                    )
                    continue
                actual = identity(path)
                record_matches(record, actual, errors, "build input {}".format(name))
                summary["artifacts"]["build_input_" + name] = actual
        compiler = build.get("compiler")
        check(
            isinstance(compiler, dict)
            and isinstance(compiler.get("version"), str)
            and bool(compiler.get("version")),
            errors,
            "compiler provenance is incomplete",
        )
        if isinstance(compiler, dict):
            summary["compiler"] = compiler

    expected_output_count = package_artifacts.get("expected_output_count")
    if isinstance(compare, dict):
        check(
            compare.get("schema_version") == 1
            and compare.get("kind") == "contiguous_range_comparison",
            errors,
            "tensor comparison schema/kind is invalid",
        )
        check(compare.get("pass") is True, errors, "tensor comparison did not pass")
        check(
            compare.get("selector") == planned["selector"],
            errors,
            "tensor comparison selector differs from plan",
        )
        source = compare.get("source")
        check(
            isinstance(source, dict)
            and source.get("pass") is True
            and source.get("actual_sha256") == EXPECTED_SOURCE["sha256"]
            and source.get("expected_sha256") == EXPECTED_SOURCE["sha256"],
            errors,
            "tensor comparison source verification is not pinned PASS",
        )
        manifest_report = compare.get("manifest")
        if (
            isinstance(manifest_report, dict)
            and package_artifacts.get("manifest") is not None
        ):
            check(
                manifest_report.get("sha256")
                == package_artifacts["manifest"]["sha256"],
                errors,
                "tensor comparison used a different manifest",
            )
        else:
            errors.append("tensor comparison manifest identity is missing")
        compare_dump = compare.get("dump")
        actual_dump = summary["artifacts"].get("dump")
        if actual_dump is not None:
            record_matches(compare_dump, actual_dump, errors, "compared dump")
        summary_record = compare.get("summary")
        check(
            isinstance(summary_record, dict),
            errors,
            "tensor comparison summary is missing",
        )
        if isinstance(summary_record, dict):
            check(
                summary_record.get("node_count") == planned["node_count"],
                errors,
                "comparison node count differs from plan",
            )
            check(
                summary_record.get("output_count") == expected_output_count
                and summary_record.get("passed_outputs") == expected_output_count
                and summary_record.get("failed_outputs") == 0,
                errors,
                "comparison did not pass every selected node output",
            )
            summary["comparison"] = summary_record
        outputs = compare.get("outputs")
        expected_records = package_manifest.get("outputs", [])
        expected_ids = [
            record.get("output_id")
            for record in expected_records
            if isinstance(record, dict)
        ]
        actual_ids = (
            [record.get("output_id") for record in outputs if isinstance(record, dict)]
            if isinstance(outputs, list)
            else []
        )
        check(
            actual_ids == expected_ids
            and all(
                isinstance(record, dict) and record.get("pass") is True
                for record in (outputs if isinstance(outputs, list) else [])
            ),
            errors,
            "comparison output records are incomplete, reordered, or failing",
        )
        result_header = compare.get("result_header")
        check(
            isinstance(result_header, dict)
            and result_header.get("status") == 0
            and result_header.get("first_node") == planned["first_node"]
            and result_header.get("last_node") == planned["last_node"]
            and result_header.get("node_count") == planned["node_count"],
            errors,
            "device result header does not prove the selected node span passed",
        )

    if isinstance(pmc, dict):
        check(pmc.get("status") == "PASS", errors, "PMC report status is not PASS")
        header = pmc.get("header")
        check(
            isinstance(header, dict)
            and header.get("status") == "PASS"
            and isinstance(header.get("active_harts"), int)
            and header.get("active_harts", 0) >= 1,
            errors,
            "PMC header is not a valid active PASS record",
        )
        memory = package_manifest.get("memory_map")
        if isinstance(header, dict) and isinstance(memory, dict):
            check(
                header.get("record_offset") == memory.get("pmc_device_offset"),
                errors,
                "PMC record offset differs from the range memory map",
            )
        harts = pmc.get("harts")
        check(
            isinstance(harts, list)
            and bool(harts)
            and all(
                isinstance(record, dict) and record.get("status") == "PASS"
                for record in harts
            ),
            errors,
            "one or more PMC hart records failed",
        )
        shared = pmc.get("shared")
        check(
            isinstance(shared, dict) and shared.get("status") == "PASS",
            errors,
            "PMC shared-counter record failed",
        )
        if isinstance(header, dict) and isinstance(harts, list):
            summary["pmc"] = {
                "active_harts": header.get("active_harts"),
                "harts": len(harts),
                "status": pmc.get("status"),
            }

    summary["pass"] = not errors
    return summary, errors


def load_result_map(path: Optional[Path]) -> Dict[str, Path]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LedgerError("cannot read result map {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise LedgerError("result map root must be an object")
    result: Dict[str, Path] = {}
    for name, raw_path in value.items():
        if not isinstance(name, str) or not isinstance(raw_path, str) or not raw_path:
            raise LedgerError("result map entries must be string paths")
        result[name] = Path(raw_path)
    return result


def coverage_summary(
    ranges: Sequence[Mapping[str, Any]],
    passing_names: Iterable[str],
) -> Dict[str, Any]:
    passing = set(passing_names)
    counts = [0] * EXPECTED_NODE_COUNT
    for planned in ranges:
        if planned["name"] not in passing:
            continue
        for node in range(planned["first_node"], planned["last_node"] + 1):
            if 0 <= node < EXPECTED_NODE_COUNT:
                counts[node] += 1
    covered = [index for index, count in enumerate(counts) if count > 0]
    gaps = [index for index, count in enumerate(counts) if count == 0]
    overlaps = [index for index, count in enumerate(counts) if count > 1]
    return {
        "expected_first_node": "N000",
        "expected_last_node": "N307",
        "expected_node_count": EXPECTED_NODE_COUNT,
        "passing_range_count": len(passing),
        "covered_node_count": len(covered),
        "covered_first_node": ("N{:03d}".format(covered[0]) if covered else None),
        "covered_last_node": ("N{:03d}".format(covered[-1]) if covered else None),
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "gaps": ["N{:03d}".format(index) for index in gaps],
        "overlaps": ["N{:03d}".format(index) for index in overlaps],
        "pass": counts == [1] * EXPECTED_NODE_COUNT,
    }


def collect(
    plan_path: Path,
    ranges_root: Path,
    results_root: Path,
    result_map: Mapping[str, Path],
) -> Dict[str, Any]:
    plan, ranges = validate_plan(plan_path)
    current_runtime = {name: identity(path) for name, path in RUNTIME_FILES.items()}
    rows: List[Dict[str, Any]] = []
    passing_names: List[str] = []

    for planned in ranges:
        name = planned["name"]
        package_dir = ranges_root / name
        package_artifacts, manifest, package_errors = audit_package(
            planned, package_dir
        )
        run_dir = result_map.get(name, results_root / name)
        run_summary, run_errors = audit_run(
            planned,
            package_dir,
            manifest,
            package_artifacts,
            run_dir,
            current_runtime,
        )
        errors = package_errors + run_errors
        passed = not errors
        if passed:
            passing_names.append(name)
        rows.append(
            {
                "range_index": planned["range_index"],
                "name": name,
                "selector": planned["selector"],
                "first_node": planned["first_node"],
                "last_node": planned["last_node"],
                "node_count": planned["node_count"],
                "pass": passed,
                "errors": errors,
                "package": package_artifacts,
                "execution": run_summary,
            }
        )

    coverage = coverage_summary(ranges, passing_names)
    passed = (
        coverage["pass"]
        and len(passing_names) == len(ranges)
        and all(row["pass"] for row in rows)
    )
    return {
        "schema_version": 1,
        "kind": "yolov10n_hf_sys_emu_complete_coverage_ledger",
        "generated_utc": utc_now(),
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "device": "sys_emu",
        "hardware": False,
        "source": dict(EXPECTED_SOURCE),
        "license": EXPECTED_LICENSE,
        "identities": {
            "collector": identity(Path(__file__)),
            "plan": identity(plan_path),
            "current_runtime": current_runtime,
        },
        "roots": {
            "ranges": str(ranges_root.resolve()),
            "results": str(results_root.resolve()),
        },
        "plan": {
            "kind": plan["plan_kind"],
            "range_count": len(ranges),
            "coverage": plan["coverage"],
        },
        "coverage": coverage,
        "summary": {
            "planned_ranges": len(ranges),
            "passing_ranges": len(passing_names),
            "failing_ranges": len(ranges) - len(passing_names),
            "all_selected_outputs_compared": passed,
            "all_pmc_records_pass": passed,
            "all_device_identities_sys_emu": passed,
        },
        "ranges": rows,
    }


def markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    summary = report["summary"]
    lines = [
        "# YOLOv10n HF system-emulator coverage ledger",
        "",
        "- Status: **{}**".format(report["status"]),
        "- Device: `sys_emu` (software emulator; hardware=false)",
        "- Pinned ONNX SHA-256: `{}`".format(report["source"]["sha256"]),
        "- Plan SHA-256: `{}`".format(report["identities"]["plan"]["sha256"]),
        "- Collector SHA-256: `{}`".format(report["identities"]["collector"]["sha256"]),
        "- Passing ranges: {}/{}".format(
            summary["passing_ranges"], summary["planned_ranges"]
        ),
        "- Covered nodes: {}/{}; gaps={}; overlaps={}".format(
            coverage["covered_node_count"],
            coverage["expected_node_count"],
            coverage["gap_count"],
            coverage["overlap_count"],
        ),
        "",
        "| Range | Nodes | Package | Execution | Outputs | PMC | ELF SHA-256 |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report["ranges"]:
        execution = row["execution"]
        artifacts = execution.get("artifacts", {})
        compare = execution.get("comparison", {})
        pmc = execution.get("pmc", {})
        package_status = (
            "PASS"
            if row.get("package")
            and not any(
                message.startswith("manifest") or "blob" in message
                for message in row["errors"]
            )
            else "FAIL"
        )
        execution_status = "PASS" if row["pass"] else "FAIL"
        output_text = "{}/{}".format(
            compare.get("passed_outputs", 0),
            compare.get("output_count", 0),
        )
        elf_sha = artifacts.get("elf", {}).get("sha256", "-")
        if elf_sha != "-":
            elf_sha = "`{}`".format(elf_sha)
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["selector"],
                row["node_count"],
                package_status,
                execution_status,
                output_text,
                pmc.get("status", "FAIL"),
                elf_sha,
            )
        )
    failures = [row for row in report["ranges"] if row["errors"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append("- `{}`:".format(row["selector"]))
            for message in row["errors"]:
                lines.append("  - {}".format(message))
    else:
        lines.extend(
            [
                "",
                "Every selected node output passed its schema-v2 comparison, "
                "every range has a valid PMC record, and the passing ranges form "
                "the exact gap-free `N000:N307` set.",
            ]
        )
    return "\n".join(lines) + "\n"


def compact_summary(
    report: Mapping[str, Any],
    detailed_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a relocatable, reviewable summary for the tracked manifest tree."""
    ranges = []
    for row in report["ranges"]:
        execution = row["execution"]
        artifacts = execution.get("artifacts", {})
        comparison = execution.get("comparison", {})
        pmc = execution.get("pmc", {})
        ranges.append(
            {
                "range_index": row["range_index"],
                "name": row["name"],
                "selector": row["selector"],
                "node_count": row["node_count"],
                "status": "PASS" if row["pass"] else "FAIL",
                "elapsed_seconds": execution.get("elapsed_seconds"),
                "output_count": comparison.get("output_count"),
                "passed_outputs": comparison.get("passed_outputs"),
                "total_elements": comparison.get("total_elements"),
                "total_mismatches": comparison.get("total_mismatches"),
                "pmc_status": pmc.get("status"),
                "pmc_active_harts": pmc.get("active_harts"),
                "sha256": {
                    name: artifacts.get(name, {}).get("sha256")
                    for name in (
                        "run_result",
                        "tensor_compare",
                        "pmc",
                        "elf",
                        "build_record",
                        "dump",
                        "launcher",
                    )
                },
                "package_manifest_sha256": row.get("package", {})
                .get("manifest", {})
                .get("sha256"),
            }
        )
    return {
        "schema_version": 1,
        "kind": "yolov10n_hf_sys_emu_complete_coverage_summary",
        "generated_utc": report["generated_utc"],
        "status": report["status"],
        "pass": report["pass"],
        "device": report["device"],
        "hardware": report["hardware"],
        "source": report["source"],
        "license": report["license"],
        "plan_sha256": report["identities"]["plan"]["sha256"],
        "collector_sha256": report["identities"]["collector"]["sha256"],
        "runtime_sha256": {
            name: record["sha256"]
            for name, record in report["identities"]["current_runtime"].items()
        },
        "detailed_ledger": {
            "artifact_policy": "ignored local-artifacts evidence",
            "bytes": detailed_identity["bytes"],
            "sha256": detailed_identity["sha256"],
        },
        "coverage": report["coverage"],
        "summary": report["summary"],
        "ranges": ranges,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--ranges-root", type=Path, default=DEFAULT_RANGES_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--result-map",
        type=Path,
        help="optional JSON object mapping canonical range names to run directories",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="ledger JSON path (default: RESULTS_ROOT/coverage_ledger.json)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="ledger Markdown path (default: RESULTS_ROOT/coverage_ledger.md)",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        help=(
            "compact relocatable PASS summary path; written only after complete "
            "coverage (recommended tracked path: {})"
        ).format(DEFAULT_COMPACT_SUMMARY),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="audit and print status without writing ledger files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result_map = load_result_map(args.result_map)
        report = collect(
            args.plan.resolve(),
            args.ranges_root.resolve(),
            args.results_root.resolve(),
            result_map,
        )
    except LedgerError as error:
        print("COVERAGE_LEDGER ERROR {}".format(error), file=sys.stderr)
        return 2

    if not args.check_only:
        json_path = (
            args.output_json
            if args.output_json is not None
            else args.results_root / "coverage_ledger.json"
        )
        markdown_path = (
            args.output_markdown
            if args.output_markdown is not None
            else args.results_root / "coverage_ledger.md"
        )
        output_paths = [json_path, markdown_path]
        if args.output_summary is not None:
            output_paths.append(args.output_summary)
        for path in output_paths:
            if path.exists():
                print(
                    "COVERAGE_LEDGER ERROR refusing to overwrite {}".format(path),
                    file=sys.stderr,
                )
                return 2
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        json_path.write_text(detailed_text, encoding="utf-8")
        markdown_path.write_text(markdown(report), encoding="utf-8")
        print("ledger_json={}".format(json_path))
        print("ledger_markdown={}".format(markdown_path))
        if args.output_summary is not None:
            if not report["pass"]:
                print(
                    "COVERAGE_LEDGER FAIL compact summary not written because "
                    "coverage is incomplete",
                    file=sys.stderr,
                )
            else:
                args.output_summary.parent.mkdir(parents=True, exist_ok=True)
                detailed_identity = identity(json_path)
                args.output_summary.write_text(
                    json.dumps(
                        compact_summary(report, detailed_identity),
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print("compact_summary={}".format(args.output_summary))

    print(
        "COVERAGE_LEDGER {} ranges={}/{} nodes={}/{} gaps={} overlaps={}".format(
            report["status"],
            report["summary"]["passing_ranges"],
            report["summary"]["planned_ranges"],
            report["coverage"]["covered_node_count"],
            report["coverage"]["expected_node_count"],
            report["coverage"]["gap_count"],
            report["coverage"]["overlap_count"],
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
