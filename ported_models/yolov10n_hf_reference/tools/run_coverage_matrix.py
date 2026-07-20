#!/usr/bin/env python3
"""Orchestrate the canonical schema-v2 host and ET range matrix.

Every action is derived from manifests/sys_emu_coverage_plan.json.  Fresh
artifact roots are mandatory except for explicit sys-emu resume/validation;
failed or partial range directories are never overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import collect_sys_emu_coverage as coverage


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_PLAN = PORT_ROOT / "manifests/sys_emu_coverage_plan.json"
DEFAULT_RANGES_ROOT = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/ranges_v2"
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_PYTHON = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/venv/bin/python"


class MatrixError(RuntimeError):
    """An orchestration contract or subprocess failed."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def compact_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise MatrixError("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise MatrixError("{} {} is not an object".format(label, path))
    return value


def require_fresh_root(path: Path) -> None:
    if path.exists():
        raise MatrixError(
            "artifact root already exists; choose a fresh path: {}".format(path)
        )
    path.mkdir(parents=True)


def select_ranges(
    ranges: Sequence[Mapping[str, Any]], requested: Sequence[str]
) -> List[Mapping[str, Any]]:
    if not requested:
        return list(ranges)
    by_key: Dict[str, Mapping[str, Any]] = {}
    for record in ranges:
        by_key[record["name"]] = record
        by_key[record["selector"]] = record
    selected: List[Mapping[str, Any]] = []
    seen = set()
    for key in requested:
        record = by_key.get(key)
        if record is None:
            raise MatrixError("unknown canonical range: {}".format(key))
        if record["name"] in seen:
            raise MatrixError("range selected more than once: {}".format(key))
        seen.add(record["name"])
        selected.append(record)
    return sorted(selected, key=lambda record: record["range_index"])


def verify_package(
    planned: Mapping[str, Any], ranges_root: Path
) -> Tuple[Path, Dict[str, Any]]:
    package_dir = (ranges_root / planned["name"]).resolve()
    package, manifest, errors = coverage.audit_package(planned, package_dir)
    if errors:
        raise MatrixError(
            "{} package failed preflight: {}".format(planned["name"], "; ".join(errors))
        )
    if manifest is None:
        raise MatrixError("{} package has no manifest".format(planned["name"]))
    return package_dir, manifest


def subprocess_to_log(
    command: Sequence[str],
    log_path: Path,
    environment: Mapping[str, str],
    timeout: int,
) -> Tuple[int, float]:
    start = datetime.datetime.now(datetime.timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if log_path.exists() else "xb"
    with log_path.open(mode) as log:
        log.write(("COMMAND {}\n".format(shlex.join(list(command)))).encode("utf-8"))
        log.flush()
        try:
            result = subprocess.run(
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                timeout=timeout,
                check=False,
            )
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            log.write("MATRIX_TIMEOUT seconds={}\n".format(timeout).encode("utf-8"))
            return_code = 124
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()
    return return_code, elapsed


def run_parallel(
    selected: Sequence[Mapping[str, Any]],
    jobs: int,
    worker: Any,
    action: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for planned in selected:
            print(
                "{} START range={} selector={}".format(
                    action, planned["name"], planned["selector"]
                ),
                flush=True,
            )
            futures[executor.submit(worker, planned)] = planned
        for future in concurrent.futures.as_completed(futures):
            planned = futures[future]
            try:
                record = future.result()
            except Exception as error:  # Preserve all other in-flight evidence.
                record = {
                    "name": planned["name"],
                    "selector": planned["selector"],
                    "status": "FAIL",
                    "error": str(error),
                }
            records.append(record)
            if record.get("status") != "PASS":
                failures += 1
            print(
                "{} {} range={} {}".format(
                    action,
                    record.get("status", "FAIL"),
                    planned["name"],
                    record.get("error", ""),
                ).rstrip(),
                flush=True,
            )
    records.sort(
        key=lambda record: next(
            planned["range_index"]
            for planned in selected
            if planned["name"] == record["name"]
        )
    )
    return records


def command_environment(args: argparse.Namespace) -> Dict[str, str]:
    environment = dict(os.environ)
    if getattr(args, "host_python", None):
        environment["YOLOV10N_HOST_PYTHON"] = str(args.host_python.resolve())
    if getattr(args, "model", None):
        environment["YOLOV10N_MODEL"] = str(args.model.resolve())
    if getattr(args, "et_platform", None):
        environment["ET_PLATFORM"] = str(args.et_platform.resolve())
    if getattr(args, "ld_library_path", None):
        environment["LD_LIBRARY_PATH"] = args.ld_library_path
    return environment


def action_host(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> int:
    output_root = args.output_root.resolve()
    require_fresh_root(output_root)
    logs_root = output_root / "_matrix_logs"
    environment = command_environment(args)
    python = args.host_python.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise MatrixError("host Python is not executable: {}".format(python))
    if not args.model.is_file():
        raise MatrixError("pinned model is missing: {}".format(args.model))

    def worker(planned: Mapping[str, Any]) -> Dict[str, Any]:
        package_dir, _ = verify_package(planned, args.ranges_root)
        result_dir = output_root / planned["name"]
        result_dir.mkdir()
        runner = result_dir / "host_range_v2_runner"
        dump = result_dir / "dump.bin"
        compare = result_dir / "tensor_compare.json"
        log = logs_root / "{}.log".format(planned["name"])
        commands = [
            [
                str(PORT_ROOT / "scripts/build_host_range_v2.sh"),
                str(package_dir),
                str(runner),
            ],
            [
                str(runner),
                str(package_dir / "inputs.bin"),
                str(package_dir / "weights.bin"),
                str(dump),
            ],
            [
                str(python),
                str(PORT_ROOT / "tools/compare_range_v2.py"),
                str(package_dir),
                str(dump),
                "--model",
                str(args.model.resolve()),
                "--json",
                str(compare),
            ],
        ]
        elapsed = 0.0
        for command in commands:
            return_code, command_elapsed = subprocess_to_log(
                command, log, environment, args.command_timeout
            )
            elapsed += command_elapsed
            if return_code != 0:
                raise MatrixError(
                    "command returned {} (log {})".format(return_code, log)
                )
        report = read_json(compare, "host comparison")
        if report.get("pass") is not True:
            raise MatrixError("host comparison did not pass")
        record = {
            "schema_version": 1,
            "status": "PASS",
            "name": planned["name"],
            "selector": planned["selector"],
            "elapsed_seconds": elapsed,
            "source_sha256": coverage.EXPECTED_SOURCE["sha256"],
            "artifacts": {
                "package_manifest": identity(package_dir / "slice_manifest.json"),
                "runner": identity(runner),
                "dump": identity(dump),
                "tensor_compare": identity(compare),
                "log": identity(log),
            },
            "comparison": report["summary"],
        }
        write_new_json(result_dir / "host_result.json", record)
        return record

    try:
        records = run_parallel(selected, args.jobs, worker, "HOST_MATRIX")
        failed = sum(record.get("status") != "PASS" for record in records)
        if failed:
            raise MatrixError(
                "HOST_MATRIX failed for {}/{} ranges".format(failed, len(records))
            )
        status = "PASS"
        return_code = 0
    except MatrixError as error:
        records = locals().get("records", [])
        status = "FAIL"
        return_code = 1
        failure = str(error)
    summary = {
        "schema_version": 1,
        "kind": "schema_v2_host_range_matrix",
        "timestamp_utc": utc_now(),
        "status": status,
        "source": coverage.EXPECTED_SOURCE,
        "plan": identity(args.plan.resolve()),
        "tool": identity(Path(__file__)),
        "selected_ranges": [record["name"] for record in selected],
        "records": records,
    }
    if status != "PASS":
        summary["error"] = failure
    write_new_json(output_root / "host_matrix_summary.json", summary)
    print(
        "HOST_MATRIX {} root={}".format(status, output_root),
        file=sys.stderr if return_code else sys.stdout,
    )
    return return_code


def action_build_et(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> int:
    del plan
    output_root = args.output_root.resolve()
    require_fresh_root(output_root)
    logs_root = output_root / "_matrix_logs"
    environment = command_environment(args)

    def worker(planned: Mapping[str, Any]) -> Dict[str, Any]:
        package_dir, _ = verify_package(planned, args.ranges_root)
        build_dir = output_root / planned["name"]
        build_dir.mkdir()
        elf = build_dir / "yolov10n_hf_range.elf"
        log = logs_root / "{}.log".format(planned["name"])
        command = [
            str(PORT_ROOT / "scripts/build_et_slice.sh"),
            str(package_dir),
            str(elf),
        ]
        return_code, elapsed = subprocess_to_log(
            command, log, environment, args.command_timeout
        )
        if return_code != 0:
            raise MatrixError("ET build returned {} (log {})".format(return_code, log))
        build_record_path = Path(str(elf) + ".build.json")
        build = read_json(build_record_path, "ET build record")
        elf_id = identity(elf)
        recorded_elf = build.get("elf")
        if (
            not isinstance(recorded_elf, dict)
            or recorded_elf.get("bytes") != elf_id["bytes"]
            or recorded_elf.get("sha256") != elf_id["sha256"]
        ):
            raise MatrixError("ET build record does not anchor its ELF")
        record = {
            "schema_version": 1,
            "status": "PASS",
            "name": planned["name"],
            "selector": planned["selector"],
            "elapsed_seconds": elapsed,
            "source_sha256": coverage.EXPECTED_SOURCE["sha256"],
            "artifacts": {
                "package_manifest": identity(package_dir / "slice_manifest.json"),
                "elf": elf_id,
                "build_record": identity(build_record_path),
                "log": identity(log),
            },
            "compiler": build.get("compiler"),
        }
        write_new_json(build_dir / "matrix_build_result.json", record)
        return record

    try:
        records = run_parallel(selected, args.jobs, worker, "ET_BUILD_MATRIX")
        failed = sum(record.get("status") != "PASS" for record in records)
        if failed:
            raise MatrixError(
                "ET_BUILD_MATRIX failed for {}/{} ranges".format(failed, len(records))
            )
        status = "PASS"
        return_code = 0
    except MatrixError as error:
        records = locals().get("records", [])
        status = "FAIL"
        return_code = 1
        failure = str(error)
    summary = {
        "schema_version": 1,
        "kind": "schema_v2_et_build_matrix",
        "timestamp_utc": utc_now(),
        "status": status,
        "source": coverage.EXPECTED_SOURCE,
        "plan": identity(args.plan.resolve()),
        "tool": identity(Path(__file__)),
        "selected_ranges": [record["name"] for record in selected],
        "records": records,
    }
    if status != "PASS":
        summary["error"] = failure
    write_new_json(output_root / "et_build_matrix_summary.json", summary)
    print(
        "ET_BUILD_MATRIX {} root={}".format(status, output_root),
        file=sys.stderr if return_code else sys.stdout,
    )
    return return_code


def verify_et_build(
    planned: Mapping[str, Any], build_root: Path
) -> Tuple[Path, Dict[str, Any]]:
    elf = build_root / planned["name"] / "yolov10n_hf_range.elf"
    build_record_path = Path(str(elf) + ".build.json")
    if not elf.is_file() or not build_record_path.is_file():
        raise MatrixError(
            "{} lacks ELF/build record under {}".format(planned["name"], build_root)
        )
    build = read_json(build_record_path, "ET build record")
    elf_id = identity(elf)
    recorded = build.get("elf")
    if (
        not isinstance(recorded, dict)
        or recorded.get("bytes") != elf_id["bytes"]
        or recorded.get("sha256") != elf_id["sha256"]
    ):
        raise MatrixError("{} ELF differs from build record".format(planned["name"]))
    return elf.resolve(), build


def validation_reports_pass(run_dir: Path) -> bool:
    try:
        run = read_json(run_dir / "run_result.json", "run result")
        compare = read_json(run_dir / "tensor_compare.json", "tensor comparison")
        pmc = read_json(run_dir / "pmc.json", "PMC report")
    except MatrixError:
        return False
    outputs = compare.get("outputs")
    return (
        run.get("status") == "pass"
        and run.get("device") == "sys_emu"
        and run.get("hardware") is False
        and compare.get("pass") is True
        and isinstance(outputs, list)
        and bool(outputs)
        and all(
            isinstance(record, dict) and record.get("pass") is True
            for record in outputs
        )
        and pmc.get("status") == "PASS"
    )


def executable(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise MatrixError("{} is not executable: {}".format(label, path))
    return path


def validate_one(
    planned: Mapping[str, Any],
    package_dir: Path,
    run_dir: Path,
    environment: Mapping[str, str],
    timeout: int,
    log_path: Path,
    reuse_reports: bool,
) -> Dict[str, Any]:
    compare_exists = (run_dir / "tensor_compare.json").exists()
    pmc_exists = (run_dir / "pmc.json").exists()
    if compare_exists != pmc_exists:
        raise MatrixError(
            "partial validation reports already exist in {}; preserving them".format(
                run_dir
            )
        )
    if compare_exists:
        if not reuse_reports:
            raise MatrixError(
                "validation reports already exist in {}; refusing overwrite".format(
                    run_dir
                )
            )
        if not validation_reports_pass(run_dir):
            raise MatrixError("saved validation reports are not PASS")
        return {
            "schema_version": 1,
            "status": "PASS",
            "name": planned["name"],
            "selector": planned["selector"],
            "mode": "verified_saved_reports",
            "artifacts": {
                "run_result": identity(run_dir / "run_result.json"),
                "tensor_compare": identity(run_dir / "tensor_compare.json"),
                "pmc": identity(run_dir / "pmc.json"),
            },
        }
    command = [
        str(PORT_ROOT / "scripts/validate_device_run.sh"),
        str(package_dir),
        str(run_dir),
        "sys_emu",
    ]
    return_code, elapsed = subprocess_to_log(command, log_path, environment, timeout)
    if return_code != 0:
        raise MatrixError(
            "device validation returned {} (log {})".format(return_code, log_path)
        )
    if not validation_reports_pass(run_dir):
        raise MatrixError("device validation did not produce strict PASS reports")
    return {
        "schema_version": 1,
        "status": "PASS",
        "name": planned["name"],
        "selector": planned["selector"],
        "mode": "new_validation",
        "elapsed_seconds": elapsed,
        "artifacts": {
            "run_result": identity(run_dir / "run_result.json"),
            "tensor_compare": identity(run_dir / "tensor_compare.json"),
            "pmc": identity(run_dir / "pmc.json"),
        },
    }


def write_invocation(
    root: Path, kind: str, suffix: str, value: Mapping[str, Any]
) -> Path:
    directory = root / "_matrix_invocations"
    path = directory / "{}_{}_{}.json".format(compact_utc(), kind, suffix)
    counter = 1
    while path.exists():
        path = directory / "{}_{}_{}_{}.json".format(
            compact_utc(), kind, suffix, counter
        )
        counter += 1
    write_new_json(path, value)
    return path


def sys_emu_config(
    args: argparse.Namespace,
    launcher: Path,
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "schema_v2_sys_emu_matrix_config",
        "source": coverage.EXPECTED_SOURCE,
        "plan": identity(args.plan.resolve()),
        "tool": identity(Path(__file__)),
        "ranges_root": str(args.ranges_root.resolve()),
        "build_root": str(args.build_root.resolve()),
        "launcher": identity(launcher),
        "outer_timeout": args.outer_timeout,
        "launcher_timeout": args.launcher_timeout,
        "validation_timeout": args.validation_timeout,
        "ET_PLATFORM": environment.get("ET_PLATFORM"),
        "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH"),
    }


def ensure_sys_emu_root(
    args: argparse.Namespace,
    launcher: Path,
    environment: Mapping[str, str],
) -> Path:
    root = args.output_root.resolve()
    config = sys_emu_config(args, launcher, environment)
    config_path = root / "matrix_config.json"
    if not root.exists():
        root.mkdir(parents=True)
        write_new_json(config_path, config)
        return root
    if not args.resume:
        raise MatrixError(
            "result root exists; choose a fresh path or pass --resume: {}".format(root)
        )
    saved = read_json(config_path, "matrix config")
    stable_fields = (
        "source",
        "plan",
        "ranges_root",
        "build_root",
        "launcher",
        "outer_timeout",
        "launcher_timeout",
        "validation_timeout",
        "ET_PLATFORM",
        "LD_LIBRARY_PATH",
    )
    differences = [
        field for field in stable_fields if saved.get(field) != config.get(field)
    ]
    if differences:
        raise MatrixError(
            "resume configuration changed fields: {}".format(", ".join(differences))
        )
    return root


def action_sys_emu(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> int:
    del plan
    launcher = executable(args.launcher, "system-emulator launcher")
    host_python = executable(args.host_python, "host Python")
    del host_python
    environment = command_environment(args)
    output_root = ensure_sys_emu_root(args, launcher, environment)
    logs_root = output_root / "_matrix_logs"

    preflight: Dict[str, Tuple[Path, Path]] = {}
    for planned in selected:
        package_dir, _ = verify_package(planned, args.ranges_root)
        elf, _ = verify_et_build(planned, args.build_root.resolve())
        preflight[planned["name"]] = (package_dir, elf)

    started = {
        "schema_version": 1,
        "kind": "schema_v2_sys_emu_matrix_invocation",
        "timestamp_utc": utc_now(),
        "status": "RUNNING",
        "resume": args.resume,
        "jobs": args.jobs,
        "selected_ranges": [record["name"] for record in selected],
    }
    started_path = write_invocation(output_root, "sys_emu", "started", started)

    def worker(planned: Mapping[str, Any]) -> Dict[str, Any]:
        package_dir, elf = preflight[planned["name"]]
        run_dir = output_root / planned["name"]
        orchestration_log = logs_root / "{}.log".format(planned["name"])
        if run_dir.exists():
            if not args.resume:
                raise MatrixError("range result already exists: {}".format(run_dir))
            if validation_reports_pass(run_dir):
                return {
                    "schema_version": 1,
                    "status": "PASS",
                    "name": planned["name"],
                    "selector": planned["selector"],
                    "mode": "resumed_validated",
                    "artifacts": {
                        "run_result": identity(run_dir / "run_result.json"),
                        "tensor_compare": identity(run_dir / "tensor_compare.json"),
                        "pmc": identity(run_dir / "pmc.json"),
                    },
                }
            run_result_path = run_dir / "run_result.json"
            if run_result_path.is_file():
                run_result = read_json(run_result_path, "run result")
                reports_absent = (
                    not (run_dir / "tensor_compare.json").exists()
                    and not (run_dir / "pmc.json").exists()
                )
                if (
                    run_result.get("status") == "pass"
                    and run_result.get("device") == "sys_emu"
                    and reports_absent
                ):
                    return validate_one(
                        planned,
                        package_dir,
                        run_dir,
                        environment,
                        args.validation_timeout,
                        orchestration_log,
                        False,
                    )
            raise MatrixError(
                "partial/failing range directory is preserved; retry in a "
                "fresh matrix root: {}".format(run_dir)
            )

        command = [
            str(PORT_ROOT / "scripts/run_et_slice.sh"),
            "--device",
            "sys_emu",
            "--slice-dir",
            str(package_dir),
            "--elf",
            str(elf),
            "--launcher",
            str(launcher),
            "--output-dir",
            str(run_dir),
            "--outer-timeout",
            str(args.outer_timeout),
            "--launcher-timeout",
            str(args.launcher_timeout),
        ]
        return_code, elapsed = subprocess_to_log(
            command,
            orchestration_log,
            environment,
            args.outer_timeout + 120,
        )
        if return_code != 0:
            raise MatrixError(
                "sys-emu run returned {} after {:.1f}s (log {})".format(
                    return_code, elapsed, orchestration_log
                )
            )
        validation = validate_one(
            planned,
            package_dir,
            run_dir,
            environment,
            args.validation_timeout,
            orchestration_log,
            False,
        )
        validation["mode"] = "new_run_and_validation"
        validation["run_elapsed_seconds"] = elapsed
        return validation

    try:
        records = run_parallel(selected, args.jobs, worker, "SYS_EMU_MATRIX")
        failed = sum(record.get("status") != "PASS" for record in records)
        if failed:
            raise MatrixError(
                "SYS_EMU_MATRIX failed for {}/{} ranges".format(failed, len(records))
            )
        status = "PASS"
        return_code = 0
        error = None
    except MatrixError as caught:
        records = locals().get("records", [])
        status = "FAIL"
        return_code = 1
        error = str(caught)
    result = {
        "schema_version": 1,
        "kind": "schema_v2_sys_emu_matrix_invocation",
        "timestamp_utc": utc_now(),
        "status": status,
        "started_record": identity(started_path),
        "resume": args.resume,
        "jobs": args.jobs,
        "selected_ranges": [record["name"] for record in selected],
        "records": records,
    }
    if error is not None:
        result["error"] = error
    result_path = write_invocation(output_root, "sys_emu", "result", result)
    print(
        "SYS_EMU_MATRIX {} root={} result={}".format(status, output_root, result_path),
        file=sys.stderr if return_code else sys.stdout,
    )
    return return_code


def load_result_map(path: Optional[Path]) -> Dict[str, Path]:
    return coverage.load_result_map(path)


def action_validate(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> int:
    del plan
    environment = command_environment(args)
    results_root = args.results_root.resolve()
    result_map = load_result_map(args.result_map)
    logs_root = results_root / "_matrix_validation_logs"

    def worker(planned: Mapping[str, Any]) -> Dict[str, Any]:
        package_dir, _ = verify_package(planned, args.ranges_root)
        run_dir = result_map.get(planned["name"], results_root / planned["name"])
        if not run_dir.is_dir():
            raise MatrixError("run directory missing: {}".format(run_dir))
        log = logs_root / "{}.log".format(planned["name"])
        return validate_one(
            planned,
            package_dir,
            run_dir,
            environment,
            args.validation_timeout,
            log,
            True,
        )

    try:
        records = run_parallel(selected, args.jobs, worker, "VALIDATE_MATRIX")
        failed = sum(record.get("status") != "PASS" for record in records)
        if failed:
            raise MatrixError(
                "VALIDATE_MATRIX failed for {}/{} ranges".format(failed, len(records))
            )
        status = "PASS"
        return_code = 0
        error = None
    except MatrixError as caught:
        records = locals().get("records", [])
        status = "FAIL"
        return_code = 1
        error = str(caught)
    result = {
        "schema_version": 1,
        "kind": "schema_v2_sys_emu_validation_invocation",
        "timestamp_utc": utc_now(),
        "status": status,
        "selected_ranges": [record["name"] for record in selected],
        "records": records,
    }
    if error is not None:
        result["error"] = error
    result_path = write_invocation(results_root, "validate", "result", result)
    print(
        "VALIDATE_MATRIX {} result={}".format(status, result_path),
        file=sys.stderr if return_code else sys.stdout,
    )
    return return_code


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--ranges-root", type=Path, default=DEFAULT_RANGES_ROOT)
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        help="canonical name or selector; repeat to run a subset",
    )
    parser.add_argument("--jobs", type=int, default=1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    check_parser = subparsers.add_parser(
        "check-plan", help="validate exact canonical package coverage"
    )
    add_common(check_parser)

    host = subparsers.add_parser(
        "host", help="build and validate every selected range on the host"
    )
    add_common(host)
    host.add_argument("--output-root", type=Path, required=True)
    host.add_argument("--host-python", type=Path, default=DEFAULT_PYTHON)
    host.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    host.add_argument("--command-timeout", type=int, default=600)

    build = subparsers.add_parser(
        "build-et", help="build fresh ET ELFs for every selected range"
    )
    add_common(build)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--command-timeout", type=int, default=900)
    build.add_argument("--et-platform", type=Path)

    sys_emu = subparsers.add_parser(
        "sys-emu", help="run and validate a bounded parallel sys-emu matrix"
    )
    add_common(sys_emu)
    sys_emu.add_argument("--build-root", type=Path, required=True)
    sys_emu.add_argument("--output-root", type=Path, required=True)
    sys_emu.add_argument("--launcher", type=Path, required=True)
    sys_emu.add_argument("--host-python", type=Path, default=DEFAULT_PYTHON)
    sys_emu.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    sys_emu.add_argument("--et-platform", type=Path)
    sys_emu.add_argument("--ld-library-path")
    sys_emu.add_argument("--outer-timeout", type=int, default=7200)
    sys_emu.add_argument("--launcher-timeout", type=int, default=7140)
    sys_emu.add_argument("--validation-timeout", type=int, default=600)
    sys_emu.add_argument("--resume", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="validate existing sys-emu range artifacts"
    )
    add_common(validate)
    validate.add_argument("--results-root", type=Path, required=True)
    validate.add_argument("--result-map", type=Path)
    validate.add_argument("--host-python", type=Path, default=DEFAULT_PYTHON)
    validate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    validate.add_argument("--validation-timeout", type=int, default=600)

    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    for name in (
        "command_timeout",
        "outer_timeout",
        "launcher_timeout",
        "validation_timeout",
    ):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.action == "sys-emu" and args.launcher_timeout >= args.outer_timeout:
        parser.error("--launcher-timeout must be less than --outer-timeout")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        plan, ranges = coverage.validate_plan(args.plan.resolve())
        selected = select_ranges(ranges, args.range)
        for planned in selected:
            verify_package(planned, args.ranges_root.resolve())
        if args.action == "check-plan":
            print(
                "MATRIX_PLAN PASS ranges={} nodes=N000:N307 packages={}".format(
                    len(selected), args.ranges_root.resolve()
                )
            )
            return 0
        if args.action == "host":
            return action_host(args, plan, selected)
        if args.action == "build-et":
            return action_build_et(args, plan, selected)
        if args.action == "sys-emu":
            return action_sys_emu(args, plan, selected)
        if args.action == "validate":
            return action_validate(args, plan, selected)
        raise MatrixError("unhandled action {}".format(args.action))
    except MatrixError as error:
        print("MATRIX ERROR {}".format(error), file=sys.stderr)
        return 2
    except coverage.LedgerError as error:
        print("MATRIX ERROR {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
