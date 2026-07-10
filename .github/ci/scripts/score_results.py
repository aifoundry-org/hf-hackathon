#!/usr/bin/env python3
"""Score sys-emu benchmark output for one model."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config_helpers import load_config as load_benchmark_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"

YOLO_MAGIC = 0x10500001
SUMMARY = struct.Struct("<16I")
YOLO_DETECTION = struct.Struct("<I5f")


def int_cfg(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def load_config() -> dict:
    return load_benchmark_config(os.environ.get("BENCHMARK_CONFIG", CONFIG_PATH))


def wait_seconds(log_path: Path) -> float | None:
    text = log_path.read_text(errors="ignore")
    match = re.search(r"Kernel wait seconds:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def dump_summary(path: Path, model: str = "") -> dict:
    data = path.read_bytes()[: 0x1000 + SUMMARY.size]
    fields = SUMMARY.unpack_from(data, 0x1000)
    return {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
    }


def read_dump_bytes(path: Path, offset: int, count: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(count)
    if len(data) != count:
        raise ValueError(f"short dump read at 0x{offset:x}: got {len(data)} bytes, expected {count}")
    return data


def with_metrics(metrics: dict, **updates) -> dict:
    merged = dict(metrics)
    merged.update(updates)
    return merged


def load_uint8_npy(path: Path) -> tuple[tuple[int, ...], bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("reference is not a .npy file")
    major = data[6]
    if major == 1:
        header_len = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported .npy version {major}.{data[7]}")

    header = ast.literal_eval(data[header_start : header_start + header_len].decode("latin1").strip())
    if header.get("fortran_order"):
        raise ValueError("fortran-order .npy references are not supported")
    if header.get("descr") not in ("|u1", "u1"):
        raise ValueError(f"expected uint8 .npy reference, got {header.get('descr')}")
    shape = tuple(int(v) for v in header["shape"])
    count = math.prod(shape)
    payload = data[header_start + header_len :]
    if len(payload) != count:
        raise ValueError(f"reference payload has {len(payload)} bytes, expected {count}")
    return shape, payload


def artifact_root() -> Path:
    return Path(
        os.environ.get("BENCHMARK_ARTIFACT_ROOT")
        or os.environ.get("AMP_ROOT")
        or REPO_ROOT / "local-artifacts" / "model-port-benchmarks"
    )


def resolve_reference(paths: list[str]) -> Path | None:
    roots = [artifact_root(), REPO_ROOT]
    for rel in paths:
        candidate = Path(rel)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        for root in roots:
            path = root / rel
            if path.is_file():
                return path
    return None


def byte_diff_metrics(actual: bytes, expected: bytes) -> tuple[int, float]:
    if len(actual) != len(expected):
        raise ValueError(f"length mismatch: actual {len(actual)} bytes, expected {len(expected)} bytes")
    max_abs = 0
    total = 0
    for a, b in zip(actual, expected):
        diff = abs(a - b)
        total += diff
        if diff > max_abs:
            max_abs = diff
    mean_abs = total / len(actual) if actual else 0.0
    return max_abs, mean_abs


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def max_box_abs_diff(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def read_yolo_detections(path: Path, offset: int, max_detections: int) -> list[dict]:
    count = struct.unpack("<I", read_dump_bytes(path, offset, 4))[0]
    if count > max_detections:
        raise ValueError(f"detection count {count} exceeds configured max {max_detections}")
    payload = read_dump_bytes(path, offset + 4, count * YOLO_DETECTION.size)
    detections = []
    for idx in range(count):
        class_id, score, x1, y1, x2, y2 = YOLO_DETECTION.unpack_from(payload, idx * YOLO_DETECTION.size)
        detections.append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "box": (float(x1), float(y1), float(x2), float(y2)),
            }
        )
    return detections


def yolo_host_reference_path() -> Path:
    configured = os.environ.get("YOLO_HOST_REFERENCE_JSON")
    if configured:
        return Path(configured)
    output_root = Path(os.environ.get("BENCHMARK_OUTPUT", REPO_ROOT / ".ci-work" / "benchmark-output"))
    return output_root / "yolo-host-reference.json"


def load_yolo_contract(mcfg: dict) -> tuple[Path, dict]:
    contract_value = mcfg.get("reference_contract") or mcfg.get("validation", {}).get(
        "reference_contract"
    )
    if not contract_value:
        raise ValueError("YOLO reference contract is not configured")
    contract_path = Path(contract_value)
    if not contract_path.is_absolute():
        contract_path = REPO_ROOT / contract_path
    return contract_path, json.loads(contract_path.read_text())


def yolo_benchmark_contract_error(mcfg: dict, cases: list[dict], contract: dict) -> str | None:
    if mcfg.get("runner", "elf") != "elf":
        return "YOLO benchmark runner must remain elf"
    score = mcfg.get("score", {})
    if score.get("metric") != "kernel_wait_s" or score.get("higher_is_better") is not False:
        return "YOLO must remain a lower-is-better kernel_wait_s benchmark"
    source = str(mcfg.get("source") or "")
    if not source.startswith("ported_models/yolo/src/"):
        return "YOLO benchmark source must stay under ported_models/yolo/src"

    fixtures = contract["fixtures"]["cases"]
    if [str(case.get("name")) for case in cases] != [str(item["name"]) for item in fixtures]:
        return "YOLO benchmark cases do not match the fixed COCO reference contract"
    abi = contract["board_abi"]
    detection_offset = int_cfg(abi["detections_offset"])
    max_detections = int(abi["max_detections"])
    if int_cfg(mcfg["dump_size"]) < detection_offset + 4 + max_detections * YOLO_DETECTION.size:
        return "YOLO dump_size does not include the fixed detection output"

    for case, fixture in zip(cases, fixtures):
        name = str(fixture["name"])
        accuracy = case.get("accuracy", {})
        if (
            accuracy.get("kind") != "yolo_reference_detections"
            or accuracy.get("reference_case") != name
            or int_cfg(accuracy.get("offset", -1)) != detection_offset
            or int(accuracy.get("max_detections", 0)) != max_detections
        ):
            return f"{name}: accuracy ABI does not match the YOLO reference contract"
        expected_asset = str(fixture["asset"]).removeprefix("ported_models/yolo/assets/")
        input_loads = [
            load
            for load in case.get("file_loads", [])
            if int_cfg(load.get("address", -1)) == int_cfg(abi["input_address"])
        ]
        if not any(expected_asset in (load.get("paths") or [load.get("path")]) for load in input_loads):
            return f"{name}: input asset does not match the fixed COCO reference contract"
    return None


def load_yolo_reference_case(mcfg: dict, case_name: str) -> tuple[list[dict], dict, str]:
    reference_path = yolo_host_reference_path()
    if not reference_path.is_file():
        raise ValueError(f"missing host reference {reference_path}")
    payload = json.loads(reference_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported YOLO host-reference schema")

    contract_path, contract = load_yolo_contract(mcfg)
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if payload.get("contract_sha256") != contract_sha:
        raise ValueError("host reference was generated from a different YOLO contract")

    fixture = next(
        (item for item in contract["fixtures"]["cases"] if item.get("name") == case_name),
        None,
    )
    case = payload.get("cases", {}).get(case_name)
    if fixture is None or not isinstance(case, dict):
        raise ValueError(f"host reference has no fixed case {case_name}")
    if case.get("input_sha256") != fixture.get("asset_sha256"):
        raise ValueError(f"host reference input hash mismatch for {case_name}")
    checkpoint_sha = payload.get("reference", {}).get("checkpoint_sha256")
    if checkpoint_sha != contract["model"]["source"]["sha256"]:
        raise ValueError("host reference checkpoint hash does not match the YOLO contract")
    detections = case.get("detections")
    if not isinstance(detections, list):
        raise ValueError(f"host reference detections missing for {case_name}")
    description = (
        f"{contract['model']['source']['repo']}@"
        f"{contract['model']['source']['revision'][:12]}:{case_name}"
    )
    return detections, payload["agreement"], description


def compare_yolo_detections(
    actual: list[dict], reference: list[dict], agreement: dict
) -> tuple[bool, list[str], dict]:
    score_threshold = float(agreement["comparison_score_threshold"])
    matching_iou = float(agreement["matching_iou_threshold"])
    actual = [item for item in actual if float(item["score"]) >= score_threshold]
    reference = [item for item in reference if float(item["score"]) >= score_threshold]
    if len(reference) < int(agreement["minimum_reference_detections_per_case"]):
        raise ValueError("host reference has too few compared detections")

    pairs = []
    for reference_index, expected in enumerate(reference):
        expected_box = tuple(float(value) for value in expected["box"])
        for actual_index, candidate in enumerate(actual):
            if int(candidate["class_id"]) != int(expected["class_id"]):
                continue
            iou = box_iou(candidate["box"], expected_box)
            if iou >= matching_iou:
                pairs.append((iou, reference_index, actual_index))
    pairs.sort(reverse=True)

    used_reference: set[int] = set()
    used_actual: set[int] = set()
    matches = []
    for iou, reference_index, actual_index in pairs:
        if reference_index in used_reference or actual_index in used_actual:
            continue
        used_reference.add(reference_index)
        used_actual.add(actual_index)
        expected = reference[reference_index]
        candidate = actual[actual_index]
        matches.append(
            {
                "iou": iou,
                "score_abs": abs(float(candidate["score"]) - float(expected["score"])),
                "box_abs": max_box_abs_diff(candidate["box"], tuple(float(value) for value in expected["box"])),
            }
        )

    true_positives = len(matches)
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    mean_iou = sum(item["iou"] for item in matches) / true_positives if matches else 0.0
    score_mae = sum(item["score_abs"] for item in matches) / true_positives if matches else None
    box_abs_values = [item["box_abs"] for item in matches]

    failures = []
    if precision < float(agreement["minimum_precision"]):
        failures.append(
            f"precision={precision:.3f} below {float(agreement['minimum_precision']):.3f}"
        )
    if recall < float(agreement["minimum_recall"]):
        failures.append(f"recall={recall:.3f} below {float(agreement['minimum_recall']):.3f}")
    if mean_iou < float(agreement["minimum_mean_iou"]):
        failures.append(
            f"mean_iou={mean_iou:.3f} below {float(agreement['minimum_mean_iou']):.3f}"
        )
    if score_mae is None or score_mae > float(agreement["maximum_score_mae"]):
        score_text = "none" if score_mae is None else f"{score_mae:.3f}"
        failures.append(
            f"score_mae={score_text} above {float(agreement['maximum_score_mae']):.3f}"
        )

    missing = [reference[index] for index in range(len(reference)) if index not in used_reference]
    unexpected = [actual[index] for index in range(len(actual)) if index not in used_actual]
    if missing:
        failures.append("missing classes=" + ",".join(str(item["class_id"]) for item in missing))
    if unexpected:
        failures.append("unexpected classes=" + ",".join(str(item["class_id"]) for item in unexpected))

    return not failures, failures, {
        "accuracy_true_positives": true_positives,
        "accuracy_reference_count": len(reference),
        "accuracy_candidate_count": len(actual),
        "accuracy_precision": precision,
        "accuracy_recall": recall,
        "accuracy_mean_iou": mean_iou,
        "accuracy_score_mae": score_mae,
        "accuracy_max_abs": max(box_abs_values) if box_abs_values else None,
        "accuracy_mean_abs": (
            sum(box_abs_values) / len(box_abs_values) if box_abs_values else None
        ),
    }


def validate_accuracy(
    model: str,
    dump_path: Path | None,
    mcfg: dict,
    summary: dict | None,
    accuracy_cfg: dict | None = None,
) -> tuple[bool, str, dict]:
    acfg = accuracy_cfg if accuracy_cfg is not None else mcfg.get("accuracy")
    if not acfg:
        return True, "no accuracy gate configured", {
            "valid_accuracy": True,
            "accuracy_kind": None,
            "accuracy_max_abs": None,
            "accuracy_mean_abs": None,
            "accuracy_reference": None,
        }

    kind = acfg.get("kind")
    metrics = {
        "valid_accuracy": False,
        "accuracy_kind": kind,
        "accuracy_max_abs": None,
        "accuracy_mean_abs": None,
        "accuracy_reference": None,
        "accuracy_true_positives": None,
        "accuracy_reference_count": None,
        "accuracy_candidate_count": None,
        "accuracy_precision": None,
        "accuracy_recall": None,
        "accuracy_mean_iou": None,
        "accuracy_score_mae": None,
    }

    try:
        if kind == "checksum":
            if summary is None:
                return False, "accuracy check failed: missing dump summary", metrics
            expected = int_cfg(acfg["expected_output_sum"])
            actual = int(summary.get("output_sum", -1))
            ok = actual == expected
            note = f"accuracy checksum {'valid' if ok else 'failed'} output_sum={actual} expected={expected}"
            return ok, note, with_metrics(metrics, valid_accuracy=ok)

        if kind == "sha256":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            count = int_cfg(acfg["count"])
            expected = str(acfg["expected_sha256"]).lower()
            actual = hashlib.sha256(read_dump_bytes(dump_path, offset, count)).hexdigest()
            ok = actual == expected
            note = (
                f"accuracy sha256 {'valid' if ok else 'failed'} "
                f"actual={actual[:12]} expected={expected[:12]}"
            )
            return ok, note, with_metrics(metrics, valid_accuracy=ok)

        if kind == "constant_u8":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            count = int_cfg(acfg["count"])
            expected_value = int_cfg(acfg["expected_value"])
            max_allowed = int_cfg(acfg.get("max_abs", 0))
            actual = read_dump_bytes(dump_path, offset, count)
            expected = bytes([expected_value]) * count
            max_abs, mean_abs = byte_diff_metrics(actual, expected)
            ok = max_abs <= max_allowed
            note = (
                f"accuracy {'valid' if ok else 'failed'} constant_u8 "
                f"max_abs={max_abs} mean_abs={mean_abs:.6f} gate={max_allowed}"
            )
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_max_abs=max_abs,
                accuracy_mean_abs=mean_abs,
            )

        if kind == "uint8_npy":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            ref_paths = acfg.get("reference_paths") or [acfg.get("reference_path")]
            ref_paths = [str(path) for path in ref_paths if path]
            ref_path = resolve_reference(ref_paths)
            if ref_path is None:
                return False, "accuracy check failed: missing reference " + ", ".join(ref_paths), metrics
            shape, expected = load_uint8_npy(ref_path)
            cfg_shape = tuple(int(v) for v in acfg.get("shape", shape))
            if shape != cfg_shape:
                return False, f"accuracy check failed: reference shape {shape} != configured {cfg_shape}", metrics
            actual = read_dump_bytes(dump_path, int_cfg(acfg["offset"]), len(expected))
            max_abs, mean_abs = byte_diff_metrics(actual, expected)
            max_allowed = int_cfg(acfg.get("max_abs", 0))
            ok = max_abs <= max_allowed
            rel = str(ref_path)
            try:
                rel = str(ref_path.relative_to(REPO_ROOT))
            except ValueError:
                pass
            note = (
                f"accuracy {'valid' if ok else 'failed'} uint8_npy "
                f"max_abs={max_abs} mean_abs={mean_abs:.6f} gate={max_allowed}"
            )
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_max_abs=max_abs,
                accuracy_mean_abs=mean_abs,
                accuracy_reference=rel,
            )

        if kind == "yolo_reference_detections":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            case_name = str(acfg.get("reference_case") or "")
            if not case_name:
                return False, "accuracy check failed: missing reference_case", metrics
            detections = read_yolo_detections(
                dump_path,
                int_cfg(acfg["offset"]),
                int_cfg(acfg.get("max_detections", 64)),
            )
            expected, agreement, description = load_yolo_reference_case(mcfg, case_name)
            ok, failures, comparison = compare_yolo_detections(detections, expected, agreement)
            note = (
                f"accuracy yolo_reference_detections {'valid' if ok else 'failed'} "
                f"tp={comparison['accuracy_true_positives']}/"
                f"{comparison['accuracy_reference_count']} "
                f"precision={comparison['accuracy_precision']:.3f} "
                f"recall={comparison['accuracy_recall']:.3f} "
                f"mean_iou={comparison['accuracy_mean_iou']:.3f}"
            )
            if comparison["accuracy_score_mae"] is not None:
                note += f" score_mae={comparison['accuracy_score_mae']:.4f}"
            if failures:
                note += "; " + "; ".join(failures)
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_reference=description,
                **comparison,
            )

        return False, f"accuracy check failed: unknown kind {kind}", metrics
    except Exception as exc:
        return False, f"accuracy check failed: {exc}", metrics


def validate_dump(model: str, dump_path: Path | None, magic_cfg: str | None) -> tuple[bool, str]:
    if magic_cfg is None:
        return True, "no dump magic configured"
    if dump_path is None or not dump_path.is_file():
        return False, "missing dump.bin"
    magic_expected = int(magic_cfg, 0)
    s = dump_summary(dump_path, model)
    ok = (
        s["magic"] == magic_expected
        and s["done_count"] == s["active_harts"]
        and s["output_sum"] == s["slot_sum"]
    )
    if not ok:
        return False, (
            f"dump check failed magic=0x{s['magic']:x} "
            f"done={s['done_count']} harts={s['active_harts']} "
            f"out={s['output_sum']} slot={s['slot_sum']}"
        )
    return True, "dump valid"


def benchmark_image_count(mcfg: dict) -> int | None:
    for section in ("accuracy", "validation"):
        cfg = mcfg.get(section, {})
        if not isinstance(cfg, dict) or "image_count" not in cfg:
            continue
        try:
            value = int(cfg["image_count"])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def find_job_dir(results_dir: Path, model: str, variant: str) -> Path | None:
    jobs = results_dir / "jobs"
    if not jobs.is_dir():
        return None
    for job in sorted(jobs.iterdir()):
        if variant in job.name and model in job.name:
            return job
    return None


def row_paths(results_dir: Path, row: dict, job_dir: Path | None = None) -> tuple[Path | None, Path | None]:
    log_path = (results_dir / row["log"]) if row.get("log") else None
    dump_path = (results_dir / row["dump"]) if row.get("dump") else None
    if job_dir and (job_dir / "run.log").is_file():
        log_path = job_dir / "run.log"
    if job_dir and (job_dir / "dump.bin").is_file():
        dump_path = job_dir / "dump.bin"
    return log_path, dump_path


def evaluate_row(
    model: str,
    mcfg: dict,
    row: dict,
    results_dir: Path,
    magic_cfg: str | None,
    accuracy_cfg: dict | None = None,
    job_dir: Path | None = None,
) -> dict:
    status = row.get("status", "")
    log_path, dump_path = row_paths(results_dir, row, job_dir)
    if status == "fail" and log_path and log_path.is_file():
        log_text = log_path.read_text(errors="ignore")
        if "kernel execution timed out" in log_text or "timed out" in log_text.lower():
            status = "timeout"

    kernel_wait = row.get("kernel_wait_s") or ""
    if not kernel_wait and log_path and log_path.is_file():
        w = wait_seconds(log_path)
        kernel_wait = f"{w:.6f}" if w is not None else ""

    valid_dump, valid_note = validate_dump(model, dump_path, magic_cfg)
    parsed_summary = dump_summary(dump_path, model) if dump_path and dump_path.is_file() else None
    valid_accuracy, accuracy_note, accuracy_metrics = validate_accuracy(
        model,
        dump_path,
        mcfg,
        parsed_summary,
        accuracy_cfg=accuracy_cfg,
    )
    combined_note = valid_note
    if accuracy_note:
        combined_note = f"{valid_note}; {accuracy_note}"
    passed = status == "pass" and bool(kernel_wait) and valid_dump and valid_accuracy
    return {
        "case": row.get("case") or "",
        "status": "pass" if passed else (status if status and status != "pass" else "fail"),
        "passed": passed,
        "kernel_wait_s": float(kernel_wait) if kernel_wait else None,
        "valid_dump": valid_dump,
        "valid_accuracy": valid_accuracy,
        "valid_note": combined_note,
        "accuracy_metrics": accuracy_metrics,
        "emu_cycle_last": row.get("emu_cycle_last") or None,
        "elapsed_s": float(row["elapsed_s"]) if row.get("elapsed_s") else None,
        "note": row.get("note") or "",
    }


def score_benchmark_cases(
    model: str,
    mcfg: dict,
    variant: str,
    magic_cfg: str | None,
    rows: list[dict],
    results_dir: Path,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
) -> dict | None:
    cases = [case for case in mcfg.get("benchmark_cases", []) if isinstance(case, dict) and case.get("name")]
    if model == "yolo":
        try:
            contract_path, contract = load_yolo_contract(mcfg)
            contract_error = yolo_benchmark_contract_error(mcfg, cases, contract)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return fail_payload(
                model, variant, sha, ref, actor, run_url, f"invalid YOLO reference contract: {exc}"
            )
        if contract_error:
            return fail_payload(
                model,
                variant,
                sha,
                ref,
                actor,
                run_url,
                contract_error,
            )
    if not cases:
        return None

    rows_by_case = {
        row.get("case"): row
        for row in rows
        if row.get("model") == model and row.get("variant") == variant and row.get("case")
    }
    case_results = []
    for case in cases:
        name = str(case["name"])
        row = rows_by_case.get(name)
        if row is None:
            case_results.append(
                {
                    "case": name,
                    "status": "missing",
                    "passed": False,
                    "kernel_wait_s": None,
                    "valid_dump": False,
                    "valid_accuracy": False,
                    "valid_note": "missing benchmark case row",
                    "accuracy_metrics": {
                        "accuracy_kind": case.get("accuracy", {}).get("kind"),
                        "accuracy_max_abs": None,
                        "accuracy_mean_abs": None,
                        "accuracy_reference": case.get("accuracy", {}).get("reference"),
                    },
                    "emu_cycle_last": None,
                    "elapsed_s": None,
                    "note": "missing benchmark case row",
                }
            )
            continue
        case_results.append(
            evaluate_row(
                model,
                mcfg,
                row,
                results_dir,
                magic_cfg,
                accuracy_cfg=case.get("accuracy") if isinstance(case.get("accuracy"), dict) else None,
            )
        )

    passed = all(result["passed"] for result in case_results)
    waits = [result["kernel_wait_s"] for result in case_results if isinstance(result["kernel_wait_s"], float)]
    kernel_wait_value = sum(waits) / len(waits) if waits else None
    elapsed_values = [result["elapsed_s"] for result in case_results if isinstance(result["elapsed_s"], float)]
    elapsed_s = sum(elapsed_values) if elapsed_values else None
    failed_notes = [
        f"{result['case']}: {result['valid_note'] or result['status']}"
        for result in case_results
        if not result["passed"]
    ]
    valid_note = (
        f"{sum(1 for result in case_results if result['passed'])}/{len(case_results)} "
        "YOLO COCO cases valid"
    )
    if failed_notes:
        valid_note += "; " + "; ".join(failed_notes[:3])

    accuracy_metrics = [result["accuracy_metrics"] for result in case_results]
    max_abs_values = [
        metric.get("accuracy_max_abs")
        for metric in accuracy_metrics
        if isinstance(metric.get("accuracy_max_abs"), (int, float))
    ]
    mean_abs_values = [
        metric.get("accuracy_mean_abs")
        for metric in accuracy_metrics
        if isinstance(metric.get("accuracy_mean_abs"), (int, float))
    ]
    true_positives = sum(
        int(metric.get("accuracy_true_positives") or 0) for metric in accuracy_metrics
    )
    reference_count = sum(
        int(metric.get("accuracy_reference_count") or 0) for metric in accuracy_metrics
    )
    candidate_count = sum(
        int(metric.get("accuracy_candidate_count") or 0) for metric in accuracy_metrics
    )
    matched_metrics = [
        metric
        for metric in accuracy_metrics
        if int(metric.get("accuracy_true_positives") or 0) > 0
    ]
    mean_iou = (
        sum(
            float(metric["accuracy_mean_iou"]) * int(metric["accuracy_true_positives"])
            for metric in matched_metrics
        )
        / true_positives
        if true_positives
        else None
    )
    score_mae = (
        sum(
            float(metric["accuracy_score_mae"]) * int(metric["accuracy_true_positives"])
            for metric in matched_metrics
            if metric.get("accuracy_score_mae") is not None
        )
        / true_positives
        if true_positives
        else None
    )
    first_kind = next((metric.get("accuracy_kind") for metric in accuracy_metrics if metric.get("accuracy_kind")), None)
    first_reference = next(
        (metric.get("accuracy_reference") for metric in accuracy_metrics if metric.get("accuracy_reference")),
        None,
    )
    return {
        "model": model,
        "variant": variant,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "kernel_wait_s": kernel_wait_value,
        "kernel_wait_per_image_s": kernel_wait_value,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": all(result["valid_dump"] for result in case_results),
        "valid_note": valid_note,
        "valid_accuracy": all(result["valid_accuracy"] for result in case_results),
        "accuracy_kind": first_kind,
        "accuracy_max_abs": max(max_abs_values) if max_abs_values else None,
        "accuracy_mean_abs": sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else None,
        "accuracy_reference": first_reference,
        "accuracy_true_positives": true_positives,
        "accuracy_reference_count": reference_count,
        "accuracy_candidate_count": candidate_count,
        "accuracy_precision": true_positives / candidate_count if candidate_count else None,
        "accuracy_recall": true_positives / reference_count if reference_count else None,
        "accuracy_mean_iou": mean_iou,
        "accuracy_score_mae": score_mae,
        "validation_contract_sha256": (
            hashlib.sha256(contract_path.read_bytes()).hexdigest()
            if model == "yolo"
            else None
        ),
        "case_results": [
            {
                **{key: value for key, value in result.items() if key != "accuracy_metrics"},
                "accuracy": result["accuracy_metrics"],
            }
            for result in case_results
        ],
        "emu_cycle_last": None,
        "elapsed_s": elapsed_s,
        "note": "",
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def score_from_results(
    model: str,
    results_dir: Path,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
) -> dict:
    cfg = load_config()
    mcfg = cfg["models"][model]
    variant = mcfg["canonical_variant"]
    magic_cfg = mcfg.get("dump_magic")

    results_tsv = results_dir / "results.tsv"
    if not results_tsv.is_file():
        return fail_payload(model, variant, sha, ref, actor, run_url, "missing results.tsv")

    rows = []
    with results_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    case_score = score_benchmark_cases(
        model,
        mcfg,
        variant,
        magic_cfg,
        rows,
        results_dir,
        sha,
        ref,
        actor,
        run_url,
    )
    if case_score is not None:
        return case_score

    row = None
    for r in rows:
        if r.get("model") == model and r.get("variant") == variant:
            row = r
            break

    if row is None:
        return fail_payload(model, variant, sha, ref, actor, run_url, "canonical variant not in results.tsv")

    job_dir = find_job_dir(results_dir, model, variant)
    evaluated = evaluate_row(model, mcfg, row, results_dir, magic_cfg, job_dir=job_dir)
    kernel_wait_value = evaluated["kernel_wait_s"]
    image_count = benchmark_image_count(mcfg)
    kernel_wait_per_image = (
        kernel_wait_value / image_count
        if kernel_wait_value is not None and image_count
        else None
    )

    return {
        "model": model,
        "variant": variant,
        "status": evaluated["status"],
        "passed": evaluated["passed"],
        "kernel_wait_s": kernel_wait_value,
        "kernel_wait_per_image_s": kernel_wait_per_image,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": evaluated["valid_dump"],
        "valid_note": evaluated["valid_note"],
        **evaluated["accuracy_metrics"],
        "validation_contract_sha256": None,
        "emu_cycle_last": evaluated["emu_cycle_last"],
        "elapsed_s": evaluated["elapsed_s"],
        "note": evaluated["note"],
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def fail_payload(
    model: str,
    variant: str,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
    note: str,
    status: str = "fail",
) -> dict:
    return {
        "model": model,
        "variant": variant,
        "status": status,
        "passed": False,
        "kernel_wait_s": None,
        "kernel_wait_per_image_s": None,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": False,
        "valid_accuracy": False,
        "accuracy_kind": None,
        "accuracy_max_abs": None,
        "accuracy_mean_abs": None,
        "accuracy_reference": None,
        "accuracy_true_positives": None,
        "accuracy_reference_count": None,
        "accuracy_candidate_count": None,
        "accuracy_precision": None,
        "accuracy_recall": None,
        "accuracy_mean_iou": None,
        "accuracy_score_mae": None,
        "validation_contract_sha256": None,
        "valid_note": note,
        "emu_cycle_last": None,
        "elapsed_s": None,
        "note": note,
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status")
    parser.add_argument("--note", default="")
    parser.add_argument("--sha", default="local")
    parser.add_argument("--ref", default="local")
    parser.add_argument("--actor", default="local")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()

    if args.status:
        cfg = load_config()
        variant = cfg["models"][args.model]["canonical_variant"]
        payload = fail_payload(
            args.model,
            variant,
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
            args.note,
            status=args.status,
        )
    else:
        if not args.results_dir:
            print("error: --results-dir required unless --status is set", file=sys.stderr)
            return 2
        payload = score_from_results(
            args.model,
            Path(args.results_dir),
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
