#!/usr/bin/env python3
"""Decide whether a paired trusted SmolVLM2 board evaluation may merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def cycle_value(score: dict[str, Any]) -> float | None:
    value = score.get("pmc_cycles")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def wall_value(score: dict[str, Any]) -> float | None:
    value = score.get("median_end_to_end_s")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def hardware_key(score: dict[str, Any]) -> tuple[Any, ...] | None:
    hardware = score.get("hardware")
    if not isinstance(hardware, dict):
        return None
    values = (
        score.get("hardware_epoch"),
        hardware.get("board_id"),
        hardware.get("boot_id"),
        hardware.get("minion_frequency_mhz"),
        hardware.get("noc_frequency_mhz"),
        hardware.get("tdp_w"),
    )
    if any(value is None or value == "" for value in values):
        return None
    return values


def validation_errors(
    score: dict[str, Any],
    contract: dict[str, Any],
    contract_sha: str,
    *,
    require_quality: bool,
) -> list[str]:
    errors: list[str] = []
    if not score.get("passed"):
        errors.append(str(score.get("valid_note") or score.get("note") or "benchmark failed"))
    if score.get("validation_contract_sha256") != contract_sha:
        errors.append("validation contract hash is stale or missing")
    if score.get("task_accuracy") != 1.0:
        errors.append("public task accuracy must be 1.0")
    if score.get("vision_fallback_ops") != []:
        errors.append("vision graph used CPU fallback operations")
    if require_quality:
        if (
            score.get("trusted_cpu_reference") is not True
            or score.get("cpu_perplexity_reference_executed") is not True
        ):
            errors.append("CPU perplexity reference was not executed by the current-main runtime")
        ppl = score.get("perplexity")
        cpu_ppl = score.get("cpu_perplexity")
        ceiling = float(contract["quality"]["perplexity"]["maximum_perplexity"])
        if not isinstance(ppl, (int, float)) or float(ppl) > ceiling:
            errors.append(f"perplexity must be <= {ceiling:.4f}")
        if not isinstance(cpu_ppl, (int, float)) or float(cpu_ppl) > ceiling:
            errors.append(f"trusted CPU perplexity must be <= {ceiling:.4f}")
    if cycle_value(score) is None:
        errors.append("firmware-cycle measurement is missing")
    if wall_value(score) is None:
        errors.append("end-to-end request measurement is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--baseline-before", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-after", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = load(args.contract)
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    before = load(args.baseline_before)
    candidate = load(args.candidate)
    after = load(args.baseline_after)
    before_errors = validation_errors(before, contract, contract_sha, require_quality=False)
    after_errors = validation_errors(after, contract, contract_sha, require_quality=False)
    candidate_errors = validation_errors(candidate, contract, contract_sha, require_quality=True)
    infrastructure_errors = [f"baseline before: {item}" for item in before_errors]
    infrastructure_errors.extend(f"baseline after: {item}" for item in after_errors)
    before_hardware = hardware_key(before)
    candidate_hardware = hardware_key(candidate)
    after_hardware = hardware_key(after)
    if before_hardware is None:
        infrastructure_errors.append("baseline before has no complete hardware identity")
    if after_hardware is None or after_hardware != before_hardware:
        infrastructure_errors.append("paired main runs are not from the same board boot")
    if candidate_hardware is None or candidate_hardware != before_hardware:
        infrastructure_errors.append("candidate is not from the paired main board boot")

    before_cycles = cycle_value(before)
    after_cycles = cycle_value(after)
    candidate_cycles = cycle_value(candidate)
    baseline_cycles: float | None = None
    drift: float | None = None
    if before_cycles is not None and after_cycles is not None:
        baseline_cycles = statistics.mean([before_cycles, after_cycles])
        drift = abs(after_cycles - before_cycles) / baseline_cycles
        maximum_drift = float(contract["performance"]["maximum_paired_baseline_drift"])
        if drift > maximum_drift:
            infrastructure_errors.append(
                f"paired main drift {drift:.2%} exceeds {maximum_drift:.2%}"
            )

    required_cycles: float | None = None
    if baseline_cycles is not None:
        improvement = float(contract["performance"]["minimum_relative_improvement"])
        required_cycles = baseline_cycles * (1.0 - improvement)
        if candidate_cycles is None or candidate_cycles >= required_cycles:
            candidate_errors.append(f"cycles must be < {required_cycles:,.0f}")

    before_wall = wall_value(before)
    after_wall = wall_value(after)
    candidate_wall = wall_value(candidate)
    baseline_wall: float | None = None
    wall_drift: float | None = None
    required_wall: float | None = None
    if before_wall is not None and after_wall is not None:
        baseline_wall = statistics.mean([before_wall, after_wall])
        wall_drift = abs(after_wall - before_wall) / baseline_wall
        maximum_wall_drift = float(contract["performance"]["maximum_paired_end_to_end_drift"])
        if wall_drift > maximum_wall_drift:
            infrastructure_errors.append(
                f"paired main end-to-end drift {wall_drift:.2%} exceeds {maximum_wall_drift:.2%}"
            )
        wall_improvement = float(contract["performance"]["minimum_end_to_end_improvement"])
        required_wall = baseline_wall * (1.0 - wall_improvement)
        if candidate_wall is None or candidate_wall >= required_wall:
            candidate_errors.append(f"end-to-end request time must be < {required_wall:.3f}s")

    result = "infrastructure error" if infrastructure_errors else "fail" if candidate_errors else "pass"
    lines = [
        "## Trusted SmolVLM2 Gate",
        "",
        "The main-owned harness ran current main, the candidate, and current main again on ET-SoC1.",
        "",
        "| Run | Firmware cycles | End to end | PPL | Accuracy | Host agreement |",
        "|-----|----------------:|-----------:|----:|---------:|---------------:|",
    ]
    for name, score in (("main before", before), ("candidate", candidate), ("main after", after)):
        cycles = cycle_value(score)
        ppl = score.get("perplexity")
        accuracy = score.get("task_accuracy")
        agreement = score.get("host_agreement")
        wall = wall_value(score)
        lines.append(
            f"| {name} | {cycles:,.0f} | {wall:.3f}s | "
            f"{float(ppl):.4f} | "
            f"{float(accuracy):.2f} | "
            f"{float(agreement):.2f} |"
            if cycles is not None
            and wall is not None
            and isinstance(ppl, (int, float))
            and isinstance(accuracy, (int, float))
            and isinstance(agreement, (int, float))
            else f"| {name} | {cycles:,.0f} | {wall:.3f}s | "
            f"{float(ppl):.4f} | {float(accuracy):.2f} | - |"
            if cycles is not None
            and wall is not None
            and isinstance(ppl, (int, float))
            and isinstance(accuracy, (int, float))
            else f"| {name} | {cycles:,.0f} | {wall:.3f}s | - | {float(accuracy):.2f} | - |"
            if cycles is not None and wall is not None and isinstance(accuracy, (int, float))
            else f"| {name} | - | - | - | - | - |"
        )
    lines.extend(["", f"Result: {result}."])
    if drift is not None:
        lines.append(f"Paired-main drift: {drift:.2%}.")
    if wall_drift is not None:
        lines.append(f"Paired-main end-to-end drift: {wall_drift:.2%}.")
    if infrastructure_errors:
        lines.append("No candidate verdict was produced: " + "; ".join(infrastructure_errors) + ".")
    elif candidate_errors:
        lines.append("Do not merge: " + "; ".join(candidate_errors) + ".")
    else:
        lines.append(
            f"Candidate may merge: {candidate_cycles:,.0f} cycles is below the required "
            f"{required_cycles:,.0f} cycles and {candidate_wall:.3f}s is below the required "
            f"{required_wall:.3f}s, with correctness and PPL preserved."
        )
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 2 if infrastructure_errors else 1 if candidate_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
