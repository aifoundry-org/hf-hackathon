#!/usr/bin/env python3
"""Decide whether a trusted Llama 3.2 1B candidate is mergeable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def leaderboard_entries(model: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / "data" / f"{model}.json"
    if not path.is_file():
        return []
    value = load_json(path)
    entries = value.get("entries", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def best(entries: list[dict[str, Any]], key: str, *, higher: bool) -> float | None:
    values = [float(entry[key]) for entry in entries if isinstance(entry.get(key), (int, float))]
    if not values:
        return None
    return max(values) if higher else min(values)


def fmt(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.4f}{suffix}"


def cell(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def regression_floor(reference: float, max_relative_regression: float) -> float:
    return reference * (1.0 - max_relative_regression)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--track-policy", required=True)
    parser.add_argument("--mode", choices=("competition", "regression"), required=True)
    parser.add_argument("--baseline-score", required=True)
    parser.add_argument("--candidate-scores-dir", required=True)
    parser.add_argument("--regression-models", default="")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--candidate-metadata", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--result-output", default="")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = load_json(contract_path)
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    policy_path = Path(args.track_policy)
    policy = load_json(policy_path)
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if policy.get("model") != contract.get("model"):
        raise RuntimeError("track policy and validation contract model do not match")
    model = str(contract["model"])
    baseline = load_json(Path(args.baseline_score))
    scores_dir = Path(args.candidate_scores_dir)
    candidate = load_json(scores_dir / f"score-{model}.json")
    min_relative = float(contract["performance"]["min_relative_improvement"])
    max_speed_regression = float(
        policy["shared_runtime"]["max_relative_throughput_regression"]
    )
    max_ppl_regression = float(contract["quality"]["max_ppl_regression"])
    failures: list[str] = []
    result_rows: list[dict[str, Any]] = []

    lines = [
        "## Trusted Llama 3.2 1B Gate",
        "",
        (
            f"Mode: **{args.mode}**. The candidate uses the main-owned model contract, CPU "
            "reference, ET quality check, PP256/TG128 benchmark, and shared-runtime "
            "regression set."
        ),
        "",
        "| Model | Decode tok/s | Baseline/best | PPL | Verdict | Notes |",
        "|-------|--------------|---------------|-----|---------|-------|",
    ]

    baseline_speed = baseline.get("tokens_per_second")
    candidate_speed = candidate.get("tokens_per_second")
    candidate_ppl = candidate.get("perplexity")
    historical = leaderboard_entries(model)
    contract_historical = [
        entry
        for entry in historical
        if entry.get("validation_contract_sha256") == contract_sha
    ]
    historical_speed = best(contract_historical, "tokens_per_second", higher=True)
    historical_ppl = best(historical, "perplexity", higher=False)
    quality_reference = historical_ppl
    if quality_reference is None and isinstance(baseline.get("perplexity"), (int, float)):
        quality_reference = float(baseline["perplexity"])

    target_notes: list[str] = []
    target_ok = True
    if not baseline.get("passed"):
        target_ok = False
        target_notes.append("current-main paired baseline failed")
    if baseline.get("validation_contract_sha256") != contract_sha:
        target_ok = False
        target_notes.append("paired baseline contract hash is stale or missing")
    if not candidate.get("passed"):
        target_ok = False
        target_notes.append(str(candidate.get("valid_note") or "candidate validation failed"))
    if candidate.get("validation_contract_sha256") != contract_sha:
        target_ok = False
        target_notes.append("candidate score contract hash is stale or missing")
    if candidate.get("team") != args.participant:
        target_ok = False
        target_notes.append("candidate score is not attributed to the canonical PR login")
    if candidate.get("sha") != args.head_sha:
        target_ok = False
        target_notes.append("candidate score does not identify the exact PR head commit")
    if not isinstance(baseline_speed, (int, float)) or not isinstance(candidate_speed, (int, float)):
        target_ok = False
        target_notes.append("paired baseline or candidate has no decode throughput")
    else:
        if args.mode == "competition":
            required = float(baseline_speed) * (1.0 + min_relative)
            if float(candidate_speed) <= required:
                target_ok = False
                target_notes.append(f"requires paired decode > {required:.4f}")
            if historical_speed is not None and float(candidate_speed) <= historical_speed:
                target_ok = False
                target_notes.append(f"requires leaderboard decode > {historical_speed:.4f}")
        else:
            required = regression_floor(float(baseline_speed), max_speed_regression)
            if float(candidate_speed) < required:
                target_ok = False
                target_notes.append(f"requires paired decode >= {required:.4f}")
    if not isinstance(candidate_ppl, (int, float)):
        target_ok = False
        target_notes.append("candidate has no PPL")
    elif quality_reference is not None:
        max_ppl = quality_reference * (1.0 + max_ppl_regression)
        if float(candidate_ppl) > max_ppl:
            target_ok = False
            target_notes.append(
                f"PPL {float(candidate_ppl):.4f} exceeds quality ceiling {max_ppl:.4f}"
            )
        else:
            target_notes.append(f"PPL ceiling {max_ppl:.4f}")
    if not target_ok:
        failures.append(model)
    reference_text = fmt(float(baseline_speed) if isinstance(baseline_speed, (int, float)) else None)
    if historical_speed is not None:
        reference_text += f" / best {historical_speed:.4f}"
    lines.append(
        f"| {model} | {fmt(float(candidate_speed) if isinstance(candidate_speed, (int, float)) else None)} "
        f"| {reference_text} | {fmt(float(candidate_ppl) if isinstance(candidate_ppl, (int, float)) else None)} "
        f"| {'pass' if target_ok else 'fail'} | {cell('; '.join(target_notes) or 'passed')} |"
    )
    result_rows.append(
        {
            "model": model,
            "passed": target_ok,
            "tokens_per_second": candidate_speed,
            "reference_tokens_per_second": baseline_speed,
            "historical_best_tokens_per_second": historical_speed,
            "perplexity": candidate_ppl,
            "notes": target_notes,
        }
    )

    regression_models = [item for item in args.regression_models.replace(",", " ").split() if item]
    for regression_model in regression_models:
        score_path = scores_dir / f"score-{regression_model}.json"
        score = load_json(score_path) if score_path.is_file() else {}
        entries = leaderboard_entries(regression_model)
        current_speed = best(entries, "tokens_per_second", higher=True)
        current_ppl = best(entries, "perplexity", higher=False)
        speed = score.get("tokens_per_second")
        ppl = score.get("perplexity")
        notes: list[str] = []
        ok = True
        if not score.get("passed"):
            ok = False
            notes.append(str(score.get("valid_note") or score.get("note") or "candidate failed"))
        if current_speed is None:
            ok = False
            notes.append("main has no trusted speed baseline")
        elif not isinstance(speed, (int, float)):
            ok = False
            notes.append("candidate has no decode throughput")
        else:
            minimum_speed = regression_floor(current_speed, max_speed_regression)
            if float(speed) < minimum_speed:
                ok = False
                notes.append(f"requires decode >= {minimum_speed:.4f}")
        if current_ppl is not None:
            max_ppl = current_ppl * (1.0 + max_ppl_regression)
            if not isinstance(ppl, (int, float)) or float(ppl) > max_ppl:
                ok = False
                notes.append(f"requires PPL <= {max_ppl:.4f}")
        if not ok:
            failures.append(regression_model)
        lines.append(
            f"| {regression_model} | {fmt(float(speed) if isinstance(speed, (int, float)) else None)} "
            f"| {fmt(current_speed)} | {fmt(float(ppl) if isinstance(ppl, (int, float)) else None)} "
            f"| {'pass' if ok else 'fail'} | {cell('; '.join(notes) or 'passed')} |"
        )
        result_rows.append(
            {
                "model": regression_model,
                "passed": ok,
                "tokens_per_second": speed,
                "reference_tokens_per_second": current_speed,
                "perplexity": ppl,
                "notes": notes,
            }
        )

    lines.extend(
        [
            "",
            (
                "Result: pass. Candidate may merge."
                if not failures
                else "Result: fail. Do not merge; failing models: " + ", ".join(failures) + "."
            ),
        ]
    )
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    if args.result_output:
        metadata = (
            load_json(Path(args.candidate_metadata)) if args.candidate_metadata else {}
        )
        result = {
            "schema_version": 1,
            "track": policy["track"],
            "mode": args.mode,
            "passed": not failures,
            "eligible_for_standings": args.mode == "competition" and not failures,
            "participant_login": args.participant,
            "participant_head_sha": args.head_sha,
            "submission_id": metadata.get("submission_id"),
            "runtime_revision": metadata.get("runtime_revision"),
            "runtime_url": metadata.get("runtime_url"),
            "candidate_variant": metadata.get("candidate_variant") or candidate.get("variant"),
            "candidate_quantization": metadata.get("candidate_quantization"),
            "validation_contract_sha256": contract_sha,
            "track_policy_sha256": policy_sha,
            "run_url": candidate.get("run_url"),
            "score_sha": candidate.get("sha"),
            "results": result_rows,
        }
        Path(args.result_output).write_text(json.dumps(result, indent=2) + "\n")
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
