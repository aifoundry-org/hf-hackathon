#!/usr/bin/env python3
"""Gate PR board scores against the current leaderboard baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_config_helpers import load_config, parse_model_selection

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


def split_items(value: str | None) -> list[str]:
    return [item for item in (value or "").replace(",", " ").split() if item]


def metric_config(cfg: dict[str, Any], model: str) -> tuple[str, str, bool]:
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", cfg.get("primary_metric", "kernel_wait_s"))
    label = score_cfg.get("label", "Kernel wait" if metric == "kernel_wait_s" else metric)
    higher = bool(score_cfg.get("higher_is_better", not cfg.get("lower_is_better", True)))
    return str(metric), str(label), higher


def load_json_from_ref(path: str, base_ref: str) -> Any | None:
    if base_ref:
        proc = subprocess.run(
            ["git", "show", f"{base_ref}:{path}"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)

    local = REPO_ROOT / path
    if not local.is_file():
        return None
    return json.loads(local.read_text())


def leaderboard_entries(model: str, base_ref: str) -> list[dict[str, Any]]:
    data = load_json_from_ref(f"data/{model}.json", base_ref)
    if data is None:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        entries = data.get("entries", [])
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def best_entry(entries: list[dict[str, Any]], metric: str, higher: bool) -> dict[str, Any] | None:
    candidates = [entry for entry in entries if isinstance(entry.get(metric), (int, float))]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry[metric]) if higher else min(candidates, key=lambda entry: entry[metric])


def score_value(score: dict[str, Any], metric: str) -> float | None:
    value = score.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def beats_baseline(value: float, baseline: float, higher: bool, min_relative: float) -> bool:
    if higher:
        return value > baseline * (1.0 + min_relative)
    return value < baseline * (1.0 - min_relative)


def fmt_metric(value: float | int | None, metric: str) -> str:
    if value is None:
        return "-"
    if metric == "kernel_wait_s":
        return f"{float(value):.6f}s"
    if metric.endswith("tokens_per_second") or metric == "tokens_per_second":
        return f"{float(value):.4f}"
    return f"{float(value):.4f}"


def cell(value: Any, limit: int = 120) -> str:
    flat = " ".join(str(value).split()).replace("|", "\\|")
    return (flat[: limit - 1] + "...") if len(flat) > limit else flat


def write_markdown(path: str, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    if path:
        Path(path).write_text(text)
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--unregistered", default="")
    parser.add_argument("--target", choices=("all", "sysemu", "board"), default="board")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=float(os.environ.get("LEADERBOARD_MIN_RELATIVE_IMPROVEMENT", "0")),
        help="Require this fractional improvement over the current best score, e.g. 0.01 for 1%.",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    models = parse_model_selection(args.models, cfg, target=args.target) if args.models.strip() else []
    unregistered = split_items(args.unregistered)
    scores_dir = Path(args.scores_dir)
    failed = False

    lines = [
        "## Leaderboard Gate",
        "",
        "Policy: every selected model must pass board CI and beat the current base-branch leaderboard value for its primary metric.",
        "",
        "| Model | Metric | PR score | Current best | Verdict | Notes |",
        "|-------|--------|----------|--------------|---------|-------|",
    ]

    if not models and not unregistered:
        lines.append("| - | - | - | - | pass | No changed board leaderboard models were selected. |")

    for port in unregistered:
        failed = True
        lines.append(
            f"| {cell(port)} | - | - | - | fail | New port has no benchmark config entry, so it cannot be gated. |"
        )

    for model in models:
        metric, label, higher = metric_config(cfg, model)
        score_path = scores_dir / f"score-{model}.json"
        baseline = best_entry(leaderboard_entries(model, args.base_ref), metric, higher)
        baseline_value = float(baseline[metric]) if baseline else None
        baseline_text = fmt_metric(baseline_value, metric)
        baseline_team = baseline.get("team") if baseline else None
        if baseline_team:
            baseline_text = f"{baseline_text} ({cell(baseline_team, limit=40)})"

        if not score_path.is_file():
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | - | {baseline_text} | fail | Missing score artifact. |"
            )
            continue

        try:
            score = json.loads(score_path.read_text())
        except json.JSONDecodeError as exc:
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | - | {baseline_text} | fail | Invalid score JSON: {cell(exc)}. |"
            )
            continue

        value = score_value(score, metric)
        score_text = fmt_metric(value, metric)
        if not score.get("passed"):
            failed = True
            note = score.get("valid_note") or score.get("note") or score.get("status") or "board score did not pass"
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | {cell(note)} |"
            )
            continue
        if value is None:
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | - | {baseline_text} | fail | Passing score has no `{metric}` value. |"
            )
            continue
        if baseline_value is None:
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | none | pass | First valid leaderboard score for this model. |"
            )
            continue

        if beats_baseline(value, baseline_value, higher, args.min_relative_improvement):
            direction = "higher" if higher else "lower"
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | pass | New score is {direction} than current best. |"
            )
        else:
            failed = True
            comparator = ">" if higher else "<"
            required = baseline_value * (
                1.0 + args.min_relative_improvement if higher else 1.0 - args.min_relative_improvement
            )
            note = f"Requires {comparator} {fmt_metric(required, metric)}."
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | {note} |"
            )

    lines.append("")
    if failed:
        lines.append("Result: fail. Do not merge until every touched leaderboard model passes and improves.")
    else:
        lines.append("Result: pass. The touched leaderboard models all improve or establish new baselines.")

    write_markdown(args.output, lines)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
