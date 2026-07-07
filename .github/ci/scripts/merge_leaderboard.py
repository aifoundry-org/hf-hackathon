#!/usr/bin/env python3
"""Merge CI scores into per-model leaderboard JSON under data/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config_helpers import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
LEADERBOARD_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


def configured_models() -> list[str]:
    cfg = load_config(CONFIG_PATH)
    return list(cfg.get("models", {}).keys())


def metric_config(model: str) -> tuple[str, str, bool]:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", cfg.get("primary_metric", "kernel_wait_s"))
    label = score_cfg.get("label", "Kernel wait" if metric == "kernel_wait_s" else metric)
    higher = bool(score_cfg.get("higher_is_better", not cfg.get("lower_is_better", True)))
    return metric, label, higher


def selected_models(value: str | None) -> list[str]:
    names = configured_models()
    if not value:
        return names
    selected = [item for item in value.replace(",", " ").split() if item]
    unknown = [item for item in selected if item not in names]
    if unknown:
        raise SystemExit(
            "unknown benchmark model(s): "
            + ", ".join(unknown)
            + ". configured models: "
            + ", ".join(names)
        )
    return selected


def load_board(model: str) -> list:
    path = LEADERBOARD_DIR / f"{model}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("entries", [])


def save_board(model: str, entries: list) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{model}.json"
    metric, _, higher = metric_config(model)
    payload = {
        "model": model,
        "metric": metric,
        "lower_is_better": not higher,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def merge_entry(entries: list, score: dict) -> list:
    if not score.get("passed"):
        return entries

    metric, _, higher_is_better = metric_config(score["model"])
    metric_value = score.get(metric)
    if metric_value is None:
        return entries
    team = score.get("team") or "unknown"
    sha = score.get("sha") or ""

    new = {
        "team": team,
        "variant": score.get("variant"),
        "kernel_wait_s": score.get("kernel_wait_s"),
        "tokens_per_second": score.get("tokens_per_second"),
        "prompt_tokens_per_second": score.get("prompt_tokens_per_second"),
        "prompt_tokens": score.get("prompt_tokens"),
        "completion_tokens": score.get("completion_tokens"),
        "total_tokens": score.get("total_tokens"),
        "perplexity": score.get("perplexity"),
        "perplexity_error": score.get("perplexity_error"),
        "perplexity_tokens": score.get("perplexity_tokens"),
        "perplexity_prompt_tokens_per_second": score.get("perplexity_prompt_tokens_per_second"),
        "sha": sha,
        "ref": score.get("ref"),
        "run_url": score.get("run_url"),
        "scored_at": score.get("scored_at"),
    }

    # Replace same team's prior entry, then sort.
    entries = [e for e in entries if e.get("team") != team]
    entries.append(new)
    entries.sort(
        key=lambda e: e.get(metric) if e.get(metric) is not None else (-1e99 if higher_is_better else 1e99),
        reverse=higher_is_better,
    )
    return entries[:50]


def fmt_metric_value(metric: str, value: object) -> str:
    if value is None:
        return "-"
    if metric == "kernel_wait_s":
        return f"{float(value):.6f}s"
    if metric.endswith("tokens_per_second") or metric == "tokens_per_second":
        return f"{float(value):.2f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def fmt_ppl(entry: dict) -> str:
    value = entry.get("perplexity")
    if value is None:
        return ""
    text = f"{float(value):.2f}"
    error = entry.get("perplexity_error")
    if error is not None:
        return f"{text} (+/- {float(error):.2f})"
    return text


def same_best_entry(left: dict | None, right: dict | None, metric: str) -> bool:
    if not left or not right:
        return left == right
    keys = ("team", "variant", "sha", metric)
    return all(left.get(key) == right.get(key) for key in keys)


def announcement_for_change(model: str, before: list, after: list) -> str | None:
    if not after:
        return None
    metric, label, _ = metric_config(model)
    old_best = before[0] if before else None
    new_best = after[0]
    if same_best_entry(old_best, new_best, metric):
        return None

    team = new_best.get("team") or "unknown"
    metric_text = f"{label} {fmt_metric_value(metric, new_best.get(metric))}"
    ppl = fmt_ppl(new_best)
    if ppl:
        metric_text += f", PPL {ppl}"

    if old_best:
        old_team = old_best.get("team") or "unknown"
        old_metric = fmt_metric_value(metric, old_best.get(metric))
        return f"{model} has been beaten by {team}: {metric_text} (was {old_metric} by {old_team})."
    return f"{model} has its first leaderboard score by {team}: {metric_text}."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument(
        "--announcements-output",
        default="",
        help="Optional JSON file of short messages for newly beaten leaderboard bests.",
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    changed = False
    announcements: list[str] = []
    for model in selected_models(args.models):
        score_path = scores_dir / f"score-{model}.json"
        if not score_path.is_file():
            continue
        score = json.loads(score_path.read_text())
        before = load_board(model)
        after = merge_entry(before, score)
        if after != before:
            announcement = announcement_for_change(model, before, after)
            if announcement:
                announcements.append(announcement)
            save_board(model, after)
            changed = True
            print(f"updated leaderboard for {model}")
        else:
            print(f"no leaderboard change for {model}")

    if args.announcements_output:
        out = Path(args.announcements_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(announcements, indent=2) + "\n")

    subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github" / "ci" / "scripts" / "update_readme_leaderboard.py")],
        check=True,
    )

    return 0 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
