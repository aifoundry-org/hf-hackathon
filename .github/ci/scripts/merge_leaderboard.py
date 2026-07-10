#!/usr/bin/env python3
"""Merge CI scores into per-model leaderboard JSON under data/."""

from __future__ import annotations

import argparse
import hashlib
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


def metric_config(model: str) -> tuple[str, bool]:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", cfg.get("primary_metric", "kernel_wait_s"))
    higher = bool(score_cfg.get("higher_is_better", not cfg.get("lower_is_better", True)))
    return metric, higher


def baseline_variant(model: str) -> str | None:
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    variant = score_cfg.get("baseline_variant")
    return str(variant) if variant else None


def validation_contract_sha256(model: str) -> str | None:
    cfg = load_config(CONFIG_PATH)
    value = cfg.get("models", {}).get(model, {}).get("reference_contract")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    entries = data if isinstance(data, list) else data.get("entries", [])
    required_variant = baseline_variant(model)
    if required_variant:
        entries = [entry for entry in entries if entry.get("variant") == required_variant]
    required_contract_sha = validation_contract_sha256(model)
    if required_contract_sha:
        entries = [
            entry
            for entry in entries
            if entry.get("validation_contract_sha256") == required_contract_sha
        ]
    return entries


def save_board(model: str, entries: list) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{model}.json"
    metric, higher = metric_config(model)
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

    metric, higher_is_better = metric_config(score["model"])
    metric_value = score.get(metric)
    if metric_value is None:
        return entries
    team = score.get("team") or "unknown"
    sha = score.get("sha") or ""

    new = {
        "team": team,
        "variant": score.get("variant"),
        "kernel_wait_s": score.get("kernel_wait_s"),
        "kernel_wait_per_image_s": score.get("kernel_wait_per_image_s"),
        "tokens_per_second": score.get("tokens_per_second"),
        "prompt_tokens_per_second": score.get("prompt_tokens_per_second"),
        "prompt_tokens": score.get("prompt_tokens"),
        "completion_tokens": score.get("completion_tokens"),
        "total_tokens": score.get("total_tokens"),
        "perplexity": score.get("perplexity"),
        "perplexity_error": score.get("perplexity_error"),
        "perplexity_tokens": score.get("perplexity_tokens"),
        "perplexity_prompt_tokens_per_second": score.get("perplexity_prompt_tokens_per_second"),
        "validation_contract_sha256": score.get("validation_contract_sha256"),
        "sha": sha,
        "ref": score.get("ref"),
        "run_url": score.get("run_url"),
        "scored_at": score.get("scored_at"),
    }
    if metric not in new:
        new[metric] = score.get(metric)

    # Replace same team's prior entry, then sort.
    entries = [e for e in entries if e.get("team") != team]
    entries.append(new)
    entries.sort(
        key=lambda e: e.get(metric) if e.get(metric) is not None else (-1e99 if higher_is_better else 1e99),
        reverse=higher_is_better,
    )
    return entries[:50]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--models", default="")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    changed = False
    for model in selected_models(args.models):
        score_path = scores_dir / f"score-{model}.json"
        if not score_path.is_file():
            continue
        score = json.loads(score_path.read_text())
        before = load_board(model)
        after = merge_entry(before, score)
        if after != before:
            save_board(model, after)
            changed = True
            print(f"updated leaderboard for {model}")
        else:
            print(f"no leaderboard change for {model}")

    subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github" / "ci" / "scripts" / "update_readme_leaderboard.py")],
        check=True,
    )

    return 0 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
