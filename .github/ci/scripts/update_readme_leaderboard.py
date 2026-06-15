#!/usr/bin/env python3
"""Render the board leaderboard block in README.md from data/."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_config_helpers import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"
LEADERBOARD_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
START = "<!-- leaderboard:start -->"
END = "<!-- leaderboard:end -->"


def config() -> dict:
    return load_config(CONFIG_PATH)


def model_names() -> list[str]:
    names = []
    for name, model_cfg in config().get("models", {}).items():
        if model_cfg.get("benchmark_default", True) or (LEADERBOARD_DIR / f"{name}.json").is_file():
            names.append(name)
    return names


def metric_config(model: str) -> tuple[str, str, bool]:
    data = config()
    model_cfg = data.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", data.get("primary_metric", "kernel_wait_s"))
    label = score_cfg.get("label", "Kernel wait" if metric == "kernel_wait_s" else metric)
    higher = bool(score_cfg.get("higher_is_better", not data.get("lower_is_better", True)))
    return metric, label, higher


def load_best(model: str) -> dict | None:
    path = LEADERBOARD_DIR / f"{model}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    entries = data.get("entries", [])
    return entries[0] if entries else None


def fmt_ppl(entry: dict | None) -> str:
    if not entry or entry.get("perplexity") is None:
        return "-"
    value = f"{entry['perplexity']:.2f}"
    error = entry.get("perplexity_error")
    if error is not None:
        return f"{value} (+/- {error:.2f})"
    return value


def fmt_metric(model: str, entry: dict | None) -> tuple[str, str]:
    metric, label, _ = metric_config(model)
    if not entry or entry.get(metric) is None:
        return label, "-"
    value = entry[metric]
    if metric == "kernel_wait_s":
        return label, f"{value:.6f}s"
    if metric.endswith("tokens_per_second") or metric == "tokens_per_second":
        return label, f"{value:.2f}"
    return label, f"{value:.3f}" if isinstance(value, float) else str(value)


def fmt_run(entry: dict | None) -> str:
    if not entry:
        return "-"
    run_url = entry.get("run_url")
    sha = (entry.get("sha") or "")[:7]
    if run_url:
        return f"[{sha or 'run'}]({run_url})"
    return sha or "-"


def render_block() -> str:
    lines = [
        START,
        "## ET-SoC1 Board Leaderboard",
        "",
        "Results are from real ET-SoC1 silicon via the main-branch board workflow. Each model uses its own primary metric.",
        "",
        "| Model | Best participant | Variant | Metric | Score | PPL | Run |",
        "|-------|------------------|---------|--------|-------|-----|-----|",
    ]
    for model in model_names():
        entry = load_best(model)
        team = entry.get("team", "-") if entry else "-"
        variant = entry.get("variant", "-") if entry else "-"
        metric_label, metric_value = fmt_metric(model, entry)
        lines.append(
            f"| {model} | {team} | `{variant}` | {metric_label} | {metric_value} | {fmt_ppl(entry)} | {fmt_run(entry)} |"
        )
    lines.extend(["", "Full JSON data lives in [`data/`](data/).", END])
    return "\n".join(lines)


def main() -> int:
    text = README.read_text()
    block = render_block()
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        new = before.rstrip() + "\n\n" + block + "\n\n" + after.lstrip("\n")
    else:
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            new = lines[0] + "\n\n" + block + "\n\n" + "\n".join(lines[1:]).lstrip() + "\n"
        else:
            new = block + "\n\n" + text
    README.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
