#!/usr/bin/env python3
"""Build markdown PR comment from per-model score JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_config_helpers import (
    load_config,
    model_names as configured_model_names,
    parse_model_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


def selected_model_names(target: str = "all", selection: str | None = None) -> list[str]:
    data = load_config(CONFIG_PATH)
    if selection is None:
        return configured_model_names(data, target, default_only=False)
    if not selection.strip():
        return []
    return parse_model_selection(selection, data, target=target)


def metric_config(model: str) -> tuple[str, str]:
    data = load_config(CONFIG_PATH)
    model_cfg = data.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", data.get("primary_metric", "kernel_wait_s"))
    label = score_cfg.get("label", "Kernel wait" if metric == "kernel_wait_s" else metric)
    return metric, label


def load_score(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def cell(text: str, limit: int = 100) -> str:
    """Make a value safe for a single markdown table cell.

    Collapses newlines/whitespace (a raw \\n breaks the whole table from that
    row down) and escapes pipes, then truncates. Failed-model notes carry
    multi-line server logs, so this is essential.
    """
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return (flat[: limit - 1] + "…") if len(flat) > limit else flat


def is_yolo_real_image_path(path: str) -> bool:
    return path.startswith("ported_models/yolo/src/")


def uncovered_note(path: str) -> str:
    if is_yolo_real_image_path(path):
        return (
            "YOLO source changed; no configured five-image COCO host-reference "
            "benchmark covers it"
        )
    return "changed runtime source is not built by any configured benchmark row"


def fmt_ppl(score: dict) -> str:
    value = score.get("perplexity")
    if value is None:
        return "—"
    text = f"{value:.2f}"
    error = score.get("perplexity_error")
    if error is not None:
        return f"{text} (+/- {error:.2f})"
    return text


def fmt_score(model: str, score: dict) -> tuple[str, str]:
    metric, label = metric_config(model)
    value = score.get(metric)
    if value is None:
        return label, "—"
    if metric in {"kernel_wait_s", "kernel_wait_per_image_s"}:
        return label, f"{value:.6f}s"
    if metric.endswith("tokens_per_second") or metric == "tokens_per_second":
        return label, f"{value:.2f}"
    return label, f"{value:.3f}" if isinstance(value, float) else str(value)


def status_badge(score: dict) -> str:
    if score.get("status") == "skipped":
        return "⏭️ skipped"
    if score.get("passed"):
        return "✅ pass"
    st = score.get("status", "fail")
    if st == "timeout":
        return "⏱️ timeout"
    return f"❌ {st}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha", default="")
    parser.add_argument("--ref", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--target", choices=("all", "sysemu", "board"), default="all")
    parser.add_argument("--models", default=None)
    parser.add_argument(
        "--unregistered",
        default="",
        help="Space/comma-separated ported_models ports touched by the PR that have no benchmark_config.json entry.",
    )
    parser.add_argument(
        "--uncovered",
        default="",
        help="Space/comma-separated runtime source files touched by the PR that are not built by any benchmark row.",
    )
    parser.add_argument("--title", default="ET-SoC1 sys-emu benchmark")
    parser.add_argument(
        "--footer",
        default="Sys-emu is slower than silicon; timeouts do not always mean the kernel is wrong.",
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    models = selected_model_names(args.target, args.models)
    scores = {m: load_score(scores_dir / f"score-{m}.json") for m in models}
    unregistered = [p for p in args.unregistered.replace(",", " ").split() if p]
    uncovered = [p for p in args.uncovered.replace(",", " ").split() if p]

    lines = [
        f"## {args.title}",
        "",
    ]
    if args.run_url:
        lines.append(f"Workflow: [{args.run_url}]({args.run_url})")
        lines.append("")
    if args.sha:
        lines.append(f"Commit: `{args.sha[:12]}` · ref `{args.ref}`")
        lines.append("")

    lines.extend(
        [
            "| Model | Variant | Status | Metric | Score | PPL | Notes |",
            "|-------|---------|--------|--------|-------|-----|-------|",
        ]
    )

    for model in models:
        score = scores.get(model)
        if score is None:
            lines.append(f"| {model} | — | ⚠️ missing | — | — | — | no score artifact |")
            continue
        note = score.get("valid_note") or score.get("note") or ""
        if score.get("status") == "skipped":
            note = score.get("note") or "skipped"
        metric, value = fmt_score(model, score)
        lines.append(
            "| {model} | `{variant}` | {badge} | {metric} | {value} | {ppl} | {note} |".format(
                model=model,
                variant=cell(score.get("variant", "—"), limit=48),
                badge=status_badge(score),
                metric=metric,
                value=value,
                ppl=fmt_ppl(score),
                note=cell(note),
            )
        )

    for port in unregistered:
        lines.append(
            f"| {port} | — | ⚠️ not registered | — | — | — | "
            "new port; no benchmark_config.json entry |"
        )

    for path in uncovered:
        lines.append(
            f"| {cell(path, limit=80)} | — | ⚠️ not benchmarked | — | — | — | "
            f"{cell(uncovered_note(path))} |"
        )

    passed = [m for m in models if scores.get(m) and scores[m].get("passed")]
    lines.append("")
    if passed:
        lines.append(
            f"**Passed:** {', '.join(passed)}. "
            "Leaderboard updates on push to `main` after merge."
        )
    elif models:
        lines.append(
            "No model passed the canonical smoke variant yet. "
            "See workflow logs for `run.log` artifacts."
        )

    if unregistered:
        lines.append("")
        lines.append(
            "**New ports detected without a benchmark entry:** "
            + ", ".join(f"`{p}`" for p in unregistered)
            + ". Add a `models` entry in `.github/ci/benchmark_config.json` "
            "(pointing at the port's `benchmark.json`) so CI benchmarks it."
        )

    if uncovered:
        lines.append("")
        lines.append(
            "**Changed runtime source is not covered by an adequate benchmark row:** "
            + ", ".join(f"`{p}`" for p in uncovered)
            + ". Update the configured benchmark source, add a benchmark row for the new variant, "
            "or keep the change out of the submission. YOLO changes require a "
            "five-image category detection row."
        )

    lines.append("")
    if args.footer:
        lines.append(f"_{args.footer}_")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
