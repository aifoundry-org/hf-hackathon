#!/usr/bin/env python3
"""Gate PR board scores against the current leaderboard baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_config_helpers import load_config, parse_model_selection

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
BENCHMARK_CONFIG_REL = ".github/ci/benchmark_config.json"
VALIDATION_ONLY_MODEL_CONFIG_KEYS = {"accuracy", "dump_magic", "dump_size"}
YOLO_VALIDATION_ONLY_MODEL_CONFIG_KEYS = {
    "accuracy",
    "benchmark_cases",
    "dump_magic",
    "dump_size",
    "reference_contract",
    "validation",
}
YOLO_INPUT_ADDRESS = 0x04A00000


def split_items(value: str | None) -> list[str]:
    return [item for item in (value or "").replace(",", " ").split() if item]


def norm(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


def is_under(path: str, prefix: str) -> bool:
    path = norm(path)
    prefix = norm(prefix).rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def repo_rel(path: str | Path | None) -> str | None:
    if not path:
        return None
    value = Path(path)
    if value.is_absolute():
        try:
            return norm(value.relative_to(REPO_ROOT))
        except ValueError:
            return norm(value)
    return norm(value)


def changed_files(base_ref: str) -> list[str]:
    if not base_ref:
        return []
    for rev_range in (f"{base_ref}...HEAD", f"{base_ref}..HEAD"):
        proc = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", rev_range],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            return [norm(line) for line in proc.stdout.splitlines() if line.strip()]
    return []


def raw_benchmark_config_from_ref(base_ref: str) -> dict[str, Any]:
    data = load_json_from_ref(BENCHMARK_CONFIG_REL, base_ref)
    return data if isinstance(data, dict) else {}


def raw_benchmark_config_local() -> dict[str, Any]:
    try:
        data = json.loads((REPO_ROOT / BENCHMARK_CONFIG_REL).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def strip_validation_only_keys(model_cfg: Any, model: str = "") -> Any:
    if not isinstance(model_cfg, dict):
        return model_cfg
    ignored = (
        YOLO_VALIDATION_ONLY_MODEL_CONFIG_KEYS
        if model == "yolo"
        else VALIDATION_ONLY_MODEL_CONFIG_KEYS
    )
    stripped = {
        key: value
        for key, value in model_cfg.items()
        if key not in ignored
    }
    if model == "yolo" and isinstance(stripped.get("file_loads"), list):
        stripped["file_loads"] = [
            load
            for load in stripped["file_loads"]
            if int(str(load.get("address", "-1")), 0) != YOLO_INPUT_ADDRESS
        ]
    return stripped


def model_config_submission_changed(model: str, base_ref: str) -> bool:
    old = raw_benchmark_config_from_ref(base_ref)
    new = raw_benchmark_config_local()
    old_model = old.get("models", {}).get(model, {}) if isinstance(old.get("models"), dict) else {}
    new_model = new.get("models", {}).get(model, {}) if isinstance(new.get("models"), dict) else {}
    return strip_validation_only_keys(old_model, model) != strip_validation_only_keys(
        new_model, model
    )


def is_yolo_validation_only_path(path: str) -> bool:
    return path == "ported_models/yolo/tools/host_reference.py" or (
        path.startswith("ported_models/yolo/assets/yolo/coco_")
        and path.endswith("_raw_480x640x3_uint8_rgb.bin")
    )


def source_model_root(model_cfg: dict[str, Any]) -> str | None:
    source = repo_rel(model_cfg.get("source"))
    if not source:
        return None
    parts = source.split("/")
    if len(parts) >= 2 and parts[0] == "ported_models":
        return "/".join(parts[:2])
    return str(Path(source).parent)


def configured_asset_paths(model_cfg: dict[str, Any]) -> set[str]:
    root = source_model_root(model_cfg)
    if not root or not root.startswith("ported_models/"):
        return set()

    rels: set[str] = set()
    loads = list(model_cfg.get("file_loads", []))
    for case in model_cfg.get("benchmark_cases", []):
        if isinstance(case, dict):
            loads.extend(case.get("file_loads", []))
    for load in loads:
        paths = load.get("paths") or [load.get("path")]
        for path in paths:
            if path:
                rels.add(norm(path))

    accuracies = [model_cfg.get("accuracy", {})]
    for case in model_cfg.get("benchmark_cases", []):
        if isinstance(case, dict):
            accuracies.append(case.get("accuracy", {}))
    for accuracy in accuracies:
        if not isinstance(accuracy, dict):
            continue
        paths = accuracy.get("reference_paths") or [accuracy.get("reference_path")]
        for path in paths:
            if path:
                rels.add(norm(path))

    return {norm(f"{root}/assets/{rel}") for rel in rels if rel}


def is_model_code_path(path: str) -> bool:
    name = Path(path).name.lower()
    if name in {"readme.md", "model.md", "third_party.md"}:
        return False
    if "/docs/" in path:
        return False
    return Path(path).suffix in {
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".s",
        ".S",
        ".sh",
        ".py",
        ".json",
        ".txt",
    }


def model_submission_changed(
    cfg: dict[str, Any],
    model: str,
    files: list[str],
    base_ref: str,
) -> bool:
    if not base_ref:
        return True
    model_cfg = cfg.get("models", {}).get(model, {})
    source = repo_rel(model_cfg.get("source"))
    config_path = repo_rel(model_cfg.get("_config_path"))
    artifacts_path = repo_rel(model_cfg.get("_artifacts_path"))
    root = source_model_root(model_cfg)

    for path in files:
        if model == "yolo" and is_yolo_validation_only_path(path):
            continue
        if path == BENCHMARK_CONFIG_REL:
            if model_config_submission_changed(model, base_ref):
                return True
            continue
        if config_path and path == config_path:
            return True
        if artifacts_path and path == artifacts_path:
            return True
        if source and path == source:
            return True
        if path in configured_asset_paths(model_cfg):
            return True
        if root and is_under(path, root) and is_model_code_path(path):
            return True

        artifacts = model_cfg.get("artifacts", {})
        if isinstance(artifacts, dict):
            for artifact in artifacts.values():
                if not isinstance(artifact, dict):
                    continue
                source_path = repo_rel(artifact.get("submodule_path"))
                if source_path and is_under(path, source_path):
                    return True

    return False


def metric_config(cfg: dict[str, Any], model: str) -> tuple[str, str, bool]:
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    metric = score_cfg.get("metric", cfg.get("primary_metric", "kernel_wait_s"))
    label = score_cfg.get("label", "Kernel wait" if metric == "kernel_wait_s" else metric)
    higher = bool(score_cfg.get("higher_is_better", not cfg.get("lower_is_better", True)))
    return str(metric), str(label), higher


def baseline_variant(cfg: dict[str, Any], model: str) -> str | None:
    model_cfg = cfg.get("models", {}).get(model, {})
    score_cfg = model_cfg.get("score", {})
    variant = score_cfg.get("baseline_variant")
    return str(variant) if variant else None


def validation_contract_sha256(cfg: dict[str, Any], model: str) -> str | None:
    model_cfg = cfg.get("models", {}).get(model, {})
    value = model_cfg.get("reference_contract")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_allows_legacy_quality_baseline(cfg: dict[str, Any], model: str) -> bool:
    model_cfg = cfg.get("models", {}).get(model, {})
    value = model_cfg.get("reference_contract")
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(contract.get("legacy_quality_baseline_compatible", False))


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


def model_requires_ppl(cfg: dict[str, Any], model: str) -> bool:
    model_cfg = cfg.get("models", {}).get(model, {})
    return bool(model_cfg.get("llama_server", {}).get("perplexity", {}).get("enabled", False))


def best_ppl_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [entry for entry in entries if isinstance(entry.get("perplexity"), (int, float))]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: entry["perplexity"])


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
    if metric in {"kernel_wait_s", "kernel_wait_per_image_s"}:
        return f"{float(value):.6f}s"
    if metric.endswith("tokens_per_second") or metric == "tokens_per_second":
        return f"{float(value):.4f}"
    return f"{float(value):.4f}"


def cell(value: Any, limit: int = 120) -> str:
    flat = " ".join(str(value).split()).replace("|", "\\|")
    return (flat[: limit - 1] + "...") if len(flat) > limit else flat


def is_yolo_real_image_path(path: str) -> bool:
    return path.startswith("ported_models/yolo/src/")


def uncovered_note(path: str) -> str:
    if is_yolo_real_image_path(path):
        return (
            "YOLO source changed, but no configured five-image COCO host-reference "
            "benchmark covers it. The gate must validate classes, scores, and boxes "
            "against the pinned reference model."
        )
    return "Changed runtime source is not built by any configured benchmark row. Add or update a benchmark row before merging."


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
    parser.add_argument("--uncovered", default="")
    parser.add_argument("--target", choices=("all", "sysemu", "board"), default="board")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--expected-ref", default="")
    parser.add_argument("--expected-run-url", default="")
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=float(os.environ.get("LEADERBOARD_MIN_RELATIVE_IMPROVEMENT", "0")),
        help="Require this fractional improvement over the current best score, e.g. 0.01 for 1%.",
    )
    parser.add_argument(
        "--max-ppl-regression",
        type=float,
        default=float(os.environ.get("LEADERBOARD_MAX_PPL_REGRESSION", "0.20")),
        help="Allow at most this fractional PPL regression from the best seen PPL for perplexity-gated models.",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Fail selected models that have no compatible base-branch leaderboard entry.",
    )
    parser.add_argument(
        "--force-submission-models",
        default="",
        help="Treat these models as participant submissions even when the trusted tree contains no source diff.",
    )
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    models = parse_model_selection(args.models, cfg, target=args.target) if args.models.strip() else []
    unregistered = split_items(args.unregistered)
    uncovered = split_items(args.uncovered)
    scores_dir = Path(args.scores_dir)
    failed = False

    lines = [
        "## Leaderboard Gate",
        "",
        (
            "Policy: every selected model must pass board CI and beat the current base-branch "
            "leaderboard value for its primary metric. Models with llama-perplexity enabled "
            f"must also stay within {args.max_ppl_regression:.0%} of the best seen PPL. "
            "CI/scoring-only changes must pass board CI but do not need to improve runtime. "
            "Fidelity-sensitive paths need their model-specific validation row; YOLO changes "
            "require five-image COCO agreement with the pinned host reference."
        ),
        "",
        "| Model | Metric | PR score | Current best | Verdict | Notes |",
        "|-------|--------|----------|--------------|---------|-------|",
    ]

    if not models and not unregistered and not uncovered:
        lines.append("| - | - | - | - | pass | No changed board leaderboard models were selected. |")

    files = changed_files(args.base_ref)
    submission_changed = {
        model: model_submission_changed(cfg, model, files, args.base_ref)
        for model in models
    }
    for model in split_items(args.force_submission_models):
        if model in submission_changed:
            submission_changed[model] = True

    for port in unregistered:
        failed = True
        lines.append(
            f"| {cell(port)} | - | - | - | fail | New port has no benchmark config entry, so it cannot be gated. |"
        )

    for path in uncovered:
        failed = True
        lines.append(
            f"| {cell(path)} | - | - | - | fail | {cell(uncovered_note(path))} |"
        )

    for model in models:
        metric, label, higher = metric_config(cfg, model)
        score_path = scores_dir / f"score-{model}.json"
        entries = leaderboard_entries(model, args.base_ref)
        quality_entries = list(entries)
        required_variant = baseline_variant(cfg, model)
        if required_variant:
            entries = [entry for entry in entries if entry.get("variant") == required_variant]
        required_contract_sha = validation_contract_sha256(cfg, model)
        if required_contract_sha:
            entries = [
                entry
                for entry in entries
                if entry.get("validation_contract_sha256") == required_contract_sha
            ]
            if not contract_allows_legacy_quality_baseline(cfg, model):
                quality_entries = list(entries)
        baseline = best_entry(entries, metric, higher)
        baseline_value = float(baseline[metric]) if baseline else None
        baseline_text = fmt_metric(baseline_value, metric)
        baseline_team = baseline.get("team") if baseline else None
        if baseline_team:
            baseline_text = f"{baseline_text} ({cell(baseline_team, limit=40)})"
        requires_ppl = model_requires_ppl(cfg, model)
        baseline_ppl = best_ppl_entry(quality_entries)
        baseline_ppl_value = float(baseline_ppl["perplexity"]) if baseline_ppl else None

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

        provenance = {
            "sha": args.expected_sha,
            "ref": args.expected_ref,
            "run_url": args.expected_run_url,
        }
        mismatches = [
            key
            for key, expected in provenance.items()
            if expected and score.get(key) != expected
        ]
        if mismatches:
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | - | {baseline_text} | fail | "
                f"Score provenance does not match this workflow run: {cell(', '.join(mismatches))}. |"
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
        if required_contract_sha and score.get("validation_contract_sha256") != required_contract_sha:
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | Score was not produced by the current validation contract. |"
            )
            continue
        if value is None:
            failed = True
            lines.append(
                f"| {model} | {cell(label)} | - | {baseline_text} | fail | Passing score has no `{metric}` value. |"
            )
            continue

        ppl_note = ""
        if requires_ppl:
            ppl_value = score.get("perplexity")
            if not isinstance(ppl_value, (int, float)):
                failed = True
                lines.append(
                    f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | Passing score has no PPL value. |"
                )
                continue
            if baseline_ppl_value is not None:
                max_allowed_ppl = baseline_ppl_value * (1.0 + args.max_ppl_regression)
                if float(ppl_value) > max_allowed_ppl:
                    failed = True
                    lines.append(
                        f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | PPL {float(ppl_value):.2f} is worse than allowed max {max_allowed_ppl:.2f}; best seen PPL is {baseline_ppl_value:.2f}. |"
                    )
                    continue
                ppl_note = f" PPL {float(ppl_value):.2f} is within {args.max_ppl_regression:.0%} of best seen {baseline_ppl_value:.2f}."
            else:
                ppl_note = f" PPL {float(ppl_value):.2f} recorded for new baseline."

        if baseline_value is None:
            if args.require_baseline:
                failed = True
                lines.append(
                    f"| {model} | {cell(label)} | {score_text} | none | fail | Main has no compatible validated baseline; wait for the main-branch board update and re-run this gate. |"
                )
            else:
                lines.append(
                    f"| {model} | {cell(label)} | {score_text} | none | pass | First valid leaderboard score for this model.{ppl_note} |"
                )
            continue

        if not submission_changed.get(model, True):
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | pass | Board score passed for a CI/scoring-only change; no leaderboard improvement required.{ppl_note} |"
            )
            continue

        if beats_baseline(value, baseline_value, higher, args.min_relative_improvement):
            direction = "higher" if higher else "lower"
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | pass | New score is {direction} than current best.{ppl_note} |"
            )
        else:
            failed = True
            comparator = ">" if higher else "<"
            required = baseline_value * (
                1.0 + args.min_relative_improvement if higher else 1.0 - args.min_relative_improvement
            )
            if model == "yolo" and not higher:
                comparison = "slower than" if value > baseline_value else "not faster than"
                note = (
                    "Correctness passed on all 5 images, but "
                    f"{fmt_metric(value, metric)} is {comparison} main's "
                    f"{fmt_metric(baseline_value, metric)}."
                )
                if args.min_relative_improvement:
                    note += f" Requires {comparator} {fmt_metric(required, metric)}."
            else:
                note = f"Requires {comparator} {fmt_metric(required, metric)}."
            lines.append(
                f"| {model} | {cell(label)} | {score_text} | {baseline_text} | fail | {note} |"
            )

    lines.append("")
    if failed:
        lines.append("Result: fail. Do not merge until every touched leaderboard model passes and improves.")
    else:
        lines.append("Result: pass. The touched leaderboard models satisfy the applicable board gate.")

    write_markdown(args.output, lines)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
