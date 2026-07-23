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


def hardware_epoch() -> str:
    value = load_config(CONFIG_PATH).get("board", {}).get("hardware", {}).get("epoch")
    if not value:
        raise RuntimeError("benchmark config has no board.hardware.epoch")
    return str(value)


def validate_score_set(scores_dir: Path, models: list[str]) -> dict[str, dict]:
    """Load a complete, passing score set before mutating leaderboard files."""
    cfg = load_config(CONFIG_PATH)
    policy = cfg.get("board", {}).get("hardware", {})
    expected_hardware = {
        "hardware_epoch": policy.get("epoch"),
        "board_id": policy.get("board_id"),
        "minion_frequency_mhz": policy.get("minion_frequency_mhz"),
        "noc_frequency_mhz": policy.get("noc_frequency_mhz"),
        "tdp_w": policy.get("tdp_w"),
    }
    scores: dict[str, dict] = {}
    errors: list[str] = []

    for model in models:
        score_path = scores_dir / f"score-{model}.json"
        if not score_path.is_file():
            errors.append(f"{model}: missing score artifact")
            continue
        try:
            score = json.loads(score_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{model}: invalid score artifact ({exc})")
            continue
        if not isinstance(score, dict):
            errors.append(f"{model}: score artifact is not an object")
            continue

        model_errors: list[str] = []
        if score.get("model") != model:
            model_errors.append(f"model={score.get('model')!r}")
        if not score.get("passed"):
            model_errors.append("passed is not true")
        if score.get("ref") != "refs/heads/main":
            model_errors.append(f"ref={score.get('ref')!r}")
        if not score.get("sha"):
            model_errors.append("sha is missing")
        if not score.get("run_url"):
            model_errors.append("run_url is missing")

        metric, _ = metric_config(model)
        if not isinstance(score.get(metric), (int, float)):
            model_errors.append(f"{metric} is missing or non-numeric")

        hardware = score.get("hardware")
        actual_hardware = {
            "hardware_epoch": score.get("hardware_epoch"),
            "board_id": hardware.get("board_id") if isinstance(hardware, dict) else None,
            "minion_frequency_mhz": (
                hardware.get("minion_frequency_mhz")
                if isinstance(hardware, dict)
                else None
            ),
            "noc_frequency_mhz": (
                hardware.get("noc_frequency_mhz")
                if isinstance(hardware, dict)
                else None
            ),
            "tdp_w": hardware.get("tdp_w") if isinstance(hardware, dict) else None,
        }
        mismatches = [
            key
            for key, expected in expected_hardware.items()
            if actual_hardware.get(key) != expected
        ]
        if mismatches:
            model_errors.append("hardware mismatch: " + ", ".join(mismatches))
        if not isinstance(hardware, dict) or not hardware.get("boot_id"):
            model_errors.append("hardware boot_id is missing")

        required_variant = baseline_variant(model)
        if required_variant and score.get("variant") != required_variant:
            model_errors.append(
                f"variant={score.get('variant')!r}, expected {required_variant!r}"
            )
        required_contract_sha = validation_contract_sha256(model)
        if (
            required_contract_sha
            and score.get("validation_contract_sha256") != required_contract_sha
        ):
            model_errors.append("validation contract does not match")

        if model_errors:
            errors.append(f"{model}: " + "; ".join(model_errors))
        else:
            scores[model] = score

    if errors:
        raise SystemExit(
            "refusing a partial or untrusted leaderboard update:\n- "
            + "\n- ".join(errors)
        )
    return scores


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


def eligible_entries(model: str, entries: list) -> list:
    values = [entry for entry in entries if isinstance(entry, dict)]
    required_variant = baseline_variant(model)
    if required_variant:
        values = [entry for entry in values if entry.get("variant") == required_variant]
    required_contract_sha = validation_contract_sha256(model)
    if required_contract_sha:
        values = [
            entry
            for entry in values
            if entry.get("validation_contract_sha256") == required_contract_sha
        ]
    return values


def load_board_state(model: str) -> tuple[list, list]:
    path = LEADERBOARD_DIR / f"{model}.json"
    if not path.is_file():
        return [], []
    data = json.loads(path.read_text())
    entries = data if isinstance(data, list) else data.get("entries", [])
    existing_legacy = [] if isinstance(data, list) else data.get("legacy_entries", [])
    current_epoch = hardware_epoch()
    active_candidates = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get("hardware_epoch") == current_epoch
    ]
    active = eligible_entries(model, active_candidates)
    active_ids = {id(entry) for entry in active}
    legacy = [entry for entry in existing_legacy if isinstance(entry, dict)]
    legacy.extend(
        entry for entry in entries
        if isinstance(entry, dict) and id(entry) not in active_ids
    )
    return active, legacy


def load_board(model: str) -> list:
    entries, _ = load_board_state(model)
    return entries


def save_board(model: str, entries: list, legacy_entries: list | None = None) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{model}.json"
    metric, higher = metric_config(model)
    payload = {
        "model": model,
        "metric": metric,
        "lower_is_better": not higher,
        "hardware_epoch": hardware_epoch(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    if legacy_entries:
        payload["legacy_entries"] = legacy_entries
    path.write_text(json.dumps(payload, indent=2) + "\n")


def bootstrap_participant(model: str, score: dict, legacy_entries: list) -> str | None:
    """Keep the incumbent owner when main establishes a new hardware epoch."""
    if score.get("ref") != "refs/heads/main":
        return None
    if score.get("hardware_epoch") != hardware_epoch():
        return None
    candidates = eligible_entries(model, legacy_entries)
    metric, higher = metric_config(model)
    candidates = [
        entry for entry in candidates if isinstance(entry.get(metric), (int, float))
    ]
    if not candidates:
        return None
    incumbent = (max if higher else min)(
        candidates, key=lambda entry: float(entry[metric])
    )
    return str(incumbent.get("participant_login") or incumbent.get("team") or "") or None


def merge_entry(
    entries: list, score: dict, *, participant_login: str | None = None
) -> list:
    if not score.get("passed"):
        return entries

    metric, higher_is_better = metric_config(score["model"])
    metric_value = score.get(metric)
    if metric_value is None:
        return entries
    team = participant_login or score.get("team") or "unknown"
    participant_login = participant_login or team
    sha = score.get("sha") or ""

    new = {
        "team": team,
        "participant_login": participant_login,
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
        "hardware_epoch": score.get("hardware_epoch"),
        "hardware": score.get("hardware"),
        "sha": sha,
        "ref": score.get("ref"),
        "run_url": score.get("run_url"),
        "scored_at": score.get("scored_at"),
    }
    if metric not in new:
        new[metric] = score.get(metric)

    participant_entries = [
        entry
        for entry in entries
        if (entry.get("participant_login") or entry.get("team"))
        == participant_login
    ]
    participant_metrics = [
        entry.get(metric)
        for entry in participant_entries
        if entry.get(metric) is not None
    ]
    if participant_metrics:
        incumbent = (
            max(participant_metrics)
            if higher_is_better
            else min(participant_metrics)
        )
        improved = (
            metric_value > incumbent
            if higher_is_better
            else metric_value < incumbent
        )
        if not improved:
            return entries

    # Replace the same canonical participant's prior entry only when the new
    # score is strictly better, then sort.
    entries = [
        e
        for e in entries
        if (e.get("participant_login") or e.get("team")) != participant_login
    ]
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
    parser.add_argument("--participant-login", default="")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    models = selected_models(args.models)
    scores = validate_score_set(scores_dir, models)
    changed = False
    for model in models:
        score = scores[model]
        before, legacy_entries = load_board_state(model)
        participant_login = args.participant_login or None
        if not before:
            participant_login = (
                bootstrap_participant(model, score, legacy_entries)
                or participant_login
            )
        after = merge_entry(
            before,
            score,
            participant_login=participant_login,
        )
        if after != before:
            save_board(model, after, legacy_entries)
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
