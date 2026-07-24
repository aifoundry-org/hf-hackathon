#!/usr/bin/env python3
"""Render deterministic individual model-port standings from the event ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_port_claim import (
    load_json,
    parse_time,
    validate_credit_inventory,
    validate_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
START = "<!-- model-port-standings:start -->"
END = "<!-- model-port-standings:end -->"


def standings(
    policy: dict[str, Any],
    ledger: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    validate_policy(policy)
    credits = validate_credit_inventory(ledger, policy, registry)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for credit in credits.values():
        grouped.setdefault(str(credit["participant_login"]), []).append(credit)
    rows = []
    for login, participant_credits in grouped.items():
        ordered = sorted(
            participant_credits,
            key=lambda item: (
                parse_time(item["merged_at"], "credit.merged_at"),
                item["identity_id"],
            ),
        )
        rows.append(
            {
                "participant_login": login,
                "credits": len(ordered),
                "identity_ids": [item["identity_id"] for item in ordered],
                "benchmark_models": [item["benchmark_model"] for item in ordered],
                "final_qualifying_merge_at": ordered[-1]["merged_at"],
                "credit_ids": [item["credit_id"] for item in ordered],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["credits"],
            parse_time(
                row["final_qualifying_merge_at"],
                "final_qualifying_merge_at",
            ),
            row["participant_login"].lower(),
        )
    )
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return {
        "schema_version": 1,
        "track": policy["track"],
        "activation_mode": policy["activation_mode"],
        "metric": "active_unique_model_port_credits",
        "tie_break": policy["tie_break"],
        "standings": rows,
    }


def readme_section(payload: dict[str, Any], credits: dict[str, dict[str, Any]]) -> str:
    lines = [START, "## Most Models Ported by One Individual", ""]
    if payload["activation_mode"] == "shadow":
        lines.extend(
            [
                "The trusted credit system is in shadow mode while contest dates and the",
                "historical identity inventory are reviewed. No award credits have been issued.",
            ]
        )
    elif not payload["standings"]:
        lines.append("No eligible model-port credits have been issued yet.")
    else:
        by_credit = {record["credit_id"]: record for record in credits.values()}
        lines.extend(
            [
                "| Rank | GitHub participant | Distinct ports | Credited models |",
                "|------|--------------------|----------------|-----------------|",
            ]
        )
        for row in payload["standings"]:
            model_links = []
            for credit_id in row["credit_ids"]:
                credit = by_credit[credit_id]
                model_links.append(
                    f"[{credit['benchmark_model']}]({credit['trusted_run']['url']})"
                )
            lines.append(
                f"| {row['rank']} | [@{row['participant_login']}](https://github.com/{row['participant_login']}) "
                f"| {row['credits']} | {', '.join(model_links)} |"
            )
    lines.extend(["", END])
    return "\n".join(lines)


def replace_section(readme: Path, section: str) -> None:
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise RuntimeError("README model-port standings markers are missing or duplicated")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + section + after)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", default=str(REPO_ROOT / ".github/ci/reference/model_ports_track.json")
    )
    parser.add_argument("--ledger", default=str(REPO_ROOT / "data/model-port-credits.json"))
    parser.add_argument(
        "--identities", default=str(REPO_ROOT / "data/model-port-identities.json")
    )
    parser.add_argument("--output", default=str(REPO_ROOT / "data/model-port-standings.json"))
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    args = parser.parse_args()

    policy = load_json(Path(args.policy))
    ledger = load_json(Path(args.ledger))
    registry = load_json(Path(args.identities))
    payload = standings(policy, ledger, registry)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    credits = validate_credit_inventory(ledger, policy, registry)
    if args.readme:
        replace_section(Path(args.readme), readme_section(payload, credits))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
