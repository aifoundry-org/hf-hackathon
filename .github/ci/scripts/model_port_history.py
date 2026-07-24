#!/usr/bin/env python3
"""Validate the organizer-reviewed historical model-port award backfill."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from model_port_claim import (
    ClaimError,
    GITHUB_LOGIN_RE,
    HEX40_RE,
    HEX64_RE,
    IDENTITY_RE,
    MODEL_RE,
    file_sha256,
    git_blob,
    git_json,
    load_json,
    parse_time,
    run_git,
    safe_path,
    validate_credit_inventory,
    validate_policy,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_PATH = REPO_ROOT / "data" / "model-port-historical-review.json"
INELIGIBLE_OUTCOMES = {
    "baseline_port_root",
    "existing_execution_family",
    "seed_benchmark_model",
    "missing_model_oracle",
    "organizer_baseline",
}


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ClaimError(f"{label} is missing required fields: {', '.join(missing)}")


def _validate_git_provenance(
    repo: Path,
    *,
    participant_head_sha: str,
    merge_sha: str,
    reviewed_through_sha: str,
    require_head_parent: bool,
) -> None:
    for field, sha in (
        ("participant_head_sha", participant_head_sha),
        ("merge_sha", merge_sha),
    ):
        if not HEX40_RE.fullmatch(sha):
            raise ClaimError(f"historical decision has an invalid {field}")
        run_git(repo, "cat-file", "-e", f"{sha}^{{commit}}")
    parents = run_git(repo, "show", "-s", "--format=%P", merge_sha).split()
    if require_head_parent and participant_head_sha not in parents:
        raise ClaimError(
            f"historical participant head {participant_head_sha} is not a parent of {merge_sha}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", merge_sha, reviewed_through_sha],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ClaimError(
            f"historical merge {merge_sha} is not contained in reviewed-through commit"
        )


def _matching_score_entry(
    repo: Path,
    *,
    reviewed_through_sha: str,
    decision: dict[str, Any],
    credit: dict[str, Any],
) -> dict[str, Any]:
    model = credit["benchmark_model"]
    board = git_json(repo, reviewed_through_sha, f"data/{model}.json")
    assert board is not None
    entries = board.get("entries")
    if not isinstance(entries, list):
        raise ClaimError(f"historical score data for {model} has no entries list")
    trusted = credit["trusted_run"]
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("participant_login") == decision["participant_login"]
        and entry.get("sha") == decision["merge_sha"]
        and entry.get("run_url") == trusted["url"]
        and entry.get(trusted["metric"]) == trusted["metric_value"]
        and entry.get("validation_contract_sha256")
        == trusted["validation_contract_sha256"]
    ]
    if len(matches) != 1:
        raise ClaimError(
            f"historical credit {decision['review_id']} does not match exactly one board score"
        )
    return matches[0]


def validate_historical_review(
    *,
    repo: Path,
    policy: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    validate_policy(policy)
    identities = validate_registry(registry, policy)
    credits = validate_credit_inventory(ledger, policy, registry)

    if review.get("schema_version") != 1 or review.get("track") != policy["track"]:
        raise ClaimError("historical review has an unsupported schema or track")
    if review.get("historical_review_complete") is not True:
        raise ClaimError("historical review must explicitly be complete")
    if policy.get("historical_review_complete") is not True:
        raise ClaimError("policy does not mark the historical review complete")
    if review.get("baseline_sha") != policy["baseline_sha"]:
        raise ClaimError("historical review baseline does not match policy")
    if review.get("contest_start") != policy["contest_start"]:
        raise ClaimError("historical review contest start does not match policy")
    if review.get("contest_end") != policy["contest_end"]:
        raise ClaimError("historical review contest end does not match policy")
    parse_time(str(review.get("reviewed_at") or ""), "reviewed_at")

    reviewed_through_sha = str(review.get("reviewed_through_sha") or "")
    if not HEX40_RE.fullmatch(reviewed_through_sha):
        raise ClaimError("historical review requires a full reviewed_through_sha")
    run_git(repo, "cat-file", "-e", f"{reviewed_through_sha}^{{commit}}")

    decisions = review.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ClaimError("historical review requires a non-empty decisions list")
    review_ids: set[str] = set()
    credited_review_ids: set[str] = set()
    start = parse_time(policy["contest_start"], "contest_start")
    end = parse_time(policy["contest_end"], "contest_end")

    for decision in decisions:
        if not isinstance(decision, dict):
            raise ClaimError("every historical decision must be an object")
        _require_keys(
            decision,
            {
                "review_id",
                "pr_number",
                "participant_login",
                "participant_head_sha",
                "merge_sha",
                "merged_at",
                "identity_id",
                "benchmark_models",
                "outcome",
                "reason",
            },
            "historical decision",
        )
        review_id = str(decision["review_id"])
        if not IDENTITY_RE.fullmatch(review_id) or review_id in review_ids:
            raise ClaimError(f"invalid or duplicate historical review_id: {review_id}")
        review_ids.add(review_id)
        if not isinstance(decision["pr_number"], int) or decision["pr_number"] <= 0:
            raise ClaimError(f"historical decision {review_id} has an invalid PR number")
        login = str(decision["participant_login"])
        if not GITHUB_LOGIN_RE.fullmatch(login):
            raise ClaimError(f"historical decision {review_id} has an invalid participant")
        merged_at = parse_time(str(decision["merged_at"]), "historical.merged_at")
        if merged_at < start or merged_at > end:
            raise ClaimError(f"historical decision {review_id} is outside the contest window")
        _validate_git_provenance(
            repo,
            participant_head_sha=str(decision["participant_head_sha"]),
            merge_sha=str(decision["merge_sha"]),
            reviewed_through_sha=reviewed_through_sha,
            require_head_parent=decision["outcome"] == "credited",
        )

        identity_id = str(decision["identity_id"])
        identity = identities.get(identity_id)
        if identity is None:
            raise ClaimError(f"historical decision {review_id} uses unknown identity {identity_id}")
        models = decision["benchmark_models"]
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(model, str) and MODEL_RE.fullmatch(model) for model in models)
        ):
            raise ClaimError(f"historical decision {review_id} has invalid benchmark models")
        if any(model not in identity["benchmark_models"] for model in models):
            raise ClaimError(
                f"historical decision {review_id} contains a model outside identity {identity_id}"
            )
        if not str(decision["reason"]).strip():
            raise ClaimError(f"historical decision {review_id} requires a reason")

        outcome = decision["outcome"]
        if outcome == "credited":
            if identity.get("eligible") is not True:
                raise ClaimError(f"historical credit {review_id} uses an ineligible identity")
            credit_id = str(decision.get("credit_id") or "")
            if not HEX64_RE.fullmatch(credit_id):
                raise ClaimError(f"historical credit {review_id} requires a credit_id")
            credit = credits.get(identity_id)
            if credit is None or credit["credit_id"] != credit_id:
                raise ClaimError(f"historical credit {review_id} is missing from the ledger")
            expected = {
                "issuance": "historical_backfill",
                "historical_review_id": review_id,
                "participant_login": login,
                "pr_number": decision["pr_number"],
                "participant_head_sha": decision["participant_head_sha"],
                "merge_sha": decision["merge_sha"],
                "merged_at": decision["merged_at"],
            }
            mismatches = [key for key, value in expected.items() if credit.get(key) != value]
            if mismatches:
                raise ClaimError(
                    f"historical credit {review_id} ledger mismatch: {', '.join(mismatches)}"
                )
            oracle = decision.get("oracle")
            if not isinstance(oracle, dict):
                raise ClaimError(f"historical credit {review_id} requires oracle evidence")
            contract_path = safe_path(
                oracle.get("validation_contract"), field="historical.validation_contract"
            )
            if contract_path != identity["validation_contract"]:
                raise ClaimError(f"historical credit {review_id} uses the wrong oracle")
            contract = git_blob(repo, reviewed_through_sha, contract_path)
            assert contract is not None
            contract_sha = file_sha256(contract)
            if contract_sha != identity["validation_contract_sha256"]:
                raise ClaimError(f"historical credit {review_id} oracle hash is stale")
            if oracle.get("validation_contract_sha256") != contract_sha:
                raise ClaimError(f"historical credit {review_id} records the wrong oracle hash")
            if oracle.get("trusted_run_url") != credit["trusted_run"]["url"]:
                raise ClaimError(f"historical credit {review_id} records the wrong run URL")
            _matching_score_entry(
                repo=repo,
                reviewed_through_sha=reviewed_through_sha,
                decision=decision,
                credit=credit,
            )
            credited_review_ids.add(review_id)
        elif outcome not in INELIGIBLE_OUTCOMES:
            raise ClaimError(f"historical decision {review_id} has an unsupported outcome")

    ledger_review_ids = {
        str(credit.get("historical_review_id"))
        for credit in credits.values()
        if credit.get("issuance") == "historical_backfill"
    }
    if ledger_review_ids != credited_review_ids:
        raise ClaimError("historical review and backfilled ledger credits do not match")
    return {
        "reviewed_through_sha": reviewed_through_sha,
        "decisions": len(decisions),
        "credited": len(credited_review_ids),
        "ineligible": len(decisions) - len(credited_review_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument(
        "--policy", default=str(REPO_ROOT / ".github/ci/reference/model_ports_track.json")
    )
    parser.add_argument(
        "--identities", default=str(REPO_ROOT / "data/model-port-identities.json")
    )
    parser.add_argument("--ledger", default=str(REPO_ROOT / "data/model-port-credits.json"))
    parser.add_argument("--review", default=str(REVIEW_PATH))
    args = parser.parse_args()
    try:
        result = validate_historical_review(
            repo=Path(args.repo).resolve(),
            policy=load_json(Path(args.policy)),
            registry=load_json(Path(args.identities)),
            ledger=load_json(Path(args.ledger)),
            review=load_json(Path(args.review)),
        )
    except ClaimError as exc:
        print(f"historical model-port review invalid: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
