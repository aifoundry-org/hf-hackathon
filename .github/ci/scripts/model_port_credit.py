#!/usr/bin/env python3
"""Issue idempotent model-port credits from trusted main-branch board scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from model_port_claim import (
    ClaimError,
    GITHUB_LOGIN_RE,
    HEX40_RE,
    RUN_URL_RE,
    inspect_claims,
    load_json,
    parse_time,
    validate_credit_inventory,
)
from render_model_port_standings import readme_section, replace_section, standings

REPO_ROOT = Path(__file__).resolve().parents[3]


def record_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_score(
    *,
    score: dict[str, Any],
    claim: dict[str, Any],
    actor: str,
    main_sha: str,
    expected_run_url: str,
    required_device: str,
) -> None:
    expected = {
        "model": claim["benchmark_model"],
        "variant": claim["canonical_variant"],
        "sha": main_sha,
        "ref": "refs/heads/main",
        "team": actor,
        "run_url": expected_run_url,
        "benchmark_device": required_device,
        "validation_contract_sha256": claim["validation_contract_sha256"],
    }
    mismatches = [key for key, value in expected.items() if score.get(key) != value]
    if mismatches:
        raise ClaimError(
            f"{claim['benchmark_model']} score provenance mismatch: " + ", ".join(mismatches)
        )
    if score.get("passed") is not True:
        note = score.get("valid_note") or score.get("note") or "score did not pass"
        raise ClaimError(f"{claim['benchmark_model']} trusted board score failed: {note}")
    value = score.get(claim["metric"])
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ClaimError(
            f"{claim['benchmark_model']} passing score has no numeric {claim['metric']}"
        )


def issue_credits(
    *,
    repo: Path,
    before: str,
    head: str,
    actor: str,
    pr_number: int,
    participant_head_sha: str,
    merged_at: str,
    expected_run_url: str,
    scores_dir: Path,
    policy: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
    expected_claim_paths: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = inspect_claims(
        repo=repo,
        base=before,
        head=head,
        actor=actor,
        pr_number=pr_number,
        policy=policy,
        registry=registry,
        ledger=ledger,
        merged_at=merged_at,
        allow_existing_credit=True,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "track": policy["track"],
        "activation_mode": policy["activation_mode"],
        "targeted": result["targeted"],
        "passed": result["passed"],
        "issued": [],
        "would_issue": [],
        "idempotent": [],
        "errors": list(result["errors"]),
    }
    if not result["targeted"]:
        return ledger, report
    if not GITHUB_LOGIN_RE.fullmatch(actor):
        report["errors"].append("credit issuer requires a canonical GitHub login")
    if not HEX40_RE.fullmatch(head) or not HEX40_RE.fullmatch(participant_head_sha):
        report["errors"].append(
            "credit issuer requires full main and participant commit SHAs"
        )
    if pr_number <= 0:
        report["errors"].append("credit issuer requires a merged pull request number")
    try:
        parse_time(merged_at, "merged_at")
    except ClaimError as exc:
        report["errors"].append(str(exc))
    if not RUN_URL_RE.fullmatch(expected_run_url):
        report["errors"].append(
            "credit issuer requires a canonical GitHub Actions run URL"
        )
    if result["targeted"] and expected_claim_paths is not None:
        actual_claim_paths = set(result["claim_paths"])
        if actual_claim_paths != expected_claim_paths:
            report["errors"].append(
                "main-push claims do not exactly match the resolved pull request files"
            )
            report["passed"] = False
    if not result["passed"]:
        return ledger, report
    if report["errors"]:
        report["passed"] = False
        return ledger, report

    records = list(ledger["records"])
    current = validate_credit_inventory(ledger, policy, registry)
    new_records: list[dict[str, Any]] = []
    for claim in result["claims"]:
        score_path = scores_dir / f"score-{claim['benchmark_model']}.json"
        if not score_path.is_file():
            report["errors"].append(f"missing trusted score for {claim['benchmark_model']}")
            continue
        try:
            score = load_json(score_path)
            validate_score(
                score=score,
                claim=claim,
                actor=actor,
                main_sha=head,
                expected_run_url=expected_run_url,
                required_device=policy["required_device"],
            )
            credit_id = hashlib.sha256(
                f"{policy['track']}\0{claim['identity_id']}".encode()
            ).hexdigest()
            existing = current.get(claim["identity_id"])
            if existing:
                same_event = (
                    existing.get("credit_id") == credit_id
                    and existing.get("participant_login") == actor
                    and existing.get("pr_number") == pr_number
                    and existing.get("merge_sha") == head
                )
                if same_event:
                    report["idempotent"].append(claim["identity_id"])
                    continue
                raise ClaimError(
                    f"identity {claim['identity_id']} was already credited to "
                    f"@{existing.get('participant_login')} in PR #{existing.get('pr_number')}"
                )
            record_without_id = {
                "record_type": "credit",
                "issuance": "trusted_merge",
                "credit_id": credit_id,
                "identity_id": claim["identity_id"],
                "benchmark_model": claim["benchmark_model"],
                "participant_login": actor,
                "pr_number": pr_number,
                "participant_head_sha": participant_head_sha,
                "merge_sha": head,
                "merged_at": merged_at,
                "source": claim["source"],
                "recipe": claim["recipe"],
                "benchmark_config_sha256": claim["benchmark_config_sha256"],
                "trusted_run": {
                    "url": expected_run_url,
                    "score_sha": score["sha"],
                    "validation_contract_sha256": claim[
                        "validation_contract_sha256"
                    ],
                    "benchmark_device": score["benchmark_device"],
                    "runner": claim["runner"],
                    "metric": claim["metric"],
                    "metric_value": score[claim["metric"]],
                },
            }
            record = {"record_id": record_hash(record_without_id), **record_without_id}
            new_records.append(record)
            report["would_issue"].append(claim["identity_id"])
        except (ClaimError, json.JSONDecodeError) as exc:
            report["errors"].append(str(exc))

    report["passed"] = not report["errors"]
    if report["passed"] and policy["activation_mode"] == "enforce":
        records.extend(new_records)
        report["issued"] = [record["identity_id"] for record in new_records]
        ledger = {**ledger, "records": records}
    return ledger, report


def markdown(report: dict[str, Any]) -> str:
    lines = ["## Model-port credit issuance", ""]
    if not report["targeted"]:
        lines.append("No model-port claim was merged in this push.")
    elif report["errors"]:
        lines.append("Result: fail.")
        lines.extend(f"- {error}" for error in report["errors"])
    elif report["activation_mode"] == "shadow":
        values = ", ".join(report["would_issue"]) or "none"
        lines.append(f"Shadow result: would issue credits for {values}; ledger was not modified.")
    else:
        issued = ", ".join(report["issued"]) or "none (idempotent re-run)"
        lines.append(f"Result: pass. Issued: {issued}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--before", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--participant-head-sha", required=True)
    parser.add_argument("--merged-at", required=True)
    parser.add_argument("--expected-run-url", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument(
        "--expected-claim-paths",
        help="newline-delimited claim paths from the canonical merged pull request",
    )
    parser.add_argument(
        "--policy", default=str(REPO_ROOT / ".github/ci/reference/model_ports_track.json")
    )
    parser.add_argument(
        "--identities", default=str(REPO_ROOT / "data/model-port-identities.json")
    )
    parser.add_argument("--ledger", default=str(REPO_ROOT / "data/model-port-credits.json"))
    parser.add_argument(
        "--standings", default=str(REPO_ROOT / "data/model-port-standings.json")
    )
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    policy = load_json(Path(args.policy))
    registry = load_json(Path(args.identities))
    original_ledger = load_json(Path(args.ledger))
    try:
        ledger, report = issue_credits(
            repo=Path(args.repo).resolve(),
            before=args.before,
            head=args.head,
            actor=args.actor,
            pr_number=args.pr_number,
            participant_head_sha=args.participant_head_sha,
            merged_at=args.merged_at,
            expected_run_url=args.expected_run_url,
            scores_dir=Path(args.scores_dir),
            policy=policy,
            registry=registry,
            ledger=original_ledger,
            expected_claim_paths=(
                {
                    line.strip()
                    for line in Path(args.expected_claim_paths).read_text().splitlines()
                    if line.strip()
                }
                if args.expected_claim_paths
                else None
            ),
        )
    except ClaimError as exc:
        report = {
            "schema_version": 1,
            "track": policy.get("track", "most_models_ported"),
            "activation_mode": policy.get("activation_mode", "unknown"),
            "targeted": True,
            "passed": False,
            "issued": [],
            "would_issue": [],
            "idempotent": [],
            "errors": [str(exc)],
        }
        ledger = original_ledger

    if report["passed"] and policy["activation_mode"] == "enforce":
        Path(args.ledger).write_text(json.dumps(ledger, indent=2) + "\n")
        payload = standings(policy, ledger, registry)
        Path(args.standings).write_text(json.dumps(payload, indent=2) + "\n")
        replace_section(
            Path(args.readme),
            readme_section(
                payload,
                validate_credit_inventory(ledger, policy, registry),
            ),
        )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
