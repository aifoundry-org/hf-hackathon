#!/usr/bin/env python3
"""Apply an eligible standalone model port to a trusted main checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from model_port_claim import (
    BENCHMARK_CONFIG,
    ClaimError,
    IDENTITIES_PATH,
    LEDGER_PATH,
    POLICY_PATH,
    changed_files,
    git_json,
    inspect_claims,
    load_json,
    run_git,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def blob(repo: Path, ref: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ClaimError(f"cannot read participant file {path}")
    return proc.stdout


def regular_mode(repo: Path, ref: str, path: str) -> int:
    fields = run_git(repo, "ls-tree", ref, "--", path).strip().split()
    if len(fields) < 4 or fields[1] != "blob" or fields[0] not in ("100644", "100755"):
        raise ClaimError(f"participant path must be a regular file: {path}")
    return 0o755 if fields[0] == "100755" else 0o644


def write_participant_file(repo: Path, ref: str, path: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob(repo, ref, path))
    os.chmod(target, regular_mode(repo, ref, path))


def prepare(
    *,
    repo: Path,
    base: str,
    head: str,
    main_ref: str,
    actor: str,
    pr_number: int,
    policy_path: Path,
    identities_path: Path,
    ledger_path: Path,
) -> dict:
    if run_git(repo, "rev-parse", "HEAD").strip() != run_git(
        repo, "rev-parse", main_ref
    ).strip():
        raise ClaimError("trusted model-port preparation must start from the selected main commit")
    result = inspect_claims(
        repo=repo,
        base=base,
        head=head,
        actor=actor,
        pr_number=pr_number,
        policy=load_json(policy_path),
        registry=load_json(identities_path),
        ledger=load_json(ledger_path),
    )
    if not result["targeted"]:
        raise ClaimError("pull request does not add a model-port credit claim")
    if not result["passed"]:
        raise ClaimError("model-port claim is ineligible: " + "; ".join(result["errors"]))

    changed = changed_files(repo, base, head)
    allowed: set[str] = set(result["claim_paths"])
    for claim in result["claims"]:
        allowed.update(claim["implementation_files"])
    applied: list[str] = []
    for path in sorted(allowed):
        if changed.get(path) != "A":
            raise ClaimError(f"trusted candidate may only add model-port files: {path}")
        write_participant_file(repo, head, path)
        applied.append(path)

    main_config = json.loads((repo / BENCHMARK_CONFIG).read_text())
    candidate_config = git_json(repo, head, BENCHMARK_CONFIG)
    assert candidate_config is not None
    for model in result["models"]:
        main_config.setdefault("models", {})[model] = candidate_config["models"][model]
    (repo / BENCHMARK_CONFIG).write_text(json.dumps(main_config, indent=2) + "\n")
    applied.append(BENCHMARK_CONFIG + " (claimed model entries only)")

    ignored = sorted(path for path in changed if path not in allowed and path != BENCHMARK_CONFIG)
    return {
        **result,
        "trusted_main_sha": run_git(repo, "rev-parse", main_ref).strip(),
        "applied_paths": applied,
        "ignored_paths": ignored,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--policy", default=str(POLICY_PATH))
    parser.add_argument("--identities", default=str(IDENTITIES_PATH))
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    try:
        metadata = prepare(
            repo=Path(args.repo).resolve(),
            base=args.base,
            head=args.head,
            main_ref=args.main,
            actor=args.actor,
            pr_number=args.pr_number,
            policy_path=Path(args.policy),
            identities_path=Path(args.identities),
            ledger_path=Path(args.ledger),
        )
    except ClaimError as exc:
        raise SystemExit(f"error: {exc}")
    path = Path(args.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
