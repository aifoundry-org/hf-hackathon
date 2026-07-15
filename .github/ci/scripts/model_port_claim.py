#!/usr/bin/env python3
"""Validate trusted claims for the individual model-port track."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / ".github/ci/reference/model_ports_track.json"
IDENTITIES_PATH = REPO_ROOT / "data/model-port-identities.json"
LEDGER_PATH = REPO_ROOT / "data/model-port-credits.json"
BENCHMARK_CONFIG = ".github/ci/benchmark_config.json"
MODEL_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,63}")
IDENTITY_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,79}")
HF_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
HEX40_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
GITHUB_LOGIN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
METRIC_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
RUN_URL_RE = re.compile(r"https://github\.com/[^/]+/[^/]+/actions/runs/[0-9]+")


class ClaimError(RuntimeError):
    pass


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise ClaimError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def git_blob(repo: Path, ref: str, path: str, *, required: bool = True) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout
    if required:
        raise ClaimError(f"{path} does not exist at {ref}")
    return None


def git_json(repo: Path, ref: str, path: str, *, required: bool = True) -> dict[str, Any] | None:
    raw = git_blob(repo, ref, path, required=required)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClaimError(f"{path} is invalid JSON at {ref}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimError(f"{path} must contain a JSON object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimError(f"{path} must contain a JSON object")
    return value


def safe_path(value: Any, *, field: str) -> str:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or str(path) != text:
        raise ClaimError(f"{field} must be a normalized repository-relative path")
    return text


def is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def file_sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def changed_files(repo: Path, base: str, head: str) -> dict[str, str]:
    raw = run_git(
        repo,
        "diff",
        "--name-status",
        "--diff-filter=ACDMRTUXB",
        base,
        head,
    )
    changed: dict[str, str] = {}
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0][0]
        path = fields[-1]
        changed[path] = status
    return changed


def parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ClaimError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ClaimError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != 1 or policy.get("track") != "most_models_ported":
        raise ClaimError("model-port policy has an unsupported schema or track")
    if policy.get("activation_mode") not in ("shadow", "enforce"):
        raise ClaimError("model-port activation_mode must be shadow or enforce")
    if not isinstance(policy.get("historical_review_complete"), bool):
        raise ClaimError("historical_review_complete must be boolean")
    if not HEX40_RE.fullmatch(str(policy.get("baseline_sha") or "")):
        raise ClaimError("model-port baseline_sha must be a full commit SHA")
    parse_time(str(policy.get("contest_start") or ""), "contest_start")
    if policy.get("contest_end") is not None:
        end = parse_time(str(policy["contest_end"]), "contest_end")
        if end <= parse_time(str(policy["contest_start"]), "contest_start"):
            raise ClaimError("contest_end must be later than contest_start")
    if policy.get("activation_mode") == "enforce":
        if policy.get("contest_end") is None:
            raise ClaimError("enforcement requires a frozen contest_end")
        if policy.get("historical_review_complete") is not True:
            raise ClaimError("enforcement requires completed historical review")
    if policy.get("tie_break") != "earliest_final_qualifying_merge":
        raise ClaimError("unsupported model-port tie-break policy")
    if policy.get("credit_owner") != "pull_request_author":
        raise ClaimError("model-port credits must belong to the pull request author")
    if policy.get("required_device") != "soc1sim":
        raise ClaimError("model-port credits must require the ET-SoC1 soc1sim device")
    safe_path(policy.get("claim_root"), field="claim_root")
    excluded = policy.get("excluded_logins")
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value for value in excluded
    ):
        raise ClaimError("excluded_logins must be a string list")
    runners = policy.get("allowed_runners")
    if not isinstance(runners, list) or not runners or not all(isinstance(v, str) for v in runners):
        raise ClaimError("allowed_runners must be a non-empty string list")
    for field in ("max_added_files", "max_added_bytes"):
        if not isinstance(policy.get(field), int) or int(policy[field]) <= 0:
            raise ClaimError(f"{field} must be a positive integer")
    for field in (
        "require_new_standalone_root",
        "require_main_owned_validation_contract",
    ):
        if policy.get(field) is not True:
            raise ClaimError(f"{field} must remain enabled")


def validate_registry(registry: dict[str, Any], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if registry.get("schema_version") != 1 or registry.get("track") != policy["track"]:
        raise ClaimError("model-port identity registry has an unsupported schema or track")
    identities = registry.get("identities")
    if not isinstance(identities, list):
        raise ClaimError("model-port identity registry requires an identities list")
    baseline_roots = registry.get("baseline_port_roots")
    if not isinstance(baseline_roots, list) or not baseline_roots:
        raise ClaimError("model-port identity registry requires frozen baseline_port_roots")
    normalized_roots = [safe_path(value, field="baseline_port_roots") for value in baseline_roots]
    if len(set(normalized_roots)) != len(normalized_roots) or not all(
        root != "ported_models" and is_under(root, "ported_models")
        for root in normalized_roots
    ):
        raise ClaimError("baseline_port_roots must be unique paths under ported_models")
    by_id: dict[str, dict[str, Any]] = {}
    claimed_models: dict[str, str] = {}
    execution_families: dict[str, str] = {}
    identity_names: dict[str, str] = {}
    for raw in identities:
        if not isinstance(raw, dict):
            raise ClaimError("every model-port identity must be an object")
        identity_id = str(raw.get("identity_id") or "")
        if not IDENTITY_RE.fullmatch(identity_id) or identity_id in by_id:
            raise ClaimError(f"invalid or duplicate model-port identity_id: {identity_id}")
        models = raw.get("benchmark_models")
        if not isinstance(models, list) or not models:
            raise ClaimError(f"identity {identity_id} requires benchmark_models")
        for model in models:
            if not isinstance(model, str) or not MODEL_RE.fullmatch(model):
                raise ClaimError(f"identity {identity_id} has an invalid benchmark model")
            if model in claimed_models:
                raise ClaimError(
                    f"benchmark model {model} belongs to both {claimed_models[model]} and {identity_id}"
                )
            claimed_models[model] = identity_id
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(v, str) for v in aliases):
            raise ClaimError(f"identity {identity_id} aliases must be strings")
        for name in [identity_id, *models, *aliases]:
            normalized = name.lower()
            if not IDENTITY_RE.fullmatch(normalized):
                raise ClaimError(f"identity {identity_id} has an invalid alias or model name: {name}")
            if normalized in identity_names and identity_names[normalized] != identity_id:
                raise ClaimError(
                    f"identity name or alias {name} belongs to both "
                    f"{identity_names[normalized]} and {identity_id}"
                )
            identity_names[normalized] = identity_id
        execution_family = str(raw.get("execution_family") or "")
        if not IDENTITY_RE.fullmatch(execution_family):
            raise ClaimError(f"identity {identity_id} requires a safe execution_family")
        if execution_family in execution_families:
            raise ClaimError(
                f"execution family {execution_family} belongs to both "
                f"{execution_families[execution_family]} and {identity_id}"
            )
        execution_families[execution_family] = identity_id
        if raw.get("eligible") is True:
            source = raw.get("canonical_source")
            if not isinstance(source, dict):
                raise ClaimError(f"eligible identity {identity_id} requires canonical_source")
            if not HF_REPO_RE.fullmatch(str(source.get("repo") or "")):
                raise ClaimError(f"eligible identity {identity_id} requires a Hugging Face repo")
            if not HEX40_RE.fullmatch(str(source.get("revision") or "")):
                raise ClaimError(f"eligible identity {identity_id} requires a pinned source revision")
            if not str(source.get("license") or ""):
                raise ClaimError(f"eligible identity {identity_id} requires source license metadata")
            if not HEX64_RE.fullmatch(str(raw.get("benchmark_config_sha256") or "")):
                raise ClaimError(f"eligible identity {identity_id} requires benchmark_config_sha256")
            contract = safe_path(raw.get("validation_contract"), field="validation_contract")
            if not is_under(contract, ".github/ci/reference"):
                raise ClaimError(f"identity {identity_id} contract must be main-owned CI reference data")
            if raw.get("approved_runner") not in policy["allowed_runners"]:
                raise ClaimError(f"identity {identity_id} uses an unapproved runner")
        elif not str(raw.get("ineligible_reason") or ""):
            raise ClaimError(f"ineligible identity {identity_id} requires a reason")
        by_id[identity_id] = raw
    return by_id


def active_credits(ledger: dict[str, Any], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if ledger.get("schema_version") != 1 or ledger.get("track") != policy["track"]:
        raise ClaimError("model-port ledger has an unsupported schema or track")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ClaimError("model-port ledger requires a records list")
    credits: dict[str, dict[str, Any]] = {}
    inactive: set[str] = set()
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ClaimError("every model-port ledger record must be an object")
        record_id = str(record.get("record_id") or "")
        if not HEX64_RE.fullmatch(record_id) or record_id in record_ids:
            raise ClaimError("model-port ledger record IDs must be unique SHA-256 values")
        record_ids.add(record_id)
        hashed = {key: value for key, value in record.items() if key != "record_id"}
        if canonical_sha256(hashed) != record_id:
            raise ClaimError("model-port ledger record content does not match record_id")
        kind = record.get("record_type")
        if kind == "credit":
            credit_id = str(record.get("credit_id") or "")
            identity_id = str(record.get("identity_id") or "")
            if not HEX64_RE.fullmatch(credit_id) or not IDENTITY_RE.fullmatch(identity_id):
                raise ClaimError("invalid credit record")
            expected_credit_id = hashlib.sha256(
                f"{policy['track']}\0{identity_id}".encode()
            ).hexdigest()
            if credit_id != expected_credit_id:
                raise ClaimError(f"credit ID does not match identity {identity_id}")
            benchmark_model = str(record.get("benchmark_model") or "")
            if not MODEL_RE.fullmatch(benchmark_model):
                raise ClaimError("credit record has an invalid benchmark model")
            if not GITHUB_LOGIN_RE.fullmatch(str(record.get("participant_login") or "")):
                raise ClaimError("credit record has an invalid GitHub participant login")
            if not isinstance(record.get("pr_number"), int) or record["pr_number"] <= 0:
                raise ClaimError("credit record has an invalid pull request number")
            for field in ("participant_head_sha", "merge_sha"):
                if not HEX40_RE.fullmatch(str(record.get(field) or "")):
                    raise ClaimError(f"credit record has an invalid {field}")
            parse_time(str(record.get("merged_at") or ""), "credit.merged_at")
            trusted_run = record.get("trusted_run")
            if not isinstance(trusted_run, dict):
                raise ClaimError("credit record requires trusted_run provenance")
            if not RUN_URL_RE.fullmatch(str(trusted_run.get("url") or "")):
                raise ClaimError("credit record has an invalid trusted run URL")
            if not HEX40_RE.fullmatch(str(trusted_run.get("score_sha") or "")):
                raise ClaimError("credit record has an invalid trusted score SHA")
            if trusted_run.get("benchmark_device") != policy["required_device"]:
                raise ClaimError("credit record was not produced on the required device")
            if not HEX64_RE.fullmatch(
                str(trusted_run.get("validation_contract_sha256") or "")
            ):
                raise ClaimError("credit record has an invalid validation contract hash")
            metric_value = trusted_run.get("metric_value")
            if (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(metric_value)
            ):
                raise ClaimError("credit record has no numeric trusted metric")
            if not METRIC_RE.fullmatch(str(trusted_run.get("metric") or "")):
                raise ClaimError("credit record has an invalid metric name")
            source = record.get("source")
            if not isinstance(source, dict) or set(source) != {"repo", "revision", "license"}:
                raise ClaimError("credit record has invalid source provenance")
            if not HF_REPO_RE.fullmatch(str(source.get("repo") or "")):
                raise ClaimError("credit record has an invalid source repository")
            if not HEX40_RE.fullmatch(str(source.get("revision") or "")):
                raise ClaimError("credit record has an invalid source revision")
            if not str(source.get("license") or ""):
                raise ClaimError("credit record has no source license")
            safe_path(record.get("recipe"), field="credit.recipe")
            if not HEX64_RE.fullmatch(str(record.get("benchmark_config_sha256") or "")):
                raise ClaimError("credit record has an invalid benchmark configuration hash")
            if identity_id in credits:
                raise ClaimError(f"ledger contains duplicate credits for {identity_id}")
            credits[identity_id] = record
        elif kind in ("revocation", "supersession"):
            target = str(record.get("target_credit_id") or "")
            if not HEX64_RE.fullmatch(target) or not str(record.get("reason") or ""):
                raise ClaimError(f"invalid {kind} record")
            inactive.add(target)
        else:
            raise ClaimError(f"unsupported model-port ledger record type: {kind}")
    known_credit_ids = {record["credit_id"] for record in credits.values()}
    unknown_targets = inactive - known_credit_ids
    if unknown_targets:
        raise ClaimError("ledger revocation or supersession targets an unknown credit")
    return {
        identity: record
        for identity, record in credits.items()
        if record["credit_id"] not in inactive
    }


def protected_track_change(path: str) -> bool:
    if path == BENCHMARK_CONFIG:
        return False
    return (
        is_under(path, ".github/workflows")
        or is_under(path, ".github/ci/scripts")
        or is_under(path, ".github/ci/reference")
        or path in {
            "data/model-port-identities.json",
            "data/model-port-credits.json",
            "data/model-port-standings.json",
            ".github/CODEOWNERS",
        }
    )


def effective_model_config(
    repo: Path,
    ref: str,
    benchmark_model: str,
    benchmark_config: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = git_json(repo, ref, BENCHMARK_CONFIG)
    assert root is not None
    models = root.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(benchmark_model), dict):
        raise ClaimError(f"{benchmark_model} is not configured at the candidate commit")
    entry = models[benchmark_model]
    include = entry.get("config")
    if include:
        if benchmark_config != include:
            raise ClaimError("claim benchmark_config does not match the configured model include")
        included = git_json(repo, ref, benchmark_config)
        assert included is not None
        override = {key: value for key, value in entry.items() if key != "config"}
        merged = deep_merge(included, override)
        merged["config"] = include
        return merged, merged, canonical_sha256(merged)
    if benchmark_config != BENCHMARK_CONFIG:
        raise ClaimError("inline model entries must claim .github/ci/benchmark_config.json")
    return dict(entry), dict(entry), canonical_sha256(entry)


def claim_paths(policy: dict[str, Any], changed: dict[str, str]) -> tuple[list[str], list[str]]:
    root = str(policy["claim_root"]).rstrip("/")
    pattern = re.compile(rf"{re.escape(root)}/([a-z0-9][a-z0-9_-]{{1,63}})\.json")
    candidates = sorted(path for path in changed if is_under(path, root))
    valid = [path for path in candidates if pattern.fullmatch(path)]
    invalid = [path for path in candidates if path not in valid]
    return valid, invalid


def inspect_claims(
    *,
    repo: Path,
    base: str,
    head: str,
    actor: str,
    pr_number: int,
    policy: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
    merged_at: str = "",
    allow_existing_credit: bool = False,
) -> dict[str, Any]:
    validate_policy(policy)
    identities = validate_registry(registry, policy)
    credits = active_credits(ledger, policy)
    changed = changed_files(repo, base, head)
    paths, invalid_claim_paths = claim_paths(policy, changed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "track": policy["track"],
        "activation_mode": policy["activation_mode"],
        "targeted": bool(paths or invalid_claim_paths),
        "passed": True,
        "actor": actor,
        "pr_number": pr_number,
        "base_sha": base,
        "head_sha": head,
        "claim_paths": paths,
        "models": [],
        "claims": [],
        "errors": [],
    }
    if invalid_claim_paths:
        result["errors"].append(
            "invalid files under the model-port claim root: " + ", ".join(invalid_claim_paths)
        )
        result["passed"] = False
    if not paths:
        return result

    global_errors: list[str] = []
    protected = sorted(path for path in changed if protected_track_change(path))
    if protected:
        global_errors.append(
            "claimed model-port PRs may not change trusted track/measurement files: "
            + ", ".join(protected)
        )
    excluded = {str(v).lower() for v in policy.get("excluded_logins", [])}
    if not GITHUB_LOGIN_RE.fullmatch(actor):
        global_errors.append("model-port credits require a canonical GitHub login")
    if actor.lower() in excluded:
        global_errors.append(f"@{actor} is excluded from individual model-port credits")
    if pr_number <= 0:
        global_errors.append("model-port credits require a GitHub pull request")
    if merged_at:
        merged_time = parse_time(merged_at, "merged_at")
        if merged_time < parse_time(policy["contest_start"], "contest_start"):
            global_errors.append("pull request merged before the contest start")
        if policy.get("contest_end") is not None and merged_time > parse_time(
            policy["contest_end"], "contest_end"
        ):
            global_errors.append("pull request merged after the contest deadline")

    baseline = str(policy["baseline_sha"])
    baseline_root = git_json(repo, baseline, BENCHMARK_CONFIG, required=False) or {}
    baseline_models = baseline_root.get("models", {})
    base_root = git_json(repo, base, BENCHMARK_CONFIG, required=False) or {}
    base_models = base_root.get("models", {})
    head_root = git_json(repo, head, BENCHMARK_CONFIG, required=False) or {}
    head_models = head_root.get("models", {})
    if not all(isinstance(value, dict) for value in (baseline_models, base_models, head_models)):
        global_errors.append("benchmark configuration requires a models object")
        baseline_models = {}
        base_models = {}
        head_models = {}
    else:
        claimed_model_names = {PurePosixPath(path).stem for path in paths}
        base_globals = {key: value for key, value in base_root.items() if key != "models"}
        head_globals = {key: value for key, value in head_root.items() if key != "models"}
        if base_globals != head_globals:
            global_errors.append(
                "claimed model-port PRs may not change global benchmark configuration"
            )
        changed_existing = sorted(
            model for model, value in base_models.items() if head_models.get(model) != value
        )
        removed_existing = sorted(set(base_models) - set(head_models))
        unexpected_added = sorted(set(head_models) - set(base_models) - claimed_model_names)
        if changed_existing or removed_existing or unexpected_added:
            affected = sorted(set(changed_existing + removed_existing + unexpected_added))
            global_errors.append(
                "claimed model-port PRs may only add their claimed benchmark entries: "
                + ", ".join(affected)
            )

    for path in paths:
        errors = list(global_errors)
        model_from_path = PurePosixPath(path).stem
        try:
            if changed[path] != "A":
                raise ClaimError("a model-port claim must be newly added, not modified")
            claim = git_json(repo, head, path)
            assert claim is not None
            allowed_keys = {
                "schema_version",
                "track",
                "benchmark_model",
                "identity_id",
                "source",
                "implementation_paths",
                "benchmark_config",
                "recipe",
            }
            unknown = sorted(set(claim) - allowed_keys)
            if unknown:
                raise ClaimError("claim contains unsupported keys: " + ", ".join(unknown))
            if claim.get("schema_version") != 1 or claim.get("track") != policy["track"]:
                raise ClaimError("claim has an unsupported schema or track")
            model = str(claim.get("benchmark_model") or "")
            if model != model_from_path or not MODEL_RE.fullmatch(model):
                raise ClaimError("claim filename and benchmark_model must be the same safe slug")
            identity_id = str(claim.get("identity_id") or "")
            identity = identities.get(identity_id)
            if identity is None:
                raise ClaimError(
                    f"identity {identity_id!r} is not main-owned; request identity/contract approval first"
                )
            if identity.get("eligible") is not True:
                raise ClaimError(
                    f"identity {identity_id} is ineligible: {identity.get('ineligible_reason')}"
                )
            if model not in identity["benchmark_models"]:
                raise ClaimError(f"identity {identity_id} does not authorize benchmark model {model}")
            if identity_id in credits and not allow_existing_credit:
                raise ClaimError(f"identity {identity_id} already has an active port credit")
            if model in baseline_models:
                raise ClaimError(f"benchmark model {model} existed at the contest baseline")
            if model in base_models:
                raise ClaimError(f"benchmark model {model} already exists on main")

            source = claim.get("source")
            if not isinstance(source, dict):
                raise ClaimError("claim requires a source object")
            if set(source) != {"repo", "revision", "license"}:
                raise ClaimError("claim source must contain only repo, revision, and license")
            normalized_source = {
                "repo": str(source.get("repo") or ""),
                "revision": str(source.get("revision") or ""),
                "license": str(source.get("license") or ""),
            }
            if not HF_REPO_RE.fullmatch(normalized_source["repo"]):
                raise ClaimError("source.repo must be a Hugging Face owner/repository")
            if not HEX40_RE.fullmatch(normalized_source["revision"]):
                raise ClaimError("source.revision must be a pinned 40-character commit")
            if not normalized_source["license"]:
                raise ClaimError("source.license is required")
            if normalized_source != identity["canonical_source"]:
                raise ClaimError("claim source does not match the main-owned canonical identity")

            implementation_root = f"ported_models/{model}"
            if is_under(str(policy["claim_root"]), implementation_root):
                raise ClaimError("benchmark model collides with the reserved claim directory")
            implementation_paths = claim.get("implementation_paths")
            if implementation_paths != [implementation_root]:
                raise ClaimError(
                    f"implementation_paths must be exactly [{implementation_root!r}] for a standalone port"
                )
            if git_blob(repo, base, implementation_root, required=False) is not None:
                raise ClaimError(f"implementation root {implementation_root} already exists on main")
            root_changes = [p for p in changed if is_under(p, implementation_root)]
            if not root_changes:
                raise ClaimError("claim has no changed implementation files")
            non_added = [p for p in root_changes if changed[p] != "A"]
            if non_added:
                raise ClaimError("new standalone ports may only add files: " + ", ".join(non_added))
            if len(root_changes) > int(policy["max_added_files"]):
                raise ClaimError("new standalone port exceeds the added-file limit")
            for changed_path in root_changes:
                if ".git" in PurePosixPath(changed_path).parts:
                    raise ClaimError("implementation paths may not contain .git components")
                tree_fields = run_git(
                    repo, "ls-tree", head, "--", changed_path
                ).strip().split()
                if (
                    len(tree_fields) < 4
                    or tree_fields[0] not in ("100644", "100755")
                    or tree_fields[1] != "blob"
                ):
                    raise ClaimError(
                        f"implementation path must be a regular file: {changed_path}"
                    )
            total_bytes = sum(
                int(run_git(repo, "cat-file", "-s", f"{head}:{changed_path}").strip())
                for changed_path in root_changes
            )
            if total_bytes > int(policy["max_added_bytes"]):
                raise ClaimError("new standalone port exceeds the committed-byte limit")

            recipe = safe_path(claim.get("recipe"), field="recipe")
            if not is_under(recipe, implementation_root) or changed.get(recipe) != "A":
                raise ClaimError("recipe must be a newly added file under the implementation root")
            if not (git_blob(repo, head, recipe) or "").strip():
                raise ClaimError("recipe must not be empty")
            benchmark_config = safe_path(claim.get("benchmark_config"), field="benchmark_config")
            effective, hashed_config, config_sha = effective_model_config(
                repo, head, model, benchmark_config
            )
            if config_sha != identity["benchmark_config_sha256"]:
                raise ClaimError(
                    "benchmark configuration differs from the organizer-approved configuration hash"
                )
            runner = str(effective.get("runner") or effective.get("framework", {}).get("runner") or "elf")
            if runner != identity["approved_runner"] or runner not in policy["allowed_runners"]:
                raise ClaimError(f"runner {runner} is not approved for this identity")
            contract = effective.get("reference_contract")
            if not contract and isinstance(effective.get("validation"), dict):
                contract = effective["validation"].get("reference_contract")
            contract_path = safe_path(contract, field="reference_contract")
            if contract_path != identity["validation_contract"]:
                raise ClaimError("candidate validation contract is not the approved contract")
            base_contract = git_blob(repo, base, contract_path, required=False)
            head_contract = git_blob(repo, head, contract_path, required=False)
            if base_contract is None or head_contract != base_contract:
                raise ClaimError("validation contract must already exist unchanged on main")
            score_cfg = effective.get("score", {})
            if not isinstance(score_cfg, dict):
                raise ClaimError("approved benchmark configuration requires a score object")
            metric = str(score_cfg.get("metric") or "kernel_wait_s")
            if not METRIC_RE.fullmatch(metric):
                raise ClaimError("approved benchmark configuration has an invalid metric")
            higher_raw = score_cfg.get("higher_is_better", False)
            if not isinstance(higher_raw, bool):
                raise ClaimError("score.higher_is_better must be boolean")
            higher = higher_raw
            variant = str(effective.get("canonical_variant") or "")
            if not variant:
                raise ClaimError("approved benchmark configuration requires canonical_variant")

            result["models"].append(model)
            result["claims"].append(
                {
                    "path": path,
                    "benchmark_model": model,
                    "identity_id": identity_id,
                    "implementation_root": implementation_root,
                    "implementation_files": sorted(root_changes),
                    "benchmark_config": benchmark_config,
                    "benchmark_config_sha256": canonical_sha256(hashed_config),
                    "runner": runner,
                    "canonical_variant": variant,
                    "metric": metric,
                    "higher_is_better": higher,
                    "validation_contract": contract_path,
                    "validation_contract_sha256": file_sha256(base_contract),
                    "source": normalized_source,
                    "recipe": recipe,
                    "passed": not errors,
                    "errors": errors,
                }
            )
        except ClaimError as exc:
            errors.append(str(exc))
            result["claims"].append(
                {
                    "path": path,
                    "benchmark_model": model_from_path,
                    "passed": False,
                    "errors": errors,
                }
            )

    result["models"] = sorted(set(result["models"]))
    result["errors"] = sorted(
        set(result["errors"])
        | {error for claim in result["claims"] for error in claim.get("errors", [])}
    )
    result["passed"] = not result["errors"] and all(
        claim.get("passed") for claim in result["claims"]
    )
    return result


def markdown_summary(result: dict[str, Any]) -> str:
    lines = ["## Trusted model-port credit eligibility", ""]
    if not result["targeted"]:
        lines.append("No model-port credit claim was added; this check is a no-op.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            f"Policy mode: `{result['activation_mode']}`",
            "",
            "| Model | Identity | Verdict | Notes |",
            "|-------|----------|---------|-------|",
        ]
    )
    for claim in result["claims"]:
        errors = "; ".join(claim.get("errors", [])) or "eligible; trusted board result still required"
        errors = errors.replace("|", "\\|")
        lines.append(
            f"| {claim.get('benchmark_model', '-')} | {claim.get('identity_id', '-')} | "
            f"{'pass' if claim.get('passed') else 'fail'} | {errors} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--merged-at", default="")
    parser.add_argument("--policy", default=str(POLICY_PATH))
    parser.add_argument("--identities", default=str(IDENTITIES_PATH))
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    try:
        result = inspect_claims(
            repo=Path(args.repo).resolve(),
            base=args.base,
            head=args.head,
            actor=args.actor,
            pr_number=args.pr_number,
            policy=load_json(Path(args.policy)),
            registry=load_json(Path(args.identities)),
            ledger=load_json(Path(args.ledger)),
            merged_at=args.merged_at,
        )
    except ClaimError as exc:
        result = {
            "schema_version": 1,
            "track": "most_models_ported",
            "activation_mode": "unknown",
            "targeted": True,
            "passed": False,
            "actor": args.actor,
            "pr_number": args.pr_number,
            "base_sha": args.base,
            "head_sha": args.head,
            "claim_paths": [],
            "models": [],
            "claims": [],
            "errors": [str(exc)],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    summary = markdown_summary(result)
    if args.summary:
        Path(args.summary).write_text(summary)
    print(summary)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
