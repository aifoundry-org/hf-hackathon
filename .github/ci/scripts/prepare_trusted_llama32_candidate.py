#!/usr/bin/env python3
"""Build a trusted Llama 3.2 1B config from main plus a PR implementation."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

from benchmark_config_helpers import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_CONFIG = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
CONTRACT_PATH = REPO_ROOT / ".github" / "ci" / "reference" / "llama32_1b.json"
TRACK_POLICY_PATH = REPO_ROOT / ".github" / "ci" / "reference" / "llama32_1b_track.json"
MODEL = "llama32_1b"


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def blob(ref: str, path: str, *, required: bool = True) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout
    if required:
        raise RuntimeError(f"{path} does not exist at {ref}")
    return None


def changed_paths(base: str, head: str) -> list[str]:
    text = git("diff", "--name-only", f"{base}...{head}")
    return [line.strip() for line in text.splitlines() if line.strip()]


def submodule_entry(ref: str, path: str) -> tuple[str, str]:
    fields = git("ls-tree", ref, "--", path).strip().split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError(f"{path} is not a git submodule at {ref}")
    revision = fields[2]

    raw = blob(ref, ".gitmodules")
    parser = configparser.ConfigParser()
    parser.read_string(raw or "")
    source_url = ""
    for section in parser.sections():
        if parser.get(section, "path", fallback="") == path:
            source_url = parser.get(section, "url", fallback="")
            break
    if not source_url:
        raise RuntimeError(f"no .gitmodules URL found for {path}")
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", source_url):
        raise RuntimeError(f"candidate submodule URL is not an allowed public GitHub URL: {source_url}")
    return revision, source_url


def require_hex(value: Any, length: int, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", text):
        raise RuntimeError(f"{field} must be {length} lowercase hexadecimal characters")
    return text


def validate_track_claim(
    claim: dict[str, Any], policy: dict[str, Any], runtime_revision: str
) -> None:
    if claim.get("schema_version") != 1:
        raise RuntimeError("track claim must declare schema_version 1")
    if claim.get("track") != policy["track"] or claim.get("model") != MODEL:
        raise RuntimeError(
            f"track claim must declare track {policy['track']} and model {MODEL}"
        )
    if claim.get("runtime_revision") != runtime_revision:
        raise RuntimeError(
            "track claim runtime_revision must equal the candidate llama.cpp-et gitlink"
        )
    submission_id = str(claim.get("submission_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", submission_id):
        raise RuntimeError(
            "track claim submission_id must be 1-80 safe identifier characters"
        )


def evaluation_mode(
    *, runtime_changed: bool, manifest_changed: bool, claim_changed: bool
) -> str:
    if manifest_changed and not claim_changed:
        raise RuntimeError(
            "a Llama 3.2 1B candidate manifest change requires an explicit track claim"
        )
    if claim_changed:
        if not runtime_changed and not manifest_changed:
            raise RuntimeError(
                "a track claim must accompany a candidate runtime or manifest change"
            )
        return "competition"
    if runtime_changed:
        return "regression"
    return "none"


def candidate_artifact(manifest: dict[str, Any], contract: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if manifest.get("schema_version") != 1 or manifest.get("model") != MODEL:
        raise RuntimeError("candidate manifest must declare schema_version 1 and model llama32_1b")
    if manifest.get("base_model") != contract["base_model"]["name"]:
        raise RuntimeError("candidate manifest must derive from the contracted Llama 3.2 1B base model")
    license_id = str(manifest.get("license") or "")
    if license_id != contract["base_model"]["license"]:
        raise RuntimeError(
            f"candidate manifest license must be {contract['base_model']['license']}"
        )

    variant = str(manifest.get("variant") or "")
    quantization = str(manifest.get("quantization") or "")
    allowed = set(contract["artifact_policy"]["allowed_quantizations"])
    if not variant or quantization not in allowed:
        raise RuntimeError(f"candidate quantization must be one of {sorted(allowed)}")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("candidate manifest requires an artifact object")
    source = artifact.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("candidate artifact requires a source object")
    repo = str(source.get("repo") or "")
    revision = require_hex(source.get("revision"), 40, "artifact.source.revision")
    filename = str(source.get("filename") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise RuntimeError("artifact.source.repo must be a Hugging Face owner/repository")
    if not filename or Path(filename).name != filename:
        raise RuntimeError("artifact.source.filename must be a plain filename")
    sha256 = require_hex(artifact.get("sha256"), 64, "artifact.sha256")
    size_bytes = int(artifact.get("size_bytes") or 0)
    max_size = int(contract["artifact_policy"]["max_size_bytes"])
    if not 0 < size_bytes <= max_size:
        raise RuntimeError(f"candidate artifact size must be in (0, {max_size}]")
    expected_url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(filename)}"
    if source.get("url") not in (None, expected_url):
        raise RuntimeError("candidate artifact URL does not match its pinned Hugging Face identity")

    recipe = str(manifest.get("recipe") or "")
    if not recipe.startswith("ported_models/llama_cpp_et/"):
        raise RuntimeError(
            "candidate manifest recipe must be under ported_models/llama_cpp_et"
        )

    return variant, {
        "kind": "model",
        "framework": "llama.cpp-et",
        "variant": variant,
        "filename": filename,
        "source": {
            "type": "huggingface",
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "url": expected_url,
        },
        "sha256": sha256,
        "size_bytes": size_bytes,
        "local_cache": f"local-artifacts/models/llama32_1b/candidates/{sha256}-{filename}",
        "quantization": quantization,
        "license": license_id,
        "submission_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def apply_tuning(model_cfg: dict[str, Any], manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    tuning = manifest.get("tuning", {})
    if not isinstance(tuning, dict):
        raise RuntimeError("candidate tuning must be an object")
    unknown = sorted(set(tuning) - set(contract["allowed_tuning"]))
    if unknown:
        raise RuntimeError(f"unsupported candidate tuning keys: {', '.join(unknown)}")
    llama = model_cfg["llama_server"]
    for key, value in tuning.items():
        policy = contract["allowed_tuning"][key]
        if isinstance(policy, dict):
            value = int(value)
            if not int(policy["min"]) <= value <= int(policy["max"]):
                raise RuntimeError(f"candidate tuning {key}={value} is outside its allowed range")
        elif value not in policy:
            raise RuntimeError(f"candidate tuning {key}={value!r} is not allowed")
        llama[key] = value


def apply_candidate_manifest(
    model_cfg: dict[str, Any], manifest: dict[str, Any], contract: dict[str, Any]
) -> None:
    variant, artifact = candidate_artifact(manifest, contract)
    artifact_id = str(model_cfg["llama_server"]["model_artifact"])
    model_cfg["artifacts"][artifact_id] = artifact
    model_cfg["canonical_variant"] = variant
    apply_tuning(model_cfg, manifest, contract)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--baseline-config", default="")
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text())
    track_policy = json.loads(TRACK_POLICY_PATH.read_text())
    if track_policy.get("schema_version") != 1 or track_policy.get("model") != MODEL:
        raise RuntimeError("trusted Llama track policy has an unsupported schema or model")
    runtime_path = str(contract["runtime"]["submodule_path"])
    manifest_path = str(contract["candidate_manifest"])
    claim_path = str(track_policy["candidate_claim"])
    paths = changed_paths(args.base, args.head)
    try:
        base_runtime_revision, base_runtime_url = submodule_entry(args.base, runtime_path)
    except RuntimeError:
        base_runtime_revision, base_runtime_url = "", ""
    runtime_revision, runtime_url = submodule_entry(args.head, runtime_path)
    runtime_changed = (
        runtime_path in paths
        or runtime_revision != base_runtime_revision
        or runtime_url != base_runtime_url
    )
    manifest_changed = manifest_path in paths
    claim_changed = claim_path in paths
    mode = evaluation_mode(
        runtime_changed=runtime_changed,
        manifest_changed=manifest_changed,
        claim_changed=claim_changed,
    )
    targeted = mode != "none"

    claim: dict[str, Any] | None = None
    if claim_changed:
        raw_claim = blob(args.head, claim_path)
        claim = json.loads(raw_claim or "{}")
        if not isinstance(claim, dict):
            raise RuntimeError("track claim must contain a JSON object")
        validate_track_claim(claim, track_policy, runtime_revision)

    cfg = load_config(MAIN_CONFIG)
    model_cfg = cfg["models"][MODEL]
    manifest: dict[str, Any] | None = None
    committed_manifest = REPO_ROOT / manifest_path
    if committed_manifest.is_file():
        manifest = json.loads(committed_manifest.read_text())
        recipe = REPO_ROOT / str(manifest.get("recipe") or "")
        if not recipe.is_file():
            raise RuntimeError("committed candidate manifest recipe does not exist")
        apply_candidate_manifest(model_cfg, manifest, contract)

    if args.baseline_config:
        baseline_config = Path(args.baseline_config)
        baseline_config.parent.mkdir(parents=True, exist_ok=True)
        baseline_config.write_text(json.dumps(cfg, indent=2) + "\n")

    if manifest_changed:
        raw = blob(args.head, manifest_path)
        manifest = json.loads(raw or "{}")
        recipe = str(manifest.get("recipe") or "")
        if not recipe.startswith("ported_models/llama_cpp_et/") or blob(args.head, recipe, required=False) is None:
            raise RuntimeError("candidate manifest recipe must be a committed file under ported_models/llama_cpp_et")
        apply_candidate_manifest(model_cfg, manifest, contract)

    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(cfg, indent=2) + "\n")
    metadata = {
        "targeted": targeted,
        "mode": mode,
        "competitive": mode == "competition",
        "changed_paths": paths,
        "runtime_changed": runtime_changed,
        "manifest_changed": manifest_changed,
        "claim_changed": claim_changed,
        "claim_path": claim_path,
        "submission_id": claim.get("submission_id") if claim else None,
        "runtime_path": runtime_path,
        "runtime_revision": runtime_revision,
        "runtime_url": runtime_url,
        "candidate_variant": model_cfg["canonical_variant"],
        "candidate_quantization": manifest.get("quantization") if manifest else "Q8_0",
        "regression_models": contract["runtime"]["regression_models"] if runtime_changed else [],
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
