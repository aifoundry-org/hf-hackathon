#!/usr/bin/env python3
"""Build a SmolVLM2 candidate config from main-owned policy and a PR."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
from pathlib import Path

from benchmark_config_helpers import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_CONFIG = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
CONTRACT_PATH = REPO_ROOT / ".github" / "ci" / "reference" / "smolvlm2_500m_video.json"
MODEL = "smolvlm2_500m_video"

PROTECTED_PREFIXES = (
    ".github/ci/",
    ".github/workflows/",
    "ported_models/llama_cpp_et/assets/",
    "ported_models/llama_cpp_et/benchmarks/",
    "ported_models/llama_cpp_et/submissions/",
)
PROTECTED_PATHS = {
    "ported_models/llama_cpp_et/artifacts.json",
}


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def blob(ref: str, path: str, *, required: bool = True) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout
    if required:
        raise RuntimeError(f"{path} does not exist at {ref}")
    return None


def changed_paths(base: str, head: str) -> list[str]:
    return [
        line.strip()
        for line in git("diff", "--name-only", f"{base}...{head}").splitlines()
        if line.strip()
    ]


def validate_candidate_paths(paths: list[str], *, runtime_changed: bool) -> None:
    """Reject protected-path changes only for an actual runtime candidate.

    The trusted workflow runs for every pull request. Unrelated configuration
    and benchmark PRs must receive a no-op success even though those paths are
    protected from being overlaid into a SmolVLM2 runtime candidate.
    """
    if not runtime_changed:
        return
    blocked_paths = [
        path
        for path in paths
        if path in PROTECTED_PATHS or path.startswith(PROTECTED_PREFIXES)
    ]
    if blocked_paths:
        raise RuntimeError(
            "candidate changes protected SmolVLM2 gate paths: " + ", ".join(blocked_paths)
        )


def submodule_entry(ref: str, path: str) -> tuple[str, str]:
    fields = git("ls-tree", ref, "--", path).strip().split()
    if len(fields) < 4 or fields[0:2] != ["160000", "commit"]:
        raise RuntimeError(f"{path} is not a git submodule at {ref}")
    raw = blob(ref, ".gitmodules") or ""
    parser = configparser.ConfigParser()
    parser.read_string(raw)
    source_url = next(
        (
            parser.get(section, "url", fallback="")
            for section in parser.sections()
            if parser.get(section, "path", fallback="") == path
        ),
        "",
    )
    if not re.fullmatch(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", source_url
    ):
        raise RuntimeError(f"candidate runtime URL is not a public GitHub repository: {source_url}")
    return fields[2], source_url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--baseline-config", default="")
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text())
    runtime_path = str(contract["runtime"]["submodule_path"])
    paths = changed_paths(args.base, args.head)
    base_revision, base_url = submodule_entry(args.base, runtime_path)
    runtime_revision, runtime_url = submodule_entry(args.head, runtime_path)
    runtime_changed = runtime_path in paths or (runtime_revision, runtime_url) != (base_revision, base_url)
    validate_candidate_paths(paths, runtime_changed=runtime_changed)

    cfg = load_config(MAIN_CONFIG)
    model_cfg = cfg["models"][MODEL]
    if args.baseline_config:
        path = Path(args.baseline_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")

    output = Path(args.output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2) + "\n")
    metadata = {
        "targeted": runtime_changed,
        "changed_paths": paths,
        "runtime_changed": runtime_changed,
        "runtime_path": runtime_path,
        "runtime_revision": runtime_revision,
        "runtime_url": runtime_url,
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
