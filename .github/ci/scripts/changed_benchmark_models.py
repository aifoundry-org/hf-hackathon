#!/usr/bin/env python3
"""Select benchmark models touched by a PR diff."""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_config_helpers import (
    load_config,
    model_names as configured_model_names,
    model_supports_target,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"

FRAMEWORK_ARTIFACT_KINDS = {
    "framework_source",
    "framework_workdir",
    "framework_binary",
}

GENERIC_BOARD_INFRA_PATHS = {
    ".github/workflows/benchmark-board.yml",
    ".github/ci/scripts/benchmark_config_helpers.py",
    ".github/ci/scripts/changed_benchmark_models.py",
    ".github/ci/scripts/run_model_benchmark.sh",
    ".github/ci/scripts/score_results.py",
}

RUNNER_INFRA_PATHS = {
    ".github/ci/scripts/run_llama_server_benchmark.py": "llama_server",
}

MODEL_CODE_SUFFIXES = {
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


def norm(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return posixpath.normpath(value)


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


def git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_changed_files(base: str, head: str) -> list[str]:
    ranges = [f"{base}...{head}", f"{base}..{head}"]
    last_error = ""
    for rev_range in ranges:
        proc = git(
            ["diff", "--name-only", "--diff-filter=ACMRTUXB", rev_range],
            check=False,
        )
        if proc.returncode == 0:
            return [norm(line) for line in proc.stdout.splitlines() if line.strip()]
        last_error = proc.stderr.strip()
    raise RuntimeError(f"git diff failed for {base}..{head}: {last_error}")


def git_show_json(ref: str, path: str) -> dict[str, Any] | None:
    proc = git(["show", f"{ref}:{path}"], check=False)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def read_json(path: str) -> dict[str, Any] | None:
    try:
        return json.loads((REPO_ROOT / path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def json_changed_keys(old: dict[str, Any] | None, new: dict[str, Any], key: str) -> set[str]:
    old_values = old.get(key, {}) if old else {}
    new_values = new.get(key, {})
    if not isinstance(old_values, dict) or not isinstance(new_values, dict):
        return set(new_values.keys())
    changed = {name for name, value in new_values.items() if old_values.get(name) != value}
    changed.update(name for name in old_values.keys() if name not in new_values)
    return changed


def collect_artifact_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "artifacts":
                continue
            if key.endswith("_artifact") and isinstance(nested, str):
                refs.add(nested)
            else:
                refs.update(collect_artifact_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.update(collect_artifact_refs(nested))
    return refs


def model_framework(model_cfg: dict[str, Any]) -> str | None:
    framework = model_cfg.get("framework")
    if isinstance(framework, dict) and framework.get("name"):
        return str(framework["name"])
    refs = collect_artifact_refs(model_cfg)
    artifacts = model_cfg.get("artifacts", {})
    if isinstance(artifacts, dict):
        for artifact_id in refs:
            artifact = artifacts.get(artifact_id, {})
            if isinstance(artifact, dict) and artifact.get("framework"):
                return str(artifact["framework"])
    return None


def default_models_for_framework(cfg: dict[str, Any], framework: str, target: str) -> list[str]:
    all_models = configured_model_names(cfg, target, default_only=False)
    candidates = [
        name
        for name in all_models
        if model_framework(cfg["models"].get(name, {})) == framework
    ]
    defaults = [
        name
        for name in candidates
        if cfg["models"].get(name, {}).get("benchmark_default", True)
    ]
    return defaults or candidates


def default_models_for_runner(cfg: dict[str, Any], runner: str, target: str) -> list[str]:
    all_defaults = configured_model_names(cfg, target, default_only=True)
    return [
        name
        for name in all_defaults
        if cfg["models"].get(name, {}).get("runner", "elf") == runner
    ]


def source_model_root(model_cfg: dict[str, Any]) -> str | None:
    source = repo_rel(model_cfg.get("source"))
    if not source:
        return None
    parts = source.split("/")
    if len(parts) >= 2 and parts[0] == "ported_models":
        return "/".join(parts[:2])
    return str(Path(source).parent)


def is_model_code_path(path: str) -> bool:
    name = Path(path).name.lower()
    if name in {"readme.md", "model.md", "third_party.md"}:
        return False
    if "/docs/" in path:
        return False
    return Path(path).suffix in MODEL_CODE_SUFFIXES


def port_of(path: str) -> str | None:
    """Return the ``ported_models/<name>`` root for a path, if it is under one."""
    parts = norm(path).split("/")
    if len(parts) >= 2 and parts[0] == "ported_models":
        return "/".join(parts[:2])
    return None


def registered_ports(cfg: dict[str, Any]) -> set[str]:
    """Ports (``ported_models/<name>``) already mapped to a configured model."""
    ports: set[str] = set()
    for model_cfg in cfg.get("models", {}).values():
        candidates = [
            model_cfg.get("source"),
            model_cfg.get("_config_path"),
            model_cfg.get("_artifacts_path"),
            model_cfg.get("_model_dir"),
        ]
        artifacts = model_cfg.get("artifacts", {})
        if isinstance(artifacts, dict):
            for artifact in artifacts.values():
                if isinstance(artifact, dict) and artifact.get("kind") == "framework_source":
                    candidates.append(artifact.get("submodule_path"))
        for candidate in candidates:
            port = port_of(repo_rel(candidate) or "") if candidate else None
            if port:
                ports.add(port)
    return ports


def unregistered_ports(cfg: dict[str, Any], changed_files: list[str]) -> list[str]:
    """New ports touched by the diff that have an artifacts.json but no model entry."""
    registered = registered_ports(cfg)
    found: set[str] = set()
    for path in changed_files:
        port = port_of(path)
        if not port or port in registered:
            continue
        if (REPO_ROOT / port / "artifacts.json").is_file():
            found.add(port)
    return sorted(name.split("/", 1)[1] for name in found)


def select_from_benchmark_config(
    cfg: dict[str, Any],
    base: str,
    target: str,
    selected: set[str],
) -> bool:
    old = git_show_json(base, ".github/ci/benchmark_config.json")
    new = read_json(".github/ci/benchmark_config.json")
    if new is None:
        return True
    old_models = old.get("models", {}) if old else {}
    new_models = new.get("models", {})
    for model in configured_model_names(cfg, target, default_only=False):
        if old_models.get(model) != new_models.get(model):
            selected.add(model)
    old_without_models = {k: v for k, v in (old or {}).items() if k != "models"}
    new_without_models = {k: v for k, v in new.items() if k != "models"}
    return old_without_models != new_without_models


def select_from_artifacts(
    cfg: dict[str, Any],
    artifacts_path: str,
    base: str,
    target: str,
    selected: set[str],
    shared_frameworks: set[str],
) -> None:
    new = read_json(artifacts_path)
    if new is None:
        return
    old = git_show_json(base, artifacts_path)
    changed_artifacts = json_changed_keys(old, new, "artifacts")
    if not changed_artifacts:
        return

    for model in configured_model_names(cfg, target, default_only=False):
        model_cfg = cfg["models"].get(model, {})
        if repo_rel(model_cfg.get("_artifacts_path")) != artifacts_path:
            continue
        artifacts = model_cfg.get("artifacts", {})
        refs = collect_artifact_refs(model_cfg)
        for artifact_id in changed_artifacts:
            if artifact_id not in refs:
                continue
            artifact = artifacts.get(artifact_id, {}) if isinstance(artifacts, dict) else {}
            if artifact.get("kind") in FRAMEWORK_ARTIFACT_KINDS:
                framework = artifact.get("framework") or model_framework(model_cfg)
                if framework:
                    shared_frameworks.add(str(framework))
            else:
                selected.add(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--target", choices=("all", "sysemu", "board"), default="board")
    parser.add_argument("--format", choices=("json", "space", "csv", "lines"), default="space")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument(
        "--scope",
        choices=("affected", "changed"),
        default="affected",
        help=(
            "affected: shared infra/workflow/global-config changes fan out to all "
            "default models (use on push to main). "
            "changed: select only models whose own files or vendored framework "
            "source changed — i.e. the models the PR is pushing (use on PRs)."
        ),
    )
    parser.add_argument(
        "--unregistered-out",
        default="",
        help="Write space-separated new ported_models ports lacking a benchmark_config.json entry to this file.",
    )
    args = parser.parse_args()
    honor_global = args.scope == "affected"

    cfg = load_config(args.config)
    changed_files = [norm(path) for path in args.changed_file]
    if not changed_files:
        if not args.base:
            print("error: --base is required unless --changed-file is used", file=sys.stderr)
            return 2
        try:
            changed_files = git_changed_files(args.base, args.head)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    selected: set[str] = set()
    shared_all = False
    shared_frameworks: set[str] = set()
    shared_runners: set[str] = set()

    config_paths = {
        model: repo_rel(model_cfg.get("_config_path"))
        for model, model_cfg in cfg.get("models", {}).items()
    }
    artifact_paths = {
        repo_rel(model_cfg.get("_artifacts_path"))
        for model_cfg in cfg.get("models", {}).values()
        if model_cfg.get("_artifacts_path")
    }

    for path in changed_files:
        if path == ".github/ci/benchmark_config.json":
            # Always pick up model-specific entry changes; only let non-model
            # (global) config changes fan out to all models in "affected" scope.
            global_change = select_from_benchmark_config(cfg, args.base, args.target, selected)
            if honor_global:
                shared_all = global_change or shared_all
            continue
        if path in GENERIC_BOARD_INFRA_PATHS or is_under(path, ".github/ci/platform/"):
            if honor_global:
                shared_all = True
            continue
        if path in RUNNER_INFRA_PATHS:
            if honor_global:
                shared_runners.add(RUNNER_INFRA_PATHS[path])
            continue
        if path in artifact_paths:
            select_from_artifacts(cfg, path, args.base, args.target, selected, shared_frameworks)
            continue

        for model, model_cfg in cfg.get("models", {}).items():
            if not model_supports_target(cfg, model, args.target):
                continue
            config_path = config_paths.get(model)
            if config_path and path == config_path:
                selected.add(model)
                continue
            source = repo_rel(model_cfg.get("source"))
            if source and (path == source or is_under(path, str(Path(source).parent))):
                selected.add(model)
                continue
            root = source_model_root(model_cfg)
            if root and is_under(path, root) and is_model_code_path(path):
                selected.add(model)

            artifacts = model_cfg.get("artifacts", {})
            if isinstance(artifacts, dict):
                for artifact in artifacts.values():
                    if not isinstance(artifact, dict):
                        continue
                    if artifact.get("kind") != "framework_source":
                        continue
                    source_path = repo_rel(artifact.get("submodule_path"))
                    if source_path and is_under(path, source_path):
                        framework = artifact.get("framework") or model_framework(model_cfg)
                        if framework:
                            shared_frameworks.add(str(framework))

    if shared_all:
        selected.update(configured_model_names(cfg, args.target, default_only=True))
    for framework in sorted(shared_frameworks):
        selected.update(default_models_for_framework(cfg, framework, args.target))
    for runner in sorted(shared_runners):
        selected.update(default_models_for_runner(cfg, runner, args.target))

    ordered = [
        model
        for model in configured_model_names(cfg, args.target, default_only=False)
        if model in selected
    ]

    unregistered = unregistered_ports(cfg, changed_files)
    if args.unregistered_out:
        Path(args.unregistered_out).write_text(" ".join(unregistered))

    print(
        "unregistered ports: "
        + (" ".join(unregistered) if unregistered else "(none)"),
        file=sys.stderr,
    )

    print(
        "changed files: "
        + (", ".join(changed_files) if changed_files else "(none)")
        + "\nselected models: "
        + (" ".join(ordered) if ordered else "(none)"),
        file=sys.stderr,
    )
    if args.format == "json":
        print(json.dumps(ordered))
    elif args.format == "csv":
        print(",".join(ordered))
    elif args.format == "lines":
        print("\n".join(ordered))
    else:
        print(" ".join(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
