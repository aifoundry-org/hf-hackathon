#!/usr/bin/env python3
"""Select benchmark models touched by a PR diff."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
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
    ".github/ci/scripts/build_leaderboard_elf.sh",
    ".github/ci/scripts/changed_benchmark_models.py",
    ".github/ci/scripts/prepare_benchmark_inputs.sh",
    ".github/ci/scripts/resolve_leaderboard_team.sh",
    ".github/ci/scripts/run_model_benchmark.sh",
    ".github/ci/scripts/score_results.py",
    ".github/ci/platform/deploy/soc3-benchmark.sh",
    "scripts/run_sysemu_model_ports.sh",
}

BOARD_LOCK_INFRA_PATHS = {
    ".github/ci/scripts/board_lock.py",
    ".github/ci/scripts/prepare_board_lock.sh",
}
BOARD_LOCK_SMOKE_MODELS = ("dncnn", "smollm2_135m")


RUNNER_INFRA_PATHS = {
    ".github/ci/scripts/run_llama_server_benchmark.py": "llama_server",
    ".github/ci/scripts/run_smolvlm2_video_benchmark.py": "smolvlm2_video",
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

RUNTIME_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".s",
    ".S",
}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
YOLO_REAL_IMAGE_DETECTIONS_VALIDATION = "yolo_real_image_detections"
YOLO_REFERENCE_DETECTIONS_ACCURACY = "yolo_reference_detections"
YOLO_REFERENCE_INFRA_PATHS = {
    ".github/ci/reference/yolo.json",
    ".github/ci/scripts/prepare_trusted_yolo_tree.py",
    ".github/ci/scripts/refresh_trusted_yolo_prs.sh",
    ".github/ci/scripts/run_yolo_host_reference.sh",
    ".github/ci/scripts/trusted_yolo_input_hash.sh",
    ".github/workflows/trusted-yolo-pr.yml",
}
SMOLVLM2_REFERENCE_INFRA_PATHS = {
    ".github/ci/reference/smolvlm2_500m_video.json",
}
LLAMA32_SUBMISSION_PATHS = {
    "ported_models/llama_cpp_et/submissions/llama32_1b.json",
    "ported_models/llama_cpp_et/submissions/llama32_1b.track.json",
}
MODEL_PORT_CLAIM_ROOT = "ported_models/submissions/model_ports"
ZERO_BLOB_PRIMARY = "zero2m.bin"


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


def normalize_zero_blob_fallbacks(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                [ZERO_BLOB_PRIMARY]
                if key == "paths"
                and isinstance(nested, list)
                and nested
                and norm(str(nested[0])) == ZERO_BLOB_PRIMARY
                else normalize_zero_blob_fallbacks(nested)
            )
            for key, nested in value.items()
            if not (
                key.startswith("requires_")
                and key.endswith("_inputs")
                and nested is False
            )
        }
    if isinstance(value, list):
        return [normalize_zero_blob_fallbacks(nested) for nested in value]
    return value


def collect_artifact_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "artifacts":
                continue
            if key.endswith("_artifact") and isinstance(nested, str):
                refs.add(nested)
            elif key.endswith("_artifacts") and isinstance(nested, list):
                refs.update(str(item) for item in nested if isinstance(item, str))
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
    return Path(path).suffix in MODEL_CODE_SUFFIXES


def is_runtime_code_path(path: str) -> bool:
    name = Path(path).name.lower()
    if name in {"readme.md", "model.md", "third_party.md"}:
        return False
    if "/docs/" in path:
        return False
    return Path(path).suffix in RUNTIME_CODE_SUFFIXES


def required_validation_for_path(path: str) -> str | None:
    """Return an extra validation requirement for fidelity-sensitive paths."""
    path = norm(path)
    if is_under(path, "ported_models/yolo/src"):
        return YOLO_REAL_IMAGE_DETECTIONS_VALIDATION
    return None


def model_satisfies_validation(model_cfg: dict[str, Any], requirement: str | None) -> bool:
    if not requirement:
        return True
    if requirement == YOLO_REAL_IMAGE_DETECTIONS_VALIDATION:
        if model_cfg.get("runner", "elf") != "elf":
            return False
        score = model_cfg.get("score", {})
        if score.get("metric") != "kernel_wait_s" or score.get("higher_is_better") is not False:
            return False
        source = repo_rel(model_cfg.get("source")) or ""
        if not is_under(source, "ported_models/yolo/src"):
            return False
        validation = model_cfg.get("validation", {})
        if not isinstance(validation, dict):
            return False
        if validation.get("kind") != YOLO_REAL_IMAGE_DETECTIONS_VALIDATION:
            return False
        cases = model_cfg.get("benchmark_cases", [])
        if not isinstance(cases, list) or not cases:
            return False
        try:
            min_image_count = int(validation.get("min_image_count", 5))
        except (TypeError, ValueError):
            return False
        contract_value = validation.get("reference_contract") or model_cfg.get(
            "reference_contract"
        )
        if not contract_value:
            return False
        contract_path = Path(contract_value)
        if not contract_path.is_absolute():
            contract_path = REPO_ROOT / contract_path
        try:
            contract = json.loads(contract_path.read_text())
            fixtures = contract["fixtures"]["cases"]
            board_abi = contract["board_abi"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        fixed_cases = {
            str(case["name"]): case
            for case in fixtures
            if isinstance(case, dict) and case.get("name") and case.get("asset")
        }
        if len(fixed_cases) < min_image_count:
            return False
        if [str(case.get("name")) for case in cases] != list(fixed_cases):
            return False
        valid_cases = 0
        for case in cases:
            if not isinstance(case, dict) or not case.get("name"):
                continue
            name = str(case["name"])
            fixture = fixed_cases.get(name)
            if fixture is None:
                continue
            accuracy = case.get("accuracy", {})
            if (
                not isinstance(accuracy, dict)
                or accuracy.get("kind") != YOLO_REFERENCE_DETECTIONS_ACCURACY
                or accuracy.get("reference_case") != name
            ):
                continue
            try:
                if int(str(accuracy.get("offset")), 0) != int(
                    str(board_abi["detections_offset"]), 0
                ):
                    continue
                if int(accuracy.get("max_detections", 0)) != int(
                    board_abi["max_detections"]
                ):
                    continue
            except (TypeError, ValueError):
                continue
            expected_asset = str(fixture["asset"])
            prefix = "ported_models/yolo/assets/"
            if not expected_asset.startswith(prefix):
                continue
            expected_load = norm(expected_asset[len(prefix) :])
            input_paths = {
                norm(path)
                for load in case.get("file_loads", [])
                if str(load.get("address")) == str(board_abi["input_address"])
                for path in (load.get("paths") or [load.get("path")])
                if path
            }
            if expected_load not in input_paths:
                continue
            valid_cases += 1
        return valid_cases >= min_image_count and valid_cases == len(fixed_cases) == len(cases)
    return False


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


def local_includes(path: str, seen: set[str] | None = None) -> set[str]:
    """Return a source file and repo-local quoted includes reachable from it."""
    path = norm(path)
    if seen is None:
        seen = set()
    if path in seen:
        return set()
    seen.add(path)

    full_path = REPO_ROOT / path
    if not full_path.is_file():
        return {path}

    covered = {path}
    try:
        text = full_path.read_text(errors="ignore")
    except OSError:
        return covered

    parent = Path(path).parent
    for include in INCLUDE_RE.findall(text):
        include_path = norm(parent / include)
        if not (REPO_ROOT / include_path).is_file():
            continue
        covered.update(local_includes(include_path, seen))
    return covered


def benchmark_runtime_coverage(model_cfg: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return exact files and directory roots that a configured benchmark builds."""
    files: set[str] = set()
    roots: set[str] = set()

    source = repo_rel(model_cfg.get("source"))
    if source:
        files.update(local_includes(source))

    artifacts = model_cfg.get("artifacts", {})
    if isinstance(artifacts, dict):
        for artifact in artifacts.values():
            if not isinstance(artifact, dict):
                continue
            if artifact.get("kind") != "framework_source":
                continue
            source_path = repo_rel(artifact.get("submodule_path"))
            if source_path:
                roots.add(source_path)

    return files, roots


def benchmark_covers_path(
    cfg: dict[str, Any],
    target: str,
    path: str,
    required_validation: str | None = None,
) -> bool:
    for model, model_cfg in cfg.get("models", {}).items():
        if not model_supports_target(cfg, model, target):
            continue
        files, roots = benchmark_runtime_coverage(model_cfg)
        if path in files and model_satisfies_validation(model_cfg, required_validation):
            return True
        if any(is_under(path, root) for root in roots) and model_satisfies_validation(
            model_cfg,
            required_validation,
        ):
            return True
    return False


def uncovered_runtime_code_paths(cfg: dict[str, Any], changed_files: list[str], target: str) -> list[str]:
    """Registered port runtime files changed by the PR but not built by any benchmark row."""
    registered = registered_ports(cfg)
    uncovered: set[str] = set()
    for path in changed_files:
        port = port_of(path)
        if not port or port not in registered:
            continue
        if not is_runtime_code_path(path):
            continue
        required_validation = required_validation_for_path(path)
        if benchmark_covers_path(cfg, target, path, required_validation):
            continue
        uncovered.add(path)
    return sorted(uncovered)


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
        old_model = normalize_zero_blob_fallbacks(old_models.get(model))
        new_model = normalize_zero_blob_fallbacks(new_models.get(model))
        if old_model != new_model:
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
            "default models (use for broad validation). "
            "changed: select only models whose own files or vendored framework "
            "source changed — i.e. the models a PR or main push is submitting."
        ),
    )
    parser.add_argument(
        "--unregistered-out",
        default="",
        help="Write space-separated new ported_models ports lacking a benchmark_config.json entry to this file.",
    )
    parser.add_argument(
        "--uncovered-out",
        default="",
        help=(
            "Write space-separated changed runtime source files not covered by "
            "an adequate benchmark row. Some paths have extra validation requirements."
        ),
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
        if path in YOLO_REFERENCE_INFRA_PATHS:
            selected.add("yolo")
            continue
        if path in SMOLVLM2_REFERENCE_INFRA_PATHS:
            selected.add("smolvlm2_500m_video")
            continue
        if path in LLAMA32_SUBMISSION_PATHS:
            selected.add("llama32_1b")
            continue
        if is_under(path, MODEL_PORT_CLAIM_ROOT) and path.endswith(".json"):
            claim = read_json(path)
            model = str((claim or {}).get("benchmark_model") or "")
            if model in cfg.get("models", {}) and model_supports_target(cfg, model, args.target):
                selected.add(model)
            continue
        if path in BOARD_LOCK_INFRA_PATHS:
            for model in BOARD_LOCK_SMOKE_MODELS:
                if model in cfg.get("models", {}) and model_supports_target(
                    cfg, model, args.target
                ):
                    selected.add(model)
            if honor_global:
                shared_all = True
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
            if path in configured_asset_paths(model_cfg):
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
    uncovered = uncovered_runtime_code_paths(cfg, changed_files, args.target)
    if (
        "yolo" in selected
        and not model_satisfies_validation(
            cfg["models"]["yolo"], YOLO_REAL_IMAGE_DETECTIONS_VALIDATION
        )
    ):
        source = repo_rel(cfg["models"]["yolo"].get("source"))
        if source and source not in uncovered:
            uncovered.append(source)
            uncovered.sort()
    if args.uncovered_out:
        Path(args.uncovered_out).write_text(" ".join(uncovered))

    print(
        "unregistered ports: "
        + (" ".join(unregistered) if unregistered else "(none)"),
        file=sys.stderr,
    )
    print(
        "uncovered or under-validated runtime code: "
        + (" ".join(uncovered) if uncovered else "(none)"),
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
