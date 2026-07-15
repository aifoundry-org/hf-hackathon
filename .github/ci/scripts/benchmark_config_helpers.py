"""Helpers for model benchmark config loading and target selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_model_path(model_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else model_dir / path


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or os.environ.get("BENCHMARK_CONFIG", DEFAULT_CONFIG))
    data = json.loads(config_path.read_text())
    models = data.get("models", {})
    expanded = {}
    for name, model_cfg in models.items():
        include = model_cfg.get("config")
        if not include:
            expanded[name] = model_cfg
            continue
        include_path = resolve_repo_path(include)
        included = json.loads(include_path.read_text())
        model_dir = include_path.parent
        artifacts_file = included.get("artifacts_file")
        if artifacts_file:
            artifacts_path = resolve_model_path(model_dir, artifacts_file)
            artifacts = json.loads(artifacts_path.read_text())
            included["artifacts"] = artifacts.get("artifacts", {})
            included["_artifacts_path"] = str(artifacts_path.relative_to(REPO_ROOT))
        included["_config_path"] = str(include_path.relative_to(REPO_ROOT))
        included["_model_dir"] = str(model_dir.relative_to(REPO_ROOT))
        override = {key: value for key, value in model_cfg.items() if key != "config"}
        merged = deep_merge(included, override)
        merged["config"] = include
        expanded[name] = merged
    data["models"] = expanded
    return data


def model_runner(cfg: dict, model: str) -> str:
    return cfg.get("models", {}).get(model, {}).get("runner", "elf")


def model_supports_target(cfg: dict, model: str, target: str) -> bool:
    if target == "all":
        return True
    runner = model_runner(cfg, model)
    if target == "sysemu":
        return runner == "elf"
    if target == "board":
        return bool(cfg.get("models", {}).get(model, {}).get("board", True))
    raise ValueError(f"unknown benchmark target: {target}")


def model_names(cfg: dict, target: str = "all", default_only: bool = False) -> list[str]:
    return [
        name
        for name in cfg.get("models", {}).keys()
        if model_supports_target(cfg, name, target)
        and (not default_only or cfg.get("models", {}).get(name, {}).get("benchmark_default", True))
    ]


def parse_model_selection(selection: str | None, cfg: dict, target: str = "all") -> list[str]:
    all_names = model_names(cfg, "all")
    names = model_names(cfg, target)
    raw = (selection or "").strip()
    if not raw or raw == "default":
        return model_names(cfg, target, default_only=True)
    if raw == "all":
        return names

    requested: list[str] = []
    for item in raw.replace(",", " ").split():
        if item and item not in requested:
            requested.append(item)

    unknown = [item for item in requested if item not in all_names]
    if unknown:
        raise ValueError(
            "unknown benchmark model(s): "
            + ", ".join(unknown)
            + ". configured models: "
            + ", ".join(all_names)
        )
    unsupported = [item for item in requested if item not in names]
    if unsupported:
        raise ValueError(
            "benchmark model(s) not supported for target "
            + target
            + ": "
            + ", ".join(unsupported)
            + ". supported models: "
            + ", ".join(names)
        )
    return requested


def ci_smoke_enabled() -> bool:
    if os.environ.get("CI_SMOKE_FAST", "").lower() in ("1", "true", "yes"):
        return True
    return bool(os.environ.get("GITHUB_ACTIONS"))


def board_mode() -> bool:
    device = os.environ.get("BENCHMARK_DEVICE", "").strip().lower()
    if device in ("soc1sim", "board"):
        return True
    if os.environ.get("BOARD_BENCHMARK", "").lower() in ("1", "true", "yes"):
        return True
    return False


def build_defines(cfg: dict, model: str) -> list[str]:
    m = cfg["models"][model]
    defines = list(m["build"].get("defines", []))
    if board_mode():
        board = cfg.get("board", {}).get("models", {}).get(model, {})
        if board.get("defines"):
            return list(board["defines"])
    if ci_smoke_enabled() and not board_mode():
        smoke = cfg.get("ci_smoke", {}).get("models", {}).get(model, {})
        if smoke.get("defines"):
            defines = list(smoke["defines"])
    return defines


def suite_manifest(cfg: dict, model: str, suite: str) -> str | None:
    m = cfg["models"][model]
    if suite == "smoke":
        return m.get("manifest")
    return m.get(f"{suite}_manifest")


def suite_rows(
    cfg: dict,
    suite: str,
    selection: str | None = None,
    target: str = "sysemu",
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for model in parse_model_selection(selection, cfg, target=target):
        manifest = suite_manifest(cfg, model, suite)
        if manifest:
            rows.append((model, cfg["models"][model]["bench_dir"], manifest))
    return rows


def model_file_loads(cfg: dict, model: str) -> list[dict]:
    default = [{"address": "0x0", "paths": ["zero2m.bin", "common/zero2m.bin"], "required": True}]
    return list(cfg["models"][model].get("file_loads", default))


def model_benchmark_cases(cfg: dict, model: str) -> list[dict]:
    cases = cfg["models"][model].get("benchmark_cases", [])
    return [case for case in cases if isinstance(case, dict) and case.get("name")]


def benchmark_case(cfg: dict, model: str, case_name: str | None) -> dict | None:
    if not case_name:
        return None
    for case in model_benchmark_cases(cfg, model):
        if str(case.get("name")) == str(case_name):
            return case
    return None


def merge_file_loads(base: list[dict], overrides: list[dict]) -> list[dict]:
    merged = [dict(item) for item in base]
    by_address = {str(item["address"]): idx for idx, item in enumerate(merged) if "address" in item}
    for override in overrides:
        item = dict(override)
        address = str(item["address"])
        if address in by_address:
            merged[by_address[address]] = item
        else:
            by_address[address] = len(merged)
            merged.append(item)
    return merged


def case_file_loads(cfg: dict, model: str, case_name: str | None = None) -> list[dict]:
    loads = model_file_loads(cfg, model)
    case = benchmark_case(cfg, model, case_name)
    if case and isinstance(case.get("file_loads"), list):
        loads = merge_file_loads(loads, case["file_loads"])
    return loads


def resolve_file_loads(
    cfg: dict,
    model: str,
    amp_root: str | Path,
    case_name: str | None = None,
) -> tuple[list[dict], list[str]]:
    root = Path(amp_root)
    resolved: list[dict] = []
    missing: list[str] = []
    for load in case_file_loads(cfg, model, case_name):
        address = str(load["address"])
        paths = load.get("paths")
        if not paths:
            paths = [load["path"]]
        found = None
        for rel in paths:
            path = root / rel
            if path.is_file():
                found = path
                break
        if found:
            resolved.append({"address": address, "path": str(found)})
        elif load.get("required", True):
            missing.append(f"missing file_load for {address}")
    return resolved, missing


def benchmark_device(cfg: dict) -> str:
    if board_mode():
        return cfg.get("board", {}).get("device", "soc1sim")
    return "sys_emu"


def sysemu_timeouts(cfg: dict) -> tuple[int, int]:
    if board_mode():
        board = cfg.get("board", {})
        launcher = int(board.get("launcher_timeout_s", 600))
        outer = int(board.get("outer_timeout_s", launcher + 60))
        launcher_cap = int(os.environ.get("TRUSTED_BOARD_LAUNCHER_TIMEOUT_CAP", "0"))
        outer_cap = int(os.environ.get("TRUSTED_BOARD_OUTER_TIMEOUT_CAP", "0"))
        if launcher_cap > 0:
            launcher = min(launcher, launcher_cap)
        if outer_cap > 0:
            outer = min(outer, outer_cap)
        if outer <= launcher:
            launcher = max(1, outer - 10)
        return outer, launcher
    smoke = cfg.get("ci_smoke", {})
    launcher = int(smoke.get("launcher_timeout_s", 5400))
    outer = int(smoke.get("outer_timeout_s", launcher + 120))
    if not ci_smoke_enabled():
        launcher = int(os.environ.get("SYSEMU_LAUNCHER_TIMEOUT", "1800"))
        outer = int(os.environ.get("SYSEMU_TIMEOUT", "3600"))
    return outer, launcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", default=os.environ.get("MODELS", ""))
    parser.add_argument("--format", choices=("json", "space", "csv", "lines"), default="space")
    parser.add_argument("--suite", choices=("smoke", "full"))
    parser.add_argument("--target", choices=("all", "sysemu", "board"), default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        if args.suite:
            rows = suite_rows(cfg, args.suite, args.models, target=args.target)
            print("\n".join("\t".join(row) for row in rows))
            return 0

        models = parse_model_selection(args.models, cfg, target=args.target)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(models))
    elif args.format == "csv":
        print(",".join(models))
    elif args.format == "lines":
        print("\n".join(models))
    else:
        print(" ".join(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
