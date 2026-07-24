#!/usr/bin/env python3
"""Remove models already covered by main-owned trusted runtime gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
LLAMA_CONTRACT = REPO_ROOT / ".github" / "ci" / "reference" / "llama32_1b.json"
LLAMA_MODEL = "llama32_1b"
SMOLVLM2_MODEL = "smolvlm2_500m_video"


def delegated_runtime_models(contract: dict[str, Any]) -> set[str]:
    regression_models = contract.get("runtime", {}).get("regression_models", [])
    if not isinstance(regression_models, list) or not all(
        isinstance(model, str) and model for model in regression_models
    ):
        raise RuntimeError("trusted Llama contract has invalid regression_models")
    return {LLAMA_MODEL, SMOLVLM2_MODEL, *regression_models}


def remaining_models(
    selected: Iterable[str], contract: dict[str, Any]
) -> list[str]:
    delegated = delegated_runtime_models(contract)
    return [model for model in selected if model not in delegated]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="")
    parser.add_argument("--contract", type=Path, default=LLAMA_CONTRACT)
    parser.add_argument("--format", choices=("space", "lines", "json"), default="space")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    models = remaining_models(args.models.replace(",", " ").split(), contract)
    if args.format == "json":
        print(json.dumps(models))
    elif args.format == "lines":
        print("\n".join(models))
    else:
        print(" ".join(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
