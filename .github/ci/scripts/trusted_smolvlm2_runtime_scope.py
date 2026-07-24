#!/usr/bin/env python3
"""Classify and record nested runtime changes for trusted SmolVLM2 attempts."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


KERNEL_ROOT = "ggml/src/ggml-et/et-kernels/src/"
KERNEL_SUFFIXES = {".c", ".h", ".S", ".inc"}
INTEGRATION_PATHS = {
    "ggml/src/ggml-et/CMakeLists.txt",
    "ggml/src/ggml-et/ggml-et-ops.cpp",
    "ggml/src/ggml-et/ggml-et-ops.h",
}
ALLOWED_STATUSES = {"A", "M"}
ALLOWED_MODE = "100644"


def classify_change(path: str, status: str, mode: str) -> Dict[str, object]:
    """Return the policy classification for one candidate-tree change."""
    scope = "rejected"
    reason = "path is outside the trusted SmolVLM2 implementation surface"

    if status not in ALLOWED_STATUSES:
        reason = "only added or modified implementation files are allowed"
    elif mode != ALLOWED_MODE:
        reason = "implementation files must be regular non-executable files"
    elif path in INTEGRATION_PATHS:
        scope = "integration"
        reason = "ET kernel registration or host dispatch"
    elif path.startswith(KERNEL_ROOT) and Path(path).suffix in KERNEL_SUFFIXES:
        scope = "kernel"
        reason = "ET kernel implementation"

    return {
        "path": path,
        "status": status,
        "mode": mode or None,
        "scope": scope,
        "allowed": scope != "rejected",
        "reason": reason,
    }


def build_report(
    base_revision: str,
    candidate_revision: str,
    changes: Iterable[Mapping[str, str]],
) -> Dict[str, object]:
    """Build the stable JSON report used by CI artifacts and tests."""
    classified = [
        classify_change(change["path"], change["status"], change.get("mode", ""))
        for change in changes
    ]
    by_scope = {
        scope: [change["path"] for change in classified if change["scope"] == scope]
        for scope in ("kernel", "integration", "rejected")
    }
    return {
        "schema_version": 1,
        "base_revision": base_revision,
        "candidate_revision": candidate_revision,
        "allowed": not by_scope["rejected"],
        "summary": {
            "total": len(classified),
            "kernel": len(by_scope["kernel"]),
            "integration": len(by_scope["integration"]),
            "rejected": len(by_scope["rejected"]),
        },
        "kernel_paths": by_scope["kernel"],
        "integration_paths": by_scope["integration"],
        "rejected_paths": by_scope["rejected"],
        "changes": classified,
    }


def _git(repo: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _candidate_mode(repo: Path, revision: str, path: str) -> str:
    entry = _git(repo, ["ls-tree", "-z", revision, "--", path])
    if not entry:
        return ""
    return entry.split(b" ", 1)[0].decode("ascii")


def inspect_changes(repo: Path, base_revision: str, candidate_revision: str) -> List[Dict[str, str]]:
    """Read status and candidate modes directly from the two runtime trees."""
    raw = _git(
        repo,
        [
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            base_revision,
            candidate_revision,
            "--",
        ],
    )
    fields = raw.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("unexpected git diff --name-status output")

    changes = []
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        path = fields[index + 1].decode("utf-8", errors="surrogateescape")
        changes.append(
            {
                "path": path,
                "status": status,
                "mode": _candidate_mode(repo, candidate_revision, path),
            }
        )
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.base,
        args.head,
        inspect_changes(args.repo, args.base, args.head),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "Trusted SmolVLM2 runtime scope: "
        "{kernel} kernel, {integration} integration, {rejected} rejected".format(
            **report["summary"]
        )
    )
    for change in report["changes"]:
        print(
            "  {scope}: {status} {path} ({mode})".format(
                scope=change["scope"],
                status=change["status"],
                path=change["path"],
                mode=change["mode"] or "deleted",
            )
        )
    return 0 if report["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
