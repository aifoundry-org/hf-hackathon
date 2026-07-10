#!/usr/bin/env python3
"""Apply only participant-owned YOLO implementation changes to a trusted tree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = "ported_models/yolo/src/"
WEIGHTS_PATH = "ported_models/yolo/assets/yolo/weights_region.bin"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".s", ".S"}
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_WEIGHTS_BYTES = 64 * 1024 * 1024
SAFE_PATH_RE = re.compile(r"[A-Za-z0-9._+/-]+")


class OverlayError(RuntimeError):
    pass


def git(
    repo: Path,
    *args: str,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise OverlayError(f"git {' '.join(args)} failed: {detail}")
    return proc


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.decode().strip()


def normalize(path: str) -> str:
    value = PurePosixPath(path).as_posix()
    if (
        value == "."
        or value.startswith(("/", "../"))
        or "/../" in value
    ):
        raise OverlayError(f"unsafe repository path: {path}")
    return value.removeprefix("./")


def is_allowed_path(path: str) -> bool:
    path = normalize(path)
    if not SAFE_PATH_RE.fullmatch(path):
        return False
    if path == WEIGHTS_PATH:
        return True
    return path.startswith(SOURCE_ROOT) and PurePosixPath(path).suffix in SOURCE_SUFFIXES


def participant_merge_base(repo: Path, base: str, head: str) -> str:
    value = git_text(repo, "merge-base", base, head)
    if not value:
        raise OverlayError(f"no merge base between {base} and {head}")
    return value


def changed_paths(repo: Path, base: str, head: str) -> list[str]:
    merge_base = participant_merge_base(repo, base, head)
    raw = git(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--diff-filter=ACDMRTUXB",
        merge_base,
        head,
    ).stdout
    return sorted(normalize(item.decode()) for item in raw.split(b"\0") if item)


def validate_staged_files(repo: Path, paths: list[str]) -> None:
    source_bytes = 0
    for path in paths:
        if not is_allowed_path(path):
            raise OverlayError(f"trusted overlay changed a non-implementation path: {path}")
        entry = git(repo, "ls-files", "--stage", "--", path).stdout.decode().strip()
        if not entry:
            continue
        mode = entry.split(maxsplit=1)[0]
        if mode != "100644":
            raise OverlayError(f"trusted overlay requires regular non-executable files: {path}")
        file_path = repo / path
        if file_path.is_symlink() or not file_path.is_file():
            raise OverlayError(f"trusted overlay requires a regular file: {path}")
        size = file_path.stat().st_size
        if path == WEIGHTS_PATH:
            if size > MAX_WEIGHTS_BYTES:
                raise OverlayError(
                    f"packed YOLO weights exceed {MAX_WEIGHTS_BYTES} bytes: {size}"
                )
        else:
            source_bytes += size
    if source_bytes > MAX_SOURCE_BYTES:
        raise OverlayError(
            f"YOLO implementation sources exceed {MAX_SOURCE_BYTES} bytes: {source_bytes}"
        )


def apply_overlay(repo: Path, base: str, head: str, main: str) -> dict[str, object]:
    for name, ref in (("base", base), ("head", head), ("main", main)):
        resolved = git_text(repo, "rev-parse", f"{ref}^{{commit}}")
        if not resolved:
            raise OverlayError(f"could not resolve {name} ref {ref}")

    current = git_text(repo, "rev-parse", "HEAD")
    resolved_main = git_text(repo, "rev-parse", f"{main}^{{commit}}")
    if current != resolved_main:
        raise OverlayError(f"trusted tree HEAD {current} is not requested main {resolved_main}")
    if git(repo, "status", "--porcelain", "--untracked-files=no").stdout:
        raise OverlayError("trusted main checkout is not clean before applying the submission")

    merge_base = participant_merge_base(repo, base, head)
    changed = changed_paths(repo, base, head)
    allowed = [path for path in changed if is_allowed_path(path)]
    ignored = [path for path in changed if path not in allowed]
    if not allowed:
        raise OverlayError("PR has no approved YOLO implementation or packed-weight changes")

    pathspecs = [f":(literal){path}" for path in allowed]
    patch = git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        merge_base,
        head,
        "--",
        *pathspecs,
    ).stdout
    if not patch:
        raise OverlayError("approved YOLO changes produced an empty patch")

    git(
        repo,
        "apply",
        "--index",
        "--3way",
        "--whitespace=nowarn",
        "-",
        input_bytes=patch,
    )
    staged_raw = git(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--no-renames",
        resolved_main,
    ).stdout
    staged = sorted(normalize(item.decode()) for item in staged_raw.split(b"\0") if item)
    if not staged:
        raise OverlayError("participant patch is already present on main; no candidate remains")
    validate_staged_files(repo, staged)

    return {
        "schema_version": 1,
        "base_sha": git_text(repo, "rev-parse", f"{base}^{{commit}}"),
        "participant_merge_base_sha": merge_base,
        "participant_sha": git_text(repo, "rev-parse", f"{head}^{{commit}}"),
        "main_sha": resolved_main,
        "participant_changed_paths": changed,
        "applied_paths": staged,
        "ignored_paths": ignored,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    try:
        metadata = apply_overlay(args.repo.resolve(), args.base, args.head, args.main)
    except OverlayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(payload)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
