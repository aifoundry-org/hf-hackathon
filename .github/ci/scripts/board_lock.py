#!/usr/bin/env python3
"""Open and hold the shared ET-SoC1 lock without creating it."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, TextIO


DEFAULT_BOARD_LOCK = "/var/lock/etsoc-shire0.lock"


def open_board_lock(path: str | Path) -> TextIO:
    """Open a provisioned lock without O_CREAT.

    Linux's fs.protected_regular policy can reject O_CREAT when a shared lock
    in a sticky directory is owned by another user, even when the file is
    mode 0666. Opening the existing inode avoids that policy while preserving
    flock interoperability between root and non-root board users.
    """

    lock_path = Path(path)
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(str(lock_path), flags)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"board lock is not provisioned: {lock_path}; "
            "run .github/ci/scripts/prepare_board_lock.sh first"
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"board lock is not readable and writable by this user: {lock_path}; "
            "run .github/ci/scripts/prepare_board_lock.sh as root"
        ) from exc
    return os.fdopen(fd, "r+")


@contextlib.contextmanager
def exclusive_board_lock(
    path: str | Path,
    *,
    timeout_s: float | None = None,
    poll_s: float = 0.05,
) -> Iterator[TextIO]:
    lock_file = open_board_lock(path)
    locked = False
    try:
        if timeout_s is None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out after {timeout_s:g}s waiting for board lock: {path}"
                        )
                    time.sleep(poll_s)
        locked = True
        yield lock_file
    finally:
        if locked:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command while holding the shared ET-SoC1 lock."
    )
    parser.add_argument(
        "--lock",
        default=os.environ.get("BOARD_LOCK", DEFAULT_BOARD_LOCK),
        help="pre-provisioned board lock path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="maximum seconds to wait for the lock",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    try:
        with exclusive_board_lock(args.lock, timeout_s=args.timeout) as lock_file:
            # Keep the descriptor in the child. If CI kills this wrapper first,
            # the bounded benchmark still owns the lock until it exits, so a
            # second runner cannot overlap it on the card.
            return subprocess.run(
                command, pass_fds=(lock_file.fileno(),)
            ).returncode
    except (RuntimeError, TimeoutError) as exc:
        print(f"board-lock: {exc}", file=sys.stderr)
        return 73


if __name__ == "__main__":
    raise SystemExit(main())
