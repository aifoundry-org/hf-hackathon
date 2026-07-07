#!/usr/bin/env python3
"""Post short leaderboard messages to a Discord webhook."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MAX_CONTENT = 2000


def load_messages(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, str):
        return [data] if data.strip() else []
    if not isinstance(data, list):
        raise ValueError("messages file must contain a JSON string or list")
    return [str(item).strip() for item in data if str(item).strip()]


def compact_content(messages: list[str]) -> str:
    content = "\n".join(messages)
    if len(content) <= MAX_CONTENT:
        return content
    return content[: MAX_CONTENT - 3].rstrip() + "..."


def post_webhook(webhook_url: str, content: str, username: str) -> None:
    payload = {
        "content": content,
        "username": username,
        "allowed_mentions": {"parse": []},
    }
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hf-hackathon-board-ci",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"Discord webhook returned HTTP {resp.status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages-file", required=True)
    parser.add_argument("--webhook-url", default="")
    parser.add_argument("--username", default="AIFoundry leaderboard")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    messages = load_messages(Path(args.messages_file))
    if not messages:
        print("No Discord leaderboard announcements.")
        return 0

    content = compact_content(messages)
    webhook_url = (
        args.webhook_url
        or os.environ.get("DISCORD_WEBHOOK_URL")
        or os.environ.get("DISCORD_HACKATHON_GENERAL_WEBHOOK_URL")
        or ""
    )
    if args.dry_run or not webhook_url:
        print(content)
        if not webhook_url and not args.dry_run:
            print("Discord webhook URL is not configured; skipped.")
        return 0

    try:
        post_webhook(webhook_url, content, args.username)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"Discord webhook failed: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Discord webhook failed: {exc}", file=sys.stderr)
        return 1

    print("Posted Discord leaderboard announcement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
