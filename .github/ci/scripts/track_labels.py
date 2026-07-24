#!/usr/bin/env python3
"""Classify a pull request into hackathon track labels."""

import json
import re
import sys
from pathlib import Path

LABELS = {
    "yolo": "track: yolo-performance",
    "week2": "track: week-2-challenge",
    "llama32": "track: llama-3.2-1b-performance",
    "ports": "track: model-ports",
    "community": "track: community",
    "misc": "misc",
}
MANAGED_LABELS = tuple(LABELS.values())

# Week 2 challenge: SmolVLM2-500M-Video-Instruct on ET-SoC1.
# The shared llama.cpp-et gitlink is included because implementation-only
# submissions update it without changing a model-specific benchmark file.
WEEK2_PREFIXES = (
    "ported_models/llama_cpp_et/assets/smolvlm2_video",
    "ported_models/llama_cpp_et/benchmarks/smolvlm2_500m_video.json",
    "ported_models/llama_cpp_et/docs/smolvlm2_500m_video.md",
    "ported_models/llama_cpp_et/src/llama.cpp-et",
)

LLAMA32_RE = re.compile(
    r"(^|[/_.-])(?:llama32[_-]1b|llama[_-]3[._-]2[_-]1b)([/_.-]|$)", re.I
)


def clean(path):
    return str(path or "").replace("\\", "/").removeprefix("./").rstrip("/")


def under(path, prefix):
    path, prefix = clean(path), clean(prefix)
    return path == prefix or path.startswith(prefix + "/")


def is_community(path):
    path = clean(path)
    name = path.rsplit("/", 1)[-1]
    return (
        path == "README.md"
        or under(path, "docs")
        or "/docs/" in path
        or under(path, "scripts")
        or under(path, ".github/ci/scripts")
        or under(path, ".github/workflows")
        or re.fullmatch(r"(?:AGENTS|SKILLS)\.md", name, re.I)
        or re.search(r"recipe\.md$", name, re.I)
    )


def desired_labels(files, new_roots=(), week2_prefixes=WEEK2_PREFIXES):
    changed = [
        (clean(item.get("filename")), str(item.get("status", "modified")).lower())
        for item in files
    ]
    paths = [path for path, _ in changed]
    rules = (
        (LABELS["yolo"], any(under(path, "ported_models/yolo") for path in paths)),
        (LABELS["week2"], any(under(path, prefix) for path in paths for prefix in week2_prefixes)),
        (LABELS["llama32"], any(LLAMA32_RE.search(path) for path in paths)),
        (
            LABELS["ports"],
            bool(new_roots)
            or any(
                status == "added"
                and (
                    re.fullmatch(r"ported_models/llama_cpp_et/benchmarks/[^/]+\.json", path, re.I)
                    or re.fullmatch(
                        r"ported_models/submissions/model_ports/[a-z0-9][a-z0-9_-]+\.json",
                        path,
                        re.I,
                    )
                )
                for path, status in changed
            ),
        ),
        (LABELS["community"], any(is_community(path) for path in paths)),
    )
    selected = [label for label, matches in rules if matches]
    return selected or [LABELS["misc"]]


def classify(files, repo_root=Path("."), week2_prefixes=WEEK2_PREFIXES):
    ports = repo_root / "ported_models"
    existing_roots = {path.name for path in ports.iterdir() if path.is_dir()}
    changed_roots = {
        parts[1]
        for item in files
        if len(parts := clean(item.get("filename")).split("/")) > 2
        and parts[0] == "ported_models"
    }
    new_roots = sorted(changed_roots - existing_roots)
    return {
        "changed_file_count": len(files),
        "new_port_roots": new_roots,
        "managed_labels": list(MANAGED_LABELS),
        "desired_labels": desired_labels(files, new_roots, week2_prefixes),
    }


if __name__ == "__main__":
    files = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(classify(files), indent=2))
