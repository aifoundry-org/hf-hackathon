#!/usr/bin/env python3
"""Independently validate full-graph liveness and arena non-aliasing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


PORT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_PACKAGE = (
    REPO_ROOT
    / "local-artifacts/yolov10n_hf_reference/full_graph"
    / "deterministic_full308_v3"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def node_number(value: Any) -> int:
    require(
        isinstance(value, str)
        and len(value) == 4
        and value.startswith("N")
        and value[1:].isdigit(),
        "invalid node ID {!r}".format(value),
    )
    return int(value[1:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args()


def main() -> int:
    package = parse_args().package.resolve()
    try:
        manifest = json.loads(
            (package / "slice_manifest.json").read_text(encoding="utf-8")
        )
        require(manifest.get("schema_version") == 2, "schema is not v2")
        nodes = manifest.get("nodes")
        tensors = manifest.get("tensors")
        require(
            isinstance(nodes, list) and len(nodes) == 308,
            "node list is not the full graph",
        )
        require(isinstance(tensors, list), "tensor list is absent")
        tensors_by_name: Dict[str, Dict[str, Any]] = {}
        for tensor in tensors:
            name = tensor.get("name")
            require(
                isinstance(name, str) and name not in tensors_by_name,
                "invalid or duplicate tensor name",
            )
            tensors_by_name[name] = tensor

        consumers: Dict[str, List[int]] = {}
        produced = set()
        for index, node in enumerate(nodes):
            require(
                node.get("index") == index
                and node.get("node_id") == "N{:03d}".format(index),
                "node ordering differs at {}".format(index),
            )
            inputs = node.get("inputs")
            outputs = node.get("outputs")
            require(
                isinstance(inputs, list) and isinstance(outputs, list),
                "node {} lacks input/output lists".format(index),
            )
            for name in inputs:
                if name:
                    require(
                        name in tensors_by_name,
                        "node {} input is undescribed".format(index),
                    )
                    consumers.setdefault(name, []).append(index)
            for name in outputs:
                require(
                    name in tensors_by_name and name not in produced,
                    "node {} output is invalid or duplicated".format(index),
                )
                produced.add(name)

        arena_bytes = manifest["memory_plan"]["arena_bytes"]
        alignment = manifest["memory_plan"]["alignment_bytes"]
        require(
            isinstance(arena_bytes, int) and arena_bytes > 0,
            "arena byte count is invalid",
        )
        require(alignment == 64, "arena alignment is not 64 bytes")
        workspace: List[Dict[str, Any]] = []
        for tensor in tensors:
            if tensor.get("storage") != "workspace":
                continue
            name = tensor["name"]
            producer = node_number(tensor.get("producer"))
            live_start = tensor.get("live_start")
            live_end = tensor.get("live_end")
            nbytes = tensor.get("nbytes")
            allocated = tensor.get("allocated_nbytes")
            offset = tensor.get("offset")
            require(
                producer == live_start
                and nodes[producer]["outputs"].count(name) == 1,
                "{} producer/live_start differs".format(name),
            )
            expected_end = max(consumers.get(name, [producer]))
            if tensor.get("checkpoint") is True:
                expected_end = len(nodes)
            require(
                live_end == expected_end,
                "{} live_end is {}, expected {}".format(
                    name, live_end, expected_end
                ),
            )
            require(
                isinstance(nbytes, int)
                and isinstance(allocated, int)
                and allocated == (nbytes + alignment - 1) // alignment * alignment,
                "{} allocation size is not aligned".format(name),
            )
            require(
                isinstance(offset, int)
                and offset % alignment == 0
                and 0 <= offset <= arena_bytes - allocated,
                "{} allocation exceeds/misaligns the arena".format(name),
            )
            workspace.append(tensor)

        for left_index, left in enumerate(workspace):
            left_start = left["live_start"]
            left_end = left["live_end"]
            left_offset = left["offset"]
            left_limit = left_offset + left["allocated_nbytes"]
            for right in workspace[left_index + 1:]:
                lifetimes_overlap = not (
                    left_end < right["live_start"]
                    or right["live_end"] < left_start
                )
                storage_overlaps = not (
                    left_limit <= right["offset"]
                    or right["offset"] + right["allocated_nbytes"]
                    <= left_offset
                )
                require(
                    not (lifetimes_overlap and storage_overlaps),
                    "live tensors {!r} and {!r} alias".format(
                        left["name"], right["name"]
                    ),
                )
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print("FULL_MEMORY_PLAN FAIL {}".format(error), file=sys.stderr)
        return 1

    print(
        "FULL_MEMORY_PLAN PASS nodes=308 workspace_tensors={} "
        "arena_bytes={} live_aliases=0".format(len(workspace), arena_bytes)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
