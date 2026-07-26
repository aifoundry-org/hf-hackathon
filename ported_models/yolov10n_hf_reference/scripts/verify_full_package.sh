#!/usr/bin/env bash
# Validate the generated schema-v2 package before compiling or launching it.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
full_dir="${1:-$repo_root/local-artifacts/yolov10n_hf_reference/full_graph/deterministic}"
model="${2:-}"

python3 - "$full_dir" "$model" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

full_dir = Path(sys.argv[1]).resolve()
model_arg = sys.argv[2]
manifest_path = full_dir / "slice_manifest.json"
header_path = full_dir / "slice_manifest.h"

EXPECTED_SOURCE = {
    "repo": "onnx-community/yolov10n",
    "revision": "57657320425ee34056408a57ad9d29c4d4815bd8",
    "filename": "onnx/model.onnx",
    "sha256": "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b",
    "license": "AGPL-3.0",
}
EXPECTED_HEADER_BYTES = 92881
EXPECTED_HEADER_SHA256 = (
    "79be5b751842df025a3612ebb690e283813ea9ac8e373fd1bc44b706ca7a2a7e"
)
EXPECTED_STAGES = [
    ("stem", "N000", "N005"),
    ("backbone", "N006", "N090"),
    ("sppf_psa", "N091", "N128"),
    ("neck", "N129", "N207"),
    ("three_scale_head", "N208", "N270"),
    ("dfl_decode", "N271", "N288"),
    ("topk_selection", "N289", "N307"),
]


def fail(message):
    raise SystemExit("error: full-package contract failed: " + message)


def require(condition, message):
    if not condition:
        fail(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def integer(value, name):
    require(
        isinstance(value, int) and not isinstance(value, bool),
        name + " must be an integer",
    )
    return int(value)


def package_path(relative, name):
    require(
        isinstance(relative, str) and relative,
        name + " must be a non-empty relative path",
    )
    candidate = (full_dir / relative).resolve()
    try:
        candidate.relative_to(full_dir)
    except ValueError:
        fail(name + " escapes the package")
    return candidate


require(full_dir.is_dir(), "package directory does not exist")
require(manifest_path.is_file(), "slice_manifest.json is missing")
require(header_path.is_file(), "slice_manifest.h is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
require(manifest.get("schema_version") == 2, "schema_version is not 2")
require(
    manifest.get("manifest_kind") == "full_graph_liveness",
    "manifest_kind is not full_graph_liveness",
)
source = manifest.get("source")
require(isinstance(source, dict), "source is not an object")
for key, expected in EXPECTED_SOURCE.items():
    require(source.get(key) == expected, "source.{} differs".format(key))

selection = manifest.get("selection")
require(isinstance(selection, dict), "selection is not an object")
require(selection.get("selector") == "N000:N307", "selector is not N000:N307")
require(selection.get("first_node") == "N000", "first node is not N000")
require(selection.get("last_node") == "N307", "last node is not N307")
require(selection.get("inclusive") is True, "selection is not inclusive")

nodes = manifest.get("nodes")
require(isinstance(nodes, list) and len(nodes) == 308, "node count is not 308")
for index, node in enumerate(nodes):
    require(isinstance(node, dict), "node {} is not an object".format(index))
    require(node.get("index") == index, "node index order differs")
    require(
        node.get("node_id") == "N{:03d}".format(index),
        "node ID order differs",
    )

stages = manifest.get("pmc_stages")
memory = manifest.get("memory_map")
blobs = manifest.get("blobs")
require(isinstance(stages, list), "pmc_stages is not a list")
require(isinstance(memory, dict), "memory_map is not an object")
require(isinstance(blobs, dict), "blobs is not an object")
require(len(stages) == len(EXPECTED_STAGES), "PMC stage count is not seven")
pmc_base = integer(memory.get("pmc_device_offset"), "pmc_device_offset")
pmc_stride = integer(memory.get("pmc_stage_stride"), "pmc_stage_stride")
require(pmc_stride >= 4816, "PMC stride cannot hold one versioned record")
for index, (actual, expected) in enumerate(zip(stages, EXPECTED_STAGES)):
    name, first_node, last_node = expected
    require(actual.get("name") == name, "PMC stage {} name differs".format(index))
    require(
        actual.get("first_node") == first_node
        and actual.get("last_node") == last_node,
        "PMC stage {} range differs".format(index),
    )
    require(
        integer(actual.get("pmc_device_offset"), "stage offset")
        == pmc_base + index * pmc_stride,
        "PMC stage {} offset differs".format(index),
    )
require(
    integer(memory.get("pmc_stage_count"), "pmc_stage_count") == len(stages),
    "memory-map PMC count differs",
)

for name in ("inputs", "weights", "goldens"):
    record = blobs.get(name)
    require(isinstance(record, dict), "missing {} blob record".format(name))
    path = package_path(record.get("path"), "blobs.{}.path".format(name))
    require(path.is_file(), "{} blob is missing".format(name))
    expected_bytes = integer(record.get("nbytes"), name + " nbytes")
    expected_sha = record.get("sha256")
    require(
        path.stat().st_size == expected_bytes,
        "{} blob byte count differs".format(name),
    )
    require(digest(path) == expected_sha, "{} blob SHA-256 differs".format(name))

result_offset = integer(memory.get("result_device_offset"), "result offset")
workspace_bytes = integer(memory.get("workspace_bytes"), "workspace bytes")
dump_size = integer(memory.get("dump_size"), "dump size")
input_offset = integer(memory.get("input_device_offset"), "input offset")
weight_offset = integer(memory.get("weight_device_offset"), "weight offset")
mem_size = integer(memory.get("mem_size"), "mem size")
input_bytes = integer(memory.get("input_blob_bytes"), "input blob bytes")
weight_bytes = integer(memory.get("weight_blob_bytes"), "weight blob bytes")
require(result_offset == 0, "result offset is not zero")
require(128 + workspace_bytes <= pmc_base, "workspace overlaps PMC pages")
require(
    dump_size == pmc_base + len(stages) * pmc_stride,
    "dump does not end after the seven PMC pages",
)
require(input_offset >= dump_size, "input blob overlaps the dump")
require(weight_offset >= input_offset + input_bytes, "weight overlaps input")
require(mem_size >= weight_offset + weight_bytes, "mem_size truncates weights")
require(
    input_bytes == integer(blobs["inputs"]["nbytes"], "inputs nbytes")
    and weight_bytes == integer(blobs["weights"]["nbytes"], "weights nbytes"),
    "memory-map blob sizes differ",
)

tolerances = manifest.get("tolerances")
require(isinstance(tolerances, dict), "tolerances is not an object")
require(
    tolerances.get("atol") == 0.00005
    and tolerances.get("rtol") == 0.0001,
    "global tolerances differ from the validated contract",
)
overrides = tolerances.get("checkpoint_overrides")
require(
    isinstance(overrides, dict) and set(overrides) == {"N288"},
    "checkpoint tolerance overrides differ from the validated contract",
)
n288_tolerance = overrides["N288"]
require(
    isinstance(n288_tolerance, dict)
    and n288_tolerance.get("atol") == 0.0002
    and n288_tolerance.get("rtol") == 0.0001,
    "N288 tolerance differs from the validated contract",
)

generated = manifest.get("generated")
require(isinstance(generated, dict), "generated is not an object")
header_record = generated.get("header")
require(isinstance(header_record, dict), "generated.header is not an object")
generated_header_path = package_path(
    header_record.get("path"), "generated.header.path"
)
require(
    generated_header_path == header_path,
    "generated.header.path is not slice_manifest.h",
)
require(generated_header_path.is_file(), "generated header is missing")
require(
    generated_header_path.stat().st_size
    == integer(header_record.get("nbytes"), "generated.header.nbytes"),
    "generated header byte count differs",
)
require(
    integer(header_record.get("nbytes"), "generated.header.nbytes")
    == EXPECTED_HEADER_BYTES,
    "generated header byte count differs from the pinned topology",
)
header_sha256 = header_record.get("sha256")
require(
    isinstance(header_sha256, str)
    and len(header_sha256) == 64
    and all(character in "0123456789abcdef" for character in header_sha256),
    "generated.header.sha256 is invalid",
)
require(
    digest(generated_header_path) == header_sha256,
    "generated header SHA-256 differs",
)
require(
    header_sha256 == EXPECTED_HEADER_SHA256,
    "generated header SHA-256 differs from the pinned topology",
)

header = header_path.read_text(encoding="utf-8")
macros = {
    match.group(1): int(match.group(2), 0)
    for match in re.finditer(
        r"^#define[ \t]+(YR_[A-Z0-9_]+)[ \t]+(0x[0-9a-fA-F]+|[0-9]+)u?[ \t]*$",
        header,
        flags=re.MULTILINE,
    )
}
expected_macros = {
    "YR_MANIFEST_VERSION": 2,
    "YR_FIRST_NODE": 0,
    "YR_LAST_NODE": 307,
    "YR_NODE_COUNT": 308,
    "YR_RESULT_DEVICE_OFFSET": result_offset,
    "YR_INPUT_DEVICE_OFFSET": input_offset,
    "YR_WEIGHT_DEVICE_OFFSET": weight_offset,
    "YR_PMC_DEVICE_OFFSET": pmc_base,
    "YR_PMC_STAGE_COUNT": len(stages),
    "YR_PMC_STAGE_STRIDE": pmc_stride,
    "YR_INPUT_BLOB_BYTES": input_bytes,
    "YR_WEIGHT_BLOB_BYTES": weight_bytes,
    "YR_WORKSPACE_BYTES": workspace_bytes,
    "YR_DUMP_SIZE": dump_size,
    "YR_MEM_SIZE": mem_size,
}
for name, expected in expected_macros.items():
    require(
        macros.get(name) == expected,
        "{} header/JSON value differs".format(name),
    )

if model_arg:
    model = Path(model_arg).resolve()
    require(model.is_file(), "pinned ONNX model is missing")
    require(
        digest(model) == EXPECTED_SOURCE["sha256"],
        "pinned ONNX SHA-256 differs",
    )

print(
    "FULL_PACKAGE PASS selector=N000:N307 nodes=308 stages=7 "
    "workspace={} dump={} mem={} model_checked={}".format(
        workspace_bytes, dump_size, mem_size, bool(model_arg)
    )
)
PY
