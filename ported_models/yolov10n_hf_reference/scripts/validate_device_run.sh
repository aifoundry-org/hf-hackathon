#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
slice_dir="${1:?usage: validate_device_run.sh SLICE_DIR RUN_DIR}"
run_dir="${2:?usage: validate_device_run.sh SLICE_DIR RUN_DIR}"
expected_device="${3:?usage: validate_device_run.sh SLICE_DIR RUN_DIR EXPECTED_DEVICE}"
python="${YOLOV10N_HOST_PYTHON:-$repo_root/local-artifacts/yolov10n_hf_reference/venv/bin/python}"

[[ "$expected_device" == "sys_emu" || "$expected_device" == "soc1sim" ]] || {
  echo "error: EXPECTED_DEVICE must be sys_emu or soc1sim" >&2
  exit 2
}

manifest="$slice_dir/slice_manifest.json"
read -r schema_version manifest_kind pmc_offset < <(
  python3 - "$manifest" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(
    manifest["schema_version"],
    manifest.get("manifest_kind", "legacy_slice"),
    manifest["memory_map"]["pmc_device_offset"],
)
PY
)

"$python" - \
  "$manifest" "$slice_dir" "$run_dir" "$expected_device" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
slice_dir = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
expected_device = sys.argv[4]
result = json.loads((run_dir / "run_result.json").read_text())
manifest = json.loads(manifest_path.read_text())

def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def require(condition, message):
    if not condition:
        raise SystemExit("error: execution evidence check failed: " + message)

require(result["status"] == "pass", "run status is not pass")
require(result["device"] == expected_device, "device does not match expectation")
require(
    result["hardware"] is (expected_device == "soc1sim"),
    "hardware flag does not match device",
)
require(result["return_code"] == 0, "launcher return code is nonzero")
for field in (
    "launcher_identity_match",
    "completion_log_match",
    "dump_log_match",
    "dump_size_match",
):
    require(result[field] is True, field + " is not true")
if expected_device == "soc1sim":
    require(
        result.get("board_reset_match") is True,
        "board_reset_match is not true",
    )
require(
    result["source_sha256"] == manifest["source"]["sha256"],
    "source checksum differs from slice manifest",
)
require(
    result["selection"]["first_node"] == manifest["selection"]["first_node"]
    and result["selection"]["last_node"] == manifest["selection"]["last_node"],
    "selected node range differs from slice manifest",
)

local_artifacts = {
    "elf": run_dir / "slice.elf",
    "build_record": run_dir / "build_record.json",
    "slice_manifest": manifest_path,
    "inputs": slice_dir / manifest["blobs"]["inputs"]["path"],
    "weights": slice_dir / manifest["blobs"]["weights"]["path"],
    "dump": run_dir / "dump.bin",
    "log": run_dir / "run.log",
    "command": run_dir / "command.txt",
    "wrapper_command": run_dir / "wrapper_command.txt",
    "environment": run_dir / "environment.txt",
    "device_evidence": run_dir / "device_evidence.txt",
}
if expected_device == "soc1sim":
    local_artifacts["board_lock"] = run_dir / "board_lock.log"

for name, path in local_artifacts.items():
    require(path.is_file(), "{} is missing".format(path))
    stored = result["artifacts"].get(name)
    require(stored is not None, "run result lacks " + name)
    actual = identity(path)
    require(
        actual["bytes"] == stored["bytes"]
        and actual["sha256"] == stored["sha256"],
        name + " digest differs from run result",
    )

for name in ("inputs", "weights"):
    actual = identity(local_artifacts[name])
    record = manifest["blobs"][name]
    require(
        actual["bytes"] == int(record["nbytes"])
        and actual["sha256"] == record["sha256"],
        name + " digest differs from slice manifest",
    )

build = json.loads((run_dir / "build_record.json").read_text())
require(
    build["elf"]["sha256"] == identity(run_dir / "slice.elf")["sha256"],
    "saved ELF differs from build record",
)
evidence = (run_dir / "device_evidence.txt").read_text()
require(
    "backend=" + expected_device in evidence,
    "device evidence has the wrong backend",
)
if expected_device == "soc1sim":
    require(
        "/dev/et0_mgmt" in evidence
        and "/dev/et0_ops" in evidence
        and evidence.count("type=character special file") >= 2,
        "hardware character-device evidence is incomplete",
    )
    require(
        "1e0a:eb01" in evidence.lower() or "esperanto" in evidence.lower(),
        "PCIe device evidence is incomplete",
    )

print(
    "EXECUTION_EVIDENCE PASS device={} hardware={} "
    "nodes={}:{}".format(
        expected_device,
        expected_device == "soc1sim",
        manifest["selection"]["first_node"],
        manifest["selection"]["last_node"],
    )
)
PY

if [[ "$schema_version" == "2" \
      && "$manifest_kind" == "contiguous_node_range" ]]; then
  "$python" "$port_root/tools/compare_range_v2.py" \
    "$slice_dir" "$run_dir/dump.bin" \
    --json "$run_dir/tensor_compare.json"
elif [[ "$schema_version" == "1" ]]; then
  "$python" "$port_root/tools/compare_slice.py" \
    "$slice_dir" "$run_dir/dump.bin" \
    --json "$run_dir/tensor_compare.json"
else
  echo "error: unsupported device-validation manifest schema=$schema_version kind=$manifest_kind" >&2
  exit 2
fi
"$python" "$port_root/tools/decode_pmc.py" \
  "$run_dir/dump.bin" --offset "$pmc_offset" --format json \
  > "$run_dir/pmc.json"
grep -E '"status": "PASS"' "$run_dir/pmc.json" >/dev/null

echo "DEVICE_VALIDATION PASS device=$expected_device run=$run_dir compare=$run_dir/tensor_compare.json pmc=$run_dir/pmc.json"
