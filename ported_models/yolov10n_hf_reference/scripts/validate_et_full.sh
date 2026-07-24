#!/usr/bin/env bash
# Validate full-graph execution evidence, all checkpoints, output0, and PMCs.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
full_dir="${1:?usage: validate_et_full.sh FULL_DIR RUN_DIR EXPECTED_DEVICE [MODEL] [REQUIRE_DIRECT_OUTPUT]}"
run_dir="${2:?usage: validate_et_full.sh FULL_DIR RUN_DIR EXPECTED_DEVICE [MODEL] [REQUIRE_DIRECT_OUTPUT]}"
expected_device="${3:?usage: validate_et_full.sh FULL_DIR RUN_DIR EXPECTED_DEVICE [MODEL] [REQUIRE_DIRECT_OUTPUT]}"
model="${4:-${YOLOV10N_MODEL:-$repo_root/local-artifacts/yolov10n_hf_reference/model.onnx}}"
require_direct_output="${5:-${YR_REQUIRE_DIRECT_OUTPUT:-0}}"
python="${YOLOV10N_HOST_PYTHON:-$repo_root/local-artifacts/yolov10n_hf_reference/venv/bin/python}"

[[ "$expected_device" == "sys_emu" || "$expected_device" == "soc1sim" ]] || {
  echo "error: EXPECTED_DEVICE must be sys_emu or soc1sim" >&2
  exit 2
}
[[ "$require_direct_output" == "0" || "$require_direct_output" == "1" ]] || {
  echo "error: REQUIRE_DIRECT_OUTPUT must be 0 or 1" >&2
  exit 2
}
[[ -x "$python" ]] || {
  echo "error: host reference Python is not executable: $python" >&2
  exit 2
}

"$port_root/scripts/verify_full_package.sh" "$full_dir" "$model"
"$port_root/scripts/verify_full_elf.sh" \
  "$full_dir" "$run_dir/slice.elf" "$run_dir/build_record.json"

for output in \
  "$run_dir/full_compare.json" \
  "$run_dir/pmc_stages.json"; do
  [[ ! -e "$output" ]] || {
    echo "error: refusing to overwrite validation evidence: $output" >&2
    exit 2
  }
done

"$python" - \
  "$full_dir/slice_manifest.json" "$full_dir" "$run_dir" \
  "$expected_device" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
full_dir = Path(sys.argv[2]).resolve()
run_dir = Path(sys.argv[3]).resolve()
expected_device = sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise SystemExit("error: full execution evidence failed: " + message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


require(result.get("status") == "pass", "run status is not pass")
require(result.get("device") == expected_device, "device differs")
require(
    result.get("hardware") is (expected_device == "soc1sim"),
    "hardware flag differs",
)
require(result.get("return_code") == 0, "launcher returned nonzero")
for field in (
    "launcher_identity_match",
    "completion_log_match",
    "dump_log_match",
    "dump_size_match",
):
    require(result.get(field) is True, field + " is not true")
if expected_device == "soc1sim":
    require(result.get("board_reset_match") is True, "board reset is unproven")
require(
    result.get("source_sha256") == manifest["source"]["sha256"],
    "pinned source checksum differs",
)
selection = result.get("selection", {})
require(
    selection.get("selector") == "N000:N307"
    and selection.get("first_node") == "N000"
    and selection.get("last_node") == "N307"
    and selection.get("inclusive") is True,
    "run selection is not inclusive N000:N307",
)

local_artifacts = {
    "elf": run_dir / "slice.elf",
    "build_record": run_dir / "build_record.json",
    "slice_manifest": manifest_path,
    "inputs": full_dir / manifest["blobs"]["inputs"]["path"],
    "weights": full_dir / manifest["blobs"]["weights"]["path"],
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
    stored = result.get("artifacts", {}).get(name)
    require(stored is not None, "run result lacks " + name)
    actual = identity(path)
    require(
        actual["bytes"] == stored.get("bytes")
        and actual["sha256"] == stored.get("sha256"),
        name + " identity differs from run result",
    )

for name in ("inputs", "weights"):
    actual = identity(local_artifacts[name])
    expected = manifest["blobs"][name]
    require(
        actual["bytes"] == expected["nbytes"]
        and actual["sha256"] == expected["sha256"],
        name + " identity differs from full manifest",
    )
build = json.loads((run_dir / "build_record.json").read_text(encoding="utf-8"))
require(
    build.get("elf", {}).get("sha256")
    == identity(run_dir / "slice.elf")["sha256"],
    "saved ELF differs from build record",
)
evidence = (run_dir / "device_evidence.txt").read_text(encoding="utf-8")
require(
    "backend=" + expected_device in evidence,
    "device evidence names another backend",
)
if expected_device == "soc1sim":
    require(
        "meaning=real PCIe ET-SoC1 hardware" in evidence,
        "hardware meaning is absent",
    )
    require(
        "/dev/et0_mgmt" in evidence
        and "/dev/et0_ops" in evidence
        and evidence.count("type=character special file") >= 2,
        "hardware character-device evidence is incomplete",
    )
    require(
        "1e0a:eb01" in evidence.lower() or "esperanto" in evidence.lower(),
        "hardware PCIe identity is incomplete",
    )
else:
    require(
        "meaning=software system emulator" in evidence,
        "system-emulator meaning is absent",
    )
    require(
        "meaning=real PCIe ET-SoC1 hardware" not in evidence,
        "system-emulator evidence claims hardware",
    )

require(
    identity(run_dir / "dump.bin")["bytes"]
    == manifest["memory_map"]["dump_size"],
    "dump byte count differs from full manifest",
)
print(
    "FULL_EXECUTION_EVIDENCE PASS device={} hardware={} "
    "selector=N000:N307".format(
        expected_device, expected_device == "soc1sim"
    )
)
PY

compare_args=(
  "$python" "$port_root/tools/compare_full.py"
  "$full_dir" "$run_dir/dump.bin"
  --model "$model"
  --json "$run_dir/full_compare.json"
)
if [[ "$require_direct_output" == "1" ]]; then
  compare_args+=(--require-direct-output)
fi
"${compare_args[@]}"

mapfile -t pmc_rows < <(
  "$python" - "$full_dir/slice_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
for index, stage in enumerate(manifest["pmc_stages"]):
    print("{}\t{}\t{}\t{}\t{}".format(
        index,
        stage["name"],
        stage["first_node"],
        stage["last_node"],
        stage["pmc_device_offset"],
    ))
PY
)

pmc_files=()
for row in "${pmc_rows[@]}"; do
  IFS=$'\t' read -r index name first_node last_node offset <<<"$row"
  output="$run_dir/pmc_${index}_${name}.json"
  [[ ! -e "$output" ]] || {
    echo "error: refusing to overwrite validation evidence: $output" >&2
    exit 2
  }
  "$python" "$port_root/tools/decode_pmc.py" \
    "$run_dir/dump.bin" --offset "$offset" --format json > "$output"
  grep -E '"status": "PASS"' "$output" >/dev/null
  pmc_files+=("$output")
  echo "PMC_STAGE PASS name=$name nodes=$first_node:$last_node report=$output"
done

"$python" - \
  "$full_dir/slice_manifest.json" "$run_dir/pmc_stages.json" \
  "${pmc_files[@]}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.load(open(sys.argv[1]))
output = Path(sys.argv[2])
paths = [Path(value) for value in sys.argv[3:]]
if len(paths) != len(manifest["pmc_stages"]):
    raise SystemExit("error: PMC report count differs from manifest")
records = []
for stage, path in zip(manifest["pmc_stages"], paths):
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if decoded.get("status") != "PASS":
        raise SystemExit("error: PMC stage {} failed".format(stage["name"]))
    records.append(
        {
            "name": stage["name"],
            "first_node": stage["first_node"],
            "last_node": stage["last_node"],
            "pmc_device_offset": stage["pmc_device_offset"],
            "report": path.name,
            "decoded": decoded,
        }
    )
payload = {
    "schema_version": 1,
    "status": "PASS",
    "scope": "seven disjoint intervals around ONNX nodes only",
    "selector": "N000:N307",
    "stages": records,
}
output.write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

echo "ET_FULL_VALIDATION PASS device=$expected_device selector=N000:N307 require_direct_output=$require_direct_output compare=$run_dir/full_compare.json pmc=$run_dir/pmc_stages.json"
