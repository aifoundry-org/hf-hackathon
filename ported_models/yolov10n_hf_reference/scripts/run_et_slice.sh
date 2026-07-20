#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"

device="sys_emu"
slice_dir=""
elf=""
launcher="${LAUNCHER:-}"
output_dir=""
outer_timeout=300
launcher_timeout=240
lock_timeout=600
shire=0

usage() {
  cat <<'EOF'
Usage: run_et_slice.sh --slice-dir DIR --elf FILE --launcher FILE [options]

Options:
  --device NAME            sys_emu or soc1sim (real PCIe). Default: sys_emu.
  --output-dir DIR         Required result-artifact directory.
  --outer-timeout SECONDS  Bounded process timeout. Default: 300.
  --launcher-timeout SEC   Timeout passed to launcher. Default: 240.
  --lock-timeout SECONDS   Board lock wait for soc1sim. Default: 600.
  --shire INDEX            Shire index. Default: 0.
EOF
}

need_value() {
  [[ -n "${2:-}" ]] || {
    echo "error: $1 requires a value" >&2
    exit 2
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) need_value "$1" "${2:-}"; device="$2"; shift 2 ;;
    --slice-dir) need_value "$1" "${2:-}"; slice_dir="$2"; shift 2 ;;
    --elf) need_value "$1" "${2:-}"; elf="$2"; shift 2 ;;
    --launcher) need_value "$1" "${2:-}"; launcher="$2"; shift 2 ;;
    --output-dir) need_value "$1" "${2:-}"; output_dir="$2"; shift 2 ;;
    --outer-timeout) need_value "$1" "${2:-}"; outer_timeout="$2"; shift 2 ;;
    --launcher-timeout) need_value "$1" "${2:-}"; launcher_timeout="$2"; shift 2 ;;
    --lock-timeout) need_value "$1" "${2:-}"; lock_timeout="$2"; shift 2 ;;
    --shire) need_value "$1" "${2:-}"; shire="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$device" == "sys_emu" || "$device" == "soc1sim" ]] || {
  echo "error: --device must be sys_emu or soc1sim" >&2
  exit 2
}
for value in "$outer_timeout" "$launcher_timeout" "$lock_timeout" "$shire"; do
  case "$value" in ''|*[!0-9]*) echo "error: timeout/shire values must be integers" >&2; exit 2 ;; esac
done
[[ "$device" != "soc1sim" || "$shire" -eq 0 ]] || {
  echo "error: the repository board lock currently supports only soc1sim shire 0" >&2
  exit 2
}
for value in "$outer_timeout" "$launcher_timeout" "$lock_timeout"; do
  [[ "$value" -gt 0 ]] || {
    echo "error: timeout values must be greater than zero" >&2
    exit 2
  }
done
[[ -n "$slice_dir" && -n "$elf" && -n "$launcher" && -n "$output_dir" ]] || {
  usage >&2
  exit 2
}

slice_dir="$(cd "$slice_dir" && pwd)"
elf="$(readlink -f "$elf")"
launcher="$(readlink -f "$launcher")"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
manifest="$slice_dir/slice_manifest.json"
inputs="$slice_dir/inputs.bin"
weights="$slice_dir/weights.bin"
dump="$output_dir/dump.bin"
log="$output_dir/run.log"
build_record="$elf.build.json"

for artifact in \
  "$dump" "$log" "$output_dir/run_result.json" \
  "$output_dir/command.txt" "$output_dir/wrapper_command.txt" \
  "$output_dir/slice.elf" "$output_dir/build_record.json"; do
  [[ ! -e "$artifact" ]] || {
    echo "error: refusing to reuse result artifact: $artifact" >&2
    exit 2
  }
done

[[ -f "$manifest" && -f "$inputs" && -f "$weights" ]] || {
  echo "error: slice directory lacks manifest/input/weight blobs: $slice_dir" >&2
  exit 2
}
[[ -f "$elf" && -x "$launcher" ]] || {
  echo "error: missing ELF or executable launcher" >&2
  exit 2
}
[[ -f "$build_record" ]] || {
  echo "error: missing ET build provenance record: $build_record" >&2
  exit 2
}
cp "$elf" "$output_dir/slice.elf"
cp "$build_record" "$output_dir/build_record.json"
saved_elf="$output_dir/slice.elf"
saved_build_record="$output_dir/build_record.json"

read -r input_offset weight_offset mem_size dump_size first_node last_node < <(
  python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
memory = manifest["memory_map"]
selection = manifest["selection"]
print(
    memory["input_device_offset"],
    memory["weight_device_offset"],
    memory["mem_size"],
    memory["dump_size"],
    selection["first_node"],
    selection["last_node"],
)
PY
)

python3 - "$manifest" "$inputs" "$weights" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest = json.load(open(sys.argv[1]))
for name, path_arg in (("inputs", sys.argv[2]), ("weights", sys.argv[3])):
    record = manifest["blobs"][name]
    path = Path(path_arg)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != int(record["nbytes"]) or actual != record["sha256"]:
        raise SystemExit(
            "error: {} blob identity mismatch: bytes={}/{} sha256={}/{}".format(
                name, len(data), record["nbytes"], actual, record["sha256"]
            )
        )
    print("BLOB_CHECK PASS name={} bytes={} sha256={}".format(
        name, len(data), actual
    ))
PY

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "device=$device"
  echo "slice=$first_node:$last_node"
  echo "launcher=$launcher"
  echo "elf=$elf"
  echo "ET_PLATFORM=${ET_PLATFORM:-}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
} > "$output_dir/environment.txt"

if [[ "$device" == "soc1sim" ]]; then
  {
    echo "backend=soc1sim"
    echo "meaning=real PCIe ET-SoC1 hardware"
    stat -c 'device=%n type=%F major_minor=%t:%T permissions=%A owner=%U:%G' \
      /dev/et0_mgmt /dev/et0_ops
    lspci -nn | grep -Ei '1e0a:eb01|esperanto|processing accelerators'
  } > "$output_dir/device_evidence.txt"
  [[ -c /dev/et0_mgmt && -c /dev/et0_ops ]] || {
    echo "error: soc1sim requested but ET-SoC1 device nodes are not character devices" >&2
    exit 2
  }
else
  {
    echo "backend=sys_emu"
    echo "meaning=software system emulator"
    echo "ET_PLATFORM=${ET_PLATFORM:-}"
    if [[ -n "${ET_PLATFORM:-}" ]]; then
      stat "${ET_PLATFORM}/bin/sys_emu" || true
    fi
  } > "$output_dir/device_evidence.txt"
fi

command=(
  "$launcher"
  --device "$device"
  --elf-load "$elf"
  --shire "$shire"
)
if [[ -s "$inputs" ]]; then
  command+=(--file_load "$(printf '0x%x,%s' "$input_offset" "$inputs")")
fi
if [[ -s "$weights" ]]; then
  command+=(--file_load "$(printf '0x%x,%s' "$weight_offset" "$weights")")
fi
command+=(
  --dump_after "$dump"
  --timeout "$launcher_timeout"
  --mem_size "$mem_size"
  --dump_size "$dump_size"
)
printf '%q ' "${command[@]}" > "$output_dir/command.txt"
echo >> "$output_dir/command.txt"

cd "$output_dir"
start_epoch="$(date +%s)"
set +e
if [[ "$device" == "soc1sim" ]]; then
  bash "$repo_root/.github/ci/scripts/prepare_board_lock.sh" \
    "${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}" > "$output_dir/board_lock.log" 2>&1
  prepare_rc=$?
  if [[ "$prepare_rc" -eq 0 ]]; then
    printf '%q ' \
      python3 "$repo_root/.github/ci/scripts/board_lock.py" \
      --lock "${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}" \
      --timeout "$lock_timeout" -- \
      timeout --kill-after=10s "$outer_timeout" \
      "$port_root/scripts/board_reset_and_run.sh" "${command[@]}" \
      > "$output_dir/wrapper_command.txt"
    echo >> "$output_dir/wrapper_command.txt"
    python3 "$repo_root/.github/ci/scripts/board_lock.py" \
      --lock "${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}" \
      --timeout "$lock_timeout" \
      -- \
      timeout --kill-after=10s "$outer_timeout" \
      "$port_root/scripts/board_reset_and_run.sh" "${command[@]}" \
      > "$log" 2>&1
    rc=$?
  else
    rc="$prepare_rc"
    echo "board lock preparation failed with rc=$prepare_rc" > "$log"
  fi
else
  printf '%q ' timeout --kill-after=10s "$outer_timeout" "${command[@]}" \
    > "$output_dir/wrapper_command.txt"
  echo >> "$output_dir/wrapper_command.txt"
  timeout --kill-after=10s "$outer_timeout" "${command[@]}" > "$log" 2>&1
  rc=$?
fi
set -e
end_epoch="$(date +%s)"

identity_ok=0
grep -Fx "erbium_soc1sim: elf=$elf device=$device shire=$shire" \
  "$log" >/dev/null 2>&1 && identity_ok=1
completion_ok=0
grep -Fx "Kernel completed successfully" "$log" >/dev/null 2>&1 \
  && completion_ok=1
dump_log_ok=0
grep -Fx "Dumped $dump_size bytes to $dump" "$log" >/dev/null 2>&1 \
  && dump_log_ok=1
reset_ok=1
if [[ "$device" == "soc1sim" ]]; then
  reset_ok=0
  grep -F "Resetting ET-SoC1 via " "$log" >/dev/null 2>&1 \
    && reset_ok=1
fi
dump_ok=0
[[ -f "$dump" && "$(stat -c %s "$dump")" -eq "$dump_size" ]] && dump_ok=1
status="fail"
[[ "$rc" -eq 0 && "$identity_ok" -eq 1 && "$completion_ok" -eq 1 \
   && "$dump_log_ok" -eq 1 && "$reset_ok" -eq 1 \
   && "$dump_ok" -eq 1 ]] && status="pass"

python3 - \
  "$output_dir" "$manifest" "$device" "$status" "$rc" \
  "$identity_ok" "$completion_ok" "$dump_log_ok" "$reset_ok" "$dump_ok" \
  "$start_epoch" "$end_epoch" \
  "$launcher" "$saved_elf" "$saved_build_record" "$manifest" "$inputs" "$weights" \
  "$dump" "$log" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    out, manifest_path, device, status, rc, identity_ok, completion_ok,
    dump_log_ok, reset_ok, dump_ok, start, end, launcher, elf, build_record,
    manifest_file, inputs, weights, dump, log,
) = sys.argv[1:]

def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}

manifest = json.load(open(manifest_path))
payload = {
    "schema_version": 1,
    "status": status,
    "device": device,
    "hardware": device == "soc1sim",
    "launcher_identity_match": bool(int(identity_ok)),
    "completion_log_match": bool(int(completion_ok)),
    "dump_log_match": bool(int(dump_log_ok)),
    "board_reset_match": bool(int(reset_ok)),
    "dump_size_match": bool(int(dump_ok)),
    "return_code": int(rc),
    "elapsed_seconds": int(end) - int(start),
    "source_sha256": manifest["source"]["sha256"],
    "selection": manifest["selection"],
    "artifacts": {
        name: digest(path)
        for name, path in {
            "launcher": launcher,
            "elf": elf,
            "build_record": build_record,
            "slice_manifest": manifest_file,
            "inputs": inputs,
            "weights": weights,
            "dump": dump,
            "log": log,
            "command": str(Path(out) / "command.txt"),
            "wrapper_command": str(Path(out) / "wrapper_command.txt"),
            "environment": str(Path(out) / "environment.txt"),
            "device_evidence": str(Path(out) / "device_evidence.txt"),
            "board_lock": str(Path(out) / "board_lock.log"),
        }.items()
    },
}
Path(out, "run_result.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

cat "$log"
echo "DEVICE_RUN ${status^^} device=$device rc=$rc identity_ok=$identity_ok completion_ok=$completion_ok dump_log_ok=$dump_log_ok reset_ok=$reset_ok dump_ok=$dump_ok output=$output_dir"
[[ "$status" == "pass" ]]
