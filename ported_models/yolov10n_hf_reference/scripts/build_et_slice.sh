#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
slice_dir="${1:?usage: build_et_slice.sh SLICE_DIR [OUTPUT]}"
output="${2:-$slice_dir/yolov10n_hf_slice.elf}"
et_install="${ET_INSTALL:-${ET_PLATFORM:-/opt/et}}"
gcc="${ET_GCC:-$et_install/bin/riscv64-unknown-elf-gcc}"

test -f "$slice_dir/slice_manifest.h" || {
  echo "error: run tools/capture_slice.py first; missing $slice_dir/slice_manifest.h" >&2
  exit 2
}
test -x "$gcc" || {
  echo "error: supported ET compiler not executable: $gcc" >&2
  exit 2
}
if ! "$gcc" --version >/dev/null 2>&1; then
  echo "error: configured ET compiler cannot run on this host: $gcc" >&2
  echo "Use the repository's Ubuntu 24.04 Docker toolchain workflow; no fallback compiler is selected." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$repo_root/.github/ci/scripts/resolve_et_platform_paths.sh"
resolve_et_platform_paths
read -r -a include_flags <<<"${ERBIUM_GCC_INCLUDE_FLAGS}"
read -r -a extra_cflags <<<"${YR_ET_EXTRA_CFLAGS:-}"
mkdir -p "$(dirname "$output")"

compile_flags=(
  "-std=gnu11" -O1 -fno-fast-math "-ffp-contract=off" -fno-tree-vectorize
  "-march=rv64imfc" "-mabi=lp64f" "-mcmodel=medany" -nostdlib
  -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections
)
define_flags=(
  "-DNUM_HARTS=16" -DYR_PMC
)
link_flags=(
  "-Wl,--gc-sections" "-Wl,--no-warn-rwx-segments" "-Wl,--emit-relocs"
  "-Wl,--defsym=NUM_HARTS=16" "-Wl,--defsym=region0_size=0x00400000"
  -T"$ERBIUM_LD"
)
sources=(
  "$port_root/src/ref_runtime.c"
  "$port_root/src/et_slice_runner.c"
  "$repo_root/.github/ci/support/hart_report_crt.S"
  "$ERBIUM_LAYOUT"
)
command=(
  "$gcc"
  "${compile_flags[@]}"
  "${include_flags[@]}"
  -I"$port_root/src" -I"$slice_dir"
  "${define_flags[@]}"
  "${extra_cflags[@]}"
  "${link_flags[@]}"
  "${sources[@]}"
  -o "$output"
)
"${command[@]}"

compiler_output="$("$gcc" --version)"
compiler_version="${compiler_output%%$'\n'*}"
docker_image="${ET_DOCKER_IMAGE:-}"
if [[ -z "$docker_image" && "$(basename "$gcc")" == "et_gcc_docker_wrapper.sh" ]]; then
  docker_image="et-gcc:24.04"
fi
docker_image_id=""
if [[ -n "$docker_image" ]] && command -v docker >/dev/null; then
  docker_image_id="$(docker image inspect --format '{{.Id}}' "$docker_image" 2>/dev/null || true)"
fi
record="${YR_BUILD_RECORD:-$output.build.json}"
python3 - \
  "$record" "$output" "$gcc" "$compiler_version" "$docker_image" \
  "$docker_image_id" "$slice_dir/slice_manifest.h" "$ERBIUM_LD" \
  "$ERBIUM_LAYOUT" "$repo_root/.github/ci/support/hart_report_crt.S" \
  "$port_root/src/ref_runtime.c" "$port_root/src/ref_runtime.h" \
  "$port_root/src/ref_pmc.h" "$port_root/src/et_slice_runner.c" \
  "${command[@]}" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

(
    record, elf, compiler, compiler_version, docker_image, docker_image_id,
    manifest, linker, layout, crt, runtime_c, runtime_h, pmc_h, runner_c,
    *command,
) = sys.argv[1:]

def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

payload = {
    "schema_version": 1,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "compiler": {
        "path": compiler,
        "version": compiler_version,
        "docker_image": docker_image or None,
        "docker_image_id": docker_image_id or None,
    },
    "command": command,
    "inputs": {
        name: identity(path)
        for name, path in {
            "slice_manifest_header": manifest,
            "linker_script": linker,
            "layout": layout,
            "crt": crt,
            "runtime_c": runtime_c,
            "runtime_h": runtime_h,
            "pmc_h": pmc_h,
            "et_runner_c": runner_c,
        }.items()
    },
    "elf": identity(elf),
}
Path(record).write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

echo "ET_BUILD PASS compiler=$gcc output=$output record=$record"
