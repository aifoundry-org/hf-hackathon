#!/usr/bin/env bash
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
slice_dir="${1:?usage: build_et_slice.sh SLICE_DIR [OUTPUT]}"
output="${2:-$slice_dir/yolov10n_hf_slice.elf}"
et_install="${ET_INSTALL:-${ET_PLATFORM:-/opt/et}}"
gcc="${ET_GCC:-$et_install/bin/riscv64-unknown-elf-gcc}"

test -f "$slice_dir/slice_manifest.h" || {
  echo "error: generate the full graph or run tools/capture_range.py first; missing $slice_dir/slice_manifest.h" >&2
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

# Default stays the 16-hart configuration already validated tonight. Override
# YR_ET_NUM_HARTS (and pass -DYR_NHART=... via YR_ET_EXTRA_CFLAGS) to build a
# single-hart image, e.g. for the tensor-unit fast path in
# yr_conv_tensor_et.c, which only ever runs with yr_hart_count() == 1.
et_num_harts="${YR_ET_NUM_HARTS:-16}"
# The linker script defaults to 1024 bytes of stack per hart, which the -O2
# frames overrun. yr_conv alone takes 496 bytes and yr_run_node_span another
# 416 once they stop being folded together, and a hart that runs off the end
# writes into its neighbour's stack, which shows up as a kernel runtime
# failure late in the graph rather than as a crash at the overrun. Measured
# with -fstack-usage. 4096 keeps NUM_HARTS * STACK_SIZE a multiple of 4 KiB,
# which the linker script asserts on.
et_stack_size="${YR_ET_STACK_SIZE:-4096}"
link_flags=(
  "-Wl,--gc-sections" "-Wl,--no-warn-rwx-segments" "-Wl,--emit-relocs"
  "-Wl,--defsym=NUM_HARTS=${et_num_harts}" "-Wl,--defsym=region0_size=0x00400000"
  "-Wl,--defsym=STACK_SIZE=${et_stack_size}"
  -T"$ERBIUM_LD"
)

# -O2 measured 22 percent faster than -O1 on the board for the full graph and
# leaves the dump bit-identical to the host, because none of the flags below
# let the compiler reassociate floating point. The three companions are not
# optional at -O2. -fno-tree-loop-distribute-patterns stops the zeroing loops
# turning into calls to memset, which does not exist in a -nostdlib build.
# -fno-strict-aliasing is required because the runtime reaches tensors through
# byte pointers cast to float pointers, which the aliasing rules do not allow.
# -funroll-loops was worth a further 2 seconds on top of plain -O2. -O3 was
# tried and came out slower than -O2, so it is deliberately not used.
compile_flags=(
  "-std=gnu11" -O2 -funroll-loops -fno-tree-loop-distribute-patterns
  -fno-strict-aliasing
  -fno-fast-math "-ffp-contract=off" -fno-tree-vectorize
  "-march=rv64imfc" "-mabi=lp64f" "-mcmodel=medany" -nostdlib
  -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections
)
define_flags=(
  "-DNUM_HARTS=${et_num_harts}" -DYR_PMC
)
sources=(
  "$port_root/src/ref_runtime.c"
  "$port_root/src/yr_conv_tensor_et.c"
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
  "$port_root/src/yr_conv_tensor_et.c" \
  "${command[@]}" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

(
    record, elf, compiler, compiler_version, docker_image, docker_image_id,
    manifest, linker, layout, crt, runtime_c, runtime_h, pmc_h, runner_c,
    conv_tensor_c,
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
            "conv_tensor_c": conv_tensor_c,
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
