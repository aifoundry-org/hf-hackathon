# CORE-ET Quickstart on ET-SoC1 Boards

This guide is for a fresh workstation plus an assigned ET-SoC1 board host.
It covers the current CORE-ET model-porting flow on ET-SoC1 boards. It avoids
personal paths and lab-specific hostnames. Everything configurable is expressed
through environment variables.

The working repo is the model-port repo. AIFoundry and OpenHW repositories are
used for the platform support, CORE-ET RTL context, and RISC-V ET toolchain.
Hugging Face model references are downloaded on demand. Generated ET-SoC1 ELFs
are built locally from the checked-in port sources.

## Ported Models

The shared `ported_models` set for this flow is:

```text
ported_models = [
  "yolo",
  "lfm25",
]
```

These are the models with current board sweep manifests:

- `yolo`: YOLO-style detection kernels.
- `lfm25`: board-only LFM2.5 GGUF through the ET-backed `llama.cpp` framework.

## Variables

Pick locations that make sense for the machine:

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/et-soc1-work}"
export MODEL_PORT_REPO="$WORK_ROOT/hf-hackathon"
export ET_PLATFORM_SRC="$WORK_ROOT/et-platform"
export CORE_ET_SRC="$WORK_ROOT/core-et"
export ET_INSTALL="$WORK_ROOT/et"
export BUILD_ROOT="$WORK_ROOT/build"
export ARTIFACT_ROOT="$MODEL_PORT_REPO/local-artifacts"
export BENCHMARK_ARTIFACT_ROOT="$ARTIFACT_ROOT/model-port-benchmarks"

# Set this to the ET-SoC1 board host assigned to you.
export BOARD_HOST="<your-et-soc1-board-host>"
export BOARD_USER="${BOARD_USER:-root}"
export BOARD_SSH="$BOARD_USER@$BOARD_HOST"

# Pick a writable directory on the board host for this work.
export REMOTE_ROOT="${REMOTE_ROOT:-/root/aifoundry/et-soc1-model-ports}"
export REMOTE_MODELS="$REMOTE_ROOT/ported_models"
```

Do not use placeholder values literally. Set `BOARD_HOST` to the host you were
given by your lab or cluster.

## Clone Source Repos

Clone the model-port repo, AIFoundry platform repo, and OpenHW CORE-ET RTL repo:

```bash
mkdir -p "$WORK_ROOT"

git clone https://github.com/aifoundry-org/hf-hackathon.git "$MODEL_PORT_REPO"
git clone https://github.com/aifoundry-org/et-platform.git "$ET_PLATFORM_SRC"
git clone https://github.com/openhwgroup/core-et.git "$CORE_ET_SRC"

git -C "$MODEL_PORT_REPO" branch --show-current
git -C "$ET_PLATFORM_SRC" branch --show-current
git -C "$CORE_ET_SRC" branch --show-current
```

Prefer each repo's default branch unless the project gives you a specific
revision.

## Install Toolchain

Install the RISC-V ET toolchain through the AIFoundry `et-platform` helper.
The model-port kernels expect `rv64imfc/lp64f`.

```bash
"$ET_PLATFORM_SRC/docker/get_toolchain.sh" \
  --repo aifoundry-org/riscv-gnu-toolchain \
  --install-dir "$ET_INSTALL" \
  --base-distro ubuntu \
  --distro-version 22.04 \
  --arch rv64imfc \
  --abi lp64f \
  --install-deps true
```

Adjust `--base-distro` and `--distro-version` for the host OS.

Set the environment:

```bash
export ET_PLATFORM="$ET_INSTALL"
export PATH="$ET_INSTALL/bin:$PATH"
```

Verify:

```bash
cd "$MODEL_PORT_REPO"
riscv64-unknown-elf-gcc --version | head -1
.github/ci/scripts/resolve_et_platform_paths.sh
```

## Download Hugging Face References

Download the pinned Hugging Face base models used for model inspection and
host-side comparison. These are external references, not generated ET-SoC1 ELFs.
Generated outputs stay under `local-artifacts/` and are ignored by git.

```bash
cd "$MODEL_PORT_REPO"
scripts/download_hf_refs.sh "$ARTIFACT_ROOT/hf_refs"
```

See [`HF_REFERENCES.md`](HF_REFERENCES.md) for the exact repos, revisions, and
files. New submissions should pin their own Hugging Face base model the same
way.

## Build The Argbuf Launcher

The model-port runs use `erbium_soc1sim_argbuf_dynmem`. Build the launcher from
the checked-in source under `.github/ci/launcher`. Do not build it from
`local-artifacts/`; that tree contains generated benchmark outputs and will not
exist in a fresh clone.

```bash
cd "$MODEL_PORT_REPO"
export ET_PLATFORM="$ET_INSTALL"

.github/ci/scripts/build_launcher.sh

export LOCAL_ARGBUF="$BUILD_ROOT/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem"
test -x "$LOCAL_ARGBUF"
"$LOCAL_ARGBUF" --help 2>&1 | head -3
```

## Build Smoke Artifacts

`$BENCHMARK_ARTIFACT_ROOT` is created by the benchmark build flow. Build the
canonical smoke ELFs before running sys-emu or staging files to the board:

```bash
cd "$MODEL_PORT_REPO"

.github/ci/scripts/prepare_benchmark_inputs.sh all
.github/ci/scripts/build_leaderboard_elf.sh yolo

find "$BENCHMARK_ARTIFACT_ROOT" -maxdepth 2 -type f \
  \( -name '*.elf' -o -name '*_variants.txt' -o -name 'zero2m.bin' \) \
  | sort
```

## Local Sys-Emu Run

Use sys-emu before taking board time. It catches missing artifacts and loader
issues. The script writes logs and dumps under `/tmp` by default.

```bash
cd "$MODEL_PORT_REPO"

scripts/run_sysemu_model_ports.sh \
  --launcher "$LOCAL_ARGBUF" \
  --suite smoke \
  --model yolo \
  --variant yolo_m30 \
  --list
```

Run the canonical YOLO smoke job:

```bash
scripts/run_sysemu_model_ports.sh \
  --launcher "$LOCAL_ARGBUF" \
  --suite smoke \
  --model yolo \
  --variant yolo_m30 \
  --timeout 1800 \
  --launcher-timeout 1800
```

Larger suites need their matching variant ELFs and manifests under
`$BENCHMARK_ARTIFACT_ROOT`. Check what is ready before running them:

```bash
scripts/run_sysemu_model_ports.sh --launcher "$LOCAL_ARGBUF" --suite focused20 --model yolo --list
scripts/run_sysemu_model_ports.sh --launcher "$LOCAL_ARGBUF" --suite focused20b --model yolo --list
scripts/run_sysemu_model_ports.sh --launcher "$LOCAL_ARGBUF" --suite full --model yolo --list
```

Sys-emu can be much slower than the board for the larger hand-C kernels.
`timeout` means the emulator did not complete inside your wall-clock budget;
it does not automatically mean the kernel is broken.

## Board Connectivity

Check SSH and create the remote work directory:

```bash
ssh "$BOARD_SSH" 'hostname'
ssh "$BOARD_SSH" "mkdir -p '$REMOTE_ROOT' '$REMOTE_MODELS'"
```

Stage the launcher and runtime libraries:

```bash
rsync -aq "$LOCAL_ARGBUF" "$BOARD_SSH:$REMOTE_MODELS/erbium_soc1sim_argbuf"

# These are the common runtime dependencies for the argbuf launcher.
rsync -aq "$ET_INSTALL/lib"/libetrt.so* "$BOARD_SSH:$REMOTE_ROOT/" || true
rsync -aq "$ET_INSTALL/lib"/libg3log.so* "$BOARD_SSH:$REMOTE_ROOT/" || true

ssh "$BOARD_SSH" "
  set -e
  export LD_LIBRARY_PATH='$REMOTE_ROOT:'\"\${LD_LIBRARY_PATH:-}\"
  '$REMOTE_MODELS/erbium_soc1sim_argbuf' --help | head
"
```

## Stage A Smoke Suite

These examples assume the smoke artifacts above exist under
`$BENCHMARK_ARTIFACT_ROOT`.

Common zero blob used by ELF smoke kernels:

```bash
ssh "$BOARD_SSH" "mkdir -p '$REMOTE_MODELS'"
rsync -aq \
  "$BENCHMARK_ARTIFACT_ROOT/zero2m.bin" \
  "$BOARD_SSH:$REMOTE_MODELS/"
```

YOLO example:

```bash
ssh "$BOARD_SSH" "mkdir -p '$REMOTE_MODELS/yolo-bench' '$REMOTE_MODELS/yolo'"
rsync -aq \
  "$BENCHMARK_ARTIFACT_ROOT/yolo-bench/yolo_m30.elf" \
  "$BENCHMARK_ARTIFACT_ROOT/yolo-bench/yolo_variants.txt" \
  "$BOARD_SSH:$REMOTE_MODELS/yolo-bench/"
rsync -aq \
  "$BENCHMARK_ARTIFACT_ROOT/yolo/weights_region.bin" \
  "$BENCHMARK_ARTIFACT_ROOT/yolo/"*_raw_480x640x3_uint8_rgb.bin \
  "$BOARD_SSH:$REMOTE_MODELS/yolo/"
```

## Run One Kernel On The Board

```bash
ssh "$BOARD_SSH" "
set -euo pipefail
export LD_LIBRARY_PATH='$REMOTE_ROOT:$REMOTE_MODELS:'\"\${LD_LIBRARY_PATH:-}\"
cd '$REMOTE_MODELS/yolo-bench'
STAMP=\$(date +%Y%m%d-%H%M%S)
'$REMOTE_MODELS/erbium_soc1sim_argbuf' \
    --elf-load ./yolo_m30.elf \
    --shire 0 \
    --file_load 0x0,../zero2m.bin \
    --file_load 0x02000000,../yolo/weights_region.bin \
    --file_load 0x04A00000,../yolo/coco_room_000139_raw_480x640x3_uint8_rgb.bin \
    --dump_after dump_yolo_m30_\$STAMP.bin \
    --timeout 240 \
  2>&1 | tee run_yolo_m30_\$STAMP.log
"
```

Check results:

```bash
ssh "$BOARD_SSH" "grep -R 'Kernel wait seconds' '$REMOTE_MODELS'/*-bench/run_*.log 2>/dev/null || true"
```

Pull logs and dumps:

```bash
mkdir -p "$BUILD_ROOT/board-results"
rsync -aq "$BOARD_SSH:$REMOTE_MODELS/yolo-bench/"'run_*.log' "$BUILD_ROOT/board-results/" || true
rsync -aq "$BOARD_SSH:$REMOTE_MODELS/yolo-bench/"'dump_*.bin' "$BUILD_ROOT/board-results/" || true
```

## Run A Smoke Manifest On The Board

For a staged suite, run each variant from its manifest:

```bash
ssh "$BOARD_SSH" "
set -euo pipefail
export LD_LIBRARY_PATH='$REMOTE_ROOT:$REMOTE_MODELS:'\"\${LD_LIBRARY_PATH:-}\"
cd '$REMOTE_MODELS/yolo-bench'
STAMP=\$(date +%Y%m%d-%H%M%S)
while read -r variant; do
  [ -n \"\$variant\" ] || continue
  echo \"=== \$variant ===\"
  '$REMOTE_MODELS/erbium_soc1sim_argbuf' \
      --elf-load ./\"\$variant\".elf \
      --shire 0 \
      --file_load 0x0,../zero2m.bin \
      --file_load 0x02000000,../yolo/weights_region.bin \
      --file_load 0x04A00000,../yolo/coco_room_000139_raw_480x640x3_uint8_rgb.bin \
      --dump_after dump_\"\$variant\"_\"\$STAMP\".bin \
      --timeout 240 \
    > run_\"\$variant\"_\"\$STAMP\".log 2>&1
  grep 'Kernel wait seconds' run_\"\$variant\"_\"\$STAMP\".log || true
done < yolo_variants.txt
"
```

## Build A Kernel With Portable Paths

Some historical sweep scripts in artifact directories still contain old
absolute paths. For a fresh install, either parameterize those scripts before
using them or build with explicit variables like this:

```bash
GCC="$ET_INSTALL/bin/riscv64-unknown-elf-gcc"
INC="$ET_PLATFORM_SRC"
ROOT="$BENCHMARK_ARTIFACT_ROOT"

"$GCC" \
  -O2 -nostdlib \
  -march=rv64imfc -mabi=lp64f -mcmodel=medany \
  -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections \
  -I"$INC/et-common-libs/build-headers/erbium-soc1sim-staged-include" \
  -I"$INC/hal/platform/erbium/include" \
  -I"$INC/hal/platform/etsoc/include" \
  -I"$INC/et-common-libs/include" \
  -Wl,--gc-sections -Wl,--no-warn-rwx-segments \
  -Wl,--defsym=region0_size=0x04000000 \
  -T "$INC/et-common-libs/share/erbium-soc1sim/erbium.ld" \
  -o "$BUILD_ROOT/my_kernel.elf" \
  my_kernel.c \
  "$ROOT/hart-report/hart_report_crt.S" \
  "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
```

For embedded binary blobs:

```bash
OBJCOPY="$ET_INSTALL/bin/riscv64-unknown-elf-objcopy"
"$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
  --rename-section .data=.rodata,alloc,load,readonly,data,contents \
  --set-section-alignment .rodata=64 \
  weights.bin "$BUILD_ROOT/weights.o"
```

## Common Failure Modes

- SSH fails: fix `BOARD_HOST`, `BOARD_USER`, or keys.
- `erbium_soc1sim_argbuf not found`: stage the launcher under `REMOTE_MODELS`.
- Missing `libetrt` or `libg3log`: stage runtime libraries under
  `REMOTE_ROOT` and set `LD_LIBRARY_PATH`.
- No `Kernel wait seconds`: the kernel timed out, crashed, or the launcher
  was killed before completion.
- Correct log but stale `dump.bin`: the kernel likely missed
  `evict + WAIT_CACHEOPS + FENCE` before exit.
- Wrong output only under multi-hart: audit cache ownership and barriers;
  L1D is not coherent across minions.
- `tensor_load` reads wrong data: check 64-byte alignment. It rounds
  addresses.
