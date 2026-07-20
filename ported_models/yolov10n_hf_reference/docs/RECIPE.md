# Pinned-ONNX YOLOv10n reference-port recipe

This recipe reproduces the correctness-first path from a clean checkout: verify
the pinned Hugging Face ONNX, inventory its real graph, package its
initializers, capture ONNX Runtime intermediates, execute all 308 nodes through
the scalar C runtime, cross-compile either the full graph or bounded contiguous
ranges with the supported ET toolchain, and validate the host, system
emulator, and real ET-SoC1 evidence independently.

This is a complete correctness implementation but deliberately not a
leaderboard entry. The scalar runtime implements all 22 operator types and all
forms actually used by the pinned graph. It still rejects any unimplemented
attribute, type, shape, broadcast, or index contract explicitly. There is no
VPU, TFMA, fusion, tiling, threading, or fast-math path here.

Current evidence is asymmetric and must be reported that way:

- complete deterministic and real-image host inference pass;
- all 21 gap-free host ranges pass every node output;
- complete 21-range system-emulator execution is in progress;
- the complete real-image graph passes on real ET-SoC1 with all 16
  checkpoints, strict final output, result integrity, and seven stage PMCs.

Hardware and emulator evidence remain distinct. The hardware gate now passes,
but this scalar correctness port is intentionally not registered as a
leaderboard performance entry.

The model is:

- repository: `onnx-community/yolov10n`
- revision: `57657320425ee34056408a57ad9d29c4d4815bd8`
- file: `onnx/model.onnx`
- SHA-256:
  `a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b`
- upstream model-card license: AGPL-3.0

The pinned ONNX is the source of topology, weights, names, shapes, attributes,
and reference values. None of these commands exports from PyTorch.

## 1. Set checkout-local paths

Run all host commands in Bash from the repository checkout:

```bash
cd "$(git rev-parse --show-toplevel)"

export REPO_ROOT="$PWD"
export PORT_REL="ported_models/yolov10n_hf_reference"
export PORT_ROOT="$REPO_ROOT/$PORT_REL"
export CACHE_ROOT="$REPO_ROOT/local-artifacts/yolov10n_hf_reference"
export YOLOV10N_HOST_VENV="$CACHE_ROOT/venv"
export YOLOV10N_HOST_PYTHON="$YOLOV10N_HOST_VENV/bin/python"
export PY="$YOLOV10N_HOST_PYTHON"
```

`local-artifacts/` is ignored by Git. It holds the virtual environment,
downloaded ONNX, packed weight blob, instrumented ONNX copies, captured
activations, slice blobs, ELFs, dumps, logs, and validation results. Do not add
those large or machine-produced files to Git.

Treat existing package/result directories as immutable evidence. The launch
scripts refuse to reuse result artifacts, but generation tools can write a
named package again. Before regenerating an already validated name, choose a
new `--name` suffix and matching variables instead. Never delete an earlier
host, emulator, board, PMC, or failed-attempt directory to make a command
succeed.

The small, reviewable outputs under `manifests/` and the generated
`docs/ARCHITECTURE.md` are committed. Regenerating them from the pin should
leave no diff.

## 2. Create the pinned host environment

The setup script creates an isolated virtual environment and installs the
versions in `requirements-host.txt`:

```bash
bash "$PORT_ROOT/tools/setup_host_env.sh"
```

Expected terminal marker:

```text
HOST_ENV PASS numpy=1.24.4 onnx=1.16.2 onnxruntime=1.16.3
```

Python 3.8 is tested; use a Python release supported by every pinned wheel in
`requirements-host.txt` (normally 3.8 through 3.11). Package installation and
the first model download require network access.

## 3. Download and verify the exact ONNX

Download atomically into the ignored cache:

```bash
"$PY" "$PORT_ROOT/tools/download_model.py"
```

Then prove that the cached file is sufficient without network access:

```bash
"$PY" "$PORT_ROOT/tools/download_model.py" --verify-only
```

Both commands must end with this identity:

```text
CHECKSUM PASS ... bytes=9386116 expected_bytes=9386116 expected=a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b actual=a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b
```

Every later generation tool hashes the model again. A file at the expected
path is never accepted merely because it exists.

## 4. Inventory, package, and explain the real graph

Generate the stable `N000` through `N307` node IDs, `L000` through `L307`
layer IDs, per-operator IDs, shapes, attributes, initializer inventory, and
operator counts:

```bash
"$PY" "$PORT_ROOT/tools/inspect_onnx.py"
```

The required gate is:

```text
GRAPH nodes=308 initializers=187 opsets={'ai.onnx': 13} ...
CHECK PASS name=sha256 ...
CHECK PASS name=opset_ai_onnx ...
CHECK PASS name=input ...
CHECK PASS name=output ...
CHECK PASS name=node_count ...
CHECK PASS name=terminal_topk ...
CHECK PASS name=terminal_gather_elements ...
GRAPH_CHECK PASS
```

The two readable products are:

- `manifests/graph_inventory.json`: complete structured graph inventory.
- `manifests/layers.tsv`: compact node/name/operator/tensor listing, useful for
  choosing a range.

Package all 187 original ONNX initializers as aligned, headerless,
little-endian bytes with a declarative JSON manifest:

```bash
"$PY" "$PORT_ROOT/tools/pack_initializers.py"
```

Expected summary:

```text
MODEL_CHECK PASS nodes=308 initializers=187 ...
INITIALIZERS PASS count=187 dtypes={'FLOAT': 169, 'INT64': 18} data_bytes=9298228 padding_bytes=908
WEIGHTS PASS ... bytes=9299136 sha256=b6d5aa13ef3238328c19ff0f646f72841bcf44dc1651383d089f0d57e37cf850
MANIFEST PASS ... entries=187
PACK PASS
```

`manifests/weights_manifest.json` records every tensor's original name, graph
index, type, shape, offset, size, storage form, and SHA-256. The corresponding
`package/weights.bin` remains ignored.

Generate the architecture guide and its exhaustive 16-stage node map from the
inventory and ONNX constants:

```bash
"$PY" "$PORT_ROOT/tools/generate_architecture.py"
```

Expected marker:

```text
ARCHITECTURE PASS stages=16 nodes=308 ...
```

The generator also verifies graph landmarks, absence of
`NonMaxSuppression`, DFL weights exactly equal to `[0,1,...,15]`, 80 classes,
top-k equal to 300, and the 8/16/32 stride vector. Read
`docs/ARCHITECTURE.md` for the measured preprocessing boundary, backbone,
C2f-style blocks, SPPF, attention, neck, detection branches, DFL decode, and
NMS-free TopK/GatherElements path.

From a clean checkout, these reproducible products should not change:

```bash
git diff --exit-code -- \
  "$PORT_REL/manifests/graph_inventory.json" \
  "$PORT_REL/manifests/layers.tsv" \
  "$PORT_REL/manifests/weights_manifest.json" \
  "$PORT_REL/manifests/architecture_stages.json" \
  "$PORT_REL/docs/ARCHITECTURE.md"
```

## 5. Generate and validate the complete host graph

Generate the schema-v2 package directly from the checked ONNX and aligned
initializer package:

```bash
export DETERMINISTIC_FULL="$CACHE_ROOT/full_graph/deterministic_full308_v3"

"$PY" "$PORT_ROOT/tools/generate_full_graph.py" \
  --name deterministic_full308_v3
"$PORT_ROOT/scripts/verify_full_package.sh" \
  "$DETERMINISTIC_FULL" \
  "$CACHE_ROOT/model.onnx"
```

`full308_v3` is the preserved package named by this validation record. From a
checkout where that directory already exists, choose a new suffix rather than
overwriting it, and pass the matching `--execution-manifest` path if the run is
not intended to refresh the committed deterministic manifest.

Expected high-level markers:

```text
FULL_GRAPH PASS nodes=308 tensors=512 ops=22 checkpoints=16
MEMORY PASS arena=35788800 dump=0x22a0000 input=0x22a0000 weights=0x2750000 total=0x3030000
BLOBS PASS inputs=4915200 weights=9299136 ...
FULL_PACKAGE PASS selector=N000:N307 nodes=308 ...
```

The generated directory contains the readable JSON manifest, small descriptor
header, exact input/weight/golden blobs, and an ignored instrumented ONNX used
only to expose the 16 reference checkpoints. The manifest declares:

- a deterministic 64-byte-aligned first-fit liveness arena;
- output allocation before last-use inputs are released;
- 16 pinned checkpoints through `N307`;
- seven PMC intervals covering `N000:N307` exactly;
- 35,788,800 workspace bytes and 50,528,256 total target bytes.

Run all 308 nodes through the scalar host C runner:

```bash
"$PORT_ROOT/scripts/run_host_full.sh" \
  "$DETERMINISTIC_FULL"
```

Required end markers include:

```text
HOST_FULL PASS nodes=N000:N307 status=0 ...
YRF1 PASS status=ok nodes=N000:N307 ...
SELECTION_TAIL PASS mode=proven_topk_tie_discontinuity ...
FINAL_OUTPUT PASS ... unexplained_mismatches=0 ...
FULL_COMPARE PASS selector=N000:N307 checkpoints=16 runtime_status=ok
HOST_FULL_RUN PASS ...
```

The deterministic input has exact TopK cutoff ties. Its 1,091 direct
positional field mismatches are expected to remain visible in
`host_full_compare.json`; passing requires a tolerance-passing `N288`,
bit-exact independent replay of each C/ORT tail, an exact reference tie, and
zero unexplained mismatches. To prove the strict switch rejects this fixture:

```bash
set +e
"$PY" "$PORT_ROOT/tools/compare_full.py" \
  "$DETERMINISTIC_FULL" \
  "$DETERMINISTIC_FULL/host_full_dump.bin" \
  --model "$CACHE_ROOT/model.onnx" \
  --require-direct-output
rc=$?
set -e
test "$rc" -eq 1
```

Do not weaken a TopK comparator or discard the positional mismatch. The real
image below is the strict direct-output gate.

## 6. Run the checked real-image fixture end to end

Create the exact FP32 input from the repository's raw COCO-room fixture:

```bash
"$PY" "$PORT_ROOT/tools/preprocess_coco_room.py"

export REAL_INPUT="$CACHE_ROOT/fixtures/coco_room_000139/input_fp32.bin"
export REAL_FULL="$CACHE_ROOT/full_graph/coco_room_000139_full308_v3"

"$PY" "$PORT_ROOT/tools/generate_full_graph.py" \
  --name coco_room_000139_full308_v3 \
  --input-bin "$REAL_INPUT" \
  --execution-manifest "$REAL_FULL/full_execution.json"
```

Expected preprocessing identity:

```text
PREPROCESS PASS fixture=coco_room_000139 bytes=4915200 sha256=65afc38f381c09712cd9a6a78e8e7e9800c713d21679abee588b218a6a137c7b ...
```

Use a dedicated execution-manifest path for this input so the committed
deterministic `manifests/full_execution.json` remains the reproducible default.
Build, execute, and enforce direct positional output:

```bash
"$PORT_ROOT/scripts/build_host_full.sh" "$REAL_FULL"
"$REAL_FULL/host_full_runner" \
  "$REAL_FULL/inputs.bin" \
  "$REAL_FULL/weights.bin" \
  "$REAL_FULL/host_full_dump.bin"

"$PY" "$PORT_ROOT/tools/compare_full.py" \
  "$REAL_FULL" \
  "$REAL_FULL/host_full_dump.bin" \
  --model "$CACHE_ROOT/model.onnx" \
  --require-direct-output \
  --json "$REAL_FULL/host_full_compare_strict.json"
```

Required final evidence:

```text
SELECTION_TAIL PASS mode=direct_ort_tolerance ... anchor_overlap=300/300 pair_overlap=300/300 ...
FINAL_OUTPUT PASS ... direct_mismatches=0/1800 unexplained_mismatches=0 ...
FULL_COMPARE PASS selector=N000:N307 checkpoints=16 runtime_status=ok
```

The measured maximum absolute error is `0.00042724609375`. `output0` is 300
FP32 rows of `[x1,y1,x2,y2,score,class_id]` in the padded 640×640 input
canvas. `class_id` is integral-valued FP32 because of graph node `N306` Cast.
The graph applies no score threshold or IoU suppression. For this fixture
only, subtract 80 from y coordinates and clip to the original 480-pixel
height; x is unchanged.

## 7. Capture and validate bounded schema-v2 ranges

Generate the checked, gap-free execution plan:

```bash
"$PY" "$PORT_ROOT/tools/plan_sys_emu_coverage.py"
```

Expected marker:

```text
SYS_EMU_PLAN PASS ranges=21 nodes=308 gaps=0 overlaps=0 ...
```

The exact selectors are:

```bash
RANGES=(
  N000:N002 N003:N009 N010:N017 N018:N030 N031:N045
  N046:N060 N061:N079 N080:N104 N105:N137 N138:N153
  N154:N168 N169:N196 N197:N207 N208:N210 N211:N213
  N214:N223 N224:N231 N232:N249 N250:N270 N271:N288
  N289:N307
)
```

Capture and host-validate every range. Each boundary activation comes from
ORT execution of the exact full artifact, and every output of every selected
node remains materialized:

```bash
for selector in "${RANGES[@]}"; do
  first="${selector%%:*}"
  last="${selector##*:}"
  name="${first,,}_${last,,}"
  range_dir="$CACHE_ROOT/ranges_v2/$name"
  "$PY" "$PORT_ROOT/tools/capture_range_v2.py" \
    --range "$selector" \
    --name "$name"
  "$PORT_ROOT/scripts/run_host_range_v2.sh" "$range_dir"
done
```

The measured aggregate is 21/21 ranges, 324/324 outputs, 75,592,700 elements,
and zero mismatches. The ranges cover every node exactly once. The largest
range-level max absolute error is `0.0001220703125` in `N271:N288`;
`N289:N307` is exact across 795,900 FLOAT/INT64 elements.

For one-off investigation, a single selector is sufficient:

```bash
"$PY" "$PORT_ROOT/tools/capture_range_v2.py" \
  --range N271:N288 \
  --name n271_n288
"$PORT_ROOT/scripts/run_host_range_v2.sh" \
  "$CACHE_ROOT/ranges_v2/n271_n288"
```

Unlike the liveness-based full package, a range package retains every selected
output as a checkpoint. The plan targets at most 24,000,000 retained bytes and
250 million Conv/MatMul MACs. `N271:N288` is the explicit architecture-aligned
exception at 30,105,600 retained bytes; the measured maximum range workload is
249,958,400 MACs.

## 8. Current target status and evidence rule

The same schema-v2 manifests build with `build_et_slice.sh`; the complete
package has stricter wrappers in `build_et_full.sh`, `run_et_full.sh`, and
`validate_et_full.sh`. Exact target commands follow after the board-host and
toolchain setup below.

At the time of this record:

- `N289:N307` passes in the system emulator with all 22 outputs exact and PMC
  PASS;
- the remaining 20 canonical emulator ranges are in progress;
- udev restored both ET character devices without unloading the driver; the
  complete real-image ET-SoC1 run then passed strict validation and all seven
  stage PMCs.

Keep each attempt in a new output directory. Preserve partial logs and failed
dumps; never delete or overwrite earlier evidence. A plan, an ELF, or a
launcher-level `run_result.json` is not a numerical PASS until the appropriate
validator completes.

## 9. Preserved schema-v1 slice workflow

Print the currently implemented C operator set:

```bash
"$PY" "$PORT_ROOT/tools/capture_slice.py" --list-support
```

Expected:

```text
SUPPORTED_OPS Conv Sigmoid Mul Concat
```

The output above describes the intentionally frozen legacy schema-v1 capture
path, not the complete schema-v2 runtime. `--nodes` accepts one node (`N263`)
or an inclusive ascending range
(`N263:N265`). Use `manifests/layers.tsv` to select graph-order ranges.
All runtime tensors must be static FP32 with rank at most six. `Conv` is
explicit rank-4 NCHW with initializer weights/bias, 2-D
kernel/stride/dilation/pads, and validated group/channel shapes. `Mul` accepts
equal shapes or one scalar. `Concat` accepts one to four equal-rank inputs with
matching non-axis dimensions. The capture gate reports a specific reason when
a node falls outside those constraints.

The smallest recommended later-stage sample is `N263:N265`. It is not a toy
kernel: it is the P5 one-to-one classification branch's real 20×20 depthwise
3×3 Conv, followed by the ONNX's separate Sigmoid and Mul nodes that implement
SiLU. Its input activation is captured from execution of the exact pinned
graph, and its weights and bias come directly from that graph.

```bash
export SLICE="n263_n265_dw_silu"
export SLICE_DIR="$CACHE_ROOT/slices/$SLICE"

"$PY" "$PORT_ROOT/tools/capture_slice.py" \
  --nodes N263:N265 \
  --name "$SLICE"
```

Expected markers and sizes:

```text
SOURCE_CHECK PASS sha256=a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b
SLICE_CAPTURE PASS nodes=N263:N265 ops=Conv,Sigmoid,Mul
BLOBS inputs=128000 weights=3200 goldens=384000 ...
MEMORY mem_size=0x400000 dump_size=0x70000
```

The capture tool runs ONNX Runtime with graph optimization disabled and one
intra-op and inter-op thread. It adds intermediate tensors as outputs to an
ignored copy; it does not rewrite the selected operations. The full model
input uses the deterministic formula recorded in `slice_manifest.json`.
External boundary activations, original weights, and every selected node
output are stored in separate aligned blobs. `slice_manifest.h` is the small
declarative C mapping consumed by both runners.
The deterministic input is a reproducible validation fixture, not an image
preprocessing policy.

Useful additional samples are:

```bash
"$PY" "$PORT_ROOT/tools/capture_slice.py" \
  --nodes N000:N002 \
  --name n000_n002_stem_conv_silu

"$PY" "$PORT_ROOT/tools/capture_slice.py" \
  --nodes N266:N268 \
  --name n266_n268_pw_silu

"$PY" "$PORT_ROOT/tools/capture_slice.py" \
  --nodes N263:N270 \
  --name n263_n270_p5_class_join
```

They exercise, respectively, the input stem
Conv/Sigmoid/Mul, a P5 1×1 pointwise Conv/Sigmoid/Mul, and the useful
small-spatial P5 classification join including another Conv and `Concat`.
The stem sample has much larger blobs and emulator cost.

To run another sample through later sections, update both variables first:

```bash
export SLICE="n266_n268_pw_silu"
export SLICE_DIR="$CACHE_ROOT/slices/$SLICE"
```

Unsupported graph work is reported rather than approximated. For example:

```bash
set +e
"$PY" "$PORT_ROOT/tools/capture_slice.py" \
  --nodes N278 \
  --name unsupported_softmax_probe
rc=$?
set -e
test "$rc" -eq 2
```

Expected diagnostic:

```text
UNSUPPORTED node=N278 op=Softmax
```

No slice directory is a full YOLOv10n inference claim.

## 10. Compile and compare the preserved schema-v1 host runner

Build, run, and compare the selected slice in one command:

```bash
"$PORT_ROOT/scripts/run_host_slice.sh" "$SLICE" "$SLICE_DIR"
```

The comparator checks blob hashes, the versioned result header, selected node
IDs, workspace FNV-1a, finite values, every output element, and:

```text
abs(actual - reference) <= 5e-5 + 1e-4 * abs(reference)
```

For `N263:N265`, the known-good host result has zero mismatches in each 32,000
element output:

```text
HOST_BUILD PASS ...
HOST_SLICE PASS nodes=N263:N265 status=0 ...
TENSOR PASS node=N263 ... max_abs=4.76837158e-07 ... mismatches=0/32000 ...
TENSOR PASS node=N264 ... max_abs=1.78813934e-07 ... mismatches=0/32000 ...
TENSOR PASS node=N265 ... max_abs=5.96046448e-07 ... mismatches=0/32000 ...
SLICE_COMPARE PASS nodes=N263:N265 runtime_status=ok
```

The JSON report is written to `host_compare.json` in the slice directory. To
build or compare separately:

```bash
"$PORT_ROOT/scripts/build_host_slice.sh" \
  "$SLICE_DIR" "$SLICE_DIR/host_slice_runner"

"$SLICE_DIR/host_slice_runner" \
  "$SLICE_DIR/inputs.bin" \
  "$SLICE_DIR/weights.bin" \
  "$SLICE_DIR/host_dump.bin"

"$PY" "$PORT_ROOT/tools/compare_slice.py" \
  "$SLICE_DIR" "$SLICE_DIR/host_dump.bin" \
  --json "$SLICE_DIR/host_compare.json"
```

Do not weaken tolerances merely to make a mismatch pass. The report gives max
absolute error, max relative error, mean absolute error, mismatch count,
non-finite count, and the worst element for diagnosis. Relative error can be
large near an exact zero while the explicit combined tolerance still passes.

## 11. Deploy source and ignored packages to an ET execution host

The remaining commands assume the repository-supported `board-host` SSH setup
has been installed. They use `soc3-ssh.sh` so OpenSSH and Tailscale selection
matches the repository deployment flow.

Choose a unique remote root. The slim deploy intentionally excludes
`local-artifacts`, so copy the captured slice separately:

```bash
export SOC3_HOST="${SOC3_HOST:-root@board-host}"
# These defaults match the documented root board-host account. Override
# REMOTE_HOME when SOC3_HOST names a different remote user.
export REMOTE_HOME="${REMOTE_HOME:-/root}"
# Board-host example only; choose a fresh absolute directory on your ET host.
export REMOTE_ROOT="${REMOTE_ROOT:-$REMOTE_HOME/yolov10n-hf-reference-validation}"
export REMOTE_CACHE_ROOT="$REMOTE_ROOT/local-artifacts/yolov10n_hf_reference"

source "$REPO_ROOT/.github/ci/platform/deploy/soc3-ssh.sh"
read -r -a SSH_CMD <<<"$(soc3_ssh_cmd)"
export RSYNC_HOST
RSYNC_HOST="$(soc3_rsync_host)"
export RSYNC_RSH
RSYNC_RSH="$(soc3_rsync_rsh)"

"${SSH_CMD[@]}" \
  "mkdir -p '$REMOTE_ROOT'"
"$REPO_ROOT/.github/ci/platform/deploy/rsync-slim.sh" \
  "$SOC3_HOST" "$REMOTE_ROOT"
```

For a preserved schema-v1 slice, copy its package:

```bash
export REMOTE_SLICE_DIR="$REMOTE_CACHE_ROOT/slices/$SLICE"
"${SSH_CMD[@]}" \
  "mkdir -p '$REMOTE_SLICE_DIR'"
rsync -az -e "$RSYNC_RSH" \
  "$SLICE_DIR/slice_manifest.h" \
  "$SLICE_DIR/slice_manifest.json" \
  "$SLICE_DIR/inputs.bin" \
  "$SLICE_DIR/weights.bin" \
  "$RSYNC_HOST:$REMOTE_SLICE_DIR/"
```

For schema-v2 work, copy the selected full or range directory instead.
`build_et_full.sh` verifies the complete full package, including its golden
blob, before compiling; keep that blob on the build host even though the
launcher itself loads only inputs and weights:

```bash
export REMOTE_FULL_DIR="$REMOTE_CACHE_ROOT/full_graph/coco_room_000139_full308_v3"
"${SSH_CMD[@]}" "mkdir -p '$REMOTE_FULL_DIR'"
rsync -az -e "$RSYNC_RSH" \
  "$REAL_FULL/slice_manifest.h" \
  "$REAL_FULL/slice_manifest.json" \
  "$REAL_FULL/inputs.bin" \
  "$REAL_FULL/weights.bin" \
  "$REAL_FULL/goldens.bin" \
  "$RSYNC_HOST:$REMOTE_FULL_DIR/"

export RANGE_NAME="n271_n288"
export RANGE_DIR="$CACHE_ROOT/ranges_v2/$RANGE_NAME"
export REMOTE_RANGE_DIR="$REMOTE_CACHE_ROOT/ranges_v2/$RANGE_NAME"
"${SSH_CMD[@]}" "mkdir -p '$REMOTE_RANGE_DIR'"
rsync -az -e "$RSYNC_RSH" \
  "$RANGE_DIR/slice_manifest.h" \
  "$RANGE_DIR/slice_manifest.json" \
  "$RANGE_DIR/inputs.bin" \
  "$RANGE_DIR/weights.bin" \
  "$RSYNC_HOST:$REMOTE_RANGE_DIR/"
```

Keep each range's local `goldens.bin`, manifest, and pinned ONNX for post-run
comparison. Copy a range golden to the execution host too if validation will
run there.

The remote root is disposable validation state, but choose a new root or run
tag instead of overwriting another user's checkout or result directory.

### Known board-host path example

The following values are **examples from the repository's ET board host**, not
paths to commit into manifests. Override them for another supported
installation:

```bash
export REMOTE_ET_REAL_ROOT="/opt/et"
export REMOTE_ET_PLATFORM_SRC="$REMOTE_HOME/et-platform"
export REMOTE_SYS_EMU_SDK="$REMOTE_HOME/et-jobs-deploy/.ci-work/et"
export REMOTE_LAUNCHER="/opt/et/bin/erbium_soc1sim_argbuf_dynmem"
export REMOTE_LD_LIBRARY_PATH="$REMOTE_HOME/afonso/demo/host:/opt/et/host:/opt/et/lib"
export REMOTE_DOCKER_IMAGE="et-gcc:24.04"
export REMOTE_BOARD_LOCK="/var/lock/etsoc-shire0.lock"
```

Confirm the paths before building:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ET_REAL_ROOT" \
  "$REMOTE_ET_PLATFORM_SRC" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" <<'REMOTE'
set -euo pipefail
test -x "$1/bin/riscv64-unknown-elf-gcc"
test -f "$2/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld"
test -x "$3/bin/sys_emu"
test -x "$4"
echo "ET_PATHS PASS"
REMOTE
```

## 12. Cross-compile with the supported ET Docker toolchain

Do not substitute a generic RISC-V compiler. The native ET compiler can fail
on an older host GLIBC; `scripts/et_gcc_docker_wrapper.sh` mirrors the
repository's Ubuntu 24.04 board-build workflow and invokes the real ET
compiler from a read-only mount inside `et-gcc:24.04`.

Build the selected slice on the ET execution host:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$SLICE" \
  "$REMOTE_ET_REAL_ROOT" \
  "$REMOTE_ET_PLATFORM_SRC" \
  "$REMOTE_DOCKER_IMAGE" <<'REMOTE'
set -euo pipefail
root=$1
slice=$2
real_et=$3
platform_src=$4
image=$5
port="$root/ported_models/yolov10n_hf_reference"
slice_dir="$root/local-artifacts/yolov10n_hf_reference/slices/$slice"
elf="$slice_dir/yolov10n_hf_slice.elf"

cd "$root"
export ET_INSTALL="$real_et"
export ET_PLATFORM_SRC="$platform_src"
export ET_DOCKER_WORKSPACE="$root"
export ET_DOCKER_REAL_ET_ROOT="$real_et"
export ET_DOCKER_PLATFORM_SRC="$platform_src"
export ET_DOCKER_IMAGE="$image"
export ET_GCC="$port/scripts/et_gcc_docker_wrapper.sh"

"$port/scripts/build_et_slice.sh" "$slice_dir" "$elf"
REMOTE
```

Expected terminal marker:

```text
ET_BUILD PASS compiler=.../et_gcc_docker_wrapper.sh output=.../yolov10n_hf_slice.elf record=.../yolov10n_hf_slice.elf.build.json
```

Build the complete real-image package with its full-contract verifier:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$REMOTE_ET_REAL_ROOT" \
  "$REMOTE_ET_PLATFORM_SRC" \
  "$REMOTE_DOCKER_IMAGE" <<'REMOTE'
set -euo pipefail
root=$1
real_et=$2
platform_src=$3
image=$4
port="$root/ported_models/yolov10n_hf_reference"
full_dir="$root/local-artifacts/yolov10n_hf_reference/full_graph/coco_room_000139_full308_v3"
elf="$full_dir/yolov10n_hf_full.elf"

cd "$root"
export ET_INSTALL="$real_et"
export ET_PLATFORM="$platform_src"
export ET_PLATFORM_SRC="$platform_src"
export ET_DOCKER_WORKSPACE="$root"
export ET_DOCKER_REAL_ET_ROOT="$real_et"
export ET_DOCKER_PLATFORM_SRC="$platform_src"
export ET_DOCKER_IMAGE="$image"
export ET_GCC="$port/scripts/et_gcc_docker_wrapper.sh"

"$port/scripts/build_et_full.sh" "$full_dir" "$elf"
REMOTE
```

Required final markers:

```text
FULL_PACKAGE PASS selector=N000:N307 nodes=308 ...
ET_BUILD PASS ...
ET_FULL_BUILD PASS selector=N000:N307 ...
```

Build a canonical schema-v2 range with the same toolchain:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$RANGE_NAME" \
  "$REMOTE_ET_REAL_ROOT" \
  "$REMOTE_ET_PLATFORM_SRC" \
  "$REMOTE_DOCKER_IMAGE" <<'REMOTE'
set -euo pipefail
root=$1
name=$2
real_et=$3
platform_src=$4
image=$5
port="$root/ported_models/yolov10n_hf_reference"
range_dir="$root/local-artifacts/yolov10n_hf_reference/ranges_v2/$name"

cd "$root"
export ET_INSTALL="$real_et"
export ET_PLATFORM="$platform_src"
export ET_PLATFORM_SRC="$platform_src"
export ET_DOCKER_WORKSPACE="$root"
export ET_DOCKER_REAL_ET_ROOT="$real_et"
export ET_DOCKER_PLATFORM_SRC="$platform_src"
export ET_DOCKER_IMAGE="$image"
export ET_GCC="$port/scripts/et_gcc_docker_wrapper.sh"

"$port/scripts/build_et_slice.sh" \
  "$range_dir" "$range_dir/yolov10n_hf_range.elf"
REMOTE
```

Repeat that build for each of the 21 names from the coverage plan. Preserve
the adjacent `.build.json` with every ELF. `verify_full_elf.sh` rejects a full
ELF if its runtime, runner, generated header, linker, layout, CRT, or
correctness flags no longer match its build record.

The build uses scalar FP32, `-O1`, `-fno-fast-math`,
`-ffp-contract=off`, `-fno-tree-vectorize`, the canonical ET linker/layout,
and the repository CRT. The runner activates only hart 0. The adjacent JSON
build record hashes the compiler inputs, linker, layout, CRT, runtime,
generated slice header, and ELF, and records the full compile command and
Docker image identity.

PMC support is enabled by default. If a minimal system-emulator firmware does
not implement shared shire-cache or memshire PMC syscalls, rebuild a separate
ELF with the limitation recorded explicitly. Add this export inside the
remote build block, immediately before `build_et_slice.sh`:

```bash
export YR_ET_EXTRA_CFLAGS="-DYR_PMC_SAMPLE_SC=0 -DYR_PMC_SAMPLE_MS=0"
```

This leaves the six per-hart HPM CSR samples enabled. Do not use that override
for a run presented as having shared-counter coverage.

## 13. Run and validate in the system emulator

Use a unique result name because the runner refuses to reuse evidence files:

```bash
export RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
```

### Schema-v2 bounded range

Set one of the canonical range names, then execute only that inclusive range:

```bash
export RANGE_NAME="n271_n288"
export RANGE_DIR="$CACHE_ROOT/ranges_v2/$RANGE_NAME"
export REMOTE_RANGE_DIR="$REMOTE_CACHE_ROOT/ranges_v2/$RANGE_NAME"
export RANGE_RESULT_ID="${RUN_TAG}_sys_emu_${RANGE_NAME}"
export REMOTE_RANGE_RUN_DIR="$REMOTE_CACHE_ROOT/results/full_coverage_sys_emu/$RANGE_RESULT_ID"

"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$RANGE_NAME" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" \
  "$REMOTE_LD_LIBRARY_PATH" \
  "$RANGE_RESULT_ID" <<'REMOTE'
set -euo pipefail
root=$1
name=$2
sdk=$3
launcher=$4
libs=$5
result_id=$6
port="$root/ported_models/yolov10n_hf_reference"
range_dir="$root/local-artifacts/yolov10n_hf_reference/ranges_v2/$name"
elf="$range_dir/yolov10n_hf_range.elf"
out="$root/local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu/$result_id"

cd "$root"
ET_PLATFORM="$sdk" \
LD_LIBRARY_PATH="$libs" \
"$port/scripts/run_et_slice.sh" \
  --device sys_emu \
  --slice-dir "$range_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$out" \
  --outer-timeout 7200 \
  --launcher-timeout 7140
REMOTE
```

Pull and validate every selected node output plus its range-only PMC:

```bash
export LOCAL_RANGE_RUN_DIR="$CACHE_ROOT/results/full_coverage_sys_emu/$RANGE_RESULT_ID"
mkdir -p "$LOCAL_RANGE_RUN_DIR"
rsync -az -e "$RSYNC_RSH" \
  "$RSYNC_HOST:$REMOTE_RANGE_RUN_DIR/" \
  "$LOCAL_RANGE_RUN_DIR/"

"$PORT_ROOT/scripts/validate_device_run.sh" \
  "$RANGE_DIR" "$LOCAL_RANGE_RUN_DIR" sys_emu
```

Required end markers:

```text
EXECUTION_EVIDENCE PASS device=sys_emu hardware=False nodes=N271:N288
NODE_OUTPUT PASS ...
RANGE_V2_COMPARE PASS selector=N271:N288 ...
DEVICE_VALIDATION PASS device=sys_emu ...
```

Repeat for the 21 exact names from section 7. A coverage claim requires all 21
validated result directories and an exact union of `N000:N307`; an ELF build,
launcher completion, or the plan JSON alone is insufficient. A failed range
gets a new attempt suffix, for example `_attempt2`; preserve the original
directory.

The validated `N289:N307` attempt demonstrates an important target-specific
check. An earlier ELF emitted RV64 `fcvt.s.l` for `N306` and reached a system
emulator trap. The current graph-bounded Cast range-checks the INT64 value and
converts through INT32, causing supported `fcvt.s.w`; all 22 outputs then
compare exactly. Keep both attempts as regression evidence.

### Optional monolithic full graph

If emulator capacity permits, the full package may be launched in one bounded
run:

```bash
export FULL_RESULT_ID="${RUN_TAG}_sys_emu_full_coco_room"
export REMOTE_FULL_RUN_DIR="$REMOTE_CACHE_ROOT/results/$FULL_RESULT_ID"

"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" \
  "$REMOTE_LD_LIBRARY_PATH" \
  "$FULL_RESULT_ID" <<'REMOTE'
set -euo pipefail
root=$1
sdk=$2
launcher=$3
libs=$4
result_id=$5
port="$root/ported_models/yolov10n_hf_reference"
full_dir="$root/local-artifacts/yolov10n_hf_reference/full_graph/coco_room_000139_full308_v3"
elf="$full_dir/yolov10n_hf_full.elf"
out="$root/local-artifacts/yolov10n_hf_reference/results/$result_id"

cd "$root"
ET_PLATFORM="$sdk" \
LD_LIBRARY_PATH="$libs" \
"$port/scripts/run_et_full.sh" \
  --device sys_emu \
  --full-dir "$full_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$out" \
  --outer-timeout 43200 \
  --launcher-timeout 43140
REMOTE
```

After pulling the directory, validate all 16 checkpoints, final output, and
seven stage PMCs:

```bash
export LOCAL_FULL_RUN_DIR="$CACHE_ROOT/results/$FULL_RESULT_ID"
mkdir -p "$LOCAL_FULL_RUN_DIR"
rsync -az -e "$RSYNC_RSH" \
  "$RSYNC_HOST:$REMOTE_FULL_RUN_DIR/" \
  "$LOCAL_FULL_RUN_DIR/"

"$PORT_ROOT/scripts/validate_et_full.sh" \
  "$REAL_FULL" "$LOCAL_FULL_RUN_DIR" sys_emu "$CACHE_ROOT/model.onnx" 1
```

This monolithic emulator result is optional when all 21 bounded ranges are
validated. It is not a hardware run.

### Preserved schema-v1 slice command

Set its unique result path:

```bash
export SYSEMU_RESULT_ID="${RUN_TAG}_sys_emu_${SLICE}"
export REMOTE_SYSEMU_RUN_DIR="$REMOTE_CACHE_ROOT/results/$SYSEMU_RESULT_ID"
```

Run the exact selected range with bounded timeouts:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$SLICE" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" \
  "$REMOTE_LD_LIBRARY_PATH" \
  "$SYSEMU_RESULT_ID" <<'REMOTE'
set -euo pipefail
root=$1
slice=$2
sdk=$3
launcher=$4
libs=$5
result_id=$6
port="$root/ported_models/yolov10n_hf_reference"
slice_dir="$root/local-artifacts/yolov10n_hf_reference/slices/$slice"
elf="$slice_dir/yolov10n_hf_slice.elf"
out="$root/local-artifacts/yolov10n_hf_reference/results/$result_id"

cd "$root"
ET_PLATFORM="$sdk" \
LD_LIBRARY_PATH="$libs" \
"$port/scripts/run_et_slice.sh" \
  --device sys_emu \
  --slice-dir "$slice_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$out" \
  --outer-timeout 900 \
  --launcher-timeout 840
REMOTE
```

The expected log contains all of:

```text
BLOB_CHECK PASS name=inputs ...
BLOB_CHECK PASS name=weights ...
erbium_soc1sim: elf=<resolved-elf> device=sys_emu shire=0
Kernel completed successfully
Dumped <manifest-dump-size> bytes to <result-dir>/dump.bin
DEVICE_RUN PASS device=sys_emu rc=0 identity_ok=1 completion_ok=1 dump_log_ok=1 reset_ok=1 dump_ok=1 ...
```

`device_evidence.txt` must say `backend=sys_emu`; `run_result.json` must say
`"device": "sys_emu"` and `"hardware": false`.

Pull the preserved result directory and validate it against the local ORT
goldens:

```bash
export LOCAL_SYSEMU_RUN_DIR="$CACHE_ROOT/results/$SYSEMU_RESULT_ID"
mkdir -p "$LOCAL_SYSEMU_RUN_DIR"
rsync -az -e "$RSYNC_RSH" \
  "$RSYNC_HOST:$REMOTE_SYSEMU_RUN_DIR/" \
  "$LOCAL_SYSEMU_RUN_DIR/"

"$PORT_ROOT/scripts/validate_device_run.sh" \
  "$SLICE_DIR" "$LOCAL_SYSEMU_RUN_DIR" sys_emu
```

Required end markers:

```text
EXECUTION_EVIDENCE PASS device=sys_emu hardware=False nodes=N263:N265
TENSOR PASS node=N263 ... mismatches=0/32000 ...
TENSOR PASS node=N264 ... mismatches=0/32000 ...
TENSOR PASS node=N265 ... mismatches=0/32000 ...
SLICE_COMPARE PASS nodes=N263:N265 runtime_status=ok
DEVICE_VALIDATION PASS device=sys_emu ...
```

The known-good `N263:N265` emulator run has the same max absolute errors as
the host run and zero mismatches. The `N266:N268` pointwise sample is also a
useful emulator expansion; its known-good maximum absolute errors are
`1.04904175e-05`, `1.87754631e-06`, and `5.24520874e-06`, with zero
mismatches for every node.

## 14. Run and validate on real ET-SoC1 hardware

`soc1sim` is the repository launcher's name for the real PCIe ET-SoC1 device;
it is **not** the software emulator. The hardware runner refuses to proceed
unless `/dev/et0_mgmt` and `/dev/et0_ops` are character devices and PCIe
evidence is present.

### Full-graph hardware gate

Check the hardware contract before allocating a result directory:

```bash
"${SSH_CMD[@]}" bash -s <<'REMOTE'
set -euo pipefail
test -c /dev/et0_mgmt
test -c /dev/et0_ops
stat -c 'device=%n type=%F major_minor=%t:%T' \
  /dev/et0_mgmt /dev/et0_ops
lspci -nn | grep -Ei '1e0a:eb01|esperanto|processing accelerators'
echo "ET_HARDWARE_PREFLIGHT PASS"
REMOTE
```

Earlier in this session, PCI `1e0a:eb01` and the Esperanto driver were present
while both character devices were absent. The supported locked reset did not
restore them; udev was then used to restore the nodes without unloading the
driver. The preflight then passed and the documented full run completed
successfully. Do not substitute an emulator, bypass the runner check, or
reload a shared-board driver without explicit administrative authorization.

Once the preflight passes, run the checked real-image full package:

```bash
export FULL_BOARD_RESULT_ID="${RUN_TAG}_soc1sim_full_coco_room"
export REMOTE_FULL_BOARD_RUN_DIR="$REMOTE_CACHE_ROOT/results/$FULL_BOARD_RESULT_ID"

"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" \
  "$REMOTE_LD_LIBRARY_PATH" \
  "$REMOTE_BOARD_LOCK" \
  "$FULL_BOARD_RESULT_ID" <<'REMOTE'
set -euo pipefail
root=$1
sdk=$2
launcher=$3
libs=$4
lock=$5
result_id=$6
port="$root/ported_models/yolov10n_hf_reference"
full_dir="$root/local-artifacts/yolov10n_hf_reference/full_graph/coco_room_000139_full308_v3"
elf="$full_dir/yolov10n_hf_full.elf"
out="$root/local-artifacts/yolov10n_hf_reference/results/$result_id"

cd "$root"
ET_PLATFORM="$sdk" \
LD_LIBRARY_PATH="$libs" \
BOARD_LOCK="$lock" \
"$port/scripts/run_et_full.sh" \
  --device soc1sim \
  --full-dir "$full_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$out" \
  --outer-timeout 2400 \
  --launcher-timeout 2340 \
  --lock-timeout 600
REMOTE
```

Pull and validate the real hardware identity, 16 checkpoints, output, and all
seven PMC records:

```bash
export LOCAL_FULL_BOARD_RUN_DIR="$CACHE_ROOT/results/$FULL_BOARD_RESULT_ID"
mkdir -p "$LOCAL_FULL_BOARD_RUN_DIR"
rsync -az -e "$RSYNC_RSH" \
  "$RSYNC_HOST:$REMOTE_FULL_BOARD_RUN_DIR/" \
  "$LOCAL_FULL_BOARD_RUN_DIR/"

"$PORT_ROOT/scripts/validate_et_full.sh" \
  "$REAL_FULL" \
  "$LOCAL_FULL_BOARD_RUN_DIR" \
  soc1sim \
  "$CACHE_ROOT/model.onnx" \
  1
```

The gate is complete only if all of these are present:

```text
FULL_EXECUTION_EVIDENCE PASS device=soc1sim hardware=True selector=N000:N307
FINAL_OUTPUT PASS ... direct_mismatches=0/1800 unexplained_mismatches=0 ...
PMC_STAGE PASS name=stem nodes=N000:N005 ...
PMC_STAGE PASS name=backbone nodes=N006:N090 ...
PMC_STAGE PASS name=sppf_psa nodes=N091:N128 ...
PMC_STAGE PASS name=neck nodes=N129:N207 ...
PMC_STAGE PASS name=three_scale_head nodes=N208:N270 ...
PMC_STAGE PASS name=dfl_decode nodes=N271:N288 ...
PMC_STAGE PASS name=topk_selection nodes=N289:N307 ...
ET_FULL_VALIDATION PASS device=soc1sim selector=N000:N307 ...
```

`device_evidence.txt` must identify both character devices and PCI
`1e0a:eb01`; `run_result.json` must say `"device":"soc1sim"` and
`"hardware":true`; `run.log` must report `DevicePcie`, ETSOC1, and successful
kernel completion. Without those independent facts, the result is not real
hardware evidence.

The preserved `coco_room_000139_full308_v3` run passes this exact gate. Its
outer elapsed time is 737 seconds, kernel wait is 721.396 seconds,
`output0` has 0/1,800 direct mismatches with max absolute error
`0.00042724609375`, all 300 anchors and anchor/class pairs align, and every
stage PMC record reports PASS.

### Preserved schema-v1 slice command

Create another unique result ID:

```bash
export BOARD_RESULT_ID="${RUN_TAG}_soc1sim_${SLICE}"
export REMOTE_BOARD_RUN_DIR="$REMOTE_CACHE_ROOT/results/$BOARD_RESULT_ID"
```

Run with the canonical shared lock, a reset immediately before launch, and
bounded lock/launcher/process timeouts:

```bash
"${SSH_CMD[@]}" bash -s -- \
  "$REMOTE_ROOT" \
  "$SLICE" \
  "$REMOTE_SYS_EMU_SDK" \
  "$REMOTE_LAUNCHER" \
  "$REMOTE_LD_LIBRARY_PATH" \
  "$REMOTE_BOARD_LOCK" \
  "$BOARD_RESULT_ID" <<'REMOTE'
set -euo pipefail
root=$1
slice=$2
sdk=$3
launcher=$4
libs=$5
lock=$6
result_id=$7
port="$root/ported_models/yolov10n_hf_reference"
slice_dir="$root/local-artifacts/yolov10n_hf_reference/slices/$slice"
elf="$slice_dir/yolov10n_hf_slice.elf"
out="$root/local-artifacts/yolov10n_hf_reference/results/$result_id"

cd "$root"
ET_PLATFORM="$sdk" \
LD_LIBRARY_PATH="$libs" \
BOARD_LOCK="$lock" \
"$port/scripts/run_et_slice.sh" \
  --device soc1sim \
  --slice-dir "$slice_dir" \
  --elf "$elf" \
  --launcher "$launcher" \
  --output-dir "$out" \
  --outer-timeout 300 \
  --launcher-timeout 240 \
  --lock-timeout 600
REMOTE
```

The wrapper composes the repository-supported pieces:

1. `.github/ci/scripts/prepare_board_lock.sh` provisions the existing lock.
2. `.github/ci/scripts/board_lock.py` holds it across the child process.
3. `board_reset_and_run.sh` discovers the writable ET PCIe
   `soc_reset/reinitiate` sysfs control, writes `1`, waits two seconds, and
   execs the launcher.
4. The launcher is invoked with `--device soc1sim`.

In addition to the common launcher markers, the hardware log must contain:

```text
Resetting ET-SoC1 via <writable-sysfs-reset-control>
erbium_soc1sim: elf=<resolved-elf> device=soc1sim shire=0
Kernel completed successfully
DEVICE_RUN PASS device=soc1sim ... reset_ok=1 ...
```

Pull and validate:

```bash
export LOCAL_BOARD_RUN_DIR="$CACHE_ROOT/results/$BOARD_RESULT_ID"
mkdir -p "$LOCAL_BOARD_RUN_DIR"
rsync -az -e "$RSYNC_RSH" \
  "$RSYNC_HOST:$REMOTE_BOARD_RUN_DIR/" \
  "$LOCAL_BOARD_RUN_DIR/"

"$PORT_ROOT/scripts/validate_device_run.sh" \
  "$SLICE_DIR" "$LOCAL_BOARD_RUN_DIR" soc1sim
```

Required proof:

```text
EXECUTION_EVIDENCE PASS device=soc1sim hardware=True nodes=N263:N265
TENSOR PASS node=N263 ... mismatches=0/32000 ...
TENSOR PASS node=N264 ... mismatches=0/32000 ...
TENSOR PASS node=N265 ... mismatches=0/32000 ...
SLICE_COMPARE PASS nodes=N263:N265 runtime_status=ok
DEVICE_VALIDATION PASS device=soc1sim ...
```

The validation does more than trust a successful kernel message. It verifies
the saved run record and all artifact hashes, exact launcher backend identity,
completion and dump log lines, reset evidence, two ET character devices,
Esperanto/`1e0a:eb01` PCIe evidence, exact dump size, original model SHA,
selected range, ELF/build-record identity, tensor result header, and every
output element. A software-emulator run cannot satisfy the hardware evidence
contract.

The known-good hardware result for `N263:N265` has the host max absolute
errors above and zero mismatches. The expanded `N263:N270` hardware sample has
zero mismatches across all eight outputs; the largest observed absolute error
is `1.33514404e-05` at `N269` and `N270`, below the declared tolerance.

## 15. Inspect PMC measurements

For a full run, `validate_et_full.sh` decodes seven separate records and writes
`pmc_stages.json` plus one `pmc_<index>_<name>.json` per stage. Offsets come
from `slice_manifest.json`; never derive them from a hard-coded dump layout.
The required partition is:

```text
stem              N000:N005
backbone          N006:N090
sppf_psa          N091:N128
neck              N129:N207
three_scale_head  N208:N270
dfl_decode        N271:N288
topk_selection    N289:N307
```

Each record brackets only its listed ONNX nodes. Target setup, input/weight
loading, cache eviction before the begin call, dumping, and comparison are
outside. All seven records pass in the preserved full real-image hardware
run. Exact HPM and shared-counter values are recorded in `VALIDATION.md` and
the run's `pmc_stages.json`.

For a schema-v2 bounded range, `validate_device_run.sh` decodes the one PMC
record around that exact range. A range that crosses two architectural stages
is still one range measurement and must not be mislabeled as either full stage.

For a preserved schema-v1 slice, obtain its exact offset as follows.
`validate_device_run.sh` writes `pmc.json` after checking the versioned PMC
record. To print the same record as workshop-friendly text, obtain the exact
offset from the slice manifest rather than guessing it:

```bash
export PMC_OFFSET
PMC_OFFSET="$("$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["memory_map"]["pmc_device_offset"])' \
  "$SLICE_DIR/slice_manifest.json")"

"$PY" "$PORT_ROOT/tools/decode_pmc.py" \
  "$LOCAL_BOARD_RUN_DIR/dump.bin" \
  --offset "$PMC_OFFSET" \
  --format text
```

For the recommended slice the current map yields decimal offset `393216`
(`0x60000`), but the manifest lookup is authoritative for every newly
captured range.

Expected high-level marker:

```text
RESULT: PASS
```

The JSON form is:

```bash
"$PY" "$PORT_ROOT/tools/decode_pmc.py" \
  "$LOCAL_BOARD_RUN_DIR/dump.bin" \
  --offset "$PMC_OFFSET" \
  --format json \
  > "$LOCAL_BOARD_RUN_DIR/pmc-manual.json"
```

The six HPM deltas identify minion cycles, retired instructions for threads 0
and 1, L2 miss requests, minion icache requests, and icache ET-link requests.
Shared shire-cache and memshire fields carry explicit requested/supported
masks; `UNSUPPORTED` is not turned into a fabricated zero.

PMC begin/end calls bracket only the selected inclusive node range. Launcher
startup, file loads, cache eviction before the begin call, result dumping, and
host comparison are outside that interval. Counter values vary with firmware
and execution conditions and are evidence, not numerical goldens.

## 16. Preserved result artifacts

Each successful device run preserves:

- `slice.elf` and `build_record.json`;
- `command.txt` and `wrapper_command.txt`;
- `environment.txt` and `device_evidence.txt`;
- `run.log`, `dump.bin`, and `run_result.json`;
- `board_lock.log` for hardware;
- `tensor_compare.json` and `pmc.json` after validation.

Keep the directory intact when reporting a result. `run_result.json` contains
hashes for the evidence set, and validation rejects missing or modified
files.

Schema-v2 full validation instead writes `full_compare.json`,
`pmc_stages.json`, and seven individual PMC JSON files. Passing final argument
`1` to `validate_et_full.sh` makes that `full_compare.json` require direct
positional output. Range validation uses `tensor_compare.json` and `pmc.json`
like the legacy slice path, but its report lists every output of every
selected node, including exact INT64 comparisons.

The ignored evidence roots are:

```text
local-artifacts/yolov10n_hf_reference/full_graph/
local-artifacts/yolov10n_hf_reference/ranges_v2/
local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu/
local-artifacts/yolov10n_hf_reference/results/
```

Never reuse an output directory. Keep a failed attempt under its original name
and use a new suffix for the next build/run.

## 17. Reference runner versus canonical leaderboard runner

The repository's canonical full-model path is:

```text
.github/ci/platform/deploy/soc3-benchmark.sh
  -> .github/ci/scripts/run_model_benchmark.sh
  -> scripts/run_sysemu_model_ports.sh
  -> a model row in .github/ci/benchmark_config.json
```

This reference port is not currently registered in
`.github/ci/benchmark_config.json`. Consequently these are **not valid current
commands**:

```text
MODELS=yolov10n_hf_reference .github/ci/platform/deploy/soc3-benchmark.sh
scripts/run_sysemu_model_ports.sh --model yolov10n_hf_reference ...
```

They reject the unknown model rather than running this reference port. The
commands in this recipe intentionally use `scripts/run_et_slice.sh`, which
reuses the repository launcher, platform resolver, linker/layout, CRT,
board-lock, reset, and backend conventions while preserving per-node
correctness artifacts. The full wrapper adds strict schema-v2 package/ELF
verification and seven stage PMCs. Neither emits a leaderboard score.

All ONNX operators and the complete host and real-ET-SoC1 correctness runs now
pass. Canonical registration remains a separate, intentionally unperformed
performance-integration change. It would still need a benchmark-config row,
canonical variant/build mapping, input/reference contract, result parser, and
both canonical runner targets.

## 18. Troubleshooting

### Checksum or byte-count failure

Do not inspect, pack, or capture from the mismatched file. Redownload
atomically from the pinned URL:

```bash
"$PY" "$PORT_ROOT/tools/download_model.py" --force
"$PY" "$PORT_ROOT/tools/download_model.py" --verify-only
```

If it still fails, stop: the source artifact is not the pin documented by this
port.

### `UNSUPPORTED node=...`

This is the intended safety boundary. Inspect the node and adjacent shapes in
`manifests/layers.tsv`. For schema v2, every operator form in the pinned graph
is supported, so this normally means that the model, manifest, attributes,
type, or shape no longer matches the checked contract. Verify the pinned
checksum and regenerate before changing code. The frozen schema-v1
`capture_slice.py` still intentionally supports only Conv, Sigmoid, Mul, and
Concat. Do not replace an unsupported form with a nearby operation or silently
skip it.

### Host compare failure

Read the first failing `TENSOR FAIL` line and the JSON report. Confirm all blob
identity lines pass, the result header reports `runtime_status=ok`, there are
no non-finite values, and the selected IDs match. Preserve the failing dump.
An enormous relative error at a reference value near zero is not by itself a
failure; the mismatch count uses the declared absolute-plus-relative formula.

### Deterministic `output0` has many positional mismatches

Open `final_output.selection_validation` in the JSON report. The checked
deterministic fixture has exact zero margins at both TopK cutoffs, so
tie-aware validation is expected. It passes only when `N288` passes, C and ORT
tail replays each match their own output bit-for-bit, and unexplained
mismatches are zero. The real-image fixture must use
`--require-direct-output` and must report 0/1,800 direct mismatches. Never
enable tie-aware acceptance for a nonzero cutoff margin.

### Full ELF verification says a source or header is stale

Rebuild the ELF from the exact package after every runtime, runner, manifest
header, linker, layout, or CRT change. Do not copy an old build record beside
a new ELF: `verify_full_elf.sh` checks every recorded digest and the
correctness flags.

### ET compiler reports a GLIBC error or cannot execute

Set `ET_GCC` to `scripts/et_gcc_docker_wrapper.sh` and use the supported
`et-gcc:24.04` image with complete ET and platform trees. Do not fall back to
an arbitrary `/bin/riscv64-unknown-elf-gcc`; a generic compiler may not support
ET CSRs, ABI, linker layout, or runtime assumptions. If Docker access is
permission-gated, obtain narrowly scoped access to the supported build
workflow.

### Platform resolver cannot find linker/layout/includes

`ET_PLATFORM_SRC` must be a complete ET platform source tree, not merely a
runtime install. At minimum verify
`gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld`, `layout.c`, and the Erbium
include trees. Keep this source mounted read-only in the compiler container.

### Launcher fails to load shared libraries

Run `ldd "$REMOTE_LAUNCHER"` on the execution host and set
`REMOTE_LD_LIBRARY_PATH` to the repository-supported launcher library
directories. Do not copy arbitrary host libraries into the checkout.

### Emulator launcher cannot find `sys_emu`

Set `ET_PLATFORM` to a complete system-emulator SDK root whose
`bin/sys_emu` is executable. `--device sys_emu` must remain explicit, and
`device_evidence.txt` should identify that exact SDK.

### System emulator traps at `N306` Cast

Disassemble the ELF and check for `fcvt.s.l`. The supported reference path
range-checks this graph's `Mod 80` INT64 class IDs, converts through INT32, and
should emit `fcvt.s.w`. Preserve the trapped attempt and rebuild through the
supported Docker compiler; do not hide the trap by skipping Cast.

### Hardware evidence fails

Run only on the actual board host. Both `/dev/et0_mgmt` and `/dev/et0_ops`
must be character devices, `lspci` must identify the Esperanto accelerator,
and the launcher command must say `device=soc1sim`. A missing device node is
not permission to substitute `sys_emu` while claiming hardware.

If PCI `1e0a:eb01` is present but the device nodes remain absent after the
supported locked reset, stop. Driver unload/reload affects shared board state
and requires explicit authorization from the board administrator.

### Board lock or reset fails

Use the provided runner as the board user with root or passwordless-sudo
access required by `prepare_board_lock.sh`, and ensure the ET
`soc_reset/reinitiate` sysfs control is writable. Do not bypass the lock or
reset and do not launch overlapping card jobs. Increase `--lock-timeout` only
when another legitimate job owns the card.

### A run directory already contains artifacts

The refusal prevents old dumps and logs from being mistaken for a new run.
Set a new `RUN_TAG` and use a fresh directory. Do not delete another user's
evidence.

### Timeout

Keep both timeout layers bounded and make the outer timeout larger than the
launcher timeout. Larger node ranges, especially the 640×640 stem, can take
substantially longer in scalar system emulation. Preserve the timed-out log,
then raise the bounds deliberately for that range; do not remove them.

### PMC shared counters are unavailable

Inspect requested and supported masks in `pmc.json`. Unsupported firmware
features remain explicit. For an HPM-only emulator experiment, compile a
separate ELF with `YR_PMC_SAMPLE_SC=0` and `YR_PMC_SAMPLE_MS=0`, preserve its
new build record, and label the reduced measurement scope.
