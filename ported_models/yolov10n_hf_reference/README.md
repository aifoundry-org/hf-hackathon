# YOLOv10n pinned-ONNX scalar reference

This directory contains a readable FP32 C port of one exact Hugging Face ONNX
artifact. It is a correctness and workshop baseline for ET-SoC1, not an
optimized submission.

| Source fact | Pinned value |
|---|---|
| Repository | `onnx-community/yolov10n` |
| Revision | `57657320425ee34056408a57ad9d29c4d4815bd8` |
| File | `onnx/model.onnx` |
| Size | 9,386,116 bytes |
| SHA-256 | `a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b` |
| License | AGPL-3.0 |

The model is ONNX opset 13. Its interface is `images` FP32
`[1,3,640,640]` to `output0` FP32 `[1,300,6]`. The graph has 308 nodes,
187 initializers, 83 Conv nodes, and 22 operator types. It includes decode and
NMS-free Top-300 selection; it does not contain a `NonMaxSuppression` node.

Every ONNX node is kept as a separate scalar operation. There is no VPU,
TFMA, fusion, tiling, threading, fast-math, or other performance
transformation.

## Model parts

| Part | ONNX nodes | Purpose |
|---|---:|---|
| Stem | `N000:N005` | Two stride-2 Conv/SiLU steps |
| Backbone | `N006:N090` | C2f-style features and P3/P4/P5 downsampling |
| SPPF and partial attention | `N091:N128` | Spatial pyramid pooling and attention/FFN |
| Neck | `N129:N207` | Top-down and bottom-up multiscale fusion |
| Three-scale head | `N208:N270` | P3/P4/P5 box and class branches |
| DFL and decode | `N271:N288` | Distribution expectation, boxes, and class sigmoid |
| Top-300 selection | `N289:N307` | Two TopK stages and final `[box, score, class]` rows |

The first five parts are learned feature extraction and detection. The last
two are output transformation implemented inside the ONNX. Preprocessing,
score filtering, label names, drawing, and mapping boxes back to the original
image are outside the model.

`manifests/layers.tsv` provides stable `Nxxx`, high-level `Lxxx`, and
per-operator IDs. `manifests/graph_inventory.json` contains every node,
attribute, inferred tensor, and initializer.

## Supported workflows

There are only two execution modes:

1. `generate_full_graph.py` creates an end-to-end `N000:N307` package.
2. `capture_range.py` creates one node or any contiguous range, such as
   `N003:N003` or `N271:N288`.

Both use the same hand-written runtime in `src/ref_runtime.c`. Generated ONNX
instrumentation, headers, weights, inputs, goldens, ELFs, dumps, and logs stay
under ignored `local-artifacts/`.

### 1. Set up and verify the pinned model

Run from the repository root:

```bash
PORT=ported_models/yolov10n_hf_reference
PY=local-artifacts/yolov10n_hf_reference/venv/bin/python

"$PORT/tools/setup_host_env.sh"
"$PY" "$PORT/tools/download_model.py"
"$PY" "$PORT/tools/inspect_onnx.py"
"$PY" "$PORT/tools/pack_initializers.py"
```

The environment is pinned in `requirements-host.txt` and supports Python
3.8–3.11. The download and every generator reject the wrong model checksum.

### 2. Run the full graph on the host

```bash
"$PY" "$PORT/tools/generate_full_graph.py" --name deterministic
"$PORT/scripts/run_host_full.sh" \
  local-artifacts/yolov10n_hf_reference/full_graph/deterministic
```

This compares 16 architecture checkpoints and final `output0` against ONNX
Runtime with graph optimization disabled. FP32 values use
`abs(actual-reference) <= 5e-5 + 1e-4*abs(reference)`, except for the
documented `N288` decode checkpoint override. INT64 comparisons are exact.

### 3. Run one layer or a small range

This example captures and checks nodes `N003:N005`:

```bash
NAME=n003_n005
RANGE=local-artifacts/yolov10n_hf_reference/ranges/$NAME

"$PY" "$PORT/tools/capture_range.py" \
  --range N003:N005 \
  --name "$NAME"
"$PORT/scripts/run_host_range.sh" "$RANGE"
```

Every output of every selected node is retained and compared. Boundary
tensors are captured from the same pinned model through ONNX Runtime. A
single node is selected by repeating it or using it once:

```bash
"$PY" "$PORT/tools/capture_range.py" \
  --range N003 \
  --name n003
```

### 4. Build and run that range in `sys_emu`

The ET compiler requires a valid `ET_PLATFORM` or `ET_INSTALL`. Set
`LAUNCHER` to the system-emulator launcher installed on the ET host.

```bash
ELF="$RANGE/yolov10n_hf_range.elf"
RUN=local-artifacts/yolov10n_hf_reference/results/sys_emu_$NAME

"$PORT/scripts/build_et_slice.sh" "$RANGE" "$ELF"
"$PORT/scripts/run_et_slice.sh" \
  --device sys_emu \
  --slice-dir "$RANGE" \
  --elf "$ELF" \
  --launcher "$LAUNCHER" \
  --output-dir "$RUN" \
  --outer-timeout 1800 \
  --launcher-timeout 1740
"$PORT/scripts/validate_device_run.sh" "$RANGE" "$RUN" sys_emu
```

`sys_emu` is intentionally slow. Use it for one layer or a bounded range,
not for routine end-to-end inference. The validator checks the saved command,
ELF, input and weight identities, every selected output, and the PMC record.

### 5. Run the complete graph on ET-SoC1

On a configured ET board host:

```bash
FULL=local-artifacts/yolov10n_hf_reference/full_graph/deterministic
ELF="$FULL/yolov10n_hf_full.elf"
RUN=local-artifacts/yolov10n_hf_reference/results/full_board
MODEL=local-artifacts/yolov10n_hf_reference/model.onnx

"$PORT/scripts/build_et_full.sh" "$FULL" "$ELF"
"$PORT/scripts/run_et_full.sh" \
  --device soc1sim \
  --full-dir "$FULL" \
  --elf "$ELF" \
  --launcher "$LAUNCHER" \
  --output-dir "$RUN"
"$PORT/scripts/validate_et_full.sh" \
  "$FULL" "$RUN" soc1sim "$MODEL" 1
```

The real board path obtains the repository board lock, resets ET-SoC1, and
stores hash-bound run evidence. It never registers the port with the
leaderboard.

## PMCs

`src/ref_pmc.h` programs and reads these counters:

- `hpmcounter3`: minion cycles
- `hpmcounter4/5`: retired instructions on thread 0/1
- `hpmcounter6`: L2 miss requests
- `hpmcounter7`: minion I-cache requests
- `hpmcounter8`: I-cache ET-link requests

A range has one PMC interval around exactly its selected nodes. Full
execution has seven intervals matching the model-parts table above. Input
loading, launcher startup, dumping, and host comparison are outside those
intervals. `validate_device_run.sh` and `validate_et_full.sh` decode and check
the records automatically; `tools/decode_pmc.py` is available for manual
inspection.

Simulator PMCs prove execution and instrumentation. Use real-board PMCs for
performance conclusions.

## Current validation

| Path | Result |
|---|---|
| Full host, deterministic input | PASS, all 308 nodes and 16 checkpoints |
| Full host, checked real image | PASS, strict `output0` 0/1,800 mismatches |
| Host arbitrary ranges | PASS, including every supported operator |
| `sys_emu` bounded ranges | PASS, selected outputs and PMC records |
| ET-SoC1 full real image | PASS, strict `output0`, 16 checkpoints, seven PMCs |

The measured full-board kernel wait for this unoptimized scalar baseline was
721.396 seconds. This is correctness evidence, not a target latency.
`manifests/board_full_summary_strict.json` is the compact, hash-bound record;
raw binaries and logs are deliberately not committed.

## Repository contents

- `src/`: scalar runtime, host/ET runners, and PMC support.
- `tools/`: model download, graph generation, comparison, preprocessing, and
  evidence utilities.
- `scripts/`: small host and ET build/run/validate entry points.
- `manifests/`: pinned graph, layer, weight, execution, and board facts.
- `tools/tests/`: tamper and contract regression tests.
- [THIRD_PARTY.md](THIRD_PARTY.md): upstream artifact and license record.

Run lightweight source checks with:

```bash
python3 -m compileall -q "$PORT/tools"
for script in "$PORT"/scripts/*.sh "$PORT"/tools/*.sh; do
  bash -n "$script"
done
```

The complete graph and weight package are generated locally because the model
artifact remains under its upstream AGPL-3.0 terms.
