# YOLOv10n pinned-ONNX FP32 correctness reference

This is a new, isolated scalar FP32 port for ET-SoC1. Its sole model source is
the pinned Hugging Face ONNX artifact in `artifacts.json`; there is no PyTorch
re-export and no graph or weight mapping shared with `ported_models/yolo`.

| Source fact | Pinned value |
|---|---|
| Repository | `onnx-community/yolov10n` |
| Revision | `57657320425ee34056408a57ad9d29c4d4815bd8` |
| File | `onnx/model.onnx` |
| Bytes | 9,386,116 |
| SHA-256 | `a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b` |
| Upstream license | AGPL-3.0 |

The checked graph is ONNX opset 13 with input `images` FP32
`[1,3,640,640]`, output `output0` FP32 `[1,300,6]`, 308 nodes, and 187
initializers. The terminal graph contains the NMS-free two-stage
TopK/GatherElements selection; it has no `NonMaxSuppression` node.

## Implemented scope

The schema-v2 reference runtime executes the complete inclusive
`N000:N307` graph and produces `output0 [1,300,6]`. Its 22 checked operator
types are:

```text
Add Cast Concat Conv Div Flatten GatherElements MatMul MaxPool Mod Mul
ReduceMax Reshape Resize Sigmoid Softmax Split Sub Tile TopK Transpose
Unsqueeze
```

That is the complete operator set actually present in the pin: Conv×83,
Sigmoid×70, Mul×71, Concat×21, Split×13, Add×11, Reshape×8, Transpose×4,
MaxPool×3, Tile×3, GatherElements×3, MatMul×2, Softmax×2, Resize×2, TopK×2,
and one each of Sub, ReduceMax, Flatten, Mod, Div, and Cast. Implementations
are deliberately graph-contract-specific and validate shapes, attributes,
types, and broadcasts. An operator or form outside that contract still fails
explicitly; it is never silently skipped or approximated.

Every node remains a separate scalar operation. There is no VPU, TFMA,
fusion, tiling, threading, fast-math, or latency-oriented transformation.
The legacy schema-v1 four-operator slices remain available unchanged as
regression evidence alongside the full schema-v2 path.

The port remains intentionally absent from the leaderboard. The complete
real-image graph now passes on real ET-SoC1, but this scalar implementation is
a correctness reference rather than a latency submission. No registration,
publication, or leaderboard change is part of this work.

## What is validated

All comparisons use ONNX Runtime 1.16.3 with graph optimizations disabled,
one thread, and intermediates exposed from the checksum-verified artifact.
The default gate is
`abs(actual-reference) <= 5e-5 + 1e-4*abs(reference)`; the decoded `N288`
checkpoint alone has an explicit `2e-4` absolute override. INT64 outputs are
exact, and non-finite values always fail.

| Path | Scope | Current result |
|---|---|---|
| Full host, deterministic input | all 308 nodes; 16 pinned checkpoints; `output0` | PASS, zero unexplained mismatches |
| Full host, real COCO-room fixture | preprocessing through all 308 nodes and `output0` | strict direct PASS, 0/1,800 mismatches; max abs `0.00042724609375` |
| Host resumable ranges | 21 gap-free ranges covering `N000:N307`; all 324 node outputs | PASS, 0/75,592,700 mismatches |
| System emulator, original slices | `N263:N265` and `N266:N268` | PASS, zero mismatches, PMC PASS |
| System emulator, schema-v2 tail | `N289:N307` | PASS, all 22 outputs exact, PMC PASS |
| System emulator, all 21 ranges | planned gap-free coverage | in progress; do not infer completion from the plan |
| ET-SoC1 PCIe hardware, preserved slices | `N263:N265` and `N263:N270` | PASS, zero mismatches, PMC PASS |
| ET-SoC1 PCIe hardware, full graph | real image, `N000:N307`, 16 checkpoints, seven stage PMCs | PASS, strict direct 0/1,800 mismatches; all PMCs PASS |

The deterministic fixture contains exact score ties at both TopK cutoffs.
Direct positional comparison therefore differs in 1,091/1,800 fields even
though the two independently replayed `N289:N307` programs match their C and
ORT outputs bitwise and leave zero unexplained mismatches. The real-image
fixture has nonzero cutoff margins and is the strict final-output gate:
`--require-direct-output` passes with all 300 selected anchor/class pairs
aligned.

Preserved hardware slice evidence records `device=soc1sim`, both ET character
devices, PCI ID `1e0a:eb01`, saved ELF/build hashes, exact launcher commands,
board-lock/reset logs, dumps, and comparison reports. Those historical runs
remain regression evidence. The full run separately records `DevicePcie`,
`/dev/et0_ops`, ETSOC1, `hardware=true`, PCI `1e0a:eb01`, a 721.396-second
kernel wait, strict output agreement, and seven valid stage PMC records. Raw
binaries and logs remain under ignored `local-artifacts/`.

`manifests/board_full_summary_strict.json` is the tracked compact evidence
index. `tools/collect_board_summary.py` regenerates it only after rechecking
the pin, generated header, blobs, compiler/build inputs, saved run-artifact
hashes, hardware/reset log, strict 16-checkpoint comparison, detections, and
all seven PMCs.

## Full execution package and memory plan

`tools/generate_full_graph.py` emits a readable schema-v2 manifest plus a
small generated descriptor header, not a monolithic generated kernel. The
hand-written runtime consumes 512 typed tensor descriptors and the original
aligned initializer package.

- A deterministic 64-byte-aligned first-fit liveness plan allocates node
  outputs before releasing inputs whose last consumer is that node.
- Sixteen architecture checkpoints remain pinned for final comparison.
- The measured workspace arena is 35,788,800 bytes.
- The launcher dump is 36,306,944 bytes and includes the `YRF1` result,
  workspace, and seven independent 64 KiB PMC slots.
- The FP32 input starts at 36,306,944, the 9,299,136-byte weight package at
  41,222,144, and total target memory is 50,528,256 bytes.

For bounded range execution, `manifests/sys_emu_coverage_plan.json` partitions
the graph into 21 exact, non-overlapping ranges. Every selected node output is
retained and compared. Ranges target at most 24,000,000 retained-output bytes
and 250 million Conv/MatMul MACs; the deliberate `N271:N288` DFL/decode range
uses 30,105,600 retained bytes, and the measured maximum range workload is
249,958,400 MACs.

## Start here

- [RECIPE.md](docs/RECIPE.md): exact download, generation, host, emulator,
  board, validation, and troubleshooting commands.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): graph-measured workshop explanation
  of preprocessing boundaries, backbone, C2f-style blocks, SPPF, attention,
  neck, head, DFL decode, and NMS-free Top-300.
- [VALIDATION.md](docs/VALIDATION.md): measured host/emulator/hardware results
  and provenance.
- `manifests/layers.tsv`: readable stable `Nxxx`, `Lxxx`, and per-op IDs.
- `manifests/graph_inventory.json`: nodes, attributes, inferred tensors, and
  initializers.
- `manifests/weights_manifest.json`: deterministic little-endian initializer
  package layout; the binary stays in `local-artifacts/`.
- `manifests/board_full_summary_strict.json`: hash-bound strict real-ET-SoC1
  full-run evidence summary.

Minimal host setup:

```bash
ported_models/yolov10n_hf_reference/tools/setup_host_env.sh
python3 ported_models/yolov10n_hf_reference/tools/download_model.py
local-artifacts/yolov10n_hf_reference/venv/bin/python \
  ported_models/yolov10n_hf_reference/tools/inspect_onnx.py
local-artifacts/yolov10n_hf_reference/venv/bin/python \
  ported_models/yolov10n_hf_reference/tools/pack_initializers.py
local-artifacts/yolov10n_hf_reference/venv/bin/python \
  ported_models/yolov10n_hf_reference/tools/generate_full_graph.py \
  --name deterministic_full308_v3
ported_models/yolov10n_hf_reference/scripts/run_host_full.sh \
  local-artifacts/yolov10n_hf_reference/full_graph/deterministic_full308_v3
```

For the strict real-image gate, bounded system-emulator ranges, full board
command, expected markers, record semantics, and recovery instructions, use
[RECIPE.md](docs/RECIPE.md). The original
`capture_slice.py`/`run_host_slice.sh` examples remain documented there and
must continue to pass as regressions.

The model graph and weights retain AGPL-3.0 terms and are not committed. The
new runtime and tooling are first-party repository source; see
`THIRD_PARTY.md` for the exact upstream record.
