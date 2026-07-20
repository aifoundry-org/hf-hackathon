# Validation record

This record captures both the original four-operator slice milestone and the
current complete scalar-FP32 YOLOv10n Hugging Face reference implementation.
The historical slice runs, including their PMC and failed-attempt evidence,
remain intact. The schema-v2 runtime now executes every node `N000:N307` and
produces `output0 [1,300,6]` on the host.

Target claims remain deliberately narrower than implementation claims.
Complete host execution and the complete gap-free host range matrix pass.
Schema-v2 system-emulator validation currently proves the terminal
`N289:N307` range in addition to the preserved earlier slices; execution of
all 21 planned emulator ranges is still in progress. Full-graph real ET-SoC1
validation now passes: the complete real-image graph, all 16 checkpoints,
strict final output, result integrity, hardware identity, and seven stage PMC
records validated. No emulator result is presented as hardware, and the port
remains intentionally unregistered with the full-model leaderboard.

## Source provenance

The only model source used for topology, weights, tensor names, shapes, and
reference values was the pinned ONNX artifact. No PyTorch re-export was used.

| Field | Value |
|---|---|
| Hugging Face repository | `onnx-community/yolov10n` |
| Revision | `57657320425ee34056408a57ad9d29c4d4815bd8` |
| File | `onnx/model.onnx` |
| Resolve URL | `https://huggingface.co/onnx-community/yolov10n/resolve/57657320425ee34056408a57ad9d29c4d4815bd8/onnx/model.onnx?download=true` |
| Downloaded bytes | 9,386,116 |
| SHA-256 | `a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b` |
| Upstream model-card license | AGPL-3.0 |
| Local cache | `local-artifacts/yolov10n_hf_reference/model.onnx` |

The checksum was recomputed locally and matched. ONNX checker plus strict shape
inference confirmed opset 13, input `images` as FP32 `[1,3,640,640]`, output
`output0` as FP32 `[1,300,6]`, 308 nodes, and 187 initializers. The terminal
inventory contains two `TopK` and three `GatherElements` nodes and no NMS
operator. The full operator and shape evidence is in
`manifests/graph_inventory.json`.

All 187 initializers were also packed directly from this ONNX into a
little-endian, 64-byte-aligned package:

| Package fact | Value |
|---|---|
| FLOAT initializers | 169 |
| INT64 initializers | 18 |
| Tensor data bytes | 9,298,228 |
| Padding bytes | 908 |
| Total bytes | 9,299,136 |
| Package SHA-256 | `b6d5aa13ef3238328c19ff0f646f72841bcf44dc1651383d089f0d57e37cf850` |

## Reference method and pass criterion

Goldens were captured with NumPy 1.24.4, ONNX 1.16.2, and ONNX Runtime
1.16.3 using `CPUExecutionProvider`, graph optimization disabled, one intra-op
thread, and one inter-op thread. Instrumentation performed shape inference and
added intermediate tensors as graph outputs; it did not rewrite graph compute
or parameters.

The deterministic model input was generated elementwise as:

```text
((index * 1664525 + 1013904223) & 0x00ffffff) / 16777215
```

The preserved schema-v1 slices and schema-v2 ranges use this explicit
acceptance rule:

```text
abs(actual - reference) <= 5e-5 + 1e-4 * abs(reference)
```

The full-graph package uses the same `atol=5e-5`, `rtol=1e-4` default and a
single declared checkpoint override at decoded `N288`:
`atol=2e-4`, `rtol=1e-4`. The override covers measured scalar DFL/decode
accumulation without loosening score or final-output checkpoints. INT64
outputs compare exactly. The comparison reports maximum absolute error,
maximum relative error, mean absolute error, non-finite counts, and mismatch
count. A normally continuous tensor passes only when its mismatch count and
both non-finite counts are zero. Blob byte counts and SHA-256 values, the
device result header, selected node range, workspace size, and workspace
FNV-1a are checked before tensor comparison.

## Current validation matrix

| Gate | Required scope | Result as of this record |
|---|---|---|
| Full deterministic host | all 308 nodes, 16 checkpoints, tie-audited `output0` | PASS; zero unexplained mismatches |
| Full real-image host | preprocessing, all 308 nodes, strict positional `output0` | PASS; 0/1,800 mismatches |
| Host range matrix | 21 exact ranges, every node output | PASS; 324/324 outputs |
| System emulator | 21 exact ranges covering all 308 nodes | IN PROGRESS; terminal range plus legacy slices pass |
| Full ET-SoC1 hardware | real image, all 308 nodes, 16 checkpoints, seven stage PMCs | PASS; strict 0/1,800 final mismatches |
| Leaderboard registration | separate explicit performance-integration change | NOT REGISTERED |

## Complete scalar operator coverage

The full runtime's supported set is exactly the pinned graph's 22 operator
types:

| Operator | Count | Operator | Count |
|---|---:|---|---:|
| Conv | 83 | Sigmoid | 70 |
| Mul | 71 | Concat | 21 |
| Split | 13 | Add | 11 |
| Reshape | 8 | Transpose | 4 |
| MaxPool | 3 | Tile | 3 |
| GatherElements | 3 | MatMul | 2 |
| Softmax | 2 | Resize | 2 |
| TopK | 2 | Sub | 1 |
| ReduceMax | 1 | Flatten | 1 |
| Mod | 1 | Div | 1 |
| Cast | 1 |  |  |

Runtime guards continue to reject unimplemented attributes, shapes,
broadcasts, types, or index ranges explicitly. “All operators supported”
means all forms present in this exact artifact, not unrestricted ONNX
conformance.

## Full host inference

The schema-v2 full package has 512 tensor descriptors and executes the
inclusive selector `N000:N307`. Its deterministic first-fit liveness arena is
35,788,800 bytes with 64-byte alignment and 16 pinned checkpoints:

```text
N005 N020 N045 N071 N090 N100 N128 N144
N160 N178 N207 N228 N249 N270 N288 N307
```

The dump is 36,306,944 bytes. Input and weight offsets are 36,306,944 and
41,222,144; total target memory is 50,528,256 bytes. Seven 64 KiB PMC slots
are reserved after the arena for stem `N000:N005`, backbone `N006:N090`,
SPPF/PSA `N091:N128`, neck `N129:N207`, three-scale head `N208:N270`,
DFL/decode `N271:N288`, and TopK selection `N289:N307`.

The preserved `full308_v3` generated header is 92,881 bytes with SHA-256
`79be5b751842df025a3612ebb690e283813ea9ac8e373fd1bc44b706ca7a2a7e`.
The Docker-built real-image full ELF is 177,864 bytes with SHA-256
`278e1769020036ee835ba8838c1595f9f48d54dcd7ec6ff2c743fbff688409cc`.
These identities bind the validated board run to its exact build inputs; the
execution PASS additionally requires the checked dump, comparison, device
evidence, completion log, and PMC records.

### Deterministic fixture and TopK ties

All checkpoints through decoded candidate tensor `N288 [1,8400,84]` pass.
The deterministic LCG input creates exact equality at both TopK boundaries:

| Boundary | Cutoff margin | Ties at cutoff | Values strictly above |
|---|---:|---:|---:|
| First anchor TopK | 0 | 14 | 289 |
| Second flattened-score TopK | 0 | 15 | 298 |

Because TopK order among equal values is discontinuous across kernels, direct
record-by-record comparison reports 1,091/1,800 differing fields. That
discrepancy is preserved, not hidden. The comparator independently replays
the exact `N289:N307` tensor program from both the C and ORT `N288` tensors:

- the actual replay matches C `output0` bit-for-bit;
- the reference replay matches ORT `output0` bit-for-bit;
- 293/300 first-stage anchors and 294/300 final anchor/class pairs overlap;
- both reference cutoff margins are exactly zero; and
- unexplained mismatch count is zero.

The normal deterministic command therefore passes in
`proven_topk_tie_discontinuity` mode. Adding `--require-direct-output`
intentionally makes this tied fixture fail and is useful for proving that the
strict switch is active.

### Real COCO-room fixture and strict output

The checked raw fixture is
`ported_models/yolo/assets/yolo/coco_room_000139_raw_480x640x3_uint8_rgb.bin`,
SHA-256
`66b6131da00004bd2eab6a5d2fafab937289839d10d8199b3e95bfa3e76d8ca9`.
It is 480×640 HWC RGB UINT8. Preprocessing centers it unchanged in rows
80:560 of a 640×640 RGB canvas padded with 114, divides by FP32 255, and
transposes to NCHW. The resulting input SHA-256 is
`65afc38f381c09712cd9a6a78e8e7e9800c713d21679abee588b218a6a137c7b`.

This input has nonzero margins at both TopK boundaries. Strict direct
comparison passes all 1,800 fields with zero mismatches, max absolute error
`0.00042724609375`, and all 300 first-stage anchors and final anchor/class
pairs aligned. Ten records have score at least 0.25. The first five C records
are:

| Rank | x1 | y1 | x2 | y2 | score | class ID |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.693794 | 266.976929 | 155.103424 | 375.755677 | 0.899284 | 62 |
| 2 | 293.124054 | 325.730621 | 354.252319 | 445.966858 | 0.855219 | 56 |
| 3 | 361.279083 | 325.319794 | 418.113007 | 434.756165 | 0.735568 | 56 |
| 4 | 413.138000 | 256.556274 | 466.460236 | 416.080383 | 0.719135 | 0 |
| 5 | 448.336182 | 215.819183 | 461.191101 | 240.327805 | 0.517890 | 74 |

Each row is FP32 `[x1,y1,x2,y2,score,class_id]` in the padded 640×640 canvas.
The class ID is mathematically integral but is cast to FP32 at `N306`.
Human-readable label strings are not embedded in the ONNX. For this fixture
only, subtract 80 from each y coordinate and clip to `[0,480]` to map back to
the raw image; x is unchanged. The graph always returns 300 entries and does
not threshold or suppress overlapping boxes.

## Gap-free host range matrix

`manifests/sys_emu_coverage_plan.json` is also the host range test plan. Its
21 inclusive selectors are:

```text
N000:N002  N003:N009  N010:N017  N018:N030  N031:N045
N046:N060  N061:N079  N080:N104  N105:N137  N138:N153
N154:N168  N169:N196  N197:N207  N208:N210  N211:N213
N214:N223  N224:N231  N232:N249  N250:N270  N271:N288
N289:N307
```

They cover `N000:N307` exactly once: no gaps and no overlaps. Every output of
every selected node is retained, yielding 324 output tensors and 75,592,700
compared elements. All 324 pass with zero tolerance/exact mismatches. The
largest range-level maximum absolute error is `0.0001220703125` in
`N271:N288`; `N289:N307` is exact across 795,900 elements, including INT64
indices.

Each range is bounded at no more than 249,958,400 measured Conv/MatMul MACs.
Retained outputs target 24,000,000 bytes; the deliberate architecture-aligned
DFL/decode range is the sole exception at 30,105,600 bytes.

## Schema-v2 system-emulator evidence to date

The complete range plan has not yet finished in the system emulator. One
current full-operator range is validated:

| Result directory | Nodes | Outputs/elements | Tensor result | PMC |
|---|---|---:|---|---|
| `full_coverage_sys_emu/n289_n307` | `N289:N307` | 22 / 795,900 | exact, zero mismatches | PASS |

The saved `run_result.json` says `device=sys_emu`, `hardware=false`, identifies
the launcher and pinned source, and records 33 seconds outer elapsed. This
range covers Split, ReduceMax, both TopKs, Unsqueeze, Tile, all three
GatherElements nodes, Flatten, Mod, Div, Cast, and final Concat.

The prior directory `full_coverage_sys_emu/n289_n307_tail` is intentionally
preserved. Its launcher-level run completed and dumped memory, but target
validation exposed the emulator's unsupported RV64 `fcvt.s.l` instruction at
`N306` Cast; the PMC end record and final workspace checksum were not valid.
The checked Cast implementation now first range-checks the graph's INT64 class
IDs (known to be `Mod 80`) and converts through INT32, producing supported
`fcvt.s.w` while remaining exact for every value in the pinned graph. The
fresh `tail_cast32` build and the canonical `n289_n307` rerun are passing
evidence. This failed/passing sequence is regression evidence and must not be
deleted.

## Complete real ET-SoC1 validation

The real-image `full308_v3` package passed the complete hardware gate in
`local-artifacts/yolov10n_hf_reference/results/full_board/coco_room_000139_full308_v3/`.
This was not an emulator run:

- `run_result.json` says `device=soc1sim`, `hardware=true`, return code 0,
  inclusive selector `N000:N307`, and source SHA matching the pin.
- `device_evidence.txt` records both ET character devices and PCI
  `1e0a:eb01`.
- `run.log` enters `DevicePcie`, targets `/dev/et0_ops`, reports architecture
  `ETSOC1` and form factor `PCIE`, then reports successful kernel completion.
- The repository board lock and sysfs reset both passed.
- Outer elapsed time was 737 seconds; the launcher-reported kernel wait was
  721.396 seconds.
- The 36,306,944-byte dump SHA-256 is
  `b4975576d65ad90adced01a8e6ad16e456a8a644213edd8452e41bd7a50b1202`.

The versioned `YRF1` result header, complete-workspace FNV-1a, blob identities,
generated-header identity, ELF/build record, dump size, and all 16 checkpoint
segments passed. Hardware checkpoint results are:

| Node | Tensor role | Max abs | Mismatches |
|---|---|---:|---:|
| N005 | stem exit | 2.288818359375e-05 | 0 |
| N020 | P2 backbone exit | 3.135204315185547e-05 | 0 |
| N045 | P3 backbone exit | 9.179115295410156e-06 | 0 |
| N071 | P4 backbone exit | 1.1682510375976562e-05 | 0 |
| N090 | P5 backbone exit | 1.049041748046875e-05 | 0 |
| N100 | SPPF exit | 1.6808509826660156e-05 | 0 |
| N128 | partial-attention exit | 2.181529998779297e-05 | 0 |
| N144 | top-down P4 exit | 1.1682510375976562e-05 | 0 |
| N160 | top-down P3 exit | 8.940696716308594e-06 | 0 |
| N178 | bottom-up P4 exit | 1.71661376953125e-05 | 0 |
| N207 | bottom-up P5 exit | 1.8835067749023438e-05 | 0 |
| N228 | P3 head join | 9.5367431640625e-05 | 0 |
| N249 | P4 head join | 0.0006103515625 | 0 |
| N270 | P5 head join | 8.96453857421875e-05 | 0 |
| N288 | decoded candidates | 0.001251220703125 | 0 |
| N307 | `output0` | 0.00042724609375 | 0 |

`N288` uses only its declared `atol=2e-4`, `rtol=1e-4` checkpoint override;
all other FP32 rows use `atol=5e-5`, `rtol=1e-4`. The combined
absolute-plus-relative test passes every element.

The final `output0 [1,300,6]` passes strict direct comparison with 0/1,800
mismatches, max absolute error `0.00042724609375`, max relative error
`7.514777829110251e-05`, and zero unexplained mismatches. Both TopK boundaries
have nonzero margins; all 300 first-stage anchors and all 300 final
anchor/class pairs align. Independent actual/reference tail replays also
match their respective output tensors bit-for-bit.

### Full hardware stage PMCs

All seven `YRPM` records pass their magic, version, byte order, layout,
active-hart, and counter checks. Each interval brackets only its listed ONNX
nodes. Setup, input/weight loading, dumping, and host comparison are excluded.

| Stage | Nodes | Minion cycles | Retired T0 | L2 miss requests | I-cache requests | ETLink requests |
|---|---|---:|---:|---:|---:|---:|
| stem | N000:N005 | 15,903,571,548 | 4,979,902,049 | 84,187,039 | 1,175,405,374 | 65 |
| backbone | N006:N090 | 150,929,916,959 | 47,972,885,926 | 800,627,436 | 12,048,058,410 | 28 |
| SPPF/PSA | N091:N128 | 34,916,207,890 | 10,503,802,878 | 211,467,863 | 2,816,053,150 | 69 |
| neck | N129:N207 | 124,145,435,420 | 39,051,788,133 | 663,706,918 | 10,039,033,455 | 12 |
| three-scale head | N208:N270 | 102,548,474,177 | 33,249,340,626 | 534,267,342 | 8,191,298,447 | 0 |
| DFL/decode | N271:N288 | 1,916,064,054 | 431,774,722 | 5,527,353 | 135,731,010 | 2 |
| TopK selection | N289:N307 | 675,602,783 | 177,731,436 | 2,164,787 | 51,979,760 | 64 |

Thread-1 retired instructions are zero because this scalar reference activates
one hart/thread. Every stage also reports shared-counter PASS with shire-cache
mask `0x00000fff` and memory-shire mask `0x00ffffff`; the complete decoded
shared samples remain in `pmc_stages.json` and its seven referenced files.
These are correctness-instrumented scalar measurements, not a performance
target.

Earlier in the session, both ET character nodes were absent while PCI and the
driver remained present. The supported locked reset did not restore them;
udev restored the expected nodes (`10:123` and `10:122`) without unloading
the shared driver. That recovery history and all earlier passing hardware
slices below remain preserved as regression evidence.

### Tracked strict board evidence index

`manifests/board_full_summary_strict.json` is an 8,476-byte tracked summary
with SHA-256
`92dfa30264d648a527d7ffa0a3d77932e89a3e87939432486430d6bc4f819787`.
It does not merely transcribe the PASS strings. Its generator,
`tools/collect_board_summary.py`, independently rechecks:

- pinned source, generated-header, input, weight, golden, and instrumented
  ONNX identities;
- supported Docker compiler identity, ELF, and every recorded build input;
- all run-result artifact hashes and real PCIe/reset/completion log markers;
- strict direct-output mode, all 16 checkpoints, `output0`, anchor/pair
  overlap, and the real-image detection explanation; and
- all seven decoded PMC records and their exact stage boundaries.

With the path variables from `RECIPE.md` section 1, reproduce the tracked
summary without overwriting evidence:

```bash
SUMMARY_TMP="$(mktemp -d)/board_full_summary_strict.json"
"$PY" "$PORT_ROOT/tools/collect_board_summary.py" \
  "$CACHE_ROOT/full_graph/coco_room_000139_full308_v3" \
  "$CACHE_ROOT/results/full_board/coco_room_000139_full308_v3" \
  "$SUMMARY_TMP"
cmp "$SUMMARY_TMP" \
  "$PORT_ROOT/manifests/board_full_summary_strict.json"
```

Expected marker:

```text
BOARD_SUMMARY PASS selector=N000:N307 output_mismatches=0 pmc_stages=7 ...
```

The negative binding regression changes the direct-output mismatch count in a
temporary comparison report and proves that the collector rejects it:

```bash
"$PY" "$PORT_ROOT/tools/tests/test_board_summary_binding.py"
```

Expected marker:

```text
BOARD_SUMMARY_BINDING PASS positive=accepted direct_output_mismatch=rejected
```

## Preserved schema-v1 host slice correctness

The host runner uses the same readable scalar C runtime as the ET executable.
All four host samples passed with zero mismatches and zero non-finite values.

| Slice | Meaning | Nodes | Total mismatches | Largest max-abs |
|---|---|---:|---:|---:|
| `n000_n002_stem_conv_silu` | First real graph Conv followed by explicit Sigmoid and Mul | N000:N002 | 0 | 7.62939453125e-06 |
| `n263_n265_dw_silu` | P5 head 3x3 depthwise Conv followed by explicit Sigmoid and Mul | N263:N265 | 0 | 5.960464477539062e-07 |
| `n266_n268_pw_silu` | P5 head 1x1 pointwise Conv followed by explicit Sigmoid and Mul | N266:N268 | 0 | 1.049041748046875e-05 |
| `n263_n270_p5_class_join` | Contiguous late P5 head path through final class Conv and regression/class Concat | N263:N270 | 0 | 1.33514404296875e-05 |

Exact host tensor results:

| Slice | Node | Elements | Max abs | Max rel | Mean abs | Mismatches |
|---|---:|---:|---:|---:|---:|---:|
| N000:N002 | N000 | 1,638,400 | 7.62939453125e-06 | 0.029575892857142856 | 5.790451609755109e-07 | 0 |
| N000:N002 | N001 | 1,638,400 | 1.0132789611816406e-06 | 6.716264762121528e+30 | 6.496497382249499e-08 | 0 |
| N000:N002 | N002 | 1,638,400 | 7.62939453125e-06 | 1.0984088193214924e+32 | 4.2067303503204465e-07 | 0 |
| N263:N265 | N263 | 32,000 | 4.76837158203125e-07 | 0.017045454545454544 | 4.355658438726095e-08 | 0 |
| N263:N265 | N264 | 32,000 | 1.7881393432617188e-07 | 5.180445269631815e-07 | 2.2843945771455766e-08 | 0 |
| N263:N265 | N265 | 32,000 | 5.960464477539062e-07 | 0.01704558360641582 | 2.8660296791827024e-08 | 0 |
| N266:N268 | N266 | 32,000 | 1.049041748046875e-05 | 0.0034693667763157896 | 6.92990462994203e-07 | 0 |
| N266:N268 | N267 | 32,000 | 1.8775463104248047e-06 | 1.005196184063665e-05 | 1.0190182365477086e-07 | 0 |
| N266:N268 | N268 | 32,000 | 5.245208740234375e-06 | 0.0034704687506272353 | 5.404623757385707e-07 | 0 |
| N263:N270 | N263 | 32,000 | 4.76837158203125e-07 | 0.017045454545454544 | 4.355658438726095e-08 | 0 |
| N263:N270 | N264 | 32,000 | 1.7881393432617188e-07 | 5.180445269631815e-07 | 2.2843945771455766e-08 | 0 |
| N263:N270 | N265 | 32,000 | 5.960464477539062e-07 | 0.01704558360641582 | 2.8660296791827024e-08 | 0 |
| N263:N270 | N266 | 32,000 | 1.0132789611816406e-05 | 0.005146748310810811 | 7.666284800507128e-07 | 0 |
| N263:N270 | N267 | 32,000 | 1.8775463104248047e-06 | 1.005196184063665e-05 | 1.0814226698130369e-07 | 0 |
| N263:N270 | N268 | 32,000 | 5.245208740234375e-06 | 0.0051460312422438575 | 6.008876626992787e-07 | 0 |
| N263:N270 | N269 | 32,000 | 1.33514404296875e-05 | 8.199861070925284e-07 | 2.1297037601470947e-06 | 0 |
| N263:N270 | N270 | 57,600 | 1.33514404296875e-05 | 8.199861070925284e-07 | 1.1831687556372748e-06 | 0 |

The very large maximum-relative values at N001 and N002 occur where the ORT
reference is at or extremely near zero. They are reported rather than hidden;
the finite absolute errors remain below tolerance, and the combined
absolute-plus-relative criterion passes every element.

The standalone N266:N268 sample starts from its ORT-captured external
activation. The contiguous N263:N270 sample instead feeds N266 with the C
result from N265, so its later errors include propagation through the earlier
selected nodes.

## Preserved schema-v1 system-emulator results

Both requested small-spatial emulator samples completed with `device=sys_emu`.
The evidence file identifies `$ET_PLATFORM/bin/sys_emu` as a regular
executable, and the launcher logs contain `device=sys_emu`. These
runs are software-emulator evidence and are deliberately not presented as
hardware evidence.

| Run | UTC start | Nodes | Result | Outer elapsed | Kernel wait | Dump |
|---|---|---|---|---:|---:|---|
| `final_sys_emu_n263_n265` | 2026-07-20T09:31:36Z | N263:N265 | PASS | 5 s | 2.33081 s | 458,752 bytes, SHA-256 `2817d8f3df806515e551985476966a5a4caa823620cfc6c1e667b2bc7bed944e` |
| `final_sys_emu_n266_n268` | 2026-07-20T09:32:42Z | N266:N268 | PASS | 22 s | 18.9078 s | 458,752 bytes, SHA-256 `48168ce23e7004844d475509caad03d20262054533d422c515892d823a150017` |

Exact emulator tensor results:

| Run | Node | Shape | Max abs | Max rel | Mean abs | Mismatches |
|---|---:|---|---:|---:|---:|---:|
| N263:N265 | N263 | `[1,80,20,20]` | 4.76837158203125e-07 | 0.017045454545454544 | 4.355658438726095e-08 | 0 |
| N263:N265 | N264 | `[1,80,20,20]` | 1.7881393432617188e-07 | 5.180445269631815e-07 | 2.2843945771455766e-08 | 0 |
| N263:N265 | N265 | `[1,80,20,20]` | 5.960464477539062e-07 | 0.01704558360641582 | 2.8660296791827024e-08 | 0 |
| N266:N268 | N266 | `[1,80,20,20]` | 1.049041748046875e-05 | 0.0034693667763157896 | 6.92990462994203e-07 | 0 |
| N266:N268 | N267 | `[1,80,20,20]` | 1.8775463104248047e-06 | 1.005196184063665e-05 | 1.0190182365477086e-07 | 0 |
| N266:N268 | N268 | `[1,80,20,20]` | 5.245208740234375e-06 | 0.0034704687506272353 | 5.404623757385707e-07 | 0 |

For both runs, blob checks, the `YRF1` result header, workspace FNV-1a, every
tensor, and the decoded PMC record passed. Every tensor had zero non-finite
actual/reference values.

## Preserved schema-v1 real ET-SoC1 board results

The final hardware runs used the repository-supported PCIe launcher with
`--device soc1sim`; in this launcher, `soc1sim` selects the real ET-SoC1 PCIe
device layer, while `sys_emu` selects the software emulator. Hardware identity
was independently preserved:

- Hostname `esperanto-soc3`, Linux `5.15.140-release-94a7f6a-81353a4`.
- `/dev/et0_mgmt` and `/dev/et0_ops` were character devices.
- PCI function `0000:01:00.0` reported vendor/device `1e0a:eb01`.
- The runtime log reported `DevicePcie`, `PCIe target: /dev/et0_ops`,
  `Architecture revision: ETSOC1`, and `Form Factor: PCIE`.
- Each `run_result.json` says `"device": "soc1sim"` and `"hardware": true`.
- Launcher identity, completion log, dump log, board reset, and dump-size
  checks were all true.

| Run | UTC start | Nodes | Result | Outer elapsed | Kernel wait | Dump |
|---|---|---|---|---:|---:|---|
| `final_board_n263_n265_reset` | 2026-07-20T09:35:18Z | N263:N265 | PASS | 10 s | 0.0947028 s | 458,752 bytes, SHA-256 `d20483b8e0f04f4e211522947145751c50b5b8a23181309454c3935af5488dfa` |
| `final_board_n263_n270` | 2026-07-20T09:36:32Z | N263:N270 | PASS | 7 s | 1.70496 s | 1,245,184 bytes, SHA-256 `f2e3af9e389932888176da8b5a34c5eeec832a99e12b7bbe8c1870d1d68586f7` |

Exact hardware tensor results:

| Run | Node | Shape | Max abs | Max rel | Mean abs | Mismatches |
|---|---:|---|---:|---:|---:|---:|
| N263:N265 | N263 | `[1,80,20,20]` | 4.76837158203125e-07 | 0.017045454545454544 | 4.355658438726095e-08 | 0 |
| N263:N265 | N264 | `[1,80,20,20]` | 1.7881393432617188e-07 | 5.180445269631815e-07 | 2.2843945771455766e-08 | 0 |
| N263:N265 | N265 | `[1,80,20,20]` | 5.960464477539062e-07 | 0.01704558360641582 | 2.8660296791827024e-08 | 0 |
| N263:N270 | N263 | `[1,80,20,20]` | 4.76837158203125e-07 | 0.017045454545454544 | 4.355658438726095e-08 | 0 |
| N263:N270 | N264 | `[1,80,20,20]` | 1.7881393432617188e-07 | 5.180445269631815e-07 | 2.2843945771455766e-08 | 0 |
| N263:N270 | N265 | `[1,80,20,20]` | 5.960464477539062e-07 | 0.01704558360641582 | 2.8660296791827024e-08 | 0 |
| N263:N270 | N266 | `[1,80,20,20]` | 1.0132789611816406e-05 | 0.005146748310810811 | 7.666284800507128e-07 | 0 |
| N263:N270 | N267 | `[1,80,20,20]` | 1.8775463104248047e-06 | 1.005196184063665e-05 | 1.0814226698130369e-07 | 0 |
| N263:N270 | N268 | `[1,80,20,20]` | 5.245208740234375e-06 | 0.0051460312422438575 | 6.008876626992787e-07 | 0 |
| N263:N270 | N269 | `[1,80,20,20]` | 1.33514404296875e-05 | 8.199861070925284e-07 | 2.1297037601470947e-06 | 0 |
| N263:N270 | N270 | `[1,144,20,20]` | 1.33514404296875e-05 | 8.199861070925284e-07 | 1.1831687556372748e-06 | 0 |

Every hardware tensor had zero mismatches and zero non-finite
actual/reference values. The larger contiguous sample proves the selected
depthwise Conv/Sigmoid/Mul, pointwise Conv/Sigmoid/Mul, final class Conv, and
regression/class Concat in graph order using real ONNX weights and captured
external activations.

## Preserved schema-v1 PMC measurements

PMC collection begins immediately before `yr_run_selected` and ends
immediately after it. Input loading, launcher startup, result dumping, and host
comparison are outside the measured region. The decoded `YRPM` version-1
record was 4,816 bytes, selected one active hart, requested both shared-counter
groups, and passed its magic, endian, layout, and counter checks.

Per-hart deltas:

| Target/range | PMC offset | Minion cycles | Retired instructions T0 | Retired instructions T1 | L2 miss requests | Minion I-cache requests | I-cache ETLink requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| sys-emu N263:N265 | 393,216 (`0x60000`) | 11,423,376 | 11,423,375 | 0 | 0 | 0 | 0 |
| sys-emu N266:N268 | 393,216 (`0x60000`) | 122,438,114 | 122,438,113 | 0 | 0 | 0 | 0 |
| ET-SoC1 N263:N265 | 393,216 (`0x60000`) | 38,247,211 | 11,423,375 | 0 | 183,607 | 3,160,936 | 39 |
| ET-SoC1 N263:N270 | 1,179,648 (`0x120000`) | 977,878,949 | 253,332,210 | 0 | 5,839,297 | 81,947,425 | 49 |

The system emulator decoded the SC and memory-shire groups as supported
(`SC mask 0x00000fff`, `MS mask 0x00ffffff`) but returned zero for all 12 SC
and all 24 MS deltas. That is an emulator limitation/behavior and must not be
read as measured zero hardware traffic.

Exact ET-SoC1 shared-counter deltas are preserved here. Each SC row is
`cycles / all_l2_reads / all_l2_writes`:

| SC block | N263:N265 | N263:N270 |
|---:|---|---|
| 0 | 38,308,254 / 54,205 / 27,294 | 977,843,515 / 1,503,590 / 109,385 |
| 1 | 38,307,296 / 57,953 / 9,773 | 977,842,835 / 1,507,198 / 30,934 |
| 2 | 38,307,230 / 37,282 / 8,163 | 977,842,796 / 1,426,058 / 26,439 |
| 3 | 38,307,232 / 37,073 / 9,859 | 977,842,766 / 1,425,494 / 30,976 |

Each memory-shire row is `cycles / all_mesh_reads / all_mesh_writes`:

| Memory-shire block | N263:N265 | N263:N270 |
|---:|---|---|
| 0 | 59,565,821 / 68 / 0 | 1,520,519,367 / 165 / 0 |
| 1 | 59,565,245 / 0 / 0 | 1,520,504,357 / 0 / 0 |
| 2 | 59,564,647 / 0 / 0 | 1,520,489,089 / 0 / 0 |
| 3 | 59,564,013 / 0 / 0 | 1,520,473,830 / 0 / 0 |
| 4 | 59,566,179 / 1 / 0 | 1,520,534,648 / 1 / 0 |
| 5 | 59,565,572 / 0 / 0 | 1,520,519,353 / 0 / 0 |
| 6 | 59,565,024 / 0 / 0 | 1,520,504,185 / 0 / 0 |
| 7 | 59,564,416 / 0 / 0 | 1,520,488,887 / 0 / 0 |

These are unoptimized single-hart scalar-reference measurements. They are
correctness instrumentation, not a latency or throughput baseline.

## Compiler, launcher, and ELF identity

The ET binaries were compiled through the supported containerized toolchain;
an incompatible host compiler was not substituted.

| Identity | Value |
|---|---|
| Compiler | `riscv64-unknown-elf-gcc (g5115c7e44) 15.2.0` |
| Container tag | `et-gcc:24.04` |
| Container image ID | `sha256:6a811b9dcb63231c903d837fd969fbcee64aa2a6e8d685b8e0af3f9d92cfaa67` |
| ISA / ABI | `-march=rv64imfc -mabi=lp64f` |
| Correctness flags | `-O1 -fno-fast-math -ffp-contract=off -fno-tree-vectorize` |
| Link mode | `-nostdlib`, repository ET CRT/layout/linker, `region0_size=0x00400000` |
| Launcher | Provisioned `$LAUNCHER` (`erbium_soc1sim_argbuf_dynmem`) |
| Launcher bytes / SHA-256 | 98,008 / `fa2e63f1fd1e2cfd2dd6e29106e57de7d208c6e64c391098115235a7a98bfb5f` |
| ET platform | Provisioned `$ET_PLATFORM` tree |

| Selected range | ELF bytes | ELF SHA-256 | Used on |
|---|---:|---|---|
| N263:N265 | 34,192 | `88942abcef13fa9372a90fe12b4b290e655c017d8dec0af0e2e60ec6f72627ac` | sys-emu and ET-SoC1 |
| N266:N268 | 34,192 | `dc5258cee14c715d0864115ef52f1a6b58ecaa58d58804b5b9bdb25a2e9f5507` | sys-emu |
| N263:N270 | 35,080 | `0441a30f342b6d654861de2027e1b785217a3c2eecb87d3c18db651169b08f75` | ET-SoC1 |
| N000:N307 real-image v3 | 177,864 | `278e1769020036ee835ba8838c1595f9f48d54dcd7ec6ff2c743fbff688409cc` | ET-SoC1 full graph |

The complete compiler argv and SHA-256 identities for the manifest header,
linker script, layout, CRT, runtime sources, PMC header, runner, and ELF are in
each final run's `build_record.json`.

## Repository integration check

The repository selector was exercised directly against this new root:

```bash
python3 .github/ci/scripts/changed_benchmark_models.py \
  --changed-file ported_models/yolov10n_hf_reference/README.md
```

It reported `unregistered ports: yolov10n_hf_reference` and selected no
benchmark model. That remains the intended result. All operators and the
complete host and real-hardware correctness graphs now pass. This scalar port
still must not be routed into the leaderboard implicitly: registration,
benchmark policy, and scoring would be a separate deliberate change and were
not requested or performed here.

The full `.github/ci/scripts/ci_preflight.sh` was also attempted on the
repository's Python 3.8 board-host environment. Port-specific shell syntax,
ShellCheck, Python 3.8 compilation, JSON parsing, host builds, and all result
validators passed. The repository-wide preflight itself did not complete:
untouched CI scripts use Python features unavailable in 3.8, including
`str.removeprefix` and subscripted built-in types. Those unrelated CI files
were not changed as part of this port.

## Failed first hardware attempt and reset resolution

The first N263:N265 hardware attempt at 2026-07-20T09:33:42Z is intentionally
preserved as `failed_board_n263_n265_pre_reset`. It used the same 34,192-byte
ELF and the same real-PCIe launcher as the later passing run. Hardware discovery
succeeded: the log entered `DevicePcie`, selected `/dev/et0_ops`, and reported
`Architecture revision: ETSOC1`. Runtime initialization then aborted with:

```text
FATAL: Unbalanced number of abort unblockers.
```

The process returned 250 after 3 seconds. It failed before `Kernel loaded`,
before kernel launch, and before any dump was created, so this was a board
runtime-state failure rather than a C-kernel numerical failure.

The resolution was to keep the repository board lock and add the same sysfs
reset used by the canonical benchmark workflow inside that lock:

```text
board_lock.py
  -> bounded outer timeout
  -> board_reset_and_run.sh
  -> write 1 to .../soc_reset/reinitiate
  -> wait 2 seconds
  -> erbium_soc1sim_argbuf_dynmem --device soc1sim ...
```

The passing log explicitly records:

```text
Resetting ET-SoC1 via /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate
Kernel completed successfully
```

The rerun then passed all three tensors with zero mismatches, and the following
N263:N270 hardware run passed as well. Future troubleshooting should retain the
lock, bounded timeouts, and canonical reset; repeated initialization aborts
should not be misdiagnosed as model arithmetic failures when the kernel never
loads.

## Scope and limitations

- The implemented ET reference path is scalar FP32, single-active-hart, and
  intentionally has no VPU, TFMA, fusion, tiling, threading, or fast-math
  transformations.
- Host validation covers all 308 nodes, all 22 graph operator types, the
  complete real-image path, 16 full-graph checkpoints, and every node output
  across the 21-range matrix.
- Current schema-v2 target validation covers the terminal `N289:N307` range in
  the system emulator. Complete 21-range system-emulator coverage is in
  progress, not yet claimed.
- Full-graph real-image hardware validation covers all 308 nodes, 16
  checkpoints, strict `output0`, result integrity, and seven stage PMCs. The
  preserved silicon slices remain narrower regression tests.
- The late N263:N270 sample validates a useful 20x20 P5 detection-head segment.
  It remains useful regression evidence but is no longer the host
  implementation boundary.
- Range packages start from deterministic or ORT-captured exact-graph boundary
  activations. The separate real-image full package proves preprocessing
  through final records on the host.
- The runtime supports graph-confirmed operator forms only. Anything outside
  those checked types, attributes, shapes, or numerical conventions fails
  explicitly.
- Outer elapsed times include runtime and I/O overhead. Only the PMC interval
  is scoped to selected-node execution.
- The cached ONNX, packed blobs, captured activations, ELFs, dumps, and raw logs
  are ignored local artifacts and are not expected to be committed.

## Preserved raw artifacts

All paths below are repository-relative and intentionally live under the
ignored `local-artifacts/` cache.

Source and packaged weights:

```text
local-artifacts/yolov10n_hf_reference/model.onnx
local-artifacts/yolov10n_hf_reference/package/weights.bin
```

Captured slices and host comparisons:

```text
local-artifacts/yolov10n_hf_reference/slices/n000_n002_stem_conv_silu/
local-artifacts/yolov10n_hf_reference/slices/n263_n265_dw_silu/
local-artifacts/yolov10n_hf_reference/slices/n266_n268_pw_silu/
local-artifacts/yolov10n_hf_reference/slices/n263_n270_p5_class_join/
```

Schema-v2 full packages and complete host range matrix:

```text
local-artifacts/yolov10n_hf_reference/full_graph/deterministic_full308_v3/
local-artifacts/yolov10n_hf_reference/full_graph/coco_room_000139_full308_v3/
local-artifacts/yolov10n_hf_reference/ranges_v2/n000_n002/
...
local-artifacts/yolov10n_hf_reference/ranges_v2/n289_n307/
```

Current schema-v2 system-emulator tail evidence:

```text
local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu/n289_n307_tail/
local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu/n289_n307_tail_cast32/
local-artifacts/yolov10n_hf_reference/results/full_coverage_sys_emu/n289_n307/
```

Complete real-image ET-SoC1 evidence:

```text
local-artifacts/yolov10n_hf_reference/results/full_board/coco_room_000139_full308_v3/
```

Final target runs:

```text
local-artifacts/yolov10n_hf_reference/results/final_sys_emu_n263_n265/
local-artifacts/yolov10n_hf_reference/results/final_sys_emu_n266_n268/
local-artifacts/yolov10n_hf_reference/results/final_board_n263_n265_reset/
local-artifacts/yolov10n_hf_reference/results/final_board_n263_n270/
```

Failed pre-reset attempt:

```text
local-artifacts/yolov10n_hf_reference/results/failed_board_n263_n265_pre_reset/
```

Each final result directory preserves `run_result.json`, `tensor_compare.json`,
`pmc.json`, `build_record.json`, the exact launcher and wrapper command,
environment and device evidence, `run.log`, and the raw `dump.bin`. Hardware
runs also preserve `board_lock.log`. The failed pre-reset directory has no
dump, tensor comparison, or PMC result because execution never reached kernel
load.
