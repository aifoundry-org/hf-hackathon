#!/usr/bin/env python3
"""Generate the architecture guide and semantic stage map from pinned ONNX data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


PORT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PORT_ROOT.parents[1]
DEFAULT_INVENTORY = PORT_ROOT / "manifests/graph_inventory.json"
DEFAULT_MODEL = REPO_ROOT / "local-artifacts/yolov10n_hf_reference/model.onnx"
DEFAULT_STAGES = PORT_ROOT / "manifests/architecture_stages.json"
DEFAULT_DOC = PORT_ROOT / "docs/ARCHITECTURE.md"
DEFAULT_EXECUTION = PORT_ROOT / "manifests/full_execution.json"
EXPECTED_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"


STAGES: Sequence[Tuple[str, str, int, int, str, str]] = (
    ("S01", "stem", 0, 5, "stride-2 learned stem and downsample", "neural compute"),
    ("S02", "backbone_c2f_p2", 6, 20, "C2f-style residual feature block", "neural compute"),
    ("S03", "backbone_p3", 21, 45, "P3 downsample and C2f-style block", "neural compute"),
    ("S04", "backbone_p4", 46, 71, "SCDown-style downsample and C2f block", "neural compute"),
    ("S05", "backbone_p5", 72, 90, "SCDown-style downsample and C2f block", "neural compute"),
    ("S06", "sppf", 91, 100, "fast spatial pyramid pooling", "neural compute"),
    ("S07", "partial_attention", 101, 128, "partial self-attention and FFN", "neural compute"),
    ("S08", "neck_top_down_p4", 129, 144, "upsample, P4 concat, C2f-style fusion", "neural compute"),
    ("S09", "neck_top_down_p3", 145, 160, "upsample, P3 concat, C2f-style fusion", "neural compute"),
    ("S10", "neck_bottom_up_p4", 161, 178, "downsample, P4 concat, C2f-style fusion", "neural compute"),
    ("S11", "neck_bottom_up_p5", 179, 207, "SCDown, P5 concat, C2f/CIB-style fusion", "neural compute"),
    ("S12", "detect_p3", 208, 228, "stride-8 one-to-one regression/class branches", "neural compute"),
    ("S13", "detect_p4", 229, 249, "stride-16 one-to-one regression/class branches", "neural compute"),
    ("S14", "detect_p5", 250, 270, "stride-32 one-to-one regression/class branches", "neural compute"),
    ("S15", "dfl_decode", 271, 288, "multiscale merge, DFL expectation, box/class decode", "in-graph output transform"),
    ("S16", "nms_free_top300", 289, 307, "two-stage TopK/GatherElements selection", "in-graph output transform"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Keep generated metadata checkout-independent when the path is in this port."""
    try:
        return str(path.resolve().relative_to(PORT_ROOT))
    except ValueError:
        return str(path)


def shape_text(shape: Any) -> str:
    if shape is None:
        return "unknown"
    return "[" + ",".join(str(item) for item in shape) + "]"


def tensor_text(item: Dict[str, Any]) -> str:
    return f"`{item['name']}` {shape_text(item.get('shape'))}"


def op_summary(nodes: Sequence[Dict[str, Any]]) -> str:
    counts = Counter(node["op_type"] for node in nodes)
    return ", ".join(
        f"{name}×{count}" for name, count in sorted(counts.items())
    )


def check_node(
    nodes: Sequence[Dict[str, Any]], index: int, op_type: str, name_fragment: str
) -> None:
    node = nodes[index]
    if (
        node["index"] != index
        or node["node_id"] != f"N{index:03d}"
        or node["op_type"] != op_type
        or name_fragment not in node["name"]
    ):
        raise ValueError(
            f"landmark mismatch at N{index:03d}: expected "
            f"{op_type}/{name_fragment}, actual {node['op_type']}/{node['name']}"
        )


def boundary_tensors(
    nodes: Sequence[Dict[str, Any]],
    first: int,
    last: int,
    initializer_names: set,
    producers: Dict[str, int],
    consumers: Dict[str, List[int]],
    graph_outputs: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage_nodes = nodes[first : last + 1]
    entries: List[Dict[str, Any]] = []
    exits: List[Dict[str, Any]] = []
    seen = set()
    for node in stage_nodes:
        for item in node["inputs"]:
            name = item["name"]
            producer = producers.get(name)
            if (
                name
                and name not in initializer_names
                and not (producer is not None and first <= producer <= last)
                and name not in seen
            ):
                entries.append(item)
                seen.add(name)
    seen.clear()
    for node in stage_nodes:
        for item in node["outputs"]:
            name = item["name"]
            users = consumers.get(name, [])
            if (
                any(user < first or user > last for user in users)
                or name in graph_outputs
            ) and name not in seen:
                exits.append(item)
                seen.add(name)
    return entries, exits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stages", type=Path, default=DEFAULT_STAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_DOC)
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=DEFAULT_EXECUTION,
    )
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    execution = json.loads(args.execution_manifest.read_text())
    model_sha = sha256_file(args.model)
    if (
        model_sha != EXPECTED_SHA256
        or inventory["source"]["sha256"] != EXPECTED_SHA256
    ):
        raise SystemExit("error: pinned model or inventory SHA-256 mismatch")
    nodes = inventory["nodes"]
    if len(nodes) != 308:
        raise SystemExit(f"error: expected 308 nodes, got {len(nodes)}")
    if (
        execution.get("schema_version") != 2
        or execution.get("manifest_kind") != "full_graph_liveness"
        or execution.get("source", {}).get("sha256") != EXPECTED_SHA256
        or execution.get("selection", {}).get("selector") != "N000:N307"
        or len(execution.get("nodes", [])) != 308
    ):
        raise SystemExit("error: full execution manifest contract mismatch")
    execution_ops = Counter(
        node["op_type"] for node in execution["nodes"]
    )
    inventory_ops = Counter(node["op_type"] for node in nodes)
    if execution_ops != inventory_ops:
        raise SystemExit(
            "error: full execution operator inventory differs from graph"
        )
    expected_checkpoints = (
        "N005", "N020", "N045", "N071", "N090", "N100", "N128", "N144",
        "N160", "N178", "N207", "N228", "N249", "N270", "N288", "N307",
    )
    checkpoint_ids = tuple(
        item["node_id"] for item in execution.get("checkpoints", [])
    )
    if checkpoint_ids != expected_checkpoints:
        raise SystemExit("error: full execution checkpoints changed")
    expected_pmc = (
        ("stem", "N000", "N005"),
        ("backbone", "N006", "N090"),
        ("sppf_psa", "N091", "N128"),
        ("neck", "N129", "N207"),
        ("three_scale_head", "N208", "N270"),
        ("dfl_decode", "N271", "N288"),
        ("topk_selection", "N289", "N307"),
    )
    actual_pmc = tuple(
        (item["name"], item["first_node"], item["last_node"])
        for item in execution.get("pmc_stages", [])
    )
    if actual_pmc != expected_pmc:
        raise SystemExit("error: full execution PMC partition changed")

    # Fail loudly if graph landmarks used by the prose no longer match.
    for index, op_type, fragment in (
        (0, "Conv", "/model.0/"),
        (1, "Sigmoid", "/model.0/act/"),
        (2, "Mul", "/model.0/act/"),
        (94, "MaxPool", "/model.9/m/"),
        (95, "MaxPool", "/model.9/m_1/"),
        (96, "MaxPool", "/model.9/m_2/"),
        (109, "MatMul", "/model.10/attn/"),
        (111, "Softmax", "/model.10/attn/"),
        (129, "Resize", "/model.11/"),
        (145, "Resize", "/model.14/"),
        (208, "Conv", "one2one_cv2.0"),
        (229, "Conv", "one2one_cv2.1"),
        (250, "Conv", "one2one_cv2.2"),
        (274, "Concat", "/model.23/Concat_3"),
        (278, "Softmax", "/model.23/dfl/"),
        (279, "Conv", "/model.23/dfl/conv/"),
        (291, "TopK", "/model.23/TopK"),
        (294, "GatherElements", "/model.23/GatherElements"),
        (298, "TopK", "/model.23/TopK_1"),
        (303, "GatherElements", "/model.23/GatherElements_2"),
        (307, "Concat", "/model.23/Concat_6"),
    ):
        check_node(nodes, index, op_type, fragment)
    if "NonMaxSuppression" in {node["op_type"] for node in nodes}:
        raise SystemExit("error: unexpected NonMaxSuppression node")

    model = onnx.load(str(args.model), load_external_data=False)
    initializers = {item.name: item for item in model.graph.initializer}
    dfl = numpy_helper.to_array(initializers["model.23.dfl.conv.weight"])
    if not np.array_equal(dfl.reshape(-1), np.arange(16, dtype=np.float32)):
        raise SystemExit("error: DFL expectation weights are not exactly 0..15")
    k_value = numpy_helper.to_array(
        initializers["/model.23/Constant_6_output_0"]
    )
    class_count = numpy_helper.to_array(
        initializers["/model.23/Constant_13_output_0"]
    )
    strides = numpy_helper.to_array(
        initializers["/model.23/Constant_5_output_0"]
    ).reshape(-1)
    if int(k_value[0]) != 300 or int(class_count) != 80:
        raise SystemExit("error: terminal k/class constants changed")
    if not (
        np.all(strides[:6400] == 8.0)
        and np.all(strides[6400:8000] == 16.0)
        and np.all(strides[8000:] == 32.0)
    ):
        raise SystemExit("error: stride constant is not 6400×8, 1600×16, 400×32")

    initializer_names = set(initializers)
    graph_outputs = {
        item["name"] for item in inventory["model"]["outputs"]
    }
    producers = {
        item["name"]: node["index"]
        for node in nodes
        for item in node["outputs"]
    }
    consumers: Dict[str, List[int]] = {}
    for node in nodes:
        for item in node["inputs"]:
            consumers.setdefault(item["name"], []).append(node["index"])

    stage_records = []
    node_to_stage = {}
    for stage_id, name, first, last, purpose, classification in STAGES:
        subset = nodes[first : last + 1]
        entries, exits = boundary_tensors(
            nodes,
            first,
            last,
            initializer_names,
            producers,
            consumers,
            graph_outputs,
        )
        record = {
            "stage_id": stage_id,
            "name": name,
            "first_node": f"N{first:03d}",
            "last_node": f"N{last:03d}",
            "node_count": last - first + 1,
            "purpose": purpose,
            "classification": classification,
            "operator_counts": dict(
                sorted(Counter(node["op_type"] for node in subset).items())
            ),
            "entry_tensors": entries,
            "exit_tensors": exits,
            "nodes": [node["node_id"] for node in subset],
        }
        stage_records.append(record)
        for index in range(first, last + 1):
            if index in node_to_stage:
                raise SystemExit(f"error: N{index:03d} belongs to two stages")
            node_to_stage[index] = stage_id
    if set(node_to_stage) != set(range(308)):
        missing = sorted(set(range(308)) - set(node_to_stage))
        raise SystemExit(f"error: stage map does not cover nodes: {missing}")

    stage_document = {
        "schema_version": 1,
        "source_sha256": EXPECTED_SHA256,
        "generated_from": display_path(args.inventory),
        "host_boundaries": {
            "preprocess": "before N000; absent from ONNX",
            "presentation": "after N307; absent from ONNX",
        },
        "stages": stage_records,
    }
    args.stages.parent.mkdir(parents=True, exist_ok=True)
    args.stages.write_text(json.dumps(stage_document, indent=2) + "\n")

    def stage_row(record: Dict[str, Any]) -> str:
        entries = "<br>".join(tensor_text(item) for item in record["entry_tensors"])
        exits = "<br>".join(tensor_text(item) for item in record["exit_tensors"])
        return (
            f"| {record['stage_id']} | `{record['first_node']}:{record['last_node']}` "
            f"| {record['name']} | {entries or '—'} | {exits or '—'} "
            f"| {', '.join(f'{k}×{v}' for k, v in record['operator_counts'].items())} "
            f"| {record['classification']} |"
        )

    stage_table = "\n".join(stage_row(record) for record in stage_records)
    memory = execution["memory_map"]
    memory_plan = execution["memory_plan"]
    execution_tensor_count = len(execution["tensors"])
    execution_checkpoint_count = len(execution["checkpoints"])
    checkpoint_lines = (
        " ".join(checkpoint_ids[:8])
        + "\n"
        + " ".join(checkpoint_ids[8:])
    )
    markdown = f"""# YOLOv10n architecture measured from the pinned ONNX

> Generated by `tools/generate_architecture.py` from
> `manifests/graph_inventory.json`, `manifests/full_execution.json`, and the
> checksum-verified ONNX. The generator aborts if its named node landmarks,
> execution operator inventory, memory/checkpoint contract, DFL weights,
> stride vector, class count, or terminal TopK constants do not match.

## Provenance and graph boundary

- Source: `onnx-community/yolov10n` at
  `57657320425ee34056408a57ad9d29c4d4815bd8`, file `onnx/model.onnx`.
- SHA-256: `{EXPECTED_SHA256}`.
- ONNX opset 13; 308 nodes; 187 initializers.
- Input: `images`, FP32 `[1,3,640,640]`.
- Output: `output0`, FP32 `[1,300,6]`.

The ONNX begins immediately with `N000` Conv. It contains no image resize,
letterbox, channel reorder, normalization, or input cast. Those choices are
host-side and cannot be inferred from this graph. Likewise, thresholding,
mapping boxes back to an original image, class labels, drawing, and
serialization are host-side after `N307`.

### Reproducible real-image boundary used for validation

The repository's checked COCO-room fixture makes one concrete host policy
reproducible without pretending it is part of the model. Its raw input is
480×640 HWC RGB UINT8. `tools/preprocess_coco_room.py` centers it unchanged in
a 640×640 RGB canvas (source rows 80 through 559), fills the top and bottom
with RGB 114, divides by FP32 255, and transposes to `[1,3,640,640]` NCHW.
The script checks both the raw and resulting FP32 SHA-256 values.

Decoded coordinates remain in that 640×640 canvas. To map a record back to
this particular raw fixture, x is unchanged and 80 is subtracted from y,
followed by clipping to the 640×480 source bounds. That inverse is fixture
specific; it is not an ONNX node.

## Measured execution map

| Stage | Nodes | Part | Entry tensor(s) | Exit tensor(s) | Operators | Classification |
|---|---|---|---|---|---|---|
{stage_table}

The learned network is `N000:N270`. `N271:N307` is still inside ONNX, but it is
decode and selection rather than learned feature extraction. No
`NonMaxSuppression` operator or IoU calculation exists.

## Scalar C execution and memory architecture

The correctness runtime implements all 22 operator types in this graph:
Conv×83, Sigmoid×70, Mul×71, Concat×21, Split×13, Add×11, Reshape×8,
Transpose×4, MaxPool×3, Tile×3, GatherElements×3, MatMul×2, Softmax×2,
Resize×2, TopK×2, and one each of Sub, ReduceMax, Flatten, Mod, Div, and Cast.
The generated schema-v2 header contains declarative tensor/node records only;
the scalar kernels remain readable hand-written C. Every node executes in
ONNX order, and unsupported attributes, types, shapes, or broadcasts return
an explicit status.

`tools/generate_full_graph.py` derives a deterministic, 64-byte-aligned
first-fit liveness plan from producers and last consumers. Outputs are
allocated before inputs whose last use is the current node are released, so
in-place aliasing is not assumed. The manifest pins
{execution_checkpoint_count} comparison checkpoints:

```text
{checkpoint_lines}
```

Every workspace tensor record exposes its producer, `live_start`, `live_end`,
offset, logical bytes, allocated bytes, and checkpoint flag. The accompanying
event list records each allocation and release in node order, so the entire
arena plan is auditable without reverse-engineering the generated C header.

The measured manifest has {execution_tensor_count} tensor descriptors:

| Memory fact | Bytes |
|---|---:|
| Liveness arena | {memory_plan["arena_bytes"]:,} |
| Launcher dump | {memory["dump_size"]:,} |
| FP32 input blob | {memory["input_blob_bytes"]:,} |
| Input device offset | {memory["input_device_offset"]:,} |
| Aligned weight blob | {memory["weight_blob_bytes"]:,} |
| Weight device offset | {memory["weight_device_offset"]:,} |
| Total target allocation | {memory["mem_size"]:,} |

The dump starts with the versioned `YRF1` result and then the arena; seven
separate 64 KiB PMC slots follow the arena. Inputs and weights live after the
dump, so launcher I/O does not overwrite results.

The seven PMC intervals partition the graph without setup work:

| PMC stage | Inclusive nodes | Architectural contents |
|---|---|---|
| stem | `N000:N005` | two stride-2 Conv/SiLU steps |
| backbone | `N006:N090` | C2f-style features and P3/P4/P5 downsampling |
| SPPF/PSA | `N091:N128` | serial pooling and partial attention |
| neck | `N129:N207` | top-down and bottom-up multiscale fusion |
| three-scale head | `N208:N270` | P3/P4/P5 regression and classification |
| DFL/decode | `N271:N288` | DFL expectation, boxes, and class sigmoid |
| TopK selection | `N289:N307` | NMS-free shortlist and final records |

Each interval begins immediately before its first ONNX node and ends
immediately after its last. Input loading, launcher startup, dumping, and host
comparison are outside the measurement.

## Backbone: `N000:N128`

- `N000:N005` is the two-step stride-2 stem:
  `[1,3,640,640] → [1,16,320,320] → [1,32,160,160]`.
  Each learned Conv is followed by distinct Sigmoid and Mul nodes; the graph
  represents SiLU as `x * sigmoid(x)`, not a fused operator.
- `N006:N020` (`/model.2`) is a C2f-style block. A 1×1 Conv-SiLU is split
  `32→16+16`; the second half traverses two 3×3 Conv-SiLU nodes plus residual
  Add; the two original halves and residual result concatenate to 48 channels,
  then a 1×1 Conv-SiLU fuses back to 32.
- `N021:N045` produces P3 at `[1,64,80,80]`. `/model.4` repeats the split
  shell with two chained residual bottlenecks: four 32-channel tensors
  concatenate to 128 before the 1×1 fuse.
- `N046:N071` produces P4 at `[1,128,40,40]`. `N046:N049` first expands
  `64→128` with 1×1 Conv-SiLU, then uses a grouped 3×3 stride-2 Conv with
  `group=128`; this is graph evidence for an SCDown-style pointwise/depthwise
  downsample. `/model.6` then uses the C2f-style shell with two residual
  bottlenecks.
- `N072:N090` similarly produces P5 at `[1,256,20,20]`: pointwise expansion,
  depthwise stride-2 Conv (`group=256`), then a one-bottleneck C2f-style block.

“C2f-style” describes a directly observed split → bypass/transform → concat →
fuse topology. It is not an ONNX op name. Backbone bottlenecks contain Add
nodes, confirming shortcuts. The neck C2f-shaped blocks below omit those Adds,
so their bottleneck shortcuts are disabled in this export.

### SPPF: `N091:N100`

`N091:N093` reduces 256 to 128 channels at 20×20. `N094`, `N095`, and `N096`
are three serial 5×5 MaxPool nodes with stride 1 and pad 2. `N097` concatenates
the unpooled tensor and all three pool depths to `[1,512,20,20]`; `N098:N100`
fuses that to `[1,256,20,20]`. This exact serial pooling topology confirms the
fast spatial-pyramid structure.

### Partial attention: `N101:N128`

`N101:N104` applies 1×1 Conv-SiLU and splits 256 channels into two 128-channel
halves. One bypasses attention. For the processed half:

- `N105:N107` makes QKV, reshapes to `[1,2,128,400]`, and splits per head into
  Q `[1,2,32,400]`, K `[1,2,32,400]`, V `[1,2,64,400]`.
- `N108:N114` forms attention scores `[1,2,400,400]`, scales by
  `0.1767766923` (`1/sqrt(32)`), applies Softmax, multiplies by V, and restores
  `[1,128,20,20]`.
- `N115:N119` adds a depthwise 3×3 positional Conv (`group=128`) and a learned
  1×1 projection with residual Add.
- `N120:N124` is a 1×1 FFN `128→256→128` with SiLU and a second residual Add.
- `N125:N128` concatenates the bypass/processed halves and fuses back to
  `[1,256,20,20]`.

The graph names explicitly say `attn` and `ffn`; “PSA-like” is the architectural
description of that measured partial-channel wiring.

## Bidirectional neck: `N129:N207`

- `N129` nearest-neighbor resizes P5 `256×20² → 256×40²`; `N130` concatenates
  backbone P4 to 384 channels; `N131:N144` fuses to `[1,128,40,40]`.
- `N145` nearest-neighbor resizes `128×40² → 128×80²`; `N146` concatenates
  backbone P3 to 192 channels; `N147:N160` fuses to `[1,64,80,80]`.
- `N161:N178` runs bottom-up: stride-2 Conv-SiLU to 40×40, concat with the
  top-down P4, then fuse to `[1,128,40,40]`.
- `N179:N207` uses pointwise Conv-SiLU plus depthwise stride-2 Conv
  (`group=128`), concatenates attention-refined P5, and fuses to
  `[1,256,20,20]`.

The `/model.22` path `N188:N202` is CIB-like by direct kernel/group evidence:
DW3×3 (`g=128`) → PW1×1 (`128→256`) → DW7×7 (`g=256`) → PW1×1
(`256→128`) → DW3×3 (`g=128`), followed by residual Add `N203`, concat `N204`,
and 1×1 Conv-SiLU fuse `N205:N207`.

## One-to-one multiscale head: `N208:N270`

The node names explicitly contain `one2one`; there is no one-to-many branch in
this inference graph.

| Scale | Feature | Regression nodes/output | Classification nodes/output | Join |
|---|---|---|---|---|
| P3 / stride 8 | `[1,64,80,80]` | `N208:N214` → 64 | `N215:N227` → 80 | `N228` → `[1,144,80,80]` |
| P4 / stride 16 | `[1,128,40,40]` | `N229:N235` → 64 | `N236:N248` → 80 | `N249` → `[1,144,40,40]` |
| P5 / stride 32 | `[1,256,20,20]` | `N250:N256` → 64 | `N257:N269` → 80 | `N270` → `[1,144,20,20]` |

Each 64-channel regression output is `4×16` DFL logits. Each classification
output has 80 direct class logits; there is no extra objectness channel.
Classification branches use depthwise/pointwise Conv-SiLU paths and a final
1×1 Conv. These nodes are learned neural compute.

## DFL and box decode: `N271:N288`

`N271:N274` reshapes the three scales to `[1,144,6400]`,
`[1,144,1600]`, `[1,144,400]` and concatenates them to
`[1,144,8400]`. `N275` splits 64 regression and 80 class channels.

`N276:N280` reshapes regression to `[1,4,16,8400]`, transposes, applies
Softmax across the 16 bins, then runs a fixed 1×1 Conv whose verified weights
are exactly `[0,1,…,15]`. This is the expectation of each distance
distribution. `N281:N285` computes

```text
(grid_x - d0, grid_y - d1, grid_x + d2, grid_y + d3) × stride
```

for a verified stride vector of 6400 eights, 1600 sixteens, and 400
thirty-twos. `N286` applies class sigmoid; `N287:N288` produces
`[1,8400,84]`. Decode is in-graph output transformation, not host processing.

## NMS-free Top-300 selection: `N289:N307`

1. `N289` splits boxes `[1,8400,4]` and scores `[1,8400,80]`.
2. `N290:N291` ReduceMax across classes and TopK with verified `k=300` select
   the best 300 candidate locations.
3. `N292:N296` Tile/GatherElements collect their boxes and all 80 scores.
4. `N297:N298` flatten the 300×80 scores and take a second global TopK 300.
5. `N299 Mod 80` recovers class IDs; `N300 Div 80` recovers shortlist slots.
6. `N301:N306` gather boxes and form FP32 score/class columns.
7. `N307` concatenates `[box4, score1, class1]` as `output0 [1,300,6]`.

This path has no score threshold, IoU calculation, or NonMaxSuppression node.
It always emits 300 sorted candidate-class entries. A host may filter them or
map coordinates back through its preprocessing transform, but those policies
are outside this ONNX.

Each final FP32 row is `[x1, y1, x2, y2, score, class_id]` in the 640×640
input-canvas coordinate system. `class_id` is numerically integral but is
FP32 because `N306` casts it before `N307` concatenates one homogeneous
tensor. The graph proves 80 class indices, but human-readable label strings
are external metadata. Repeated or overlapping boxes are expected because
this export does not perform IoU suppression.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(
        f"ARCHITECTURE PASS stages={len(stage_records)} nodes={len(node_to_stage)} "
        f"doc={args.output} map={args.stages}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
