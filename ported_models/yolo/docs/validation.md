# YOLO End-to-End Validation

The benchmark loads:

- `yolo/weights_region.bin` at `0x02000000`
- one of five raw RGB images at `0x04A00000`

The committed image cases are:

- `coco_room_000139`
- `coco_cat_524280`
- `coco_giraffes_296969`
- `coco_elephants_445248`
- `coco_baseball_043816`

All five are public COCO `val2017` images. Their image IDs, source URLs, source
hashes, committed raw-fixture hashes, and resize recipe are fixed in
`.github/ci/reference/yolo.json`; CI rejects a case whose bytes do not match.

The kernel writes detections at `0x01D00000`:

```c
uint32_t count;
struct {
    uint32_t class_id;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
} detections[count];
```

Before the board run, `run_yolo_host_reference.sh` downloads and hash-checks
`kadirnar/yolov10n@9fa42234fbcdb13b78fa57ebaac6c50e6dd2eb21`, then runs it on
the host with the same preprocessing as the C kernel. The generated reference
JSON is retained in the benchmark output. For detections with score at least
0.35, every case requires:

- precision and recall of 1.0 after class-aware matching at IoU 0.5
- mean matched IoU of at least 0.85
- mean absolute confidence error no greater than 0.10
- at least one reference detection

This catches zero activations, missing detections, invented categories, shifted
boxes, and material confidence drift. The contract constrains observable model
behavior, not the storage layout, so fused weights and scales remain valid when
the board output passes.

Score artifacts and leaderboard entries carry the SHA256 of this validation
contract. The leaderboard compares runtimes only within the same contract, so a
fixture or threshold change cannot be judged against an incompatible baseline.

Normal YOLO optimization PRs may change the implementation source, packed
weights, and source-owned offsets. The host oracle, scorer, runner, load map,
contract, and five COCO fixtures are CI-owned and protected from participant
changes.

The trusted PR gate constructs its candidate from current `main`, then applies
only regular C/C++/assembly/include files under `ported_models/yolo/src/` and
the exact packed `ported_models/yolo/assets/yolo/weights_region.bin` artifact.
Everything used to decide correctness and leaderboard eligibility comes from
main. Changes elsewhere in the participant branch can still run in branch CI,
but they cannot alter the trusted YOLO result.

The trusted result records the participant SHA, main harness SHA, synthetic
candidate SHA, and validation-contract hash. It fails if the compatible
main-branch leaderboard baseline is missing or if any trusted YOLO input changes
during the board run. Unrelated main changes do not make a result stale. A full
GitHub job re-run evaluates the same participant patch against the new main
harness without requiring a rebase unless the patch no longer applies. Main
pushes that change trusted YOLO inputs automatically re-run the latest trusted
Actions attempt for each open YOLO PR. The main-owned caller writes the pending
and final `trusted-yolo/main-gate` status directly on the participant head SHA.
