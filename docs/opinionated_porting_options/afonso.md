# Afonso Opinionated CORE-ET Porting Guide

This is the silicon-port workflow for the checked-in CORE-ET reference models
running on ET-SoC1 boards.

The goal is not to ask a model to write an end-to-end model port. The useful
unit of work is one operation, one layer, one block, or one audit boundary at a
time.

## Practical Unit Of Work

A useful porting task has a narrow boundary:

1. Name the op, layer, block, or stage.
2. Record the exact input and output shape.
3. Record dtype, scale, zero point, and layout.
4. Make a host reference for the boundary.
5. Write or modify the ET-SoC1 kernel for that boundary.
6. Compare silicon output against the host reference.
7. Measure wall time and PMCs.
8. Journal the result in that model's `docs/optimizations.md`.

Do not try to port a whole graph in one shot. Map operations first, then stitch
validated pieces together.

## Two Measurements

Every iteration needs two independent measurements:

| Measurement | Purpose | Typical source |
|---|---|---|
| Correctness | Proves the kernel computed the intended result | host ONNX/ORT, numpy, raw-int8 host executor, saved taps |
| Performance | Shows what to optimize next | wall time, per-stage PMC deltas, cycle counters |

Correctness gates performance. A faster wrong kernel is not progress.

## PMC And Layer Performance

For layer-by-layer performance, write a compact summary from the kernel:

- stage id
- start cycle
- end cycle
- cycle delta
- optional HPM counters
- optional byte or tile counts

Then compare stages in a TSV or compact binary summary. The exact format is less
important than consistency across variants.

Useful derived columns:

| Column | Formula |
|---|---|
| cycles | `end_cycle - start_cycle` |
| ms | `cycles / measured_freq_hz * 1000` |
| mac_per_cycle | `macs / cycles` |
| bytes_per_cycle | `bytes_moved / cycles` |

Use PMCs to rank bottlenecks, not to explain everything. If wall time and PMCs
disagree, investigate the measurement setup before optimizing.

## Host Reference

The host side is the source of truth for one boundary at a time:

- ONNXRuntime for graph or layer outputs.
- numpy for explicit math checks.
- raw-int8 host executors when the silicon path uses packed quantized tensors.
- saved taps for intermediate YOLO boundaries.

Host reference output should be deterministic and small enough to compare often.
For large outputs, compare both summary metrics and selected slices.

Recommended correctness metrics:

- `max_abs`
- `mean_abs`
- `topk` agreement
- exact token ids
- detection count and coordinates
- selected activation taps

## Silicon Work

The silicon kernel owns the hardware-specific decisions:

- packed tensor layout
- VPU or TFMA path
- DMA/cache behavior
- scratch layout
- output dump format
- PMC and stage summaries

Keep the first version boring. Add one optimization per variant after the audit
passes.

## Operation Mapping

Map model ops to ET-SoC1 kernels by shape and data movement, not by model name.

| Operation | First implementation target |
|---|---|
| Conv2D 3x3 | padded INT8/FP16 tile path |
| Conv2D 1x1 | contiguous channel tile path |
| MatMul/GEMM | block kernel with packed weights |
| LayerNorm | row kernel with stable scalar fallback, then vectorize |
| Softmax | row/block kernel with bounded working set |
| SiLU/GELU | fuse into producer or consumer when the boundary is audited |
| Resize/reshape/split/concat | prefer view or copy-minimal layout changes |
| Detection postprocess | audit taps before parallelizing |

## Optimization Order

Use this order unless the model journal proves a better path:

1. Make the boundary correct.
2. Remove avoidable layout conversions.
3. Choose the compute path.
4. Pack weights once.
5. Tile for local reuse.
6. Add vector or matrix instructions.
7. Fuse adjacent elementwise ops.
8. Reduce dump and transfer overhead.
9. Batch variants.
10. Promote only audited wins.

## Iteration Loop

For each candidate variant:

1. Build the kernel.
2. Run the smoke input.
3. Pull the compact summary and output dump.
4. Compare against host reference.
5. Capture wall time and PMCs.
6. Append the result to `docs/optimizations.md`.
7. Keep the best passing variant as the new baseline.

Suggested journal row:

```text
variant | change | correctness | wall_ms | cycles | pmc_notes | decision
```

Failed variants are useful if the failure is recorded clearly. Do not leave a
failed idea undocumented and rediscover it later.

## Sys-Emu Runs

Use the checked-in script for a local smoke or full pass:

```bash
scripts/run_sysemu_model_ports.sh --suite smoke --limit-per-model 1 --list
scripts/run_sysemu_model_ports.sh --suite smoke --limit-per-model 1
scripts/run_sysemu_model_ports.sh --suite full
```

The script covers the current model set:

- `yolo`

If a model requires external reference files, put them under `local-artifacts/`.

## Hugging Face References

Do not commit generated metadata JSON or downloaded model packages. Fetch pinned
Hugging Face reference packages into ignored local artifacts:

```bash
scripts/download_hf_refs.sh local-artifacts/hf_refs
```

See [`../HF_REFERENCES.md`](../HF_REFERENCES.md) for the current showcase base
models. New ports should document their Hugging Face repo, revision, filename,
license, and any conversion step.

## Promotion Rule

A variant is promotable only when all of these are true:

- It passes the model-specific correctness gate.
- It improves the target performance metric or removes real complexity.
- Its inputs, outputs, and artifact requirements are documented.
- Its result is recorded in the model's `docs/optimizations.md`.

If the correctness threshold changes, record why. Do not silently relax audit
criteria to keep an optimization.

## Quick Checklist

Before claiming a port works:

- Host reference exists for the claimed boundary.
- Silicon output was compared against that reference.
- Wall time was measured.
- PMCs or stage timings identify the current bottleneck.
- External references needed to reproduce are downloaded under
  `local-artifacts/`.
- The model's `docs/optimizations.md` says what was tried and what won.
