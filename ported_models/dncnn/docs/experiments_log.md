# DnCNN int8 kernel — experiment log

A running record of each optimization attempt on `dncnn3_tfma_int8.c`: the hypothesis, the
change, the **board** result, and the one thing we learned. Newest first. This is the plan
(`IMPROVEMENT_PLAYBOOK.md`) meeting reality — read the consolidated lessons before proposing
the next lever.

All results are 8-hart canonical board runs (`ivan@aifoundry2`, `soc1sim`) on the PMC build
(`int8_pmc.elf`, ~3 % probe overhead), compared like-for-like. Correctness gate = `max_abs=0`.

## Baseline (board-verified submission)

OC-major kernel, PMC build: **wall 10.73 ms**, ~15.9 M retired-inst/hart (127.1 M total),
hart0 **IPC 2.64**, MAC/retired-inst 0.23, throughput 2.75 GMAC/s, L2 acc/MAC 0.018,
DDR 37.5k rd / 36.4k wr. This is the number every experiment must beat on `kernel_wait_s`.

## Consolidated lessons (read first)

- **L1 — Measure, and measure the right axis.** The PMC pass showed the kernel is *not* stall-idle
  (hart0 IPC 2.64 is high) but **instruction-heavy** (0.23 MAC/retired-inst → ~4.3 scalar
  instructions per MAC). That reframed the plan away from "hide cache stalls."
- **L2 — But there is a memory ceiling right under the instruction ceiling.** Cutting ~12 % of
  instructions (B4) did *not* speed the kernel up — **IPC fell 2.64 → 2.14**, i.e. the freed time
  went to memory stalls. **IPC is the tell:** if it drops when you remove instructions, you've hit
  memory. Rank levers by *bytes of cache/DRAM traffic removed*, not just instruction count.
- **L3 — Some wins are mutually exclusive.** A direct NHWC `tensor_store` (deletes the scatter)
  requires spatial-major output, which forces activations into the *sparse* A operand — quadrupling
  their eviction traffic. The old scalar scatter was the *price of dense activation packing*. You
  cannot have both the direct store and dense activations with the current tensor-engine layout.
- **L4 — `tensor_load` source stride is 64-byte-granular.** So the A operand is always one row per
  SCP line; a 16-wide int8 row uses only 16 of 64 bytes → 4× eviction waste vs a dense quartet pack.
  This constrains any layout that puts a 16-element vector on the A side.
- **L5 — The store bypasses cache.** `tensor_store` writes DRAM directly; useful to know for
  traffic accounting (it shows up as DDR writes, not L2).

## Experiments

### B4 — transposed-GEMM direct store — **REJECTED (regression)** · 2026-07-07

- **Hypothesis:** the kernel is instruction-bound (L1); a transposed GEMM makes the output store a
  direct NHWC `tensor_store` (deletes the 256-iter scalar scatter) *and* makes the activation pack a
  cheap contiguous copy (deletes the 9×256 quartet weave) → fewer instructions → faster.
- **Change:** swap FMA operand roles (activations→A, weights→B), `MUL_COL`→`MUL_ROW`, direct store
  into `padout`. Field values unchanged (`CH==P==OC==16`). Full detail: `b4_transpose_result.md`.
- **Result:** correct (`max_abs=0`), retired-inst **−11.7 %** — but wall **+8.4 %** (10.73 → 11.64 ms),
  IPC **2.64 → 2.14**, DDR traffic **+3.2×**, L2 +60 %.
- **Lesson (→ L2, L3, L4):** the instruction win was real but the transpose moved the 16-line operand
  from static weights (evicted once/layer) to per-tile activations (evicted every tap, 4× the bytes),
  and the bypass store added DRAM writes. Net memory-bound regression. **Do not pursue direct-store /
  transpose again** unless activations can stay the dense operand.

## Candidate next levers (re-ranked by the memory-traffic axis)

| Lever | Removes | Traffic effect | Notes |
|---|---|---|---|
| **B7 weight/activation residency** | per-tap/per-layer operand reloads | **cuts bytes** | keep the ~7 KB weights + reused activation window on-chip across taps |
| **B1 batch the 9 per-tap evicts** | 8 of 10 `WAIT_CACHEOPS`/tile | cuts *count*, not *bytes* | measure — won't fix a byte regression alone (L2) |
| **B6 depth-first layer fusion** | 2 of 3 inter-layer DRAM round-trips | **cuts bytes** (biggest) | keep a tile resident across the 3 hidden layers; largest restructure |
| ~~B4 direct store / transpose~~ | scatter loop | **adds bytes** | rejected — see above |
