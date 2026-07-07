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
  went to memory stalls. **IPC is the tell:** if it drops when you remove instructions, a stall cost
  ate the saving. (First read as "rank by bytes removed"; B3 later disproved even that — see L6/L7.)
- **L3 — Some wins are mutually exclusive.** A direct NHWC `tensor_store` (deletes the scatter)
  requires spatial-major output, which forces activations into the *sparse* A operand — quadrupling
  their eviction traffic. The old scalar scatter was the *price of dense activation packing*. You
  cannot have both the direct store and dense activations with the current tensor-engine layout.
- **L4 — `tensor_load` source stride is 64-byte-granular.** So the A operand is always one row per
  SCP line; a 16-wide int8 row uses only 16 of 64 bytes → 4× eviction waste vs a dense quartet pack.
  This constrains any layout that puts a 16-element vector on the A side.
- **L5 — The store bypasses cache.** `tensor_store` writes DRAM directly; useful to know for
  traffic accounting (it shows up as DDR writes, not L2).
- **L6 — We are NOT bandwidth-bound; the kernel scales.** The 1/2/4/8-hart sweep gives 6.98×
  speedup / 87 % efficiency at 8 harts, so the single shire's L2/DDR is not saturated — the
  per-hart bottleneck is instruction volume. The ~13 % droop 4→8 is **not** hart0's serial evict:
  B3 removed exactly that and *regressed*, so the droop is barrier-sync latency that grows with hart
  count, not a fixable serial pass. Net: **off-critical-path traffic cuts don't move wall.**
- **L7 — Judge wall on the plain build, interleaved, multi-run.** `run_ab_scored.sh` (no probe,
  base-vs-change interleaved, medians) matches CI — baseline median 10.44 ms ≈ CI 10.4 ms. Use PMC
  builds for *counters*, plain A/B for *wall* (the probe adds ~3 %). Single manual runs carry ~1 %
  spread; require **non-overlapping** distributions before believing a wall delta.
- **L8 — The lever that worked: cut critical-path *dispatch* overhead, not bytes.** B8 folded the 9
  per-tap FMA dispatches into 3 → **−4.3 % wall** (non-overlapping A/B). It removed per-tap
  dispatch/load/`WAIT` sets 9→3 while keeping the same pack work, same evict bytes, and the baseline
  weights=A/activations=B layout. So there is a **third axis** — per-tile fixed *tensor-op* overhead —
  distinct from instruction *volume* (the pack, untouched) and *traffic* (unchanged). "Near the floor"
  (the post-B3 read) was premature: the floor was in the axes B4/B3 hit, not in per-tile overhead.
- **L9 — "Near the floor" was a false generalization; name the axis each experiment tested.** Two
  regressions (B4, B3) were over-generalized to "all single-axis pushes fail." But **B4 tested
  layout-changing instruction cuts** (which add traffic) and **B3 tested off-critical-path traffic** —
  *neither touched per-tile dispatch overhead*, the axis B8 then won on. Two compounding mistakes:
  (a) collapsing distinct mechanisms into one "instructions vs traffic" verdict; (b) reading high
  hart0 IPC (2.64) as "waits don't matter" when the 9× per-tile blocking `WAIT`s were a fixed serial
  tax that folding could cut. **Rule:** absence of a win on axes A and B says nothing about axis C —
  enumerate the *untested* axis before declaring a floor.

## Diagnostics

### ACTIVE_HARTS sweep (1/2/4/8) — 2026-07-07 · all `max_abs=0`

| harts | wall | GMAC/s | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 75.32 ms | 0.392 | 1.00× | 100 % |
| 2 | 39.48 ms | 0.747 | 1.91× | 95.4 % |
| 4 | 19.98 ms | 1.476 | 3.77× | 94.2 % |
| 8 | 10.78 ms | 2.735 | 6.98× | 87.3 % |

Near-linear → per-hart compute is the ceiling (see L6). Scripts: `local-artifacts/build_hart_sweep.sh`,
`run_hart_sweep.sh` (gitignored). Confirms overhead is per-hart instruction volume + a small
serial-sync tail, not shared-shire bandwidth.

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

### B3 — distributed halo-only evict — **REJECTED (regression)** · 2026-07-07

- **Hypothesis:** the ~13 % 8-hart droop is hart0's O(image) serial whole-buffer halo re-evict
  (Amdahl); distribute the halo so each hart evicts only its own band's halo and drop one of the two
  per-layer barriers → faster.
- **Change:** `layer_publish` only — each hart fills+evicts its own band halo (L/R columns, plus the
  top/bottom halo row for the boundary harts); a single barrier replaces barrier→hart0-serial→barrier.
- **Result:** correct (`max_abs=0`). L2 writes **−16.8 %** (the redundant re-evict is gone — the
  mechanism worked) — but wall **+3.2 %**, confirmed by interleaved plain A/B: baseline median
  10.44 ms vs B3 10.77 ms, **non-overlapping** (baseline max 10.48 < B3 min 10.71), baseline spread
  0.99 %.
- **Lesson (→ L6):** the removed serial evict was **not on the critical path** (not bandwidth-bound),
  so cutting it saved traffic but no time; distributing the halo added barrier/skew cost exceeding the
  saving. Off-critical-path traffic reduction does not move wall here.

### B8 — fold 9 taps into 3 FMA dispatches — **WIN (−4.3 %)** · 2026-07-07

- **Hypothesis:** the 9 per-tap FMA dispatch/load/evict/`WAIT` sets are per-tile fixed overhead on the
  critical path; folding taps into fewer dispatches cuts them *without* adding traffic or touching the
  memory layout that bit B4.
- **Change:** fold each 3-wide kernel row into one int8 FMA with a `GTAPS*CH` (48-element, 12-quartet)
  contraction — within the 16-quartet `acols` hardware limit. Weights repacked group-major (GTAPS taps
  per OC line, built once per layer); B packed 12 lines/group. Orientation and requant/store/scatter
  unchanged (small blast radius, no seam-surface change).
- **Result:** correct (`max_abs=0`), wall **−4.34 %** — board A/B (plain, interleaved, 5×): baseline
  median 10.43 ms vs B8 **9.98 ms**, **non-overlapping** (baseline min 10.39 > B8 max 10.02), spread
  1.28 %. Bonus: A-load traffic ~3× lower (weights loaded once/group, not once/tap). **Committed.**
- **Lesson (→ L8):** the per-tap *tensor-op* overhead was ~4 % of wall — a third axis. 3 dispatches is
  the minimum at the 4-taps/dispatch `acols` limit, so the *dispatch-count* lever is spent (but the
  per-tile *sync* on the same axis was not yet — see B1).

### B1 — batch the folded per-tile evicts — **WIN (−2.7 % on B8)** · 2026-07-07

- **Hypothesis:** after B8 each tile still does one `FENCE+evict+WAIT_CACHEOPS` per group (3 of them);
  those per-tile blocking waits are on the critical path — batch them to one.
- **Change:** pack all NGRP groups' B up front into one per-hart buffer (grew 768 B→2304 B/hart), then
  a single `FENCE+evict+WAIT_CACHEOPS` per tile; the group loop keeps its sequential load/FMA. Same
  bytes, same layout, same output path. Committed on top of B8.
- **Result:** correct (`max_abs=0`), **−2.68 %** on B8 — A/B (plain, interleaved, 5×): B8 median
  9.98 ms vs B1 **9.71 ms**, non-overlapping (B8 min 9.88 > B1 max 9.73). Bigger than the 3→1
  wait-count ratio predicted → per-tile blocking sync is a heavy cost.
- **Lesson (→ L8):** same axis as B8, one notch further. Now spent: the remaining per-tile waits are
  the *sequential* `LOAD_WAIT`/`FMA_WAIT` (3 groups), not batchable — 3 groups' A+B (3×12+16 lines)
  exceed the 48-line SCP, so the loads/FMAs must stay serialized.

## Status — B8+B1 shipped (~9.69 ms, −7.3 % end-to-end); "near floor" was premature

New best: **~9.69 ms** (was ~10.46 ms) — **direct baseline-vs-combined A/B: −7.31 %**, non-overlapping
(baseline min 10.39 > combined max 9.79), `max_abs=0`, ~3.5× the leaderboard leader (was ~3.8×). Two
stacked wins on the per-tile fixed-overhead axis: **B8** fold 9→3 dispatches (−4.3 %) + **B1** batch
3→1 evict-waits (−2.7 %). After five experiments the picture is a **three-axis** trade:

- Cut **instruction volume** by re-laying-out memory → **B4 failed** (+8 %; added traffic/latency).
- Cut **off-critical-path traffic** → **B3 failed** (+3.2 %; not on the path, added skew).
- Cut **per-tile dispatch/sync overhead** without touching bytes/layout → **B8 + B1 won** (−7.3 % total).

The per-tile fixed-overhead axis is now **spent** (3 dispatches is the `acols` minimum; the remaining
LOAD/FMA waits are serialized by the 48-line SCP budget). The remaining prize is the dominant `pack_B`
scalar gather (~47 % of instructions) — the hard, B4-trap-adjacent one: attack it **pack-locally**
(a cheaper gather, not a re-layout) and A/B-gate every step.

**Durable deliverables:** the shipped B8+B1 speedup (~−7.3 %, board-verified `max_abs=0`), the
CI-matched noise-calibrated A/B harness (`run_ab_scored.sh`, now generic ref/chal), the hart-sweep
tooling (`build_hart_sweep.sh` / `run_hart_sweep.sh`), and this log.
