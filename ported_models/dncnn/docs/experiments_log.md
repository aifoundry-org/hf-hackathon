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

## Status — consolidated (near the single-shire floor)

After **B4** (−12 % instructions, but **+8 %** wall) and **B3** (−17 % L2 writes, but **+3.2 %** wall),
both rigorously measured, plus the hart sweep (6.98×/87 %, not bandwidth-bound), the int8 DnCNN kernel
is assessed **well-balanced and near a practical single-shire floor** (~10.4 ms, ~3.8× the leaderboard
leader). Wall ≈ hart0 cycles at a fixed ~0.55 GHz, and every single-axis restructure lands on the
wrong side of the instruction / traffic / sync trade:

- Cutting **instructions** helps only if it adds no cache-op or barrier stall — **B4 failed here.**
- Cutting **traffic** doesn't help — it isn't on the critical path — **B3 failed here.**

**Decision: consolidate.** Keep the board-verified baseline; stop single-axis pushes.

**Deferred (eyes-open, A/B-gated only):** **B8** — fold the 9 tap dispatches / feed the FMA an
addressing view to cut the dominant `pack_B` instructions *without* touching the memory layout that
bit B4. The one remaining lever with a different shape; expect the trade above to fight back.

**Durable deliverables:** the CI-matched, noise-calibrated A/B harness (`run_ab_scored.sh`), the
hart-sweep tooling (`build_hart_sweep.sh` / `run_hart_sweep.sh`), and this log.
