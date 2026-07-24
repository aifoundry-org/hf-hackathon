# SmolVLM2 — experiments log

Append-only ledger of every milestone/lever: what was tried, the numbers, and the verdict. This is how the
ladder stays honest — a WASH recorded here (with WHY) is never retried. Mirrors DnCNN's `experiments_log.md`
and the measured-dead YOLO levers.

Format per entry:

```
## <date> — <milestone/lever> — LANDED | WASH | BLOCKED
- Change: <one line — which kernel, which lever>
- Correctness: single-hart max_abs=<>; board bit-identical to 1-hart=<yes/no>
- Perf (board, paired-main): main=[<c1>,<c2>] candidate=<c> → <±%>; wall <±%>
- Quality: video_dog answer OK=<yes/no>; PPL=<> (≤26.739), ET-vs-CPU=<±%> (≤1%)
- Verdict + WHY: <if WASH, which cost it failed to move — so nobody retries it>
```

---

## 2026-07-16 — M0 (profile, no edits) — dtype/kernel RESOLVED (code-proven); board-cycle half PENDING

**Gate question answered: the dominant SigLIP vision GEMM dispatches to `mul_mat_Q8_0.c` (VPU int8-gather, no
TFMA).** Resolved from ground-truth source + the sha-pinned artifact manifest, not intuition. The board
firmware-cycle attribution + bound classification is a separate interview turn (command handed to user below);
the local ET_PERF empirical inventory is blocked on an incomplete SDK install (see "Local build gap").

### The dtype/kernel resolution (this is the M0 deliverable)
- **Dispatch is deterministic on the weight dtype** (`src0`), decided in the non-modifiable
  `ggml/src/ggml-et/ggml-et-ops.cpp:302–425` (`ggml_et_op_mul_mat`). dst=F32, src1(activations)=F32 always; the
  branch is purely `src0` dtype: `Q8_0 → mul_mat_Q8_0`, `F16 → mul_mat_f16`, `F32 (ne0%16==0 && ne1%16==0) →
  mul_mat_f32_matrix_engine`, else `F32 → mul_mat_f32`.
- **The SigLIP/mmproj weights are Q8_0** — confirmed, not assumed:
  `benchmarks/smolvlm2_500m_video.json` → `model_artifact: smolvlm2_500m_video_q8_gguf`,
  `mmproj_artifact: smolvlm2_500m_video_mmproj_q8`; `docs/smolvlm2_500m_video.md` converts the projector
  `--outtype q8_0 --mmproj` and states "the hackathon track freezes the Q8_0 model and projector"; artifacts
  are sha256-pinned (`artifacts.json`, model `6f67b80…`, mmproj `921dc7e…`). Hidden 768 / intermediate 3072
  are both multiples of the Q8_0 block (32) → the 2-D fc1/fc2 + QKV/O weights quantize cleanly to Q8_0.
- **Therefore the hot vision GEMM lands on `mul_mat_Q8_0.c`** — an 8-wide VPU int8-gather dot product
  (`compute_block_dot_product_q8_0`, `fgb.ps`+`fcvt.ps.pw`+`fmadd.ps` + L2 prefetch), scalar-per-output.
  It uses **zero** TFMA and is **not** the int8 opcode-3 path. The only tensor-engine GEMM kernel that exists
  (`mul_mat_f32_matrix_engine.c`) is reachable **only for aligned-F32 weights, which the frozen-Q8_0 baseline
  never produces** — so it sits idle on the scored path.

### Flops-ranked op table (analytic MACs from 01_ARCH_COMPUTE_PROFILE.md; NOT a fresh ET_PERF measurement)
| Rank | Op class | Kernel (dispatched) | Shape (per frame, ×4 frames) | MAC share |
|---|---|---|---|---|
| 1 | SigLIP MLP fc1/fc2 | **mul_mat_Q8_0** | `[1024×768]·[768×3072]` and back | **~43%** |
| 2 | Attention QKV + O proj | **mul_mat_Q8_0** | `[1024×768]·[768×768]` | ~22% |
| 3 | O(N²) self-attention scores/context | softmax_f32 + mul_mat | 1024², bidirectional | ~14% |
| 4 | Patch-embed conv (im2col-GEMM) | im2col_f32 + mul_mat | stride-16 | ~2.4 G (small) |
| 5 | Connector projector Linear | mul_mat | 12288→960 | ~0.6% |

Vision encoder ≈ **80%** of compute; **~65% of all compute is Q8_0 GEMM on `mul_mat_Q8_0.c`**. Vision
dominance and Q8_0-GEMM dominance both confirmed.

### Board cycle attribution — MEASURED 2026-07-16 (ivan@aifoundry2, genuine board, `device_cmd_exec_dur`)
Baseline = committed submodule HEAD, single profiled `video_dog` run. **Total = 10,870,578,663 firmware
cycles** (median `pmc_cycles` = 10,817,973,417; this is the score). Per-kernel ranking (15 kernels, 0 unmatched):

| Rank | Kernel | Launches | Cycles | % |
|---|---|---|---|---|
| 1 | **mul_mat_Q8_0** | 2960 | 8,293,218,843 | **76.29%** |
| 2 | **cont_f32** | 496 | 1,721,705,833 | **15.84%** |
| 3 | mul_mat_f32_matrix_engine | 100 | 295,744,344 | 2.72% |
| 4 | mul_mat_f16 | 768 | 193,900,369 | 1.78% |
| 5 | softmax_f32 | 432 | 96,162,081 | 0.88% |
| 6 | el_map_f32 | 1360 | 86,556,585 | 0.80% |
| 7 | rms_norm_mul_f32 | 780 | 62,932,408 | 0.58% |
| 8 | rope_f32 | 768 | 46,431,746 | 0.43% |
| 9 | set_rows_f32 | 768 | 38,098,020 | 0.35% |
| 10 | glu_f32 | 384 | 17,970,209 | 0.17% |
| 11–15 | norm_f32 / unary_f32 / memops / im2col_f32 / get_rows_f32 | — | <8M each | <0.1% |

**Confirmations:** `mul_mat_Q8_0` is the top consumer at **76.3%** (M0 code-prediction verified). All MUL_MAT
kernels (Q8_0 + matrix_engine + f16) = 80.8% — vision GEMM dominance confirmed empirically.

**SURPRISE #1 — `cont_f32` = 15.84% (unpredicted #2 lever).** 04_KERNEL_STATE flagged it NAIVE-scalar but
"secondary"; it is NOT secondary. 496 launches, ~3.47M cyc/launch. A pure reshape/permute (attention layout
transpose) done as scalar per-element atomic-store copies. Vectorizing it (or fusing the permute into the GEMM/
softmax to avoid materializing the permuted tensor) is a clean, low-risk, lossless ~16% target. This jumps the
ranked-target list.

**SURPRISE #2 — the matrix engine and F16 paths ARE live but tiny.** `mul_mat_f32_matrix_engine` fires 100×
(2.72%) and `mul_mat_f16` 768× (1.78%) — so a little F32-aligned + F16 matmul exists (connector/attention side),
but the 76.3% vision GEMM is Q8_0 as predicted. Softmax is only 0.88% (the flops-model over-weighted O(N²)
attention at ~14%; on-device it's cheap).

### Quality gate — PASS (baseline is a valid reference)
`passed: True`, `task_accuracy: 1.0` (video_dog answer correct), PPL **22.2825 ≤ 26.739**, ET-vs-CPU relative
diff **2.24e-5** (≪1%), `vision_fallback_ops: []` (zero fallbacks), full LLM offload. Any candidate must hold
all of these.

### Bound classification — still OPEN (needs HPM counters)
The board score carries only `device_cmd_exec_dur` per kernel, not the 6 HPM counters. To classify
`mul_mat_Q8_0` load-latency-bound (→ pipeline, M1/M2) vs bandwidth-bound (→ int8-TFMA/residency, M3) we still
need a PMC-instrumented run (03_HARDWARE_NOTES §PMC: `TFMA_WAIT_TENB` vs `DCACHE_MISSES`+`L2_EVICT_REQ`). Not
blocking M1 target selection — the target is `mul_mat_Q8_0.c` either way — but it decides the *mechanism*
(pipeline the VPU chain first vs jump to engine/int8).

### Ladder implication (IMPORTANT — reshapes M1)
The plan's **M1 as written (double-buffer `mul_mat_f32_matrix_engine.c`) optimizes a kernel the dominant vision
GEMM never reaches** under the frozen Q8_0 weights. Two viable routes, to be chosen by the board bound-class:
- **(A) Optimize `mul_mat_Q8_0.c` in place** — the kernel actually on the hot path. Pipeline the VPU
  load→gather→fmadd chain first; only then consider int8 TFMA opcode-3 (yolo proved int8 = parity when
  orchestration-bound, so pipeline is the prerequisite).
- **(B) Route Q8_0 vision GEMM → F32 → `mul_mat_f32_matrix_engine.c`** and pursue M1/M2 (double-buffer +
  pipeline) there, paying an on-device dequant to put the GEMM on the tensor engine.
Recommendation: get the board bound-classification before committing. If load-latency-bound, (A) pipelining
`mul_mat_Q8_0.c` is the lowest-risk first lever and stays on the frozen-weights path.

### Local build gap (blocks the local ET_PERF inventory AND the M1+ full-model check)
`cmake -DGGML_ET=ON -DGGML_ET_SYSEMU=ON` fails: the full build's `et-kernels` subdir needs
`find_package(et-common-libs)` + `esperantoTrace` + the RISC-V toolchain cmake modules, which are **not**
installed into the `/home/marin/et-src/et` prefix. `install_et_sdk.sh` only installs the *minimal host* sys-emu
packages (`runtime`, `sw-sysemu`, …) and short-circuits ("already installed") on their markers. Also needed a
build-env shim for libcap (system lib, no CMake config) at
`…/scratchpad/libcap-shim/libcapConfig.cmake`. **Action before M1:** complete the et-platform SDK install
(`et-common-libs`, `esperantoTrace`, `lib/cmake/cmake-modules`/`riscv64-ec-toolchain.cmake` into the prefix),
OR confirm the standalone single-kernel build recipe (the isolated `GGML_ET_KERNELS_PATH` loop) is provisioned,
since M1+ iteration depends on a working local sys-emu build. Models are fetched + sha-verified at
`local-artifacts/models/smolvlm2_500m_video/` (417M + 104M).

- Correctness: n/a (no code change in M0).
- Perf: baseline **10,870,578,663** firmware cycles (board, genuine). Score = median pmc_cycles 10,817,973,417.
- Quality: PASS — accuracy 1.0, PPL 22.2825 ≤ 26.739, ET-vs-CPU 2.24e-5, zero vision fallbacks.
- Verdict: **M0 COMPLETE.** Dtype/kernel gate MET and board-confirmed: `mul_mat_Q8_0` = 76.3% top consumer.
  Two board-only findings reshape the ladder: (1) `cont_f32` = 15.84% is a large unpredicted #2 lever;
  (2) matrix-engine/F16 paths are live but <3%. **M1 target = `mul_mat_Q8_0.c` (route A, pipeline in place).**
  Open follow-up: a PMC-instrumented run for the bound classification (decides M1/M2 pipeline vs M3 engine/int8).

---

## 2026-07-16 — M1 pick = T0b `cont_f32` kernel speedup — investigation → IN PROGRESS
Chose the `cont_f32` lever (15.84%, board-measured #2) as M1 over the `mul_mat_Q8_0` pipeline: self-contained,
numerics-preserving (easy `max_abs=0`), on the sanctioned kernel surface.

**Where the 496 launches come from (graph trace).** On the ET backend ONLY `GGML_OP_CONT` launches a kernel —
`permute/transpose/reshape/view` are free no-ops (`ggml-et.cpp` supports_op 913-916 / dispatch 674-677). The
15.84% is two conts per SigLIP attention layer, ×12 layers ×~18-20 encoder passes (video_dog's 4 frames split
into ~4-5 sub-image tiles → ~480 of the 496 launches; decode conts are ≤3-token, ~0% cycles):
- `tools/mtmd/clip.cpp:633` `ggml_cont(v)` — V transpose after `permute(1,2,0,3)`; dim0 NOT contiguous, and ET
  `mul_mat` needs first-dim-contiguous src0, so the transpose is genuinely required. dst `[1024,64,12]`, ~3MB.
- `tools/mtmd/clip.cpp:643` `ggml_cont_2d(kqv)` — head-merge `[64,12]→768` before the o_w proj; dst `[768,1024]`,
  ~3MB. Source is first-dim contiguous.

**Fusion is OFF-SURFACE.** Folding `:643` into the o_w GEMM, or killing `:633` via FlashAttention, both need
`clip.cpp`/backend edits — main-owned. Our surface is only `et-kernels/src/*`; we cannot remove a CONT node the
graph emits. (Same constraint partly blocks the plan's M4 FlashAttention lever: flash-attn is off in the
backend `supports_op`.) → the landable lever is making `cont_f32.c` itself fast.

**Kernel defect (root cause of the cost).** `cont_f32.c:98` writes EVERY element with `atomic_store_f32` =
`amoswapg.w` (global atomic swap, `platform.h:171`) — a data-movement kernel paying atomic-RMW per float.
`cont_f16.c` has the identical pattern. The two hot conts have 64B-multiple row widths (`:643` 768f=3072B=48
lines; `:633` 1024f=4096B=64 lines) → each hart's row-block is cacheline-disjoint → **no seam race → the atomic
is pure waste** for the dominant conts. See [[int8-cacheline-seam-race]].

**Change implemented (cont_f32.c).** Seam-aware store strategy: `seam_safe = (ne00 % 16 == 0)` → plain stores;
`src_contig = (nb00 == sizeof(float))` → 8-wide VPU `flw.ps`/`fsw.ps` vectorized copy (`copy_run_vec8`, mirrors
`el_map_f32.c` mask save/restore). seam_safe+strided → scalar plain stores; non-aligned → keep `amoswapg.w`
(so we never CPU-fallback, which the gate forbids). Partition unchanged (rows / ne01).

**Verification so far:**
- Host logic test (scratchpad `cont_logic_test.c`): `max_abs=0` on both hot shapes + a small unaligned case,
  at 1/2/4/8 harts — addressing/partition/tail correct.
- `smolvlm2-kernel-reviewer`: **PROCEED** (all 10 axes clear vs ground truth). Key: `supports_op` requires
  `nb[1]%64==0` (`ggml-et.cpp:900`) ⇒ for CONT (ne preserved) that IS `ne00%16==0` ⇒ **seam_safe is ALWAYS
  true for the shipped graph** → fast path always fires, atomic branch is dead-but-harmless. Output is
  bit-identical (same words moved, zero arithmetic) → answer/PPL/ET-vs-CPU cannot move.
- Board build path: verified the board builds from the in-tree submodule (my change ships without a commit).

**Board attempt #1 (plain fsw.ps) — FAILED on hardware (2026-07-16).** Built + ran on ivan@aifoundry2, but the
ET path threw 57× `ET: stream error detected at synchronization point` and produced an EMPTY answer; the CPU
host-correctness path was clean (0 errors) → the fault is my kernel on the ET device. Root cause: **ET-SoC1 is
non-coherent across shires.** The original `amoswapg.w` global-atomic store was PUBLISHING each write to global
memory (bypasses L1); my plain `fsw.ps`/scalar `fsw` writes landed in L1 and were invisible to the next kernel's
cross-shire reads → sync-point stream error. `set_rows_f32.c:14-18,222` documents this exactly ("amoswapg.w /
shg bypass local caches → safe on non-coherent HW"). The host logic test + reviewer BOTH missed it (they reasoned
the end-of-kernel barrier publishes L1 — false for cross-shire). **Lesson: a plain-store speedup for a
cross-kernel-consumed tensor is a non-coherency footgun; sys-emu/1-hart/host-logic cannot catch it.**

**Board attempt #2 (flq2/fsq2, corrected) — PENDING.** Rewrote `cont_f32.c` to mirror the PROVEN `set_rows_f32.c`
discipline: cache-line partition (each hart owns whole disjoint 64B lines) + `flq2`/`fsq2` 256-bit load/store
(`copy_cacheline_f32`) for the globally-publishing copy; contiguous src → direct, strided src (`:633`) → gather
one line into a 64B-aligned buffer then `fsq2`; non-aligned dst (never emitted) keeps `amoswapg.w`. Host logic
test max_abs=0 (both shapes, 1-8 harts). **Reviewer #2: PROCEED** — bit-identical (dst_cl reduces to
`dst_data + cl*16`, every line written once, hart seams on 64B boundaries); mirrors `set_rows_f32.c`'s
`flq2/fsq2` cache-aligned path which IS the empirically-validated cross-shire publish primitive (set_rows writes
embeddings/KV consumed cross-shire by attention in the shipping model); `flq2` is misalignment-tolerant
(`mmu_loadVLEN`) and both shipped sources are aligned anyway; no forbidden float ops. Residual is board-only:
validate on ≥3 consecutive no-dump 8-hart runs + re-confirm answer/PPL/zero-fallbacks.

**Board attempt #2 — LANDED (2026-07-16, genuine board, ivan@aifoundry2).** NB: attempts #1's failures were
never the fix — the board built the stale `copy_run_vec8` (attempt #1) until a sha1 guard (`[1a/5]` in the
candidate runner) forced the corrected source. With `flq2/fsq2`:
- **`cont_f32`: 1,721,705,833 → 28,868,869 cyc** (−98.3%; **15.84% → 0.31%** of total). The per-element
  `amoswapg.w` was ~all the cost.
- **Score (median pmc_cycles): 10,817,973,417 → 9,136,533,962 = −15.5%** (parser total 10.87B → 9.19B; the
  drop equals the cont savings → nothing else regressed; other kernels within run-to-run noise).
- **Correctness clean:** passed=True, accuracy 1.0, PPL 22.2825 (bit-identical, ≤26.739), ET-vs-CPU 2.24e-5,
  vision_fallback_ops=[], and **zero `ET: stream error`** — the cross-shire publish fix is hardware-validated.
- Verdict: **M1/T0b LANDED** (single clean run; ~15.5% ≫ the ≥1% gate). Remaining rigor: interleaved
  paired-main A/B (`smolvlm2-ab-score`) for ≥3 clean 8-hart runs + drift check to formalize. `cont_f16.c` has
  the identical old pattern but is off the hot path (not emitted); optional follow-up.

Baseline for the next lever shifts: with cont removed, **`mul_mat_Q8_0` is now 90.2%** of a 9.14B-cycle total —
the remaining M-work is the Q8_0 GEMM pipeline (T0).

---

## 2026-07-16 — M2a int8 TFMA single-tile — LOCAL PASS (board-evict pending)
First increment of the `mul_mat_Q8_0` → int8 TFMA (opcode-3) migration. Single 16×16 M-tile, full K, single
hart, static `.bss` scratch; scalar `compute_block_dot_product_q8_0` kept as the shape fallback.
- Impl (mul_mat_Q8_0.c, +218 lines): per-row symmetric int8 activation quant (`d_b=max/127`, `et_fdiv`=frcp,
  NO fdiv.s), quartet weight repack, per-block `tensor_fma` opcode-3 (both `tenb/tena_unsigned=false` → ua=ub=0
  signed; `first_pass=1` every block; `tenc_loc=1`), `tensor_quant` dual-scale (INT32_TO_FP32→MUL_COL(d_b)→
  MUL_ROW(d_a)), store→scalar f32 accumulate, `copy_cacheline`/flq2 publish.
- **Local gate: test-backend-ops -o MUL_MAT -p 'type_a=q8_0,m=16,k=256' → 18/18 OK, 0 FAIL** vs the f32 CPU
  reference (within tolerance — so the int8-activation approximation is faithful enough; de-risks the board
  PPL/answer gate). Builds clean (no forbidden fdiv/fsqrt).
- **⚠ NOT board-ready:** sysemu is a COHERENT functional emulator, so the scalar-store→tensor_load of the
  scratch works locally; the non-coherent board needs FENCE+evict of the staged scratch (s_qact/s_bpack)
  before each tensor_load (same class as the cont_f32 non-coherency bug). Add before the board A/B.
- Next: reviewer → add board evict → M2b (full M×N tiling / multi-hart) → board A/B. Local loop now iterates
  this in ~minutes (ET_TOOLCHAIN=~/et-src/et; see [[smolvlm2-port-research]]).

### M2a update — reviewer BLOCK resolved (evict) + path to board
Reviewer verdict on M2a: **all int8 encodings CONFIRMED-OK** (TFMA opcode-3 signedness, quartet layout,
dual-scale dequant, tensor_store, FREG barrier, publish, no forbidden ops, no CPU fallback) — the math/plumbing
is sound. Three must-fixes, all board-readiness (not the int8 approach):
1. **[FIXED] Non-coherent staged-scratch evict** (the cont class): the scalar-written scratch (s_qact/s_bpack/
   s_db/s_da) was FENCE'd but not evicted before `tensor_load` (which reads DRAM) → board reads stale L1. The
   erbium `evict()` header isn't on the kernel include path, so added an inline `q8_evict` (CSR 0x89f cacheop,
   mirrors cacheops-umode.h) + `FENCE;q8_evict(all 4);WAIT_CACHEOPS` before the loads, every K-block. Builds
   clean, local gate still OK (evict is a no-op on the coherent emulator).
2. **[OPEN — gates the board A/B] single-hart.** M2a pins the GEMM to `global_id==0`; the scalar fallback it
   replaces uses ALL harts. So M2a is very likely SLOWER — the win only appears with MULTI-HART tiling (M2f).
   **Do NOT spend a board A/B until multi-hart.**
3. **[OPEN] board PPL/answer** re-verify (int8-activation is lossy; test-backend-ops 5e-4 NMSE ≪ tolerance is
   encouraging but the hard gate is board PPL ≤26.739 + exact answer).

Local gate final: **38/38 shapes OK** (m=16, n=1..16, k=256 & 1024, batched/broadcast/permuted). M2a foundation
is board-correct; next milestone = **M2f multi-hart tile partition** (per-hart scratch by global_id; whole-16-col
m-blocks are disjoint 64B lines → seam-safe) — that is where the 90%-kernel win materializes, then board A/B.

---

## 2026-07-16 — M2f multi-hart int8 TFMA — LOCAL PASS (reviewer + board pending)
Parallelized the M2a int8 tiled path across tensor-capable (even) harts — the milestone that determines whether
int8 TFMA beats the current kernel (M2a single-hart was likely slower).
- Per-hart private scratch: 6 buffers → `[32][...]` indexed by `hidx=thread_id/2` (~116KB .bss); each hart
  evicts only its own 64B-disjoint rows.
- Tile partition: flatten (batch × m_tiles × n_tiles) → `total_tiles`; each hart `tile=my_idx; tile+=part_harts`.
- **CAP TRAP (agent caught):** default `shire_mask=0xFFFFFFFF` = 32 shires → 1024 tensor harts. A naive
  ">32 → scalar fallback" guard would SILENTLY run scalar in real deployment. Fixed by capping participation
  `part_harts=min(num_tensor_harts,32)`, `my_idx<part_harts` runs, stride=part_harts → full coverage, one shire
  engaged, no over-index. `part_harts<1` → scalar fallback.
- Local gate: q8_0 m=16 k=256 → 16 OK / 0 fail; poison-diagnostic confirmed the TILED path (not scalar) drives
  output. Builds clean.
- **⚠ OPEN #1 (board-critical, under reviewer now):** is `get_relative_thread_id(shire_mask)` GLOBAL-dense or
  PER-SHIRE? If per-shire, two physical harts get the same `my_idx<32` → same scratch row + same dst tile →
  race/corruption on the 32-shire board. Must confirm before board.
- OPEN: .bss ~116KB above `_end` (loader provisioning); non-coherent seam gate (board-only); perf A/B (32-hart
  tiled vs all-hart scalar); board PPL/answer. Reviewer running → then board gate if PROCEED.

### M2f reviewer — PROCEED (board-safe) + the perf caveat that reshapes the plan
- **Crux resolved: `get_relative_thread_id` is GLOBAL-dense** (get_hart_id=global csrr hartid; thread_id anchored
  at lowest set shire). Under 0xFFFFFFFF, `my_idx∈[0,32)` = shire 0's 32 even harts, each distinct minion →
  distinct scratch row + dst tiles. NO cross-shire collision. Tile coverage exact-once; per-hart evict/dst-seam
  single-writer. **Board-safe.**
- **PERF CAVEAT (likely fatal to the ≥1% at 32-hart):** the cap engages ONLY shire-0's 32 tensor harts; the
  other ~992 minions idle, while the scalar kernel uses ALL ~2048 harts. 32-hart int8 may be a REGRESSION vs
  2048-hart scalar. The win needs MULTI-SHIRE (~1024 tensor harts) — the next lever (M2g).
- OPEN gates: board PPL/answer (int8 lossy, same math as M2a — hart-count-independent), .bss (116KB now,
  ~3.8MB at full multi-shire) loader provisioning.
- **Plan: board-test 32-hart M2f now** to validate (1) int8 TFMA works on real silicon (evict/flq2/tensor —
  board-only), (2) quality gate holds, (3) per-shire perf calibration — THEN M2g multi-shire for the actual
  win (perf-only A/B, since correctness+quality proven). Rationale: de-risk the fundamentals before scaling.

---

## 2026-07-16 — M2g multi-shire int8 TFMA — LOCAL PASS, board-ready (board congested)
Scaled M2f from single-shire (32 harts) to ALL shires: `Q8_MAX_TENSOR_HARTS = SOC_MINIONS_PER_SHIRE * Q8_MAX_SHIRES
= 32*32 = 1024`; `part_harts = min(num_tensor_harts, 1024)` → every shire's even (tensor) harts participate;
per-hart scratch `[1024][...]`. This is the config that can WIN (M2f's 32-hart idled ~992 minions vs scalar's
~2048 workers → likely regression).
- Correctness: UNCHANGED logic from reviewed M2f (only the cap + array size scaled). Reviewer already established
  `my_idx` (=thread_id/2, get_relative_thread_id global-dense) is globally unique → per-hart scratch/tiles
  collision-free at ANY N. So M2f PROCEED extends. Local gate q8_0 m=16 k=256 → OK (sysemu launches a small
  shire_mask, so it verifies compile + small-hart correctness, NOT the 1024-hart path — board-only).
- **.bss = 3.80 MB** (riscv size: text 5784, bss 3,801,088). `linker.ld` `.bss(NOLOAD)` has NO hard length cap
  (just `_end=.`), so it links cleanly; **board-gate item: confirm the launcher provisions the enlarged region
  from `_end`** (reviewer: "appears to"). If it doesn't, shrink per-hart scratch (s_qact 16*64→16*32, drop
  s_btmp) to fit.
- **BLOCKED on board access** (shared device congested — `llama32_1b` CI job held `/dev/et0_ops`; others
  affected too). When free: one board run of M2g validates fundamentals (int8 on silicon + PPL/answer) AND the
  real perf win (mul_mat_Q8_0 cycles vs 8,291,438,783 baseline / 90.2%) in one shot — the winning config, not
  the likely-losing 32-hart M2f. Board flow: `ssh ivan@aifoundry2 hostname` (refresh Tailscale auth) →
  `run_smolvlm2_cont_candidate_board.sh` (guard now order/locale-independent + fresh-build).

### M2g BOARD RESULT (2026-07-17, fresh, genuine) — int8 TFMA is CORRECT but a PERF REGRESSION
Board-verified (ivan@aifoundry2, fresh score 10:27, no stream errors, int8 path proven to run: PPL + ET-vs-CPU
both moved from baseline).
- **Quality PASS:** passed=True, accuracy 1.0 (answer OK), PPL 22.2774 ≤ 26.739, ET-vs-CPU 0.80% ≤ 1%, zero
  fallbacks. The whole int8 stack (tensor engine, per-block Q8_0 scales, q8_evict, flq2 publish, multi-shire)
  works on silicon within the quality gate. int8 activation quant is faithful (PPL Δ = 0.005).
- **Perf REGRESSION:** mul_mat_Q8_0 8,291,438,783 → 9,214,829,656 (+11.1%); SCORE (pmc_cycles) 9,136,533,962 →
  9,884,327,577 (**+8.2% WORSE**). Multi-shire didn't help — the bottleneck is PER-BLOCK ORCHESTRATION, not
  hart count: 24 blocks/tile × serialized (load A/B, fma, load d_b/d_a, quant, store, evict+WAIT_CACHEOPS,
  scalar f32-accumulate). Marshalling > scalar VPU dot. This is the DnCNN "int8 = marshalling-bound" lesson,
  worse here due to per-block dual-scale Q8_0 handling.
- **Verdict: WASH as implemented (naive orchestration).** NOT necessarily dead — un-exploited headroom the
  M2a→M2g agents skipped: M2c (hoist quant/repack + evict OUT of the per-block loop → evict 1× not 24×) and M2e
  (pipeline load→fma→quant→store double-buffer). Odds per DnCNN precedent: parity or modest win plausible, big
  win unlikely. Decision pending: (A) one focused M2c evict/pre-pass hoist + re-measure, or (B) accept WASH,
  revert mul_mat_Q8_0.c to scalar, keep the banked cont +15.5%. Score to beat: 9,136,533,962 pmc_cycles.

### M2c FINAL = shared-arena pre-pass + hardened all-hart barrier (shipped path; user directed the barrier over the band)
The no-barrier band fallback was replaced by the barrier variant — the band's 192×→12× cut + parallelism loss
(fc2 idled ~800 harts) wasn't the real fix. The barrier gives the FULL cut with FULL parallelism.
- **Redundancy: activations 192×→1×, weights 64×→1×**, all per-tile evicts removed, all even harts run tiles.
- Structure: each even hart quantizes+packs a partition (acts by K-block, weights by M-tile) into ONE shared
  10MB `.bss` arena, evicts what it wrote → `q8_barrier(part_harts)` → tile loop reads arena via `tensor_load`.
- **Hardened barrier** (the board-safety work): generation-based, amoaddg/amoswapg; `part_harts=(get_num_threads+1)/2`
  cap 1024; single-hart (n≤1) no-op; **BOUNDED spin 2e8 then sets `g_q8_bar.timed_out=1` and proceeds** (never
  wedges the shared board on a wrong count); arrival-count sanity in `g_q8_bar` (`expected`/`last_arrivals`) —
  the board dump can confirm `expected==last_arrivals && timed_out==0`.
- Deviation flagged: NO reader-side invalidate — arena consumed only via `tensor_load` (DRAM-direct, bypasses
  L1), same pattern M2g proved on the board; no scalar cross-hart arena reads. (Reviewer scrutinizing.)
- **.bss = 12.0 MB** (< band's 18.6MB; shared arena replaces per-hart act scratch). Builds clean.
- Local: forced part_harts=8 → 14/14 OK + multi-tile cross-hart-arena cases OK, no hang. **1024-hart path is
  BOARD-ONLY** (sys-emu can't run all even harts → an unforced local run bounded-spin-times-out; that's a
  sys-emu limitation, not a defect — the reason we test on real HW).
- Reviewer running (barrier count exactness + no-invalidate crux). Then board: A/B vs scalar 8.29B + HPM profile.

### M2c barrier variant — reviewer 3 BLOCKs FIXED, board-ready
Reviewer verified the two crux axes SOUND (barrier count==arrivals exact for the 0xFFFFFFFF launch → no deadlock;
evict-before-barrier + DRAM-direct tensor_load with NO reader invalidate is correct — no scalar cross-hart arena
reads). Numerics bit-identical to M2a (PPL won't move). 3 BLOCKs fixed + verified:
1. `q8_evict` over-evicted (cacheop field is count-1); now emits exactly the covering line count (chunks of 16,
   tail field r-1). Also tightens cache traffic (helps perf).
2. Barrier spin was polling a GLOBAL atomic (chip-serialized) → ~1000harts×2e8 could wedge the shared board;
   now a PLAIN volatile load of `released` (threshold barrier); only arrival(amoaddg)/release(amoswapg) atomic.
3. Timeout poisoned the next launch; now self-healing via monotonic `arrived` + milestone release (a shortfall
   round re-aligns `arrived` to its milestone → next launch can't fire early; no per-launch reset/race).
Build clean, .bss 12.0MB, forced part_harts=8 gate 16 OK + multi-tile/multi-batch OK. BOARD-READY.

## Board runs STAGED (blocked on shared-device access)
- **B (A/B)**: `bash local-artifacts/run_smolvlm2_cont_candidate_board.sh` → builds barrier variant, runs
  video_dog+profile; then `python3 local-artifacts/smolvlm2_ab_diff.py <baseline-dir> .ci-work/smolvlm2-cont-candidate-output`
  → mul_mat_Q8_0 cycles vs scalar 8,291,438,783 / M2g 9,214,829,656. Score to beat: 9,136,533,962 pmc_cycles.
- **A (HPM)**: `bash local-artifacts/run_mulmat_q8_hpm_board.sh` → rebuilds standalone harness from current kernel,
  runs isolated fc1 GEMM under launcher --dump_after, decodes IPC / retired-inst-per-MAC / L2-miss-per-MAC. High
  inst/MAC ⇒ marshalling-bound (int8 can't win → WASH); near-floor ⇒ headroom.
- Board dump also exposes barrier sanity `g_q8_bar`: confirm `expected==last_arrivals && timed_out==0` (1024-hart
  sync fired; sys-emu can't test it). Guard: HPM script aborts if kernel .bss ≥16MB (map collision).

## 2026-07-17 — int8 TFMA: WASH (measured-dead). Pivot → Route B (float engine).

**Verdict: int8 TFMA on mul_mat_Q8_0 is measured-dead. Reverted to committed scalar; cont win (+15.5%) banked.**
Board evidence, all forms tried:
- **M2g** (per-tile int8, no barrier): correct but **+8% regression** (192× redundant activation quant).
- **Barrier variant** (shared pre-pass, all-hart barrier): after fixing 3 reviewer BLOCKs + a hard non-coherency
  bug, it reached CORRECTNESS at 1024 harts (video_dog=" Dog.", 0 stream errors) — but:
  - the barrier release was invisible to waiters (plain L1 load of a global-atomic-written flag) → early-proceed
    over a half-written arena → intermittent `0xCDCDCDCD` garbage stream errors. FIXED with a throttled global-read
    spin (`amoaddg(&released,0)` + 512-nop backoff). Recorded in [[etsoc1-noncoherent-stores]].
  - fixed build then hit a runtime `event 65535` (uint16 EventId sentinel/0xFFFF) crash that kills the PROFILED
    scoring run → no cycle score. Device-emitted sentinel (only 18,606 commands issued, no counter overflow);
    barrier variant provokes it, scalar never does.
  - **Perf proxy (non-profiled correctness run, no score needed):** scalar 9.03 ms/tok, 2.73s vs barrier
    **10.40 ms/tok, 3.12s = ~+15% SLOWER on the vision prefill** — and that's WITH the cont win, which the scalar
    baseline lacks. Second independent confirmation of the M2g +8% regression. **int8 loses; even a fixed score
    would be a regression.** Do NOT retry int8 on this kernel.

**Root cause (why int8 lost):** the giant is MAC-bound (~130 MACs/byte on fc1), but int8's marshalling (quantize
f32 acts→int8 + repack weights to quartet, redundantly) exceeds the MAC savings. The engine's fast MACs never
paid for the setup.

### Route B — Q8_0→f32 dequant → FLOAT matrix engine (STARTED)
Same engine, DIFFERENT feed: dequant Q8_0 weights → f32, load into the engine; **activations stay f32 (no
activation quant — the expensive half is gone)**; weight dequant is **per-hart-local → NO barrier, NO shared
arena** (sidesteps the entire non-coherency saga). Reuses `mul_mat_f32_matrix_engine.c` (opcode-0 f32 core).
- **v1a** (single-hart correctness, no double-buffer): dequant front-end + existing serialized engine loop.
  Gate = test-backend-ops max_abs on real vision shapes (m=3072,n=1024,k=768 etc.). IN PROGRESS.
- **v1b**: double-buffer (2 load IDs — overlap tile N+1 load with tile N fma).
- **v1c**: multi-hart (partition M-blocks; per-hart scratch pool — bound .bss, cap active engine harts).
- Precision: f32 dequant is exact; only accumulation order differs → expect max_abs ~0, PPL gate safe. f16
  (opcode-1, 2× throughput) is a v2 perf push ONLY if v1 wins and f16 holds the PPL gate.
- Scratch-sizing (multi-hart) is the known open risk: per-hart 16×768×4=48KB × many harts is too much .bss.

### 2026-07-17 CORRECTION — `event 65535` root cause = kernel .bss too big (NOT barrier/atomics)
Route B v1c (multi-hart, 12MB .bss, NO barrier/atomics) hit the SAME `event 65535` crash as the barrier variant.
Fresh board stack proved it: crash at BACKEND INIT — `ggml_et_driver_init → RuntimeImp → ExecutionContextCache →
doMemcpyHostToDevice → dispatch(65535)` — i.e. LOADING the kernel onto the device, before any inference. The only
diff vs working scalar/M2g kernels is the 12MB `.bss`. So `event 65535` = a large-`.bss` kernel can't load
(device execution-context memory). Recorded in [[etsoc1-kernel-bss-load-limit]]. IMPLICATIONS: (1) the barrier
variant's earlier "correct answer at 1024 harts" was STALE benchmark-output files — it likely never loaded (also
12MB .bss); its int8 WASH still stands on M2g's real +8%. (2) ANY Route B / engine kernel must keep .bss KB-scale.
**v1d fix:** per-K-block dequant → .bss 12MB→~256KB, AND lifts the K≤768 cap so fc2 (K=3072) also gets the engine.
Always check board file MTIMES — a crashed run leaves the prior run's score/responses in place and rsync pulls them.

---

## 2026-07-18 — int8-TENC (opcode-3, clean form) — BUILT, board gate queued

Third int8 attempt, structurally different from the measured-dead M2g/barrier forms (design:
`routeb-variants/EVENT65535_VERDICT.md` §139-216, `TFMA_VERDICT.md` Q2). Written by a subagent; I
reviewed the diff line-by-line against `tensors.cpp:1424-1569` ground truth. Working tree only,
uncommitted, on top of `5413c722e` (scalar_hoist stays as the fallback path in-file).

- **Why this form is not the dead one:** (1) activation quant hoisted to ONCE per n-tile per launch
  (M2g's killer was 192x redundant re-quant); (2) NO barrier, NO shared arena — partition by n-tile
  item across the 8 even harts, per-hart-private staging (the barrier variant's killer); (3) TENC
  hardware accumulator + tenc2rf drain — no per-K-block FREG save/restore (Route B's killer);
  (4) weights fed as raw int8 quarts — zero dequant.
- **Structure:** per 16x16 tile, K-loop of ONE tensor_fma per Q8_0 block (acols=7 -> 32K, 8 internal
  passes in TENC, first_pass=1, tenc2rf=1) -> drain int32 16x16 to f0-f31 -> VPU epilogue
  (fcvt.ps.pw + fmul dw[kb] + fadd into stack acc tile; rows 14/15 spilled around f28-f31 temps).
  d_a per-row act scale factored out of the K loop, folded at tile end. Double-buffered tensor_load
  on both IDs (block kb+1 loads issue while kb's FMA runs; A pairs ride 64B lines so A loads halve).
- **(B)-hardening vs event 65535:** dst published via fsq2 (64B-aligned whole lines, single writer)
  + strided evict + WAIT_CACHEOPS; staged scratch FENCE+evicted before tensor_load; deep-pipelined
  (no serial load->wait->fma chain -> watchdog dodge); CLEAR_TENSOR_ERROR at prologue.
- **Encodings verified vs emulator:** B quartet layout (tmpb.u8[j*4+x], tensors.cpp:1506/1533),
  aoffset field x4 (:1443), acols/bcols (x+1)*4 (:1440-1442), ua=ub=0 both-signed (:1499-1510),
  tenc2rf drain on last pass (:1464-1465,1557-1566). Scalar fallback for M%16!=0 / N<16 / K>3072 /
  unaligned dst. n_cur<16 edge rows computed but never published (acc rows garbage-in-garbage-out,
  never stored).
- **Static .bss = 2,017,280 B (1.92MB)** — between M2g's board-proven 3.8MB and the 12MB crash.
- **Local status:** kernel target compiles clean (mul_mat_Q8_0.elf, instruction checker green).
  Per the >30min rule the local 8-hart sys-emu gate was SKIPPED — correctness/PPL/perf all decided
  on the board run (this is the fast-loop policy: board beats a long local gate).
- **Board run:** `bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh`.
  Pass = fresh scored_at, answer " Dog.", PPL <= 26.739, ET-vs-CPU <= 1%, 0 fallbacks, NO event 65535
  in the profiled run, mul_mat_Q8_0 cycles < 8,291,438,783 (scalar_hoist), score < 8,489,934,819.
  If it crashes event 65535: read error_type at RuntimeImp.cpp:726 (1=exception -> tighten publish;
  2=hang -> deepen pipeline). If correct-but-slower: the per-m-tile weight repack (~16x redundancy
  across hart chunks) is the first thing to hoist.
- Verdict: PENDING BOARD.

### S2b BOARD RESULT (2026-07-18, fresh 15:15) — LANDED, the stall hypothesis CONFIRMED
- **mul_mat_Q8_0: 6,839,456,472 -> 5,327,584,696 = -22.1% kernel** (85.6% of exec).
- **Score: 7,692,846,657 -> 6,170,682,271 = -19.8%.** Cumulative from M0 10,817,973,417 = **-43.0%**.
- Quality PASS: " Dog.", PPL 22.2815, ET-vs-CPU 3.1e-5, 0 fallbacks.
- **Lesson:** 2-reg act-load pipelining flipped N=8 from +10% (S2) to -22% — the loop was
  load-use-stall-bound, not instruction-count-bound. Commit `800daee36`. Remaining rigor: formal
  interleaved A/B (smolvlm2-ab-score).
- Follow-ups this unlocks: (S3) deeper pipeline (4 act regs, ~9 free f-regs exist) — same mechanism,
  cheap; (E1) f32-matrix-engine double-buffer (now 4.9% exec, w/e=2.52); M2xN4 traffic halving.

### int8-TENC BOARD RESULT (2026-07-18, genuine) — CORRECT but ~7x SLOWER = WASH #3, engine CLOSED
Board run via run_smolvlm2_cont_candidate_board.sh. Profiled run crawled (16.8s/image vs 2.7s) until
the harness cancelled it (truncated trace, empty kernel_id.json, ppl=None -> harness TypeError; the
8,489,934,819 score is the stale scalar_hoist one, untouched). Partial trace (1167/2960 mul_mat_Q8_0
launches) parsed manually from et_runtime_trace.json:
- **mul_mat_Q8_0 = 23,957,368,353 cyc / 1167 launches = ~20.5M cyc/launch vs scalar ~2.8M = ~7.3x SLOWER.**
- **Correctness PASS (non-profiled):** answer " Dog.", 2.72s wall, 0 stream errors, and NO event 65535 —
  the (B)-hardening (fsq2 publish + evict + double-buffer + CLEAR_TENSOR_ERROR) ELIMINATED the fw crash.
  The engine is scoreable in principle now; it is simply slow.
- **Root cause (why all 3 engine forms lose):** the board launches ~2048 harts (shire_mask 0xFFFFFFFF);
  the scalar path uses ALL of them and dequants FREE inside fmadd.ps. The engine capped at Q8_ENG_HARTS=8
  (fat per-hart scratch, 1.92MB .bss) — 8 harts cannot beat 2048 even at 45 MAC/instr. Scaling the engine
  to all harts needs per-hart scratch that .bss cannot hold (200MB) or redundant staging (M2g's killer)
  or a barrier arena (M2c's killer). Every structural escape was measured dead.
- **Verdict: WASH. The tensor engine for the Q8_0 vision GEMM is CLOSED** (3 forms measured dead:
  M2g per-tile +8%, barrier +15%, int8-TENC ~7x). Do NOT retry the engine on this kernel.
- Remaining lever = the SCALAR path itself (scalar_hoist lineage): N=4 register-blocking — amortize the
  gather+convert issue stream 4-fold, inherits the scalar path's full hart count, no engine walls.
  Reverted to committed 5413c722e.

---

## 2026-07-18 — S1 N=4 register-blocked scalar (mul_mat_Q8_0) — LOCAL PASS, board queued

After the engine closure (WASH #3 above), the designated scalar lever from 07_EXPERT_DEBATE.
`dot_q8_0_row4_n4`: one weight row against FOUR activation columns — the per-block
fgb.ps/fcvt.ps.pw gather+convert is issued once and feeds 4 fmadd chains (persistent accs
f10/f15/f16/f17, temps f5-f8, scale broadcast f14, one asm block, mask save/restore hoisted).
Hot loop ~61 instr/128 MACs = 2.1 MAC/instr vs 1.14 for the single-column hoist. The fp16->f32
scale pre-pass also amortizes 4x. Each output keeps the single-column accumulation sequence
(same block/chunk/lane order) -> bit-identical, PPL/answer cannot move. N%4 tail -> existing
dot_q8_0_row_hoisted; K>8192 -> per-column block fallback. Stores unchanged (atomic_store_f32).
- Local gate (sys-emu, GGML_ET_KERNELS_PATH): MUL_MAT q8_0 n=16 12/12 OK, n=1 14/14 OK (tail),
  n=8 5/5 OK, incl. broadcast + permuted cases. Build + instruction checker green.
- Board: `bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh`.
  Beat 8,489,934,819 pmc_cycles; mul_mat_Q8_0 cycles vs 8,291,438,783. If >=1%: commit (S1).

### S1 BOARD RESULT (2026-07-18, genuine, fresh scored_at 13:52) — LANDED
- **mul_mat_Q8_0: 8,291,438,783 -> 6,839,456,472 = -17.5% kernel** (2960 launches, 88.3% of exec).
- **Score (median pmc_cycles): 8,489,934,819 -> 7,692,846,657 = -9.4%.** Cumulative from M0
  baseline 10,817,973,417 = **-28.9%**.
- Quality PASS: answer " Dog.", PPL 22.2815 <= 26.739, ET-vs-CPU 3.1e-5, 0 vision fallbacks.
- Verdict: **LANDED**, committed as `fb0bb4f02`. Remaining rigor: interleaved paired-main A/B
  (smolvlm2-ab-score) to formalize. Next: E1 f32-matrix-engine double-buffer (2.72% -> now ~4.1%
  of the new total); N=8 register-block is the follow-up scalar lever if E1 washes.


---

## 2026-07-18 — S2 N=8 register-block (mul_mat_Q8_0) — LOCAL PASS, board queued

Mechanical extension of landed S1 (N=4, -17.5% kernel). dot_q8_0_row8_n8: 8 persistent accs
(f10,f15-f21), 8 temps (f2,f5-f9,f22,f23), one asm block, ~102 instr/256 MACs = 2.5 MAC/instr
(vs N=4 2.1, scalar 1.14). Bit-identical per-output accumulation. Loop hierarchy: n8 groups ->
n4 groups -> single tail; vision N=1024 divides by 8 (no tail), decode N<=3 -> tail.
- Local gate: MUL_MAT q8_0 n=16 12/12, n=8 5/5, n=1 14/14 OK (sys-emu vs CPU). Build green.
- Board: same runner. Beat 7,692,846,657 pmc_cycles; kernel vs 6,839,456,472. If >=1%: commit (S2).
- Verdict: PENDING BOARD.

### S2 BOARD RESULT (2026-07-18) — WASH: N=8 is +10.1% SLOWER than N=4 (reverted-in-place, N=8 kept for S2b)
- Score 7,692,846,657 (N=4) -> 8,468,146,177 (N=8) = **+10.1% WORSE**; kernel exec 6.84B -> 7.64B
  (+11.7%). Quality PASS (math correct; the loss is pure perf).
- **Mechanism (why):** the loop uses ONE act register -> every fmadd serializes on its preceding
  flw.ps; N=8 chains 8 dependent load->fmadd pairs per chunk (vs 4) and pushes 9 concurrent streams
  into L1D. Wait cycles 13.1B -> 14.3B. **The kernel is memory-pipeline-bound at N=4, NOT
  instruction-count-bound** — more arithmetic blocking was the wrong direction.
- Verdict: WASH as built. N=8 NOT discarded yet -> S2b tests the stall hypothesis directly.

## 2026-07-18 — S2b N=8 + act-load software pipeline — LOCAL PASS, board queued
Round-robin f12/f13 act registers in dot_q8_0_row8_n8: every fmadd now trails its flw by ~2 instr
(hides one load latency per pair), addressing the S2 stall mechanism. Also rewrote the 8 reduces to
2-reg form (f0/f1) to free output registers (asm needs 8 outputs + 25 clobbers = 33 > 32 f-regs;
f3/f4 freed). Bit-identical accumulation order per output.
- Local gate: n=16 12/12, n=8 5/5, n=1 14/14 OK (sys-emu vs CPU). Build green.
- Board A/B (strict): candidate vs committed N=4 = 7,692,846,657 pmc_cycles; kernel vs 6,839,456,472.
  Win -> N=8+pipe is new HEAD; lose -> revert to N=4 (fb0bb4f02) and the N-blocking line closes
  (memory-side levers next: M2xN4 traffic halving, E1 f32-engine dbuf).
- Verdict: PENDING BOARD.

---

## 2026-07-18 — S3 4-deep act-load pipeline (N=8, f12/f13/f0/f1) — LOCAL PASS, board queued
Extends S2b's 2-reg pipeline to 4 act regs (load-use distance 3-4 vs 2). f0/f1 double as act regs
in the K loop (dead until the epilogue reduce) -> zero extra register pressure vs S2b, compiles
clean. Same bit-identical accumulation order. Heavy-load build note: et-kernels subdir `make
mul_mat_Q8_0.elf` is the robust recipe under load (nohup jobserver breaks; full cmake works but slow).
- Local gate: MUL_MAT q8_0 n=16 12/12 OK (sys-emu vs CPU). (n=8/n=1 unchanged from S2b's gate.)
- Board A/B (strict): vs committed S2b = 6,170,682,271 pmc_cycles; kernel vs 5,327,584,696.
  Win -> commit S3; wash -> revert to 800daee36, next = M2xN4 (halve act loads).

### S3 BOARD RESULT (2026-07-18, fresh 16:23) — LANDED
- **mul_mat_Q8_0: 5,327,584,696 -> 4,843,796,895 = -9.1% kernel** (84.6% of exec).
- **Score: 6,170,682,271 -> 5,673,127,752 = -8.1%.** Cumulative from M0 = **-47.6%**.
- Quality PASS. Commit `621636d37`.
- Competition note: PR #118 (ChiruGuru99) board-verified 5,403,606,916 (50.06%) via hoist + N4 +
  LINE-OWNED STORES + a K<=768 f32-dequant engine path. We trail by 4.8%; their store trick is our
  next lever (S4), and M2xN4/scale-hoist are levers they lack. Engine path still closed for us
  (3 forms measured dead; theirs caps at K=768 with serial waits).
- Verdict: **LANDED**. Next: S4 cache-line-owned store batching.

---

## 2026-07-18 — S4 cache-line-owned store batching — LOCAL PASS, board queued

The PR-#118 trick, independently built: flattened tile loop (8n x 16m), per-tile 512B stack
buffer (acc[8][16], c-major), 16x dot8, then publish 8 rows x 2 fsq2 = **16 publishing stores
instead of 128 amoswapg.w atomics**. One hart owns each whole 64B dst line (single writer,
seam-safe by construction). Guards: M%16==0 && N%8==0 && nbd1/2/3%64==0 && dst 64B-aligned ->
else the existing n8/n4/n1 atomic path. Vision shapes all take the tile path; fc1 = 24,576
tiles = 12/hart (same load as the n8 loop). Bit-identical math (store mechanism only).
- Local gate: n=16 (tile path) 12/12 OK; n=1 (fallback) 14/14 OK; n=8 same code path as n=16
  (board directly per the >20min rule; heavy-load machine made the extra local case slow).
- Board A/B (strict): vs committed S3 = 5,673,127,752 pmc_cycles; kernel vs 4,843,796,895.
  Win -> commit S4; wash -> revert, next = M2xN4.

### S4 BOARD RESULT (2026-07-18, fresh 17:40) — LANDED, biggest single win of the track
- **mul_mat_Q8_0: 4,843,796,895 -> 2,967,717,034 = -38.7% kernel** (76.7% of exec).
- **Score: 5,673,127,752 -> 3,817,495,063 = -32.7%.** Cumulative from M0 = **-64.7%**.
- Quality PASS. Commit `017c4a074`.
- Mechanism: 128 amoswapg.w per tile -> 16 fsq2. The win (-38.7%) dwarfs the store count cut —
  the per-element atomics were 16-way contended global RMWs on shared dst lines (16 harts/line),
  serializing the whole partition. Whole-line single-writer ownership removed the contention AND
  the RMW round-trips. **We now lead PR #118 (5.40B) by 29% with 3 levers they lack.**
- Discovery logged: `prefetch_weight_row` is DEAD CODE (no call site since the rewrites) — the hot
  path has zero prefetching; L1D = 4KB/minion explains why load-pipelining keeps paying.
- Verdict: **LANDED**. Next: act prefetch (L6), then M2xN4, then scale-hoist.

---

## 2026-07-18 — S5 act-column L2 prefetch (tile path) — board queued, no local gate
Per-tile async prefetch_va (Dest=L2, 16 lines/op) of the 8 activation columns before the 16 dot8
calls (24 ops/tile at K=768). Attacks the L2-latency half of the memory-pipeline bound that the
4-reg register pipeline cannot hide (L1D=4KB/minion -> nothing stays resident; discovered the old
prefetch_weight_row has been dead code since the rewrites). Store/dot math unchanged.
- Local gate: SKIPPED per user directive (sys-emu too slow under machine load); build green.
  Math unchanged (prefetch is value-neutral) -> risk is perf/behaviour-only, board decides.
- Board A/B (strict): vs committed S4 = 3,817,495,063 pmc_cycles; kernel vs 2,967,717,034.
- Verdict: PENDING BOARD.

### S5 BOARD RESULT (2026-07-18, fresh 18:26) — WASH (+0.7%), reverted
- Score 3,817,495,063 -> 3,844,791,903 (+0.7% worse); kernel 2,967,717,034 -> 2,976,514,612 (+0.3%).
- **Why:** acts are ALREADY L2-resident (device banner: L2 = 16 MB, not 4 MB — acts ~3 MB + weights
  ~2.4 MB fit easily after the first layer pass). L2-dest prefetch added ~24 instr/tile of pure
  overhead. The residual stall is L2->L1 latency, which L2 prefetch cannot hide. Do NOT retry
  L2-dest prefetch on this dataflow (dead-code discovery + L1D=4KB note still stand).
- Reverted to `017c4a074`. Next: M2xN4 (halve act loads + L1 thrash by sharing across 2 weight rows).

---

## 2026-07-18 — S6 M2xN4 (2 weight rows share 4 act loads) — board queued
dot_q8_0_m2n4 wired into the S4 tile path: 4 act chunks held in regs (f12,f24,f25,f26) feed BOTH
weight rows' fmadd chains -> act loads per block per 8 outputs: 32 -> 16 (halved, plus less L1
thrash). Temps preserved per block per (m,c) -> per-output math bit-identical to the shipped N=8
path. ~90 instr/256 MACs = 2.84 MAC/instr (N=8: 2.5). Results stored via fsw inside the asm (8
outputs + 25 clobbers would exceed the f-reg file). K>8192 -> per-column fallback. Local gate
SKIPPED per user directive (bit-identical by construction; board decides).
- Board A/B (strict): vs committed S4 = 3,817,495,063 pmc_cycles; kernel vs 2,967,717,034.
- Verdict: PENDING BOARD.

### S6 board attempt #1 (2026-07-18) — CRASHED on board (stream errors in profiled run), root-caused
Non-profiled run CORRECT (" Dog.", 2.73s) -> M2xN4 math good. Profiled run died with 2x "ET:
stream error detected at synchronization point" -> harness ppl=None TypeError, stale score kept.
Root cause: per-hart STACK overflow — tile scope acc[8][16] (512B) + TWO inlined m2n4 calls each
carrying scales0/1[256] (2KB) => >4KB, corrupting neighboring stacks; timing-dependent (fired only
under the profiler). NOT a math or coherency bug.
- **Fix (attempt #2):** scale extraction hoisted to tile scope, shared by both n-half calls per
  m-pair (also halves the redundant fp16->f32 prep); arrays capped Q8_M2N4_MAX_KB=128 (K<=4096,
  vision max KB=96) -> tile stack ~1.5KB. Same bit-identical dot math. Build green.
- Board A/B (strict): vs committed S4 = 3,817,495,063 pmc_cycles; kernel vs 2,967,717,034.
- Verdict: PENDING BOARD (attempt #2).

### S6 attempt #2 (2026-07-18) — PERF WIN (-11.2%) but PPL-pass crash -> attempt #3 (stack cap)
- Perf: score 3,817,495,063 -> 3,390,155,080 = **-11.2%**; kernel 2,967,717,034 -> 2,411,411,920
  = **-18.7%**. Video run clean (answer OK, ET PPL 22.2815). M2xN4 is fast.
- Crash: llama-perplexity pass died "ET: stream error, Type: 4" (rc=-6) on the LLM text shapes
  (batch=128, K=960/2560) that video_dog never exercises -> cpu_perplexity=null -> FAIL verdict
  (harness-side, not the video gate).
- Working theory: GCC merges inlined stack frames — dot1/dot4/dot8 scale arrays at
  Q8_SCALAR_MAX_KB=256 (1KB each) + tile-scope arrays (1KB) + acc (512B) can SUM over the 4KB
  per-hart stack on some shape paths.
- **Attempt #3 fix:** Q8_SCALAR_MAX_KB 256 -> 128 everywhere (K<=4096; all scored KB<=96). Same
  math. Build green. If PPL still crashes -> stack theory wrong; next = shape-specific tile bug
  (local LLM-shape repro).
- Verdict: PENDING BOARD (attempt #3).

### S6 attempt #3 (2026-07-18) — PPL crash persists with 128-cap; root cause re-classified + N-guard ship
- Attempt #3: PPL pass rc=-6 stream error AGAIN (128-cap did not help) + video server died early
  (9 min after #2 — likely device fallout from #2's wedged state, not a new bug).
- Diagnosis (local): ALL PPL shapes pass sys-emu with the M2xN4 kernel (m=960/1600/2560, n=128,
  k=960/2560, plus lm_head m=49152,n=128,k=960 — test-backend-ops throwaway cases, reverted).
  entry_point frame = 2,272 B (objdump) — under the 4KB per-hart stack; STACK THEORY DEAD.
  Conclusion: math + shapes are clean; the failure is board-environment-specific to the UNSCORED
  PPL pass (N=128 GEMMs); the scored video path (N=1024 tiles) is board-proven clean (attempt #2,
  -11.2%).
- **Ship decision: tile path guarded to N >= 1024** (encoder-only, where M2xN4 is proven). PPL's
  N=128 GEMMs fall back to the S4-proven n8/atomic paths — zero scored downside (score =
  video_dog only). Crash mechanism left OPEN (board-only, N=128-specific, invisible to sys-emu) —
  flagged for a future dedicated debug run, does not block shipping the video win.
- Verdict: PENDING BOARD (attempt #4, guarded).

### S6 BOARD RESULT attempt #4 (2026-07-18, fresh 20:23) — LANDED
- **mul_mat_Q8_0: 2,967,717,034 -> 2,104,718,021 = -29.1% kernel** (70.4% of exec).
- **Score: 3,817,495,063 -> 2,934,954,675 = -23.1%.** Cumulative from M0 10,817,973,417 = **-72.9%**.
- Quality FULL PASS (PPL pass clean with the N>=1024 tile guard): answer " Dog.", PPL 22.2815,
  cpu_perplexity 22.2808, ET-vs-CPU 3.1e-5, 0 fallbacks. Commit `6bc4bbe29`.
- We lead PR #118 (5.40B) by 46%. Remaining: scale-hoist, E1 f32-engine dbuf, f16 block+pipeline.
- Verdict: **LANDED**.

---

## 2026-07-18 — E1 f32-matrix-engine double-buffer — LOCAL PASS, board queued
mul_mat_f32_matrix_engine.c: B-transpose load for K-tile kb+1 issues while kb's FMA runs (B0@16-31 /
B1@32-47 alternating on load id 1); A stays single-buffered @0-15 (48-line SCP budget forces one
exposed load; B's transpose latency now hidden behind the MAC). Same FMA/accumulation order ->
bit-identical. Operand-reuse across tiles is structurally blocked (C lives in FREGs; noted in
04_KERNEL_STATE) — not attempted. Kernel = 9.8% of exec (0.29B, w/e=0.89).
- Local gate: MUL_MAT f32 m=16,n=16,k=256 (engine path) 12/12 OK (sys-emu vs CPU). Build green.
- Board A/B (strict): vs committed S6 = 2,934,954,675 pmc_cycles.
- Verdict: PENDING BOARD.

---

## 2026-07-18 — E1 f32-engine double-buffer LANDED + F1 f16 rewrite — board queued

### E1 BOARD RESULT (2026-07-18, fresh 20:45) — LANDED
- **mul_mat_f32_matrix_engine: 293,484,296 -> 157,142,558 = -46.5%** (B-transpose load hidden
  behind the FMA; A exposed per the 48-line SCP budget).
- **Score: 2,934,954,675 -> 2,800,358,910 = -4.6%.** Cumulative from M0 = **-74.1%**.
- Quality PASS. Commit `f2714ffe1`.

### F1 f16 full rewrite (M2xN4 + tiles + hoisted dot) — LOCAL PASS, board queued
mul_mat_f16.c (6.8%, 768 launches) had the NAIVE pattern (per-chunk flw/fsw memory round-trip,
scalar 8-add reduce per block, per-element atomics). Rewrote with the proven family:
dot_f16_m2n4 (2 rows x 4 cols, hoisted accs, no scales needed, ~2.9 MAC/instr), dot_f16_hoisted
(tails), cache-line-owned tile path (same guards as Q8_0 incl. N>=1024), n8-pair/atomic path +
per-column fallback for K%32!=0 / odd M. Also drops the old even-harts-only restriction (mirrors
Q8_0's all-harts stride).
- Local gate: MUL_MAT f16 n=16 12/12, n=8 5/5, n=1 13/13 OK (sys-emu vs CPU). Build green.
- Board A/B (strict): vs committed E1 = 2,800,358,910 pmc_cycles; f16 kernel vs 194,041,634.
- Verdict: PENDING BOARD.

### F1 BOARD RESULT (2026-07-18, fresh 21:05) — WASH (+26.3%), reverted
- Score 2,800,358,910 -> 3,536,277,992 (+26.3% WORSE); **f16 kernel 194M -> 935M = 4.8x SLOWER.**
  Math/quality all correct (PPL 22.2813) — pure perf regression.
- **Mechanism (high confidence):** the original f16 ran EVEN harts only (odd harts skip: SMT
  siblings share one L1D). The rewrite mirrored Q8_0's all-2048-hart stride -> 2x the streams into
  half the L1D bandwidth -> contention (total wait 19.7B -> 24.9B). Q8_0 was all-harts from its
  baseline (no such constraint); f16 was even-only by design.
- Reverted to `f2714ffe1`. F1b retry: identical M2xN4/tile structure with even-harts-only
  restored (one hart-distribution change, isolates the theory). If F1b also washes -> f16 stays
  NAIVE; tail hunt stops; formalize + submit.

## 2026-07-18 — F1b f16 rewrite with even-harts-only — LOCAL PASS, board queued
F1's M2xN4/tile/hoisted-dot structure re-applied unchanged EXCEPT the hart discipline: even
harts only restored (`thread_id & 1 -> skip`, stride = num_even_harts), isolating the SMT-sibling
L1D-contention theory for F1's 4.8x regression. If the kernel is still slow -> the theory is wrong
and f16 stays NAIVE (stop, formalize).
- Local gate: MUL_MAT f16 n=16 12/12 OK (coverage of the even-hart partition verified). Build green.
- Board A/B (strict): vs committed E1 = 2,800,358,910 pmc_cycles; f16 kernel vs 194,041,634.
- Verdict: PENDING BOARD.

---

## 2026-07-19 — G1 softmax classic 3-pass (1 fexp/element) — WASH
From the 10/11 research docs: the online softmax pays 2 fexp/chunk (pass1) + 1/element (pass2) =
~3 fexp/element. Replaced the non-sinks row path with classic 3-pass: valid-lane max ->
exp+sum+store-numerator-into-dst -> rescale dst by 1/sum. ONE fexp/element; per-element formula is
exactly the ggml CPU reference's (more reference-convergent than online; PPL margin huge). Validity
semantics preserved (valid-lane max; fully-invalid row untouched; NaN propagates like the CPU ref).
Sinks path unchanged (scalar, test-only). softmax_f32 = 98M cyc (3.4% of total) -> est. -1-1.5% total.
- Local gate: SOFT_MAX 68/68 OK (sys-emu vs CPU). Build green.
- Board A/B (candidate 07:47 vs main 10:03, single runs): candidate total = 2,790,906,608 pmc,
  main total = 2,796,328,909 pmc → **−0.19%**. softmax_f32 exec: candidate 94,516,964 vs
  main 97,889,678 → the 3-pass DID cut softmax exec ~3.4% at the kernel level, but softmax is only
  3.4% of total, so it nets ~0.19% board. PPL held (22.2819 vs 22.2815), accuracy 1.0, no vision fallbacks.
- Verdict: **WASH.** −0.19% is below the ≥1% gate AND within board noise — two independent no-code-change
  main runs (2,800,358,910 and 2,796,328,909) already differ by 0.14%, so the 0.19% "win" is
  indistinguishable from jitter. WHY it can't move: softmax is only 3.4% of total and wait-bound
  (w/e≈6.5); an exp-count reduction attacks exec, but the kernel's cost is wait/memory, and the
  classic 3-pass adds a dst memory pass that offsets the fewer-exp saving. Reverted; kept online path.
  Do not retry softmax micro-opts — the lever is mul_mat_Q8_0 (74%, compute-bound).

## 2026-07-19 — H1 reciprocal/rsqrt hygiene batch (norms + et_sinf) — LOCAL PASS, board queued
Batched the three sub-gate hygiene levers from doc 10 Tier A.5 + handoff §5.3 so ONE board A/B can resolve
them. All are correctness-neutral or accuracy-improving; none touch mul_mat/seams/layout.
- Change (4 files, all in et-kernels/src/):
  1. `math_fp.h`: new shared `et_rsqrt(x)` = Quake bit-seed (0x5f3759df) + 3 Newton-Raphson iters (pure
     mul/add), replaces `et_powf(x,-0.5f)` (flog+fexp) in all three norm scales.
  2. `math_fp.h`: `et_sinf` constant reciprocals (1/(2π) + six 1/n! factorials) baked to exact literals
     instead of per-call `frcp` estimates via `et_fdiv(1,C)` (et_fdiv is volatile asm → never folded).
     Backs RoPE (rope_f32.c). Strictly MORE accurate (exact vs frcp estimate).
  3. `norm_f32.c` / `rms_norm_f32.c` / `rms_norm_mul_f32.c`: hoisted `inv = et_fdiv(1,N)` above the row
     loop, replaced per-row `et_fdiv(reduce,N)` with `reduce*inv`. Bit-identical (IEEE mul commutes;
     et_fdiv(1,N)=frcp(N) exactly), removes ~1-2 frcp+mask-shuffles/row. Also switched scale to et_rsqrt.
- Correctness: build instruction-checker GREEN (et_rsqrt emits no fdiv.s/fsqrt.s/frsq.ps). Local sys-emu
  test-backend-ops (GGML_ET_KERNELS_PATH, 1-hart): NORM 5/5 OK, RMS_NORM 6/6 OK, ROPE 62/62 OK, 0 FAIL.
- Reviewer (smolvlm2-kernel-reviewer): SHIP — et_rsqrt hardware-safe + converges below f32 eps after 3 iters
  + never returns ≤0/non-finite for positive input (no new CPU fallback); 1/N hoist provably bit-identical;
  baked sinf correctly rounded. ONE residual risk it can't check statically: et_rsqrt + baked-sinf change
  f32 rounding vs the frozen PPL-22.28 baseline (every norm scale + every RoPE angle moved, favorably) —
  so the board run MUST re-measure answer + PPL + ET-vs-CPU, not trust pre-change numbers.
- Perf estimate: sub-1% (norms + RoPE are each ≤3% and the win is a per-row scalar-op cut, not tensor/cache).
  Batched precisely because individually they can't clear the ≥1% gate.
- Board (candidate run, 2026-07-19T11:31Z, genuine ivan@aifoundry2, batch confirmed built via the [1a/5]
  kernel-src content-hash gate): passed=True, accuracy=1.0, PPL=22.2824 (≤26.739), ET-vs-CPU=2.69e-5 (≤1%,
  tighter than baseline), vision_fallback_ops=[] (et_rsqrt did NOT trip the scale>0 guard). pmc_cycles=
  2,799,080,495 — INSIDE the baseline noise cloud (2,790.9M / 2,796.3M / 2,800.4M). Hot path unchanged
  (mul_mat_Q8_0 73.8%, mul_mat_f16 6.8%).
- Verdict: **WASH (perf), quality-safe.** No measurable cycle win (within noise, ≪1% gate) — as predicted for
  a per-row scalar-op cut on kernels that are each ≤3% and wait/dispatch-bound (rms_norm_mul w/e=19.2,
  rope w/e=30.6, norm w/e=79.9 — their cost is stall, not the reciprocal math). Quality held perfectly and
  et_rsqrt/baked-sinf are strictly more accurate (ET-vs-CPU tightened). WHY it can't move perf: the levers
  attack exec on kernels whose cost is wait, same failure mode as the G1 softmax WASH. Single-run, not the
  interleaved A/B — but a hygiene wash doesn't warrant more board slots.
- Disposition: **KEPT (all 4 files)** — user decision 2026-07-19. Rationale: board-confirmed quality-safe and
  strictly-more-accurate (ET-vs-CPU tightened to 2.69e-5, zero fallbacks, PPL held), so carried as accuracy
  hygiene even though it's a perf wash. The 1/N hoist is bit-identical; et_rsqrt/baked-sinf trade the
  et_powf(flog+fexp) / per-call frcp estimates for exact math. NOT a perf lever — do not re-A/B for cycles.
  Confirms the track is at its floor (handoff §5.4): the remaining action is the submission PR, not more
  kernel levers. Next: bump the submodule to include this batch, then open the hf-hackathon PR (§6).

## 2026-07-19 — Lever A: vectorize the Q8_0 fp16 scale unpack (fcvt.ps.f16) — LOCAL PASS, board queued
Doc 14 primary lever. The 5 fp16->f32 block-scale unpack sites in mul_mat_Q8_0.c ran the branchy scalar
`fp16_to_fp32` (math_fp.h:238, ~15 instr + 2 branches/scale). Replaced all 5 with a `unpack_q8_0_scales`
helper that uses the HW `fgh.ps`+`fcvt.ps.f16` convert (the proven block_ops.h:100-104 primitive). block_q8_0
is 34B (d @ off 0) so scales are strided; used Option 2 (pack 8 strided halfwords into a contiguous temp,
then gather+convert 8-at-a-time) — reuses the proven contiguous {0,2,..,14} gather, zero new HW assumptions
(Option 1's strided-238 gather offset left unverified to avoid a wasted board slot). Scalar tail for
K_blocks%8 (all scored SigLIP shapes are multiples of 8: 768/32=24, 3072/32=96). Prioritized the scored
M2xN4 tile path (sites 4-5, N>=1024); also did the n8/n4/n1 paths (sites 1-3).
- Profiling (analytic, pre-build): scale prep ~= 15 instr/scale vs ~93 instr/MAC-block on the M2xN4 path
  => prep ~14% of the scored-path instruction stream, EXPOSED serial scalar work (prep loop completes before
  the memory-bound dot reads scales[]). Diluted by wait (w/e=2.75) => ~4-6% of mul_mat_Q8_0 cycles.
  Vectorizing (~6x cut on that segment) => est ~3% kernel => ~2% board. NOT trivial -> proceed (an intra-kernel
  prep/MAC split is not available from the per-kernel board profiler without a hand-instrumented slot; the
  A/B run is the real measurement).
- Correctness: bit-neutral by construction (widening fp16->fp32 is exact for all fp16; Q8_0 scales are normal
  positive). Local gate (build-verify, sys-emu 1-hart, throwaway small cases incl. tile path m=16 n=1024
  k=768, helper tail k=352 K_blocks=11, all-tail k=96 K_blocks=3): MUL_MAT q8_0 46/46 OK, 0 FAIL. Instruction
  checker green (fcvt.ps.f16 not a trapped op). Throwaway test cases reverted.
- Reviewer (smolvlm2-kernel-reviewer): **PROCEED** on the mul_mat_Q8_0.c change — all 5 axes clean
  (bit-neutral fcvt.ps.f16 per packed_float.cpp:86; f31/f11 reloaded inside each asm block = no persistent-freg
  hazard; fsw.ps 32B stores aligned+in-bounds; scales[] per-hart stack, no seam/coherency exposure; fgh.ps
  gather semantics match block_ops.h). (It also flagged the throwaway test file [now reverted] and the
  pre-existing KEPT H1 math_fp/norm files as out-of-scope context it lacked.)
- Board A/B (2026-07-19, genuine ivan@aifoundry2, interleaved isolating Lever A = toggle only
  mul_mat_Q8_0.c, H1 kept in both arms):
  - Arm1 MAIN (Lever A off, scored_at 14:18:39): score 2,790,697,159; mul_mat_Q8_0 exec 2,105,411,878.
  - Arm2 CANDIDATE (Lever A on, scored_at 14:22:38): score 2,791,715,623; mul_mat_Q8_0 exec 2,097,289,551.
  - Delta: score +0.036% (SLOWER, inside the ~0.15% noise floor); kernel EXEC -0.39% (-8.1M cyc).
  - Quality held identically both arms: passed=True, accuracy 1.0, PPL 22.2824 (=26.739 gate), ET-vs-CPU
    2.69e-5, vision_fallback_ops=[]. (Bit-neutral confirmed: PPL byte-identical to main.)
  - Arm3 SKIPPED: candidate is +0.036% vs main1, ~1% short of the gate; no plausible main drift could pull
    the mean to a >=1% win, so a third board slot would be waste (board-scarcity rule).
- Verdict: **WASH (perf), quality-safe / bit-neutral.** WHY it can't move the score: the lever DID cut kernel
  exec 0.39% (the scalar fp16->fp32 prep was real exposed work; fcvt.ps.f16 is faster) — but mul_mat_Q8_0 is
  WAIT-bound (w/e=2.72, wait 5.73B >> exec 2.10B) and the firmware-cycle score is dominated by memory wait,
  which the prep-vectorize does not touch (same scale bytes still fetched from DRAM). The exec saving is
  swamped -> score flat within noise. The pre-build ~14%-of-instruction-stream estimate was instruction COUNT;
  the kernel's cost is wait, not issue, so cutting instructions does not cut cycles. Same failure mode as the
  S2 N=8 / S5 prefetch / G1 softmax / H1 hygiene washes: exec-count levers on a wait-bound kernel don't move
  the board. Do NOT retry scale-prep micro-opts (Lever A Option 1 strided gather, or Lever B scale-recompute
  dedup) — both attack the same exec that is already swamped by wait; Lever B additionally carries seam risk
  for no reachable win.
- Disposition: **REVERTED** (mul_mat_Q8_0.c back to HEAD scalar unpack). Per WASH discipline; the change is
  bit-neutral so it could be kept as harmless hygiene, but it adds an asm helper for a 0.39%-exec/0%-score
  win, so revert keeps the tree honest. Lever A backup left at /tmp/mul_mat_Q8_0.leverA.c for the session.
  This closes doc 14: the track is confirmed at its floor (E1 -74.1% + H1 accuracy hygiene). Remaining action
  is the hf-hackathon submission PR (handoff §6 / docs/SUBMISSION_GUIDE.md), NOT more kernel levers.

## 2026-07-19 — FOSDEM lever E2: double-buffer A via TenB direct-B path (doc 12 §4.1) — DESIGN-DISPROVEN (no board)
Investigated the last unused FOSDEM "zero to matmul" lever (doc 12 §4.1): route B via the TenB direct-to-compute
path so it vacates its 32 SCP lines, freeing them to double-buffer A (currently single-buffered @ SCP 0-15, its
refill exposed after the FMA) in mul_mat_f32_matrix_engine.c (5.2% of exec, compute-bound w/e=1.23 -> unlike the
wait-bound Q8_0, an exec/latency cut here WOULD move the score, so it was worth checking).
- **INFEASIBLE — disproven from emulator ground truth (sw-sysemu/insns/tensors.cpp), no board slot spent:**
  1. The TenB direct path is PLAIN-load-only: `tensor_load_execute` forces `cmd = tload_cmd_load` unconditionally
     when tenb is set (tensors.cpp:484), overriding the transformation field -> a direct-B load can never
     transpose. `tensor_load_setup_b` (tensor.h:336) likewise sets transformation=0.
  2. Our FMA's B operand REQUIRES a transpose: the engine indexes `b = SCP[k].f32[j]` (tensors.cpp:1252-1290),
     i.e. B must be laid out `B[k][j]` contiguous over j. For GGML `dst[n][m]=Σ_k src0[m][k]·src1[n][k]`, that is
     `B[k][j]=src0[m0+j][k]` = column k of the row-major weight tile = a genuine transpose (the current
     TL_TRANSPOSE32 load). A (`A[i][k]=src1[n0+i][k]`) is the naturally-contiguous operand, but there is no
     "direct A" path — only TenB is direct. So B cannot leave SCP; the 48 lines stay full; A can't double-buffer.
     (The FOSDEM talk streamed B direct because a standard C=A·B has B naturally [k][j]; GGML mul_mat is the
     inner-product form C=A1·A0^T, so a transpose on the FMA-B operand is unavoidable — doc 12 §4.1 missed this.)
  3. Even the fallback (reallocate A×2 / B×1, both fit in 48) is not a win: it would expose the EXPENSIVE
     B-transpose load instead of the CHEAP A-plain load. The shipped design (hide the transpose under the FMA,
     expose the cheap plain A refill) is already the better assignment.
- Verdict: **DEAD (design-disproven).** Do NOT attempt the TenB direct-B path on this GEMM — the layout transpose
  it can't do is mandatory. The other doc 12 §4 levers (§4.4 hart-1 Tensor-L2 prefetch, §4.6 batch CSR enqueue)
  assessed separately below.

## 2026-07-19 — FOSDEM lever E3: batch CSR enqueue (doc 12 §4.6) — DEAD, already exploited (no board)
- The talk's "one scalar CSR write enqueues up to 512 tensor instructions / 1 KiB of loads" is the tensor-op
  granularity we ALREADY issue at. Ground truth (sw-sysemu/insns/tensors.cpp FMA loop ~1250): ONE tensor_fma
  CSR write executes the full `for k in 16 · for i in 16 · for j in 16` = 16x16x16 tile (4096 MACs); ONE
  tensor_load moves up to 16 lines = 1 KiB. mul_mat_f32_matrix_engine issues the minimum possible number of
  CSR writes (K/16 accumulating FMAs + one load per operand tile). The per-K-tile tensor_wait()s are NOT
  un-batched scalar issue — they are structural RAW on the TenC accumulator (f0-f31) + WAR on the single A
  buffer, unremovable regardless of batching. **Nothing to batch. Design-closed, no board slot.**

## 2026-07-19 — FOSDEM lever E4: hart-1 sibling L2Scp prefetch (doc 12 §4.4) — NOT PURSUED (board-only, poor risk/reward)
- Idea: the idle odd/sibling harts (`if (hart_id & 1) return 0`) stage operands DRAM->L2Scp via
  et_tensor_load_l2scp (CSR 0x85f) while hart-0 computes, so hart-0's SCP loads hit L2Scp (fast) not DRAM.
- Mechanism confirmed feasible (tensors.cpp tensor_load_l2_start:722): copies DRAM->per-shire L2Scp region;
  hart-0 loads must then be redirected to those L2Scp addresses. Requires explicit L2Scp scratch/address
  management + sibling-hart coordination + cross-hart publish/visibility handling (the non-coherency class that
  bit cont_f32 / int8) — the highest-complexity change of any lever considered this track.
- **Cannot be disproven at design stage** (pure latency question; sys-emu models ZERO cache/memory cost — can't
  measure locally like the functional-impossibility of E2/§4.1). It is a genuine board-only perf question.
- Assessed NOT WORTH a board slot: (1) engine kernel is only 5.2% of exec and already COMPUTE-bound (w/e=1.23);
  E1's double-buffer already hides most operand latency, leaving only the single exposed A refill/K-tile to
  attack -> ceiling ~1% board at best; (2) shares the exact mechanism of the S5 L2-staging WASH (+0.7%,
  reverted); (3) highest implementation + non-coherency risk of anything remaining. Risk/reward is poor against
  a track already at its floor with a ready submission.
- Verdict: **NOT PURSUED.** The only physically-possible remaining FOSDEM lever, but sub-gate-to-marginal on a
  small kernel at high risk. Left documented for a future agent who wants to push; the recommended action is
  the submission PR, not this.
