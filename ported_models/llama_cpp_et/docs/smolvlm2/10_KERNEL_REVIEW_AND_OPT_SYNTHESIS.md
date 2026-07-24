# SmolVLM2 — Kernel Review + Optimization Synthesis

**Date:** 2026-07-18
**Trigger:** code review of commit `f2714ffe1` ("perf(smolvlm2): double-buffer B load in f32 matrix engine, −46.5% kernel"), followed by a multi-agent optimization + literature + hardware sweep.
**Method:** 7 parallel review/research agents (correctness, comments, readability, missed-opt, GEMM literature, kernel-used optimization, ET-SoC-1 sim microarchitecture), reconciled against the actual code and the board-measured ledger (`EXPERIMENTS.md`).

This document is the synthesized deliverable: **(1)** the review outcome for `f2714ffe1`, **(2)** verified hardware facts that correct two prior assumptions, **(3)** where the cycles actually are, **(4)** a cited literature map, **(5)** a ranked, gate-attached optimization roadmap, and **(6)** proven dead ends.

---

## 1. Executive summary

- **`f2714ffe1` is correct and shippable.** Adversarial review verdict: SHIP. The B double-buffer is a pure *scheduling* transform — identical FMA sequence, identical K-order, identical `first_pass` semantics — so PPL and the exact `video_dog` answer are bit-identical by construction. The "−46.5% kernel" must be a *board* figure (sys-emu models zero tensor/cache cycle cost).
- **A documentation + readability cleanup was applied** to the reviewed file (comments-only / named-constants / hoisted-identical-expressions / local renames — provably behavior-preserving). The file's header docstring had drifted to describe an obsolete single-tile prototype; it now describes the real tiled/multi-hart/double-buffered kernel.
- **The reviewed kernel is NOT the hot kernel.** The frozen model ships **Q8_0 weights**, and dispatch keys on `src0` dtype, so the dominant SigLIP GEMMs land in the **scalar `mul_mat_Q8_0.c`** (board-measured ~76% of cycles), which uses *all* ~2048 harts and dequants for free inside `fmadd.ps`. `mul_mat_f32_matrix_engine.c` sees only ~2.7–5.6% of launches.
- **Two prior "dead end" assumptions were corrected by reading the ET-SoC-1 behavioral simulator** (`sw-sysemu/`):
  1. **A *can* be double-buffered.** SCP is 48 general lines (prior claim correct), but there is a **separate 16-line TenB bank (lines 48–63) with its own third load FSM**, currently unused. Moving B' into TenB frees a general slot for A's second buffer.
  2. **`tenc_loc=1` cannot hold two fp32 output tiles.** No memory-C path exists for float; the second accumulator (`TENC`) is **int8-only**. The single-accumulator wall is real for fp32/fp16.
- **The single biggest theoretical lever — routing the dominant GEMM onto the tensor engine — is empirically closed** (measured dead 3×: hart-count asymmetry + the `.bss` execution-context load wall). Realistic remaining headroom is small and concentrated in exact, in-surface fusions.

---

## 2. Review of `f2714ffe1` (the four angles)

### 2.1 Correctness — **SHIP** (verified bit-exact)
- B-buffer alternation is a perfect ping-pong: FMA reads `BSTART = (kb/TILE)&1 ? 32 : 16`; the prefetch writes the *opposite* buffer. The buffer being read (16 lines) is never the buffer being written. Prologue B0@line-16 matches the `kb=0` read.
- No load-ID hazard: one id-1 load per iteration; the loop-top `tensor_wait(TENSOR_LOAD_WAIT_1)` drains the prior one first.
- Edge cases hold: K=16 (prologue-only, prefetch branches correctly skipped), N-edge (`n_cur`), `first_pass` (kb==0 overwrite).
- A single-buffering is honored correctly: its reload issues *after* `TENSOR_FMA_WAIT` (WAR-safe).
- SCP 48/48 within the 48-line general region (arch ref §2.2; confirmed `cache.h:23`).

### 2.2 Comments, readability — **cleanup applied**
The header docstring described an obsolete "minimal single M=N=K=16 test / only hart 0 / SCP 32-of-48." Rewritten to the real design. Applied, all behavior-preserving:
- Header docstring corrected (tiled grid, even-hart partition, N-edge, K>16, SCP 48/48, B double-buffer range).
- `FENCE` "// XXX: Do we need this?" resolved to a *reason* (stores bypass L1/L2 and are non-coherent across shires — visibility barrier before return).
- The two `XXX` musings replaced by the real constraint (tensor_fma clobbers f0–f31; no live FP value may cross it unspilled).
- Named `b_read`/`b_write` (hoisted once) so the correctness-critical read/write *swap* is stated in one place.
- `/*field=*/` labels on every `tensor_load`/`tensor_fma`/`tensor_store` call.
- Named the magic literals (`TL_TRANSPOSE32`, `A_SCP`/`B_SCP0`/`B_SCP1`, `FMA_BCOLS_16` vs `STORE_SIZE_64B`).
- `nb`/`mb` → `n0`/`m0` (collision with the `nb1_1` byte-stride names).

### 2.3 Build / math gate status — **build green; f32 correctness PASS (198/198)**
- Kernel ELF **rebuilds clean** with `ET_PLATFORM=~/et-src/et`; the post-build `check_unimplemented_instructions.sh` passed (no `fdiv.s`/`fsqrt.s` — the build would fail otherwise).
- `test-backend-ops -b ET -o MUL_MAT -p type_a=f32,type_b=f32` (sys-emu, single hart) → **198/198 tests passed, Backend ET: OK.** Coverage includes large-K cases (k=1056/1057, which iterate the double-buffered K-loop dozens of times) and N-edge cases (`n=1`). Log: `local-artifacts/mulmat_f32_engine_verify.log`.
  - (The unfiltered full `MUL_MAT` sweep timed out earlier at 560s — too slow across all dtypes under sys-emu; the f32 filter isolates exactly this kernel's path.)
- **Consistent with expectation:** the applied edits are comment/rename/constant-only and provably behavior-preserving, so `max_abs` cannot move relative to the pre-edit `f2714ffe1`; the 198/198 PASS confirms it empirically.

---

## 3. Verified hardware facts (from the `sw-sysemu` behavioral model)

All evidence in `/home/marin/et-src/et-platform/sw-sysemu/`. These are *functional/structural* truths (sys-emu is cycle-cost-free, so no latency numbers, but the structural interlocks are modeled and constrain legal schedules).

| Constant | Value | Evidence |
|---|---|---|
| `L1_SCP_ENTRIES` (general SCP) | **48** (12 sets × 4 ways) | `cache.h:23` |
| `scp[]` array size | **64** = 48 + `TFMA_MAX_AROWS`(16) | `processor.h:224`, `emu_defines.h:183` |
| Load FSMs | **3**: `tload_a[0]`, `tload_a[1]`, `tload_b` | `processor.h:248-249` |
| L2 scratchpad (per shire) | **4 MB** | `emu_defines.h:138` |
| `NFREGS` | 32 (one 16×16 fp32 tile) | `state.h:58` |
| FMA K-per-issue (`acols`) | fp32 ≤16, **f16 ≤32, int8 ≤64** | `tensors.cpp:1335-1338,1440-1443` |
| Threads per minion | 2 (tensor ops **thread0-only**; `tensor_load_l2` either) | `emu_defines.h:80`, `zicsr.cpp:462,515` |

**Corrections to prior assumptions:**

1. **A double-buffering is FEASIBLE (prior review said "dead").** Lines 48–63 are a dedicated **TenB** bank reachable only via a `tensor_load` with `tenb=1` and an FMA with its `tenb` bit set (`tensors.cpp:479-483,1256`), on the **third** load FSM. The kernel today uses only ids 0/1 in the general region and `tenb_loc=false`, so **the entire TenB bank + third FSM is idle**. Putting B' in TenB frees general lines 32–47 to become A's second buffer → the exposed A-load disappears. Cost: the TenB pairing protocol is lockstep (exactly one TenB load outstanding, paired with its FMA — `tensors.cpp:490-493`), so the FMA must alternate `tenb=0`/`tenb=1`. Real restructure, but enabled.

2. **`tenc_loc=1` two-fp32-tile trick is DEAD (confirmed).** B always comes from SCP (no DRAM-B path); fp32/fp16 C is only f0–f31. The second accumulator `Core::tenc` exists **for int8 only** and must be drained to FREGS via `tenc2rf` before store (`tensors.cpp:1464-1465`). The "can't hold two output tiles" constraint stands for the float path.

**Other exploitable hardware features found:**
- **Cooperative loads** (`use_coop`, ctrl bit 62): N minions co-fetch one 16-line tile, splitting bandwidth — for shared weight tiles in a multi-hart partition (needs shire coop-mode ESR, else illegal-instruction trap).
- **On-load transforms** beyond transpose32: transpose8/16, interleave8/16 — free A/B repacking for an f16/int8 engine path (no separate `cont` kernel).
- **`tmask` / `arows`**: skip ragged-edge rows (1024/3072 not ÷16) and apply attention masks with zero wasted MACs.
- **`tensor_quant`** (chained INT32→FP32, per-row/col scale, saturate, pack): a fused dequant/requant epilogue and a ready-made per-row softmax normalize.
- **`tensor_reduce`** (cross-minion fadd/fmax tree): split-K partial-sum accumulation and distributed softmax max/denominator.
- **`tensor_load_l2`** into the 4 MB per-shire L2 SCP, issuable by the *sibling* thread → a B-panel cache prefetched in parallel with thread0's FMAs.

---

## 4. Where the cycles actually are (board-measured, `EXPERIMENTS.md`)

M0 baseline **10.87 B** firmware cycles (genuine board, `device_cmd_exec_dur`); quality gate **PPL ≤ 26.739** (M0 measured 22.2825), accuracy 1.0, ET-vs-CPU ≤1%.

| GEMM class | MAC share | Routes to | Notes |
|---|---|---|---|
| SigLIP MLP fc1/fc2 | ~43% | **`mul_mat_Q8_0.c`** (scalar) | ~76% of board cycles; already BLIS-style N=8 / M2×N4 register-blocked |
| Attention QKV/O proj | ~22% | **`mul_mat_Q8_0.c`** | same |
| O(N²) self-attention | ~14% MAC | score/context → `mul_mat`; softmax → `softmax_f32.c` | softmax grew to ~3.4% of the *shrunken* total |
| `cont_f32` | — | `cont_f32.c` | board #2 at 15.84%; **already landed** (M1, −15.5%) |
| Patch-embed im2col | ~0.5% | `im2col_f32.c` | scalar per-element gather |
| `mul_mat_f32_matrix_engine.c` (**the reviewed kernel**) | ~2.7–5.6% | F32-aligned connector/attention side | Q8_0 vision GEMM never reaches it |

**The double-buffer diff optimizes ~2.7–5.6% of the board.** Correct and worth landing, but small absolute.

---

## 5. Literature map (cited) — the math under the kernels

| Technique | Paper | Maps to | Verdict on our hardware |
|---|---|---|---|
| GEMM anatomy / operand reuse | Goto & van de Geijn, TOMS 2008 ([PDF](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf)) | `TensorLoadTranspose32` = HW pack+transpose; cache-resident panel reuse | Pack-transpose already free; panel-reuse reorder bounded by 48-line SCP |
| BLIS micro-kernel / register blocking | Van Zee & van de Geijn TOMS 2015; Smith IPDPS 2014 ([BLISlab](https://arxiv.org/pdf/1609.00076)) | **`mul_mat_Q8_0.c`** already N=8 / M2×N4 | **DEAD for the engine** (1 FMA = all 32 regs = 1 tile); alive & applied in the VPU kernel |
| Operand double-buffer / SW pipeline | CUTLASS efficient-GEMM; Lam PLDI 1988; Rau 1994 | **the reviewed diff** | Exactly this; extend to 3-buffer via TenB (§3) |
| Systolic dataflow taxonomy | Genc *Gemmini* DAC 2021 | engine = output-stationary; VPU = weight-stationary | Both already at sensible corners |
| FlashAttention | Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | fuse QKᵀ→online-softmax→·V | Exact; SigLIP is the simplest (bidirectional, no mask/KV); **off-surface today** (flash-attn disabled in `supports_op`; score materialization in main-owned `clip.cpp`) |
| Online softmax | Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) | **already in `softmax_f32.c`** | Done; enabler for a fused attention pass |
| im2col → implicit/indirect GEMM | Dukhan [arXiv:1907.02129](https://arxiv.org/pdf/1907.02129) | `im2col_f32.c` | Non-overlapping stride-16 patches ⇒ im2col is a pure **reshape**; exact, near-free |
| GELU tanh-approx | Hendrycks & Gimpel, [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | fuse GELU into fc1 epilogue | Must keep the **tanh** variant (checkpoint-matched); off-surface fusion |
| RMSNorm / NR-rsqrt | Zhang & Sennrich NeurIPS 2019; Moroz [arXiv:1802.06302](https://arxiv.org/pdf/1802.06302) | `rms_norm_f32.c` | Already centering-free; rsqrt via `exp∘log` vs `frcp`-seeded Newton-Raphson — marginal, PPL-gated |

---

## 6. Synthesized optimization roadmap (ranked by payoff × feasibility, gate attached)

### Tier A — safe, in-surface, do next
1. **C-store pipeline across the tile boundary** (`mul_mat_f32_matrix_engine.c`). Today the tile epilogue does `tensor_store` then a synchronous `tensor_wait(TENSOR_STORE_WAIT)` before the next tile's prologue loads. The store reads f-regs; the next prologue loads write SCP — disjoint. Move the store-wait to just before the next tile's first (overwriting) FMA; keep one final wait after the last tile. The in-order tensor queue (depth 4) supports posting `fma → store → next-fma`. **Payoff:** ~0.2–0.6% of board (engine is ~5.6%). **Risk:** low, bit-identical. **Gate:** build-verify (`max_abs=0`) + board A/B. No seam gate (dst tiles hart-strided).

### Tier A.5 — division / reciprocal hygiene (no-HW-float-divide levers)
Erbium traps `fdiv.s`, so every division routes through `et_fdiv` = `frcp.ps` + `fmul.s` wrapped in a mask save/restore (~5 instrs). Because `et_fdiv` is **`volatile` inline asm** (`math_fp.h:17`), the compiler **cannot** hoist or constant-fold the reciprocal — so a divisor that is loop-invariant, or even a literal constant, recomputes `frcp` every call. Found instances, most-egregious first:

1. **`et_sinf` recomputes reciprocals of *compile-time constants* every call** (`math_fp.h:191-195`): `et_fdiv(1.0f, 6.0f)`, `et_fdiv(1.0f, 120.0f)`, `et_fdiv(1.0f, 5040.0f)`, `1/362880`, `1/39916800`, plus `et_fdiv(1.0f, two_pi)` (`:166`). These are pure constants — replace with baked literals (`0.16666667f`, …). Removes **6 runtime `frcp` + 12 mask-shuffles per `et_sinf` call**; `et_sinf`/`et_cosf` back RoPE (`rope_f32.c`). **Payoff:** small board-% (RoPE is a minor kernel) but a large *per-call* cut and zero risk. **Gate:** local build-verify (should be bit-identical or closer — a baked constant is more accurate than a `frcp` estimate). **Highest ease/safety ratio of anything in this doc.**
2. **Norms divide by the loop-invariant row length via a fresh reciprocal each row**: `norm_f32.c:83,96` (`et_fdiv(sum, row_size)` twice per row) and `rms_norm_f32.c:116` / `rms_norm_mul_f32.c:125` (`et_fdiv(sum, ne0)`). Hoist `inv = et_fdiv(1.0f, N)` above the row loop, multiply. Removes ~1–2 `frcp`/row. **Payoff:** sub-0.6% board (norm kernels are small, reduce-dominated); hygiene more than board-mover. **Gate:** local build-verify.
3. **Leave alone:** `softmax_f32.c:294,384` (`et_fdiv(1.0f, sum)` — genuine per-row variable denominator); `rope_f32.c` Taylor coefficients written as `1.0f/6.0f` (compile-time-folded, already free); the integer index divides in `mul_mat_f32_matrix_engine.c:107-115` (exact, once-per-tile, `r2/r3` usually 1). The **hot** kernel `mul_mat_Q8_0` has no `et_fdiv` — its scales are precomputed Q8_0 block scales.

### Tier B — structural, in-surface, needs board + more effort
2. **Double-buffer A via the TenB bank** (corrects the prior "at floor" conclusion). Use the idle third load FSM + SCP lines 48–63 to hold B', freeing general lines 32–47 for A's second buffer, and alternate the FMA's `tenb` bit. Removes the exposed A-load — the biggest per-K structural stall in the engine kernel. **Payoff:** larger fraction of the engine kernel, but still bounded by its ~5.6% share. **Risk:** medium (TenB lockstep pairing protocol; `tensor_error` bit 6 on mismatch). **Gate:** build-verify + board A/B + seam gate (SCP layout change).
3. **Softmax exp-reduction** (`softmax_f32.c`, now ~3.4% of the shrunken total). The two-pass online softmax runs ~3 `fexp.ps` per element; a store-the-numerator scheme (pass1 emits `exp(x−localmax)`, pass2 rescales by a `fmul` instead of a fresh exp) removes pass2's exp at the cost of a row-sized buffer. **Payoff:** up to ~1% of board (above the gate). **Risk:** medium — changes float rounding order (must hold PPL ≤ 26.739 + exact answer) and needs a >L1 row buffer. **Gate:** build-verify + board A/B + PPL re-check. *Honest caveat: prior softmax micro-opts have tended to wash; validate before investing.*

### Tier C — high theoretical value, empirically or structurally closed (do NOT re-attempt)
- **Route the Q8_0 GEMM onto the tensor engine** (the "obvious" ~40% win). **Measured dead 3×**: int8 opcode-3 per-tile (+8%, 192× redundant re-quant), shared-arena+barrier (+15%, `.bss` wall), int8-TENC clean pipeline (~7× slower). Root cause is **hart-count asymmetry**: the board launches ~2048 harts and the scalar path uses all of them (free Q8_0 dequant in `fmadd.ps`); any engine path caps at ~8 harts because per-hart SCP staging blows `.bss` past the ~12 MB execution-context load limit (`event 65535`). 8 engine harts cannot beat 2048 scalar harts even at higher MAC/instr. (`f16 acols≤32`/`int8 acols≤64` are real throughput levers *per hart*, but the hart-count wall dominates.)
- **FlashAttention proper / GELU-into-fc1 fusion**: off the modifiable surface (`supports_op` disables flash-attn; score materialization and activation nodes live in main-owned graph/`clip.cpp`, not `et-kernels/src/*`).
- **A double-buffered by treating SCP as 64 flat lines**: top 16 are TenB-only (`%48` everywhere) — must use the pairing protocol (see Tier B #2, which does it correctly).
- **`tenc_loc=1` two-fp32-tile register-block**: no memory-C for float; TENC is int8-only.
- **Wider-than-16 fp32 FMA tile**: fp32 `acols`/`bcols`/`arows` all cap at 16; one FMA is already a full 16×16×16.
- **Cross-frame weight amortization**: each of the 4 frames is a separate graph launch (`ne12==1`); no cross-frame reuse to schedule.
- **im2col / norms / GLU / unary micro-opts**: exact and safe but each <0.5% of board — opportunistic only.

---

## 7. Recommended next actions

1. **Land `f2714ffe1` + the doc cleanup** once the narrowed build-verify confirms `max_abs=0` and the board A/B reconfirms the −46.5% engine-kernel win (and PPL/answer unchanged). *(No commit without explicit OK — per standing rule.)*
2. **Prototype Tier A #1 (C-store pipeline)** — smallest, safest, genuinely untried engine lever; gate locally then board A/B.
3. **Scope Tier B #2 (TenB A-double-buffer)** — the corrected, previously-missed lever; worth a design pass because it removes the last exposed load in the engine kernel, even if the absolute share is small.
4. **Do not reopen Tier C** — engine routing is a proven wash on this silicon; record any temptation as already-tested.

**Bottom line:** the reviewed change is correct and the cleanup is safe. The literature and hardware confirm the engine kernel sits near its floor and — more importantly — that it is *not* the hot path. The real cycles live in the scalar Q8_0 kernel (already a BLIS-grade micro-kernel at its register/memory floor) and in exact fusions (FlashAttention, GELU-into-fc1) that are currently **off the modifiable surface**. The honest remaining in-surface headroom is the C-store pipeline (~0.3–0.6%), the TenB A-double-buffer, and a risky softmax exp-cut (~1%) — each board-A/B-gated.
