# SmolVLM2 ET-SoC-1 — Algorithm Inventory + Literature Speedup Map

**Date:** 2026-07-19
**Purpose:** enumerate *every* algorithm implemented in the modifiable kernel surface (`ggml-et/et-kernels/src/*` + `math_fp.h`), then map each to faster options from the classic numerical-computing literature (Knuth TAOCP, Hacker's Delight, Muller, Numerical Recipes, Higham, Cody-Waite, and relevant arXiv), adapted to our hardware.

**Hardware primitives** (what any "faster" recipe must be expressed in):
- HAS: `frcp.ps` (reciprocal *estimate*), `fexp.ps` (2ˣ), `flog.ps` (log₂), `fcvt.f16.ps`, 8-wide packed `.ps` vector fmadd/fmul, the 16×16×16 tensor engine (output f0–f31).
- LACKS (traps): hardware float divide `fdiv.s`, hardware `fsqrt`, hardware `fsin/fcos`. Div, sqrt, rsqrt, sin, cos are **all software**.
- Quality gate for any change: exact `video_dog` answer + PPL ≤ 26.739 + ET-vs-CPU ≤ 1%.

> The "Literature / faster option" column is populated by the research sweep (agents: elementary-functions, GEMM/reduction/quant, softmax/norm/activation/RoPE). Sections marked ⏳ are pending until those land; the inventory (§1) is complete.

---

## 1. Algorithm inventory (complete)

### A. Linear algebra / matrix multiply
| # | Algorithm | File | Core method | Board share |
|---|---|---|---|---|
| A1 | Dense GEMM, tensor engine | `mul_mat_f32_matrix_engine.c` | 16×16×16 output-stationary tile, double-buffered B load | ~2.7–5.6% |
| A2 | Quantized int8 GEMM (**hot**) | `mul_mat_Q8_0.c` | scalar 8-wide packed-MAC, register-blocked N=8 / M2×N4, free dequant in `fmadd.ps`, all ~2048 harts | ~76% |
| A3 | f16 GEMM | `mul_mat_f16.c` | scalar dot, ported from Q8_0 family | ~1.8% |
| A4 | f32 GEMM (scalar) | `mul_mat_f32.c` | scalar dot | minor |
| A5 | Indirect / MoE matmul | `mul_mat_id_f32.c` | gather-indexed GEMM | minor |

### B. Reductions
| # | Algorithm | File | Core method |
|---|---|---|---|
| B1 | Horizontal dot / lane reduction | `block_ops.h`, in-kernel | 8-lane `.ps` horizontal sum; int32 accumulation of int8 products |
| B2 | Cross-hart reduction | (hardware `tensor_reduce`) | cross-minion fadd/fmax tree (available, mostly unused) |

### C. Normalization
| # | Algorithm | File | Core method | Board |
|---|---|---|---|---|
| C1 | LayerNorm | `norm_f32.c` | two-pass mean+variance, `scale = powf(var+eps,-0.5)` | <0.1% |
| C2 | RMSNorm | `rms_norm_f32.c`, `rms_norm_mul_f32.c` | mean-of-squares, `powf(mean+eps,-0.5)`, fused mul | ~0.58% |

### D. Softmax / attention
| # | Algorithm | File | Core method | Board |
|---|---|---|---|---|
| D1 | Online (streaming) softmax | `softmax_f32.c` | Milakov-Gimelshein running max+sum, 2-pass, ~3 `fexp` / element | ~3.4% |

### E. Elementary functions (`math_fp.h`)
| # | Algorithm | Current implementation | Cost on our HW |
|---|---|---|---|
| E1 | Division `et_fdiv(a,b)` | `frcp.ps(b)` × a — raw estimate, **no Newton-Raphson refinement** | frcp + fmul + mask save/restore (~5 instr); can't be compiler-folded (volatile asm) |
| E2 | rsqrt / sqrt | `et_sqrtf=powf(x,0.5)`, norms use `powf(x,-0.5)` = `exp(±0.5·log₂x·…)` | **two** transcendentals (flog+fexp) per sqrt |
| E3 | pow `et_powf(b,e)` | `exp(e·log(b))` via flog.ps+fexp.ps | flog+fmul+fexp; fine for general e, overkill for fixed e=±0.5 |
| E4 | exp `et_expf` | `2^(x·log2e)` via fexp.ps | 1 HW instr + const mul (near-optimal) |
| E5 | log `et_logf` | `log₂(x)·ln2` via flog.ps | 1 HW instr + const mul (near-optimal) |
| E6 | sin/cos `et_sinf`/`et_cosf` | range reduction + **degree-11 Taylor**, **6 runtime `frcp` on constant divisors** | many mul + 6 frcp (the constant-reciprocal bug) |
| E7 | fp16↔fp32 | `fp16_to_fp32` manual bit-twiddle + subnormal normalize loop; `fp32_to_fp16` HW `fcvt` | branch + while-loop on subnormals |

### F. Activations
| # | Algorithm | File | Core method | Board |
|---|---|---|---|---|
| F1 | GELU (tanh approx) | `el_map_f32.c` | `gelu_pytorch_tanh` via exp/tanh | ~0.80% |
| F2 | SiLU / sigmoid / SwiGLU | `glu_f32.c`, `unary_f32.c` | sigmoid = 1/(1+e⁻ˣ) via fexp+frcp; SwiGLU | ~0.17% |

### G. Position encoding
| # | Algorithm | File | Core method | Board |
|---|---|---|---|---|
| G1 | RoPE rotary embedding | `rope_f32.c` | per-position sin/cos rotation; `theta_scale=powf(...)`; uses E6 | minor |

### H. Data movement / layout
| # | Algorithm | File | Core method | Board |
|---|---|---|---|---|
| H1 | im2col conv lowering | `im2col_f32.c` | scalar per-element gather, ~10 int-divs/output; stride-16 non-overlapping | ~0.5% |
| H2 | Contiguous copy / transpose | `cont_f32.c`, `cont_f16.c` | vectorized cache-line copy (flq2/fsq2) | `cont_f32` was board #2 (15.84%), M1 landed −15.5% |
| H3 | Gather / scatter rows | `get_rows_f32.c`, `set_rows_f32.c` | indexed row copy | minor |
| H4 | Scale (elementwise ×) | `scale_f32.c` | vector fmul | minor |

### I. Quantization
| # | Algorithm | File | Core method |
|---|---|---|---|
| I1 | Q8_0 (de)quantization | `quants.h`, in `mul_mat_Q8_0.c` | 32-value blocks, one fp16 scale/block; dequant folded into `fmadd.ps` |

---

## 2. Literature speedup map — elementary functions (E1–E7)

Cost unit: `et_fdiv` ≈ 5 instr (frcp + fmul + 3-instr mask save/set/restore), volatile → uncompilable-away. `et_powf` = flog + fmul + fexp = **two** transcendentals.

| Alg | Literature lever | Cited source | Recipe on our HW | Payoff / risk |
|---|---|---|---|---|
| E2 **rsqrt/sqrt** | **fast-inverse-sqrt seed + Newton-Raphson** replacing `powf(x,±0.5)` | Lomont 2003; Walczyk-Moroz-Cieśliński [arXiv:1802.06302](https://arxiv.org/abs/1802.06302); Moroz [MDPI Computation 9(2):21](https://www.mdpi.com/2079-3197/9/2/21); Muller *Elementary Functions* | `i=0x5f3759df-(bits(x)>>1); y=bits(i); y=y*(1.5-0.5*x*y*y)` ×2 → full fp32. **All fmul/fmadd — no frcp, no mask dance, no transcendental**, vectorizes 8-wide. `frcp.ps` gives 1/x not a rsqrt seed, so seed with the int trick. | **~3–5× on norm-bound code**; feeds M4. HIGH safety (add-only accuracy; keep 2 NR steps for full fp32; exhaustively gateable). **Rank 1 of this cluster.** |
| E6 **sin/cos** | kill 6 constant `et_fdiv`; **minimax (Remez) < degree-11 Taylor**; Estrin over Horner; Cody-Waite range reduction | Cody & Waite 1980; Muller; Remez; Robin Green GDC2002; [minimax coeffs](https://gist.github.com/publik-void/067f7f2fef32dbe5c27d6e215f824c91) | Bake `1/6, 1/120…` as literals (−6 frcp, −6 mask dances/call). Degree-9 minimax (5 coeffs, err 5.3e-9 < fp32 eps) beats degree-11 Taylor. Estrin fills the `.ps` pipe vs Horner's serial chain. (`rope_f32.c` already bakes constants in its SIMD path — this fixes the scalar `math_fp.h` path.) | HIGH per-call, correctness-hygiene; accuracy-neutral-or-better. **Rank 2.** |
| E1 **division** | **1 fused Newton-Raphson step** `x1=x0*(2-b*x0)`; vectorize the mask dance | Markstein *IA-64 & Elementary Fns*; Goldberg 1991; [DSPRelated NR reciprocal](https://www.dsprelated.com/showcode/201.php) | HW reciprocal estimates are typically ~12-bit → **`et_fdiv` today may be only ~12-bit accurate (latent bug)**. One NR step (+2 fmadd) → full fp32; batch the frcp+NR across 8 lanes to pay the mask dance once. | Med-High. **Gated on measuring `frcp.ps` bit-width first** (decides bug vs margin). Adding NR is monotonic-safe. **Rank 3.** |
| E7 **fp16→fp32** | **branchless "magic subtract"** (no subnormal while-loop) | Fabian Giesen [half-to-float](https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/); Warren *Hacker's Delight* | Add mantissa into a float at exp=126 and subtract the magic float — IEEE renormalizes subnormals in one op, replacing the data-dependent loop → constant ~4-op, vectorizes. | Med. Bit-exact over all 65,536 half patterns (offline-verifiable). Also **check if a HW `fcvt.ps.f16` unpack exists** — would supersede this. |
| E3 **pow** | specialize fixed exponents | Knuth TAOCP Vol 2 §4.6.3 | Route e=±0.5 to the E2 rsqrt path; keep general `exp(y·log x)` for the cold `theta_scale`. | Subsumed by E2. |
| E4/E5 **exp/log** | (table-driven Tang) | Tang TOMS 1989; Muller | **HW instruction already wins — do not replace.** Only trim the `x>88`/`x<-87` guard branches *after* confirming board `fexp.ps` saturation semantics. | ~0 |

**Two hardware facts to measure (gate the ranking):** (1) `frcp.ps` estimate bit-width (E1); (2) whether a HW f16→f32 unpack exists (E7).

## 4. Literature speedup map — softmax / norm / activation / RoPE (C, D, F, G)

| Alg | Literature lever | Cited source | Verdict | Payoff / risk |
|---|---|---|---|---|
| D1 **softmax** | collapse online 2-pass (**~3 fexp/elem**) → **classic 3-pass safe softmax (1 fexp/elem)** | Milakov-Gimelshein [1805.02867](https://arxiv.org/abs/1805.02867); Dao FlashAttention [2205.14135](https://arxiv.org/abs/2205.14135) | **LIVE, best board-% algorithmic win.** The kernel pays online's correction-exp *and* re-reads src in pass 2 — worst of both. Pass1=max only (no exp); pass2=`exp(x-max)`, store, sum; pass3=`×1/sum` (already hoisted). | **~3× fewer exp on a ~3.4% line.** HIGH safety — converges to the ggml oracle. **Caveat: if M4 fuses attention (no materialized scores), keep online single-pass instead.** **Overall rank 1.** |
| G1/E6 **RoPE** | bake `et_sinf`/`et_cosf` constant reciprocals; opt. host-baked cos/sin table; opt. position recurrence | Numerical Recipes §5.5; Su et al. RoFormer [2104.09864](https://arxiv.org/abs/2104.09864) | Constant-fold = **free, bit-identical** win (also E6). Host-baked table removes all runtime trig (RoPE freqs fixed at load) but needs ABI wiring (may be off-surface). Position recurrence: RoPE angles are *geometric* in dim (current `theta*=theta_scale` is right) but *arithmetic* in position — recurrence applies position-axis; drift risk, minor board share. | Constant-fold: **rank 2** (free). Table/recurrence: low priority unless RoPE profiles. |
| C1/C2 **norms** | NR-rsqrt (E2); hoist `1/row_size`; **LayerNorm 3→2 pass** (Σx + Σx² two accumulators) | Welford/Knuth TAOCP Vol 2 §4.2.2; Zhang-Sennrich RMSNorm [1910.07467](https://arxiv.org/abs/1910.07467); Higham | Apply E2 rsqrt; hoist the per-row reciprocal (doc 10 Tier A.5). **Reject Welford** (serial divide defeats the 8-wide `fmadd`); use two-accumulator sum/sumsq. | **Polish** — norms are element-pass-dominated, <0.6% board. One-pass variance `Σx²/n−mean²` risks catastrophic cancellation → **PPL-gate**. |
| F1/F2 **activations** | minimax/Padé tanh/sigmoid | Chiluveru IET 2021; K-TANH [1909.07729](https://arxiv.org/pdf/1909.07729); Muller | **Leave as-is.** Already at the op-floor (`gelu=x·(1-1/(exp(2z)+1))`, silu=`x/(1+exp(-x))` = 1 fexp + 1 frcp). A poly would change rounding on the gated `gelu_pytorch_tanh`. Only free nit: hoist the inner-loop `fbc.ps` constant broadcast. | ~0 / rounding risk. |

## 3. Literature speedup map — GEMM / reductions / quantization (A, B, I)

**Two facts that reframe this whole section** (from reading the actual kernel, not the mental model):
- **The hot Q8_0 kernel accumulates in FP32, not int32.** Inner loop = `fgb.ps` (gather 8 int8) → `fcvt.ps.pw` (int8→f32) → `fmadd.ps`. Activations stay f32 and are never quantized; only the weight is int8→f32. "Free dequant in `fmadd.ps`" = this. The canonical llama.cpp int8×int8→int32 path was tried 3× and measured dead.
- **No cross-hart reduction exists on the hot path.** GEMM is output-partitioned (M×N tiles across ~2048 harts); each K-dot (768/3072 MACs) sums entirely inside one hart. Output-parallelism is ~1500× oversubscribed.

| Alg | Literature lever | Cited source | Verdict on our HW | Payoff |
|---|---|---|---|---|
| A2 dense GEMM | **Strassen / Winograd fast matmul** O(n^2.807) | Knuth TAOCP Vol 2 §4.6.4; [Strassen-on-FPGA int8 (arXiv:2406.02088)](https://arxiv.org/pdf/2406.02088); stability [arXiv:2402.05630](https://arxiv.org/pdf/2402.05630) | **DEAD.** Decisive reason: Strassen must *add weight sub-blocks*, but Q8_0 blocks have incompatible per-block fp16 scales — you'd have to dequant→f32→add, losing the free-dequant property (the exact marshalling that killed int8-TFMA 3×). Also: adds compete 1:1 with fmadds on a MAC-bound engine; breaks the −38.7% cache-line-owned store tiling. | negative |
| B1 dot reduction | **Kahan / Neumaier compensated summation**; **pairwise** | Knuth TAOCP Vol 2 §4.2.2; Higham *Accuracy & Stability* Ch.4 | **NO LEVER.** Accuracy isn't at risk (PPL 22.28 vs 26.739 ceiling, ET-vs-CPU 3e-5; even naive 768ε is far inside). Kahan = 4× arithmetic on a fmadd-bound engine = regression. The reduction *shape* is already optimal: log-depth pairwise lane tree (`fswizz`), **hoisted out of the K-loop** (once/output), persistent vector accumulator (no long dependency chain). | 0 |
| H1 im2col conv | **Winograd conv**; **indirect/implicit GEMM** | [Lavin & Gray (arXiv:1509.09308)](https://arxiv.org/abs/1509.09308); [Dukhan (arXiv:1907.02129)](https://arxiv.org/abs/1907.02129) | **DEAD.** Winograd's saving comes from *overlapping* filter reuse; our stride=kernel=16 patch-embed is **non-overlapping** (zero input reuse) → transform overhead with no offsetting saving. Indirect GEMM removes im2col *duplication*, but non-overlapping im2col is a pure non-duplicating permutation. And it's <0.1% of board. | ~0 |
| I1/A2 quantized matmul | int32-accumulate + **two-scales-at-end**; scale-placement grouping | [llama.cpp Q8_0 GEMM](https://chenghuawang.github.io/keep-moving-forward/tech/q8_0_q4_0_4_gemm_in_llamacpp/); [INT8 block-level FP32-accum (arXiv:2503.08040)](https://arxiv.org/html/2503.08040v1) | int32/two-scale path = **DEAD (measured 3×)**. **One marginal LIVE lever:** hoist the scale *broadcast* (`fbcx.ps`) once per block and reuse across the M-rows of the tile (vs re-broadcasting per weight-row). Bit-identical if fold order preserved. | ~1–3% of kernel / sub-1% board; **may wash if load-stall-bound** |
| B2 cross-hart reduce | tree-reduce / recursive-doubling / Rabenseifner allreduce | [Rabenseifner (HLRS)](https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf); [Swing (arXiv:2401.09356)](https://arxiv.org/pdf/2401.09356) | **INAPPLICABLE.** No cross-hart reduction on the hot path; creating one via split-K would *add* cost to raise parallelism that's already 1500× oversubscribed. (The HW `tensor_reduce` tree is the right primitive for softmax/norm reductions — §4 — not GEMM.) | 0 / negative |

**Net:** the remaining GEMM headroom is **not algorithmic** — it's the load→L1 latency the S1–S6 register-pipelining has been chipping at. The one open algorithmic lever (scale-broadcast hoist) is gated by the **still-pending M0 PMC bound-classification**: if the loop is purely load-stall-bound, it washes and the kernel is at its algorithmic floor.

## 4. Literature speedup map — softmax / norm / activation / RoPE (C, D, F, G) ⏳
*(pending research agent — softmax/norm/activation/RoPE)*

---

## 5. Unified cross-cluster ranking (all algorithms)

Deduped across the three research sweeps and weighted by board share × safety. This is the actionable output.

| Rank | Lever | Algorithm(s) | Board impact | Safety | Gate |
|---|---|---|---|---|---|
| **1** | **Softmax: online 2-pass → classic 3-pass (1 fexp/elem)** | D1 | **~3× fewer exp on ~3.4% line** — the biggest board-% algorithmic win | HIGH — converges to ggml oracle | build-verify + board A/B + PPL. **Blocked-by:** decide vs M4 (if attention gets fused, keep online single-pass instead) |
| **2** | **rsqrt: FISR seed + 2× Newton-Raphson, retire `powf(x,±0.5)`** | E2, C1, C2 | per-op ~3–5× on norm rsqrt; **feeds milestone M4**; absolute small (norms <0.6%) but the pattern is reused | HIGH (add-only accuracy, 2 NR = full fp32) | build-verify (exhaustive) + board A/B |
| **3** | **Kill the constant/invariant reciprocals** (`et_sinf` 6× `et_fdiv` on literals; per-row `1/N` in norms) | E6, E1, C1, C2, G1 | small board-%, large per-call; correctness-hygiene | HIGHEST (baked literal is bit-identical-or-more-accurate) | build-verify. **Detailed in doc 10 §Tier A.5** |
| **4** | **sin/cos: degree-9 minimax + Estrin + Cody-Waite** (scalar `math_fp.h`) | E6, G1 | minor (RoPE not bottleneck) | HIGH (accuracy-neutral) | build-verify + board A/B |
| **5** | **Division: +1 Newton-Raphson step, vectorize mask dance** | E1 | latent correctness fix if `frcp.ps` ≈ 12-bit | HIGH (monotonic) | **measure `frcp.ps` bit-width FIRST** |
| **6** | **fp16→fp32 branchless magic-subtract** | E7 | removes data-dependent loop, enables 8-wide | HIGH (65,536-case bit-exact) | build-verify. **Check for HW f16 unpack first** |
| **7** | **Q8_0 scale-broadcast hoist** (reuse `fbcx.ps` across tile M-rows) | A2, I1 | ~1–3% of the **hot** kernel / sub-1% board | Med (bit-identical if fold order kept) | build-verify + board A/B. **May wash if load-stall-bound** |
| — | LayerNorm 3→2 pass (Σx+Σx²) | C1 | polish | Med — catastrophic-cancellation risk | PPL-gate hard |
| — | **Leave as-is:** activations (rounding risk), exp/log (HW wins), CORDIC (FMA wins) | E4,E5,F1,F2 | — | — | — |
| — | **DEAD (don't attempt):** Strassen/Winograd matmul, Kahan summation, Winograd conv, cross-hart allreduce/split-K (see §3) | A2,B1,B2,H1 | negative/0 | — | — |

### Open measurements that gate several levers
1. **`frcp.ps` estimate bit-width** — decides whether E1 division is a latent correctness bug (rank 5) or just precision margin.
2. **HW f16↔f32 unpack existence** — if present, supersedes E7 (rank 6).
3. **M0 PMC bound-classification (issue-bound vs load-stall-bound)** — decides whether the Q8_0 scale-broadcast hoist (rank 7) has any headroom or washes.
4. **M4 decision (fuse attention or not)** — flips softmax (rank 1) between 3-pass-standalone and online-fused.

## 6. Sibling levers — same class as the constant-reciprocal finding

The user's finding generalizes to: **work computable outside a loop but recomputed inside it, because a `volatile`-asm block or opaque helper hides it from the compiler.** A targeted hunt for that class turned up these siblings (verified in-code), beyond the reciprocals in doc 10 §Tier A.5:

| Sibling | Location | The invariant redone per iteration | Fix | Payoff / safety |
|---|---|---|---|---|
| **Constant broadcasts in the vectorized loop** | `glu_f32.c:72-75` (and `unary_f32.c`) | 4× `fbc.ps` of literal constants (`1.0`, `0.044715`, `√(2/π)`, `2·log₂e`) re-broadcast every 8 elements; `f20/f22/f23/f24` persist across iterations | hoist the 4 `fbc.ps` above the `for` loop | small, **bit-identical**, in a real vectorized path. Best sibling. |
| **Invariant transcendental per element** | `rope_f32.c:265` (`et_logf(et_fdiv(1/freq_scale))`), also `:64`, `:293` | `freq_scale` is a call-constant → 1 frcp + 1 flog + 2 mask dances recomputed for every element with `ext_factor≠0`; the whole `mscale` correction is loop-invariant | compute the `mscale` factor once before the element loop | minor board (RoPE), safe |
| **Integer divide by invariant divisor, per element** | `im2col_f32.c:71-79` | 6 signed int64 `/` and derived `%` by `patch`, `oh*ow`, `ow`, `kh*kw`, `kw` — all fixed per call; compiler **cannot** strength-reduce division by a runtime variable | incremental "odometer" counters (each advances by a carry as `index++`), or — since patches are non-overlapping — the reshape rewrite that deletes the index decode entirely | <0.5% board; medium effort |

**Not a lever (verified):** the `mova.x.m`/`mov.m.x m0`/`mova.m.x` **mask save/set/restore dance** baked into every `math_fp.h` helper *is* loop-invariant, but the hot vectorized kernels (`softmax_f32.c`, `glu_f32.c`, …) already hoist the save/restore outside their loops and inline `fexp.ps`/`frcp.ps` directly. The per-call dance only survives in scalar/cold fallback paths (e.g. the softmax "sinks" path, deliberately scalar to match the CPU reference). Authors already did this hoist where it counts.

**Pattern summary for future hunts** — grep for these signatures of the class:
1. `fbc.ps`/`fbcx.ps` of a compile-time constant *inside* a `for` loop body.
2. `et_fdiv`/`et_powf`/`et_logf`/`et_expf`/`et_sinf` whose arguments are all call-constants or loop-invariants, sited inside a loop.
3. `/` or `%` by a variable that is invariant across the loop (compiler can't strength-reduce it; a constant it can).
4. `fp16_to_fp32`/`fcvt` of the same value repeated across iterations.
All four are the same defect: the fast HW primitive is fine; the *placement* (inside the loop, behind a volatile/opaque boundary) is the cost.

### The through-line
The **single reusable win** across the whole library is retiring the two-transcendental-and-volatile-divide patterns — `powf(x,±0.5)` → NR-rsqrt (rank 2), and every constant/invariant `et_fdiv` → baked literal or one hoisted reciprocal (rank 3). These are the "whole numbers left in a fraction" made general: the hardware has `frcp`/`fexp`/`flog` as fast primitives, but the library wraps them in volatile asm and composes them (`exp∘log` for sqrt, per-call `frcp` for constants) in ways the compiler cannot optimize — so the win is doing that composition by hand, once, in fmadd-only form. The **highest board-% single change** is the softmax exp-collapse (rank 1). Everything in the hot GEMM path is already at its algorithmic floor.
