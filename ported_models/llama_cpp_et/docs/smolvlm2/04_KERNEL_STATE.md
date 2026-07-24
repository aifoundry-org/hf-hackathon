# Current ET kernel state (audit) — SmolVLM2 modifiable surface

Audit of `ggml/src/ggml-et/et-kernels/src/*.c` at the pinned runtime (`cc4049d`). Dispatch cited from the
non-modifiable `ggml-et.cpp` / `ggml-et-ops.cpp`. **STATE ∈ {NAIVE, PARTIAL, TUNED}.**

## Two load-bearing facts

1. **Only TWO kernels touch the TFMA tensor engine** (`tensor_load`/`tensor_fma`/`tensor_store`):
   `mul_mat_f32_matrix_engine.c` and `memops.c` (bulk memset). **Every other kernel is 8-wide VPU
   (`fmadd.ps`/`fexp.ps`/`frcp.ps`) or pure scalar.** So most of the model's GEMM compute is NOT on the
   matrix engine today.

2. **✅ M0 RESOLVED (2026-07-16): the SigLIP/mmproj weights are Q8_0, not F16 → the hot vision GEMM lands on
   `mul_mat_Q8_0.c` (VPU int8-gather, no TFMA), NOT `mul_mat_f16.c`.** Proven from the sha-pinned artifact
   manifest (`benchmarks/smolvlm2_500m_video.json`, `docs/smolvlm2_500m_video.md`, `artifacts.json`) + the
   deterministic dispatch on `src0` dtype (`ggml-et-ops.cpp:302–425`). ~65% of all compute is Q8_0 GEMM on
   `mul_mat_Q8_0.c`. **Consequence:** the F16 lever (T2 below) is DEAD, and `mul_mat_f32_matrix_engine.c` (the
   only tensor-engine GEMM) is **unreachable on the scored path** because the frozen Q8_0 weights never satisfy
   the F32-aligned branch — so double-buffering it (plan M1) optimizes a kernel the vision GEMM never hits.
   See `EXPERIMENTS.md` 2026-07-16 M0 entry for the ladder re-framing (optimize `mul_mat_Q8_0.c` in place vs
   route Q8_0→F32 into the matrix-engine kernel). Board bound-classification pending (interview).

## MUL_MAT dispatch (ggml-et-ops.cpp:315–355, gate ggml-et.cpp:708–745)
- `Q8_0 × F32` → `mul_mat_Q8_0.c` (VPU int8 gather + L2 prefetch; **not** the int8 TFMA opcode-3 path)
- `F16 × F32` → `mul_mat_f16.c` (**VPU, no TFMA — NAIVE**)
- `F32 × F32` with `ne0%16==0 && ne1%16==0` → `mul_mat_f32_matrix_engine.c` (**TFMA fp32, serialized, NAIVE**)
- else `F32 × F32` → `mul_mat_f32.c` (VPU block dot, PARTIAL)
- **Dead predicate (ops.cpp:337):** the GEMV escape `src1->ne[0] != 0` tests the contraction dim K, which is
  never 0 → always true. So GEMV/mat-vec shapes (attention vs few tokens, lm_head) are wrongly forced onto the
  single-tile matrix-engine kernel. Cheap mis-route to fix — but note ops.cpp is **not modifiable**, so this
  can only be worked around inside a kernel, not by editing the gate.

## Per-file state (abridged)

| File | op(s) | engine | STATE | key un-exploited lever |
|---|---|---|---|---|
| **mul_mat_f32_matrix_engine.c** | MUL_MAT F32 aligned | **TFMA fp32** | **NAIVE** | Serialized `load0;load1;wait0;wait1;fma;wait_fma` per kb (L115–162) — **no double-buffering** though 2 load IDs exist and only 32/48 SCP lines used (L28). No operand reuse (src1 A-tile reloaded per M-tile, src0 re-transposed per kb, L128–138). One tile per hart. |
| **mul_mat_f16.c** | MUL_MAT F16×F32 (**likely SigLIP**) | **VPU, no TFMA** | **NAIVE** | F16 weights never reach fp16 TFMA (opcode 1). Per-output scalar-shaped dot + atomic store per element. Potentially THE hot vision kernel with zero engine use. |
| **mul_mat_Q8_0.c** | MUL_MAT Q8_0×F32 (LLM) | VPU int8 gather + **L2 prefetch** | PARTIAL | Best cache behavior of the GEMMs but still scalar-per-output; not on int8 TFMA opcode-3. |
| **mul_mat_f32.c** | MUL_MAT F32 unaligned | VPU block dot | PARTIAL | No tiling/reuse; atomic store per element. |
| **im2col_f32.c** | IM2COL (patch embed) | **pure scalar** | **NAIVE** | Full div/mod index decode + one `*(float*)` copy per element (L71–89); no vector load/gather. |
| **cont_f32.c / cont_f16.c** | CONT (reshape/permute) | **pure scalar** | **NAIVE** | Atomic store per element, nested 4D scalar loop. |
| **norm_f32.c** | NORM (LayerNorm, SigLIP) | VPU | TUNED-compute | Uses `et_fdiv` mean + `et_powf(var+eps,-0.5)` rsqrt (transcendental exp/log) instead of NR rsqrt; 3 passes. |
| **rms_norm_f32.c** | RMS_NORM (LLM) | VPU | PARTIAL | `et_fdiv` + `et_powf(...,-0.5)` instead of NR rsqrt; 2 passes. |
| **rms_norm_mul_f32.c** | fused RMS_NORM+MUL | VPU | TUNED | Fusion present; still `et_powf` rsqrt. |
| **softmax_f32.c** | SOFT_MAX (attention) | VPU, online 2-pass, `fexp.ps` | TUNED | Well-built (online max/sum, mask, ALiBi). Not fused with preceding SCALE; not FlashAttention (still consumes a score row). |
| **unary_f32.c** | UNARY/GELU (SigLIP MLP act) | VPU `fexp/frcp` | TUNED | Candidate to fuse into fc1 GEMM epilogue. |
| **scale_f32.c** | SCALE | VPU | TUNED | Fusable into softmax. |
| **glu_f32.c** | GLU | GEGLU vectorized / **SWIGLU scalar** | PARTIAL | SWIGLU (the LLM path) is scalar per element (`et_expf`+`et_fdiv`); not vectorized. |
| **rope_f32.c** | ROPE (LLM) | VPU rotate + transcendental freqs | PARTIAL | `et_fdiv`/`et_logf`/`et_powf` angle math recomputed per work-unit; no freq table. |
| **get_rows_f32.c / set_rows_f32.c** | GET/SET_ROWS | VPU/scalar | PARTIAL | Per-element scalar/atomic paths. |
| **memops.c** | MEMSET | **TFMA** bulk fill | TUNED | Good. |

## Op → kernel → state for the scored path

**VISION (SigLIP, ~80% of compute):** patch IM2COL → `im2col_f32.c` **NAIVE(scalar)**; QKV/O/fc1/fc2 MUL_MAT
→ **`mul_mat_f16.c` NAIVE(no-TFMA)** *or* `mul_mat_f32_matrix_engine.c` **NAIVE(serialized-TFMA)** depending on
weight dtype; LayerNorm → `norm_f32.c` TUNED-compute; GELU → `unary_f32.c` TUNED; attention SOFT_MAX →
`softmax_f32.c` TUNED; CONT → `cont_f32.c` **NAIVE(scalar)**.

**LLM (SmolLM2, ~18%):** Q8_0 MUL_MAT → `mul_mat_Q8_0.c` PARTIAL; RMS_NORM → PARTIAL/TUNED; ROPE → PARTIAL;
SwiGLU → `glu_f32.c` **scalar** PARTIAL; SOFT_MAX → TUNED.

## Ranked targets (crux: the hot vision GEMM is Q8_0 on `mul_mat_Q8_0.c` — no engine, no double-buffer)

> **M0 BOARD-MEASURED (2026-07-16), total 10.87B cycles.** Ranking: `mul_mat_Q8_0` **76.3%**, `cont_f32`
> **15.84%**, `mul_mat_f32_matrix_engine` 2.72%, `mul_mat_f16` 1.78%, softmax 0.88%, rest <0.8%. The vision GEMM
> is Q8_0 → does **not** reach `mul_mat_f32_matrix_engine.c` (F32-only, 2.72% from other matmuls) or
> `mul_mat_f16.c` (1.78%). T1/T2 are only reachable *after* a Q8_0→F32 route (option B). Lowest-risk first lever
> = **T0: pipeline `mul_mat_Q8_0.c` in place**. **NEW: `cont_f32` at 15.84% is the measured #2 lever (T0b)** —
> was mis-ranked "secondary". Pick T0 mechanism (VPU pipeline vs engine/int8) on a PMC bound-classification run.

- **T0 — Pipeline `mul_mat_Q8_0.c`** (board-confirmed hot vision kernel, **76.3%**, 2960 launches). Overlap
  load→int8-gather→`fmadd.ps` in steady state; only then weigh int8 TFMA opcode-3 (yolo: int8 = parity until
  the chain is pipelined). Stays on the frozen-weights path — no dequant, no dtype route.
- **T0b — Vectorize / fuse `cont_f32.c`** (board-measured **15.84%**, 496 launches, ~3.47M cyc/launch). Pure
  reshape/permute (attention layout transpose) done as scalar per-element atomic-store copies. Vectorize the
  copy, or fuse the permute into the consuming GEMM/softmax so the permuted tensor is never materialized.
  Low-risk, lossless, and the 2nd-biggest cycle sink — attack alongside T0.
- **T1 — Double-buffer + operand-reuse the FP32 matrix-engine GEMM** (`mul_mat_f32_matrix_engine.c:113–177`).
  It *is* the serialized floor; 2 load IDs + 16 free SCP lines are right there. **Only pays if the vision GEMM
  is first routed Q8_0→F32 into it** (M3 option B); unreachable on the stock scored path.
- **T2 — (DEAD per M0) SigLIP F16 GEMM onto the tensor engine.** Weights are Q8_0, not F16;
  `mul_mat_f16.c` is not on the scored path. Retained only for reference.
- **T3 — Operand reuse / tile blocking** (hoist the transpose, keep A resident across the M-tile loop).
- **T4 — FlashAttention fuse** of SCALE→SOFT_MAX and avoid materializing the score row.
- **T5 — Vectorize the scalar NAIVE kernels** on the vision path: `im2col_f32.c`, `cont_f32.c`; NR-rsqrt in the
  norms; secondary but real launch/overhead wins.
