# M2 design — `mul_mat_Q8_0.c` → int8 TFMA (opcode-3 / IMA8A32)

Target: the 90.2% kernel (post-cont board profile, 9.14B total). Route the Q8_0 vision GEMM off the naive
VPU scalar-per-output path onto the int8 tensor engine. Modifiable surface = `et-kernels/src/*` only.

## What makes it tractable (ground truth: `~/et-src/et-platform/sw-sysemu/insns/tensors.cpp`)

- **Both operands signed int8.** Q8_0 weights and per-block-quantized activations are symmetric signed →
  set BOTH `tena_unsigned=false` and `tenb_unsigned=false` (the args are swapped-wired, but both-false = ua=ub=0
  = both sign-extended). Simpler than DnCNN's unsigned activations.
- **`tensor_quant` applies per-row AND per-col scale in ONE op** (`INT32_TO_FP32 → MUL_COL → MUL_ROW`, auto
  SCP-line advance). MUL_COL = per-output-row scalar (activation block scale `d_b[n]`); MUL_ROW = per-output-col
  vector (weight block scale `d_a[m]`). So the rank-1 dequant `d_a[m]·d_b[n]·int32` is free on the engine —
  this is the key that solves the per-block-scale problem.
- **int32 can't accumulate across Q8_0 blocks** (distinct `d_a·d_b` each block) → `first_pass=1` on EVERY block;
  f32-accumulate the 24 blocks outside the int32 domain.

## Operand mapping (mirror `mul_mat_f32_matrix_engine.c` to avoid a store transpose)

- **A = quantized activations**, `arows = N-tile`, plain-contiguous (per output-row `n`).
- **B = quantized weights**, `bcols = M-tile = 16`, quartet-interleaved (repack pre-pass).
- `C[i][j] = dst[n][m]` → stores with no transpose to `dst + n*nbd1 + mb*4`.

## Per-32-block sequence (encodings in the Plan output; key flags)
`tensor_load` A→SCP0..15, B(8 quartet lines)→SCP16..23; `tensor_fma(opcode=3, first_pass=1, tenc_loc=1,
a_num_cols=7 [=(32/4)-1], b_num_col=3 [=(16/4)-1], a_num_rows=n_cur-1)`; **FREG-clobber barrier**;
`tensor_quant` dual-scale; store FREGs → VPU `fadd.ps` accumulate into per-hart `acc[16][16]`. After 24 blocks,
publish each output row with `copy_cacheline_f32` (flq2/fsq2 — the proven cross-shire store; NEVER plain fsw.ps).

## Cost check (make-or-break) — win survives the per-block scaling
Per 16×16 tile, K=768: naive ≈ 24.6k VPU-MAC issues; TFMA path ≈ 24 FMA + 24 quant + 24 store + 768 fadd.ps
≈ ~30× fewer VPU ops. Holds IF the accumulate stays vectorized and loads are pipelined (YOLO's "int8=parity
when orchestration-bound" lesson). Fallback if accumulate dominates: **N-tile=8** → 16 FREGs free → keep the f32
accumulator FREG-resident (FREG→FREG fadd, no memory round-trip), at 2× the FMAs. A/B the two.

## Pre-passes
- **Activation quant** (once per GEMM, amortized over M-tiles): per-column per-block symmetric int8,
  `d_b[n][b]=max|.|/127`, `qb=round(src1/d_b)` (reciprocal-multiply, NO fdiv — hangs in U-mode), per-col-contig
  pack + f32 scales.
- **Weight repack** (per-M-tile into ~12KB per-hart staging, reused across the N loop): quartet-interleave int8
  + `fp16_to_fp32(d)` extraction. (block_q8_0 is 34B → never 64B-aligned → repack mandatory; tensor_load masks
  `addr & ~0x3F`.)

## ⚠ #1 BLOCKER — no scratch arena (resolve before M2 impl)
`ggml_et_binary_params` carries only src0/src1/dst; `linker.ld` exposes only `_end` (no DnCNN-style heap). The
kernel needs ~0.75MB packed activations + per-hart staging/acc/tmp. Options:
1. **Ask main** if the backend can hand the kernel a scratch buffer (4th tensor / env field) — backend is
   main-owned, so a coordination question, not an overlay edit.
2. **Static `.bss` arena above `_end`**, sized to the M0 shape max, compile-time cap + **runtime guard that
   falls back to the existing scalar `compute_block_dot_product_q8_0`** when a shape exceeds the cap. Verify the
   loader provisions the enlarged `.bss` on the board.
Keep the scalar path in-file as the guaranteed-correct fallback regardless.

## Milestone ladder (each: test-backend-ops -o MUL_MAT at real shapes; TOLERANCE max_abs, not 0 — int8 is
approximate → final gate is PPL ≤ 26.739 + ET-vs-CPU ≤1% + exact answer on board)
- **M2a** single 16×16 tile, full K, single hart, static scratch.
- **M2b** full-K per-block scales (24 blocks, dual-scale quant, f32 accumulate). QKV shape.
- **M2c** activation-quant + weight-repack pre-passes + scratch arena. fc1/fc2 + QKV/O shapes.
- **M2d** full M×N tiling, N-edge, flq2/fsq2 publish; profile the cycle drop.
- **M2e** double-buffer + software-pipeline (single hart); A/B the N=8 FREG-resident variant. Board A/B ≥1%.
- **M2f** multi-hart tile partition (tile = global_id += NUM_HARTS), pre-pass N-band split + barrier; whole-line
  ownership = seam-safe. MANDATORY board seam gate.

## Risks
1. Per-block scale-accumulate eating the win (mitigated ~30×; measure; N=8 fallback ready).
2. **Quality gate**: int8 activation quant is lossy → PPL/answer could move. HARD gate requires PPL ≤ 26.739 AND
   ET-vs-CPU ≤1% AND exact answer. The vision encoder feeds the answer — watch closely; per-block (not
   per-whole-K) activation scales chosen to minimize error. This is the acceptance gate, not max_abs.
3. Scratch arena (#1 blocker).
4. Pre-pass overhead on the smaller QKV/O (M=768) — less amortization; confirm via profile.
5. SMT-producer deadlock → single-hart through M2e; multi-hart only at M2f behind the seam gate.

## Functions to add (all in `mul_mat_Q8_0.c`; `#include "tensor.h"`)
`quantize_pack_activations`, `repack_weight_mtile`, `fma_block_int8`, `dequant_block`, `accumulate_tile`,
`publish_tile` (reuse `copy_cacheline_f32`), `FREG_CLOBBER_BARRIER` (from `dncnn_gen_int8.c`),
`struct q8_gemm_scratch` + capped `.bss` arena + runtime-guard dispatch to the scalar fallback.

## Open go/no-go before sinking the effort
- **PMC bound-classification** (instrumented board run, DnCNN `pmc_probe.h`): weights are ALREADY int8, so int8
  TFMA does NOT cut weight bandwidth — if the GEMM is weight-bandwidth-bound the MAC gain is capped. Confirm
  compute-bound (high fmadd throughput / low L2 traffic) first. Large shapes make compute-bound likely, but
  measure.
