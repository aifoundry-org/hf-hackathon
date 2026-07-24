# SmolVLM2 on ET-SoC1 — Optimization Catalog (strictly categorized)

Every candidate is constrained to erbium ground truth: 16×16×16 TFMA (fp32 op0 / fp16 op1 / int8 op3 with
fused `tensor_quant`), 48-line L1 SCP, **no `fdiv.s`/`fsqrt.s`**, VPU `fexp.ps`/`frcp.ps`, async-batchable
cache ops (≤31 outstanding), **no in-kernel DMA** (latency-hiding = double-buffered `tensor_load` + prefetch/
lock), ~1024 in-order harts with L1D-sharing SMT siblings (hackathon runs 1 shire / 8 active harts).

**Metric** = firmware cycles, ≥1% win. **Quality gate is hard**: exact tensor inventory, frozen Q8_0 weights
+ hashes, fixed lowercase-alphanumeric answer on `video_dog`, **zero vision CPU fallbacks**, WikiText-2 PPL
≤ 26.739 and ET-vs-CPU PPL within 1%. This makes almost all value **lossless dataflow + fusion**; lossy
techniques (pruning, low-rank, aggressive approximations) are mostly gate-forbidden.

## The governing fact (from our own prior ports)

Across **DnCNN** and **YOLO-m30** the workloads were **orchestration-bound, not MAC-bound** — the wall was
the serialized per-tile chain `load→wait→fma→wait→quant→wait→store→wait`, scalar packing/marshalling, and
per-tile cache-ops/barriers. Value a technique by **how many waits / cache-ops / barriers / dispatches it
removes or hides behind compute**, not by MACs saved. Corollary: FLOP-cutting math (Winograd, FFT, low-rank,
Strassen) is low-value here even before the quality gate rejects it.

**⚠ But re-profile for SmolVLM2 before committing.** DnCNN was a tiny 64×64 tile (size-invariant overhead
dominated); YOLO was a detector. SmolVLM2's SigLIP GEMMs are **large** (`1024×768×3072`) and may be closer to
compute/bandwidth-bound than the small prior kernels — which would *raise* the value of int8 (4× MACs/dispatch)
and weight residency. **Do a PMC pass on the board first** (`ET_PERF`/`GGML_ET_PROFILE`, HPM counters). See
`03_HARDWARE_NOTES.md §PMC`.

---

## Legend for the "Ours" column
- **NEW** — not tried on any prior port.
- **LANDED** — we shipped this and it measured a win (cites dncnn/yolo).
- **WASH** — we implemented + board-measured it; no win; reverted. Don't blindly retry (but note why it may
  differ for SmolVLM2's larger GEMMs).
- **BLOCKED** — tried and hit a hardware/silicon wall.
- **GUARD** — a mandatory correctness constraint, not a perf lever.

---

## A. DATAFLOW / SCHEDULING — highest-leverage category

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **Software-pipeline the full `load→fma→quant→store` chain** | Restructure per-tile serial waits into a steady-state pipeline; while fma(n) runs, quant(n-1)/store(n-2)/load(n+1) proceed | **HIGH** | ✅ | **NEW** | The one lever both prior ports named as "the only idea that targets the real bottleneck" but **never landed**. dncnn's SMT-producer variant **deadlocked on silicon** (§3.5); a *single-hart* sw-pipeline avoids the SMT sync problem. ALCOP 2210.16691, Autocomp 2505.18574 |
| **Double-buffered `tensor_load` (two load IDs)** | Issue load(n+1) before `tensor_wait` on fma(n); hides operand fetch behind MAC | **HIGH** | ✅ | **NEW** | Lowest-effort slice of the pipeline lever. The current GEMM kernel does neither (see `04_KERNEL_STATE.md`) |
| **Operand residency / weight-stationary tiling** | Keep the 16-wide frozen-Q8_0 weight tile resident across output tiles; stream only activations | HIGH | ✅ | **WASH** (yolo Lever-1 A-resident, flat) | yolo's A-load was NOT the stall at its shapes; **SmolVLM2 MLP reuses each weight tile across 1024 tokens × 4 frames — far more reuse, so re-test.** |
| **Cache-op batching (≤31 evicts, one FENCE+WAIT drain)** | Batch evicts across many tiles; drain once per band not per tile | HIGH | ✅ | **LANDED** (dncnn B1, part of −24%) | Confirmed real; carry the pattern over |
| **Barrier reduction / hart-local accumulation** | Per-band/per-layer barriers instead of per-tile; reduce partials once | HIGH | ✅ | **WASH** (dncnn B3/B5 off-critical-path) | Was off critical path at 64×64; may matter more with 4-frame partition |
| **Multi-hart / multi-shire partition (token-band or OC split)** | Split 1024 patch tokens / 4 frames / output channels across harts | HIGH | ✅ *(if seam-padded)* | **LANDED** (both ports) | **4 frames are embarrassingly parallel** — natural partition. Mind the seam race |
| **SMT-sibling L1D sharing** | Co-schedule sibling harts on tiles sharing weight lines | MED | ✅ | **NEW** | Free reuse if partition is L1D-aware |
| **Prefetch / lock frozen weights resident** | Weights frozen → lock hot blocks for the whole layer | MED-HIGH | ✅ | **NEW** | Replaces absent in-kernel DMA |

## B. OPERATOR FUSION

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **FlashAttention-style online-softmax** | Stream QKᵀ score tiles through SCP, running max+denom, accumulate V; never materialize N×N scores; exp/recip via VPU | **HIGH** | ✅ exact | **NEW** | Big for SigLIP (1024²×4) + LLM attn. Bidirectional encoder = single fused pass, no mask. FlashAttention 2205.14135 |
| **Bias+activation fused into `tensor_quant` requant chain** | int8 path already fuses int32→uint8; fold bias-add + activation in | HIGH | ✅ bit-exact | **LANDED** (dncnn bias-fold; yolo epilogue) | Reuse the proven chain |
| **im2col-free / implicit-GEMM patch embed** | Convert stride-16 patch conv to on-the-fly GEMM; no im2col buffer | MED-HIGH | ✅ | **PARTIAL** (yolo packless-center was WASH) | yolo's pack was already cheap so removing it did nothing; but SmolVLM2's `im2col_f32.c` is **pure scalar** (see kernel state) — different starting point. Indirect Conv 1907.02129 |
| **LayerNorm→GELU→next-GEMM producer/consumer fusion** | Keep normalized+activated tile resident, hand to next `tensor_load` | MED-HIGH | ✅ *(if NR precision holds)* | **NEW** | LN needs `1/sqrt(var)` → NR/host-bake, **no fsqrt**. SOLE 2510.17189 |
| **Residual-add + norm fusion** | Consume the residual sum once from SCP | MED | ✅ | **NEW** | Streaming one-pass mean/var |

## C. MATH / ALGORITHMIC

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **Online-softmax numerics** | Enables FlashAttention; exact vs naive softmax | HIGH | ✅ | **NEW** | Sync-cutting, not FLOP-cutting |
| **Newton-Raphson reciprocal / rsqrt** | Mandatory for LN/softmax denom (no fdiv/fsqrt); seed+1–2 NR or VPU `frcp.ps` | MED-HIGH | ✅ *(enough iters)* | **LANDED** (yolo `fast_recip`, silu) | Constants host-baked. See `erbium-no-float-divide` |
| **Polynomial/tanh GELU on VPU** | No divide/sqrt → poly GELU, host-baked constants, VPU poly+`fexp` | MED | ⚠ lossy (bounded) | **NEW** | Must match `gelu_pytorch_tanh` closely + pass PPL. 2402.10118, PEANO-ViT 2406.14854 |
| **Winograd / FFT / Strassen / low-rank** | FLOP-cutting; SigLIP is a ViT (one patch conv, rest GEMM/attn) | LOW | ⚠/❌ | **REJECTED** (dncnn math sweep) | Low applicability + gate-forbidden (low-rank changes weights) |

## D. QUANTIZATION — weights FROZEN Q8_0 → activation-side + on-device repack only

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **Drive int8 TFMA (op3) for vision GEMMs vs fp32 (op0)** | uint8 act × int8 weight → int32 → fused requant; Q8_0 already int8 | **HIGH** *(if pipelined)* | ⚠ (intended Q8_0 numeric path) | **LANDED-but-PARITY** (yolo int8 = parity, not win) | yolo int8 hit **parity** because orchestration, not MACs, was the floor — int8's 4× MACs/dispatch didn't cash out. **For SmolVLM2's large GEMMs int8 may finally pay IF the chain is pipelined first.** SmoothQuant 2211.10438 |
| **On-device Q8_0 dequant/repack into TFMA layout** | Q8_0 = 32×int8 + fp16 scale; repack once into quartet layout, resident | HIGH | ✅ exact repack | **LANDED** (dncnn weight rearrange; yolo) | Attacks scalar marshalling |
| **Dynamic per-token uint8 activation quant (affine zero-point)** | Compute act scale/zp per tile; pad with Z not 0; fold zp into bias | HIGH | ✅ faithful | **LANDED** (yolo affine-uint8 de-risked sym-int8) | Mind the **ua/ub signedness wrapper bug**. Pad-with-Z trap |
| **Per-channel Q8_0 block scales in requant** | Apply block scales in the chain, not a separate pass | MED | ✅ | **LANDED** | Free vs per-tensor |
| **KV-cache int8 (decoder)** | Only matters for multi-token decode | MED | ⚠ | **N/A** | Decode = 3 tokens here → negligible. Skip |
| **FP16/BF16 MAC path** | TFMA has fp16 (op1) but **no bf16** | MED | — | **NEW** | fp16 op1 exists — a possible middle path for vision; untried by us |
| **GPTQ/AWQ / re-quantize weights** | Alters frozen Q8_0 + hashes | — | ❌ forbidden | **RULE-OUT** | Gate-forbidden |

## E. COMPRESSION / SPARSITY — almost all LOSSY / gate-forbidden

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **Token Merging (ToMe) in SigLIP** | Merge similar patch tokens between blocks → quadratically less attn | MED-HIGH *(if gate allows)* | ⚠ lossy | **NEW / risky** | The **only** sparsity bet worth gate-risk; training-free ~0.3% acc — but **changes token count** → almost certainly trips exact-inventory/fixed-answer. Validate before adopting. ToMe 2210.09461 |
| **2:4 sparsity / pruning / clustering / low-rank** | erbium TFMA has **no sparse mode**; alters frozen weights | LOW | ❌ | **RULE-OUT** | HW can't + gate-forbidden |

## F. COMPILER / CODEGEN / MICRO-ARCH

| Technique | HW mapping | Leverage | Lossless | Ours | Notes |
|---|---|---|---|---|---|
| **Compile-time tile dims → full K-unroll + strength reduction** | 16×16×16 known at compile time; specialize kernels per shape | **HIGH** | ✅ | **LANDED** (both ports; "runtime dim = slower even if it equals the constant") | Central lesson: a runtime dim that equals a constant still blocks unrolling |
| **VPU autovectorize LN/GELU/softmax elementwise** | Map elementwise+reduction to VPU (`fexp.ps`,`frcp.ps`) | MED-HIGH | ✅ | **LANDED** (yolo VPU sigmoid, −24%) | Scalar FP marshalling was 45% of a yolo layer |
| **Build-time no-divide checker** | `llama_cpp_et` kernels build `-O3` + `check_unimplemented_instructions.sh` post-build; a float `/`/`sqrtf` emits `fdiv.s`/`fsqrt.s` → **fails the build** (unlike the `-Ofast` DnCNN/YOLO regime that hid it) | HIGH *(correctness)* | ✅ mandatory | **GUARD** | Use `et_fdiv`/`et_powf` or NR/host-baked reciprocals |
| **64B-align tensor stores + seam-pad** | `PADW*CH%64==0`; store mask aligns down silently | HIGH *(correctness)* | ✅ | **GUARD** (int8 seam race) | Multi-hart write-back false-share |
| **Avoid R_RISCV_64 pointer-table relocs** | Pointer tables trap the board ELF loader | MED *(correctness)* | ✅ | **GUARD** (yolo M-i2) | Store indices into flat pools, not pointers |
| **FREG-clobber barrier around tensor ops** | tensor_fma/quant to FREG clobber f0..f31 | HIGH *(correctness)* | ✅ | **GUARD** | `__asm__("":::"memory","f0"..."f31")` |

---

## TOP-10 ranked for SmolVLM2 (lossless-first, gate-aware)

1. **Software-pipeline the `load→fma→quant→store` chain** (A) — the measured bottleneck, never landed; single-hart avoids dncnn's SMT deadlock. **NEW.**
2. **Double-buffered `tensor_load`** (A) — lowest-effort slice of #1. **NEW.**
3. **FlashAttention online-softmax** (B) — kills 1024²×4 score traffic; exact. **NEW.**
4. **PMC-profile first, then int8 TFMA on the big SigLIP MLP GEMMs** (D) — parity on yolo *because* it was orchestration-bound; SmolVLM2's large GEMMs + pipelining (#1) may finally cash out int8's 4×. **Re-test.**
5. **Cache-op batching + barrier reduction across the 4-frame partition** (A) — **LANDED** pattern.
6. **Bias+activation fused into the requant chain** (B) — **LANDED** pattern, reuse.
7. **Compile-time-specialized GEMM kernels (full unroll)** (F) — **LANDED** lesson.
8. **On-device Q8_0 repack + VPU elementwise for LN/GELU** (D/F) — **LANDED** patterns; attacks marshalling.
9. **Weight residency/lock + L1D-aware multi-hart partition** (A) — re-test (yolo WASH was small-shape).
10. **fp16 (op1) middle-path for vision GEMMs** (D) — untried; possibly better accuracy/throughput than int8 at lower risk than op3. **NEW.**

**Gate-FORBIDDEN (ruled out):** weight re-quantization (GPTQ/AWQ), pruning/clustering/low-rank, 2:4 sparsity,
linear attention. **ToMe** is the lone lossy maybe — only if it survives the fixed-answer gate (unlikely).

Full citations in the two research agent outputs; key papers: FlashAttention 2205.14135 · ALCOP 2210.16691 ·
Autocomp 2505.18574 · SmoothQuant 2211.10438 · I-BERT 2101.01321 · SOLE 2510.17189 · ToMe 2210.09461 ·
SmolVLM 2504.05299 · Idefics3 2408.12637 · SigLIP 2303.15343.
