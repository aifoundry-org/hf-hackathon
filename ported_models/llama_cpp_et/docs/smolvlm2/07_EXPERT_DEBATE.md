# 07 — Expert Debate: the next lever for the Q8_0 vision GEMM

**Date:** 2026-07-18 · **Method:** 3-expert dynamic dialogue (director-driven, converge-to-exit), unconstrained by prior work.

Experts: **Dr. Vega** (ET-SoC1 hardware/silicon-cycle), **Prof. Nakamura** (numerics/arithmetic-intensity), **Sam Okafor** (kernel-implementation). A Haiku facilitator chose each next speaker + challenge from the live transcript; a moderator adjudicated each topic; a lead strategist integrated all three. 38 agents, 0 errors, ~18 min.

> Generated from workflow `wf_a1e06474-e8d`. Raw JSON: `local-artifacts/smolvlm2_debate/debate_raw.json`. HW briefing pack: `local-artifacts/smolvlm2_debate/hw_briefing.md`.


## ⚠️ Load-bearing correction — VERIFIED in source (2026-07-18)

The debate overturned a false premise carried in our own notes/memory. The prior claim was that `mul_mat_Q8_0.c` is a **scalar-per-output** int8-gather kernel. **That is wrong.** Reading the actual hot function `compute_block_dot_product_q8_0` (`block_ops.h:24-75`) confirms it is **already 8-lane vectorized**:

- `block_ops.h:42-44` — `fgb.ps` (8-wide int8 gather) → `fcvt.ps.pw` (int8→f32 vector) → `fmadd.ps` (8-wide MAC).

- The real throttle is a **per-Q8_0-block epilogue that fires on every one of the 24 K-blocks** of an fc1 output: mask save/set (`:28-29`), `fbci` accumulator init (`:30`), gather-pattern reload (`:34`), a **7-instruction `fswizz`/`fadd` horizontal reduce** (`:55-68`), mask restore (`:71`), and a per-block scale multiply (`:73-74`); the caller then **scalar-accumulates** across blocks (`mul_mat_Q8_0.c:320`).

- Productive work per block ≈ 4 `fmadd.ps`; overhead ≈ ~11 setup/reduce ops → **~25-30% productive-instruction fraction**. This *is* the 'per-block orchestration' floor the int8-wash post-mortem pointed at — now located on the shipping kernel, not the engine path.


**Consequence:** D1's opening framing ('kernel is scalar, widen the gather onto 64 int8 lanes') is factually wrong and would reinvent the measured-dead int8 path; D3's Route-B reorder optimizes a path **not in the measured baseline** (the 76.3% is this VPU kernel, not the fp32 engine). All three dialogues, once the source was read, collapse onto the same concrete edit: the **reduction-hoist**.


## TL;DR — decision

**Headline:** The 76%-hot mul_mat_Q8_0 is ALREADY 8-lane vectorized (verified fgb.ps/fcvt.ps.pw/fmadd.ps at block_ops.h:42-44) but throttled by a per-Q8_0-block horizontal-reduce + mask/init epilogue firing 24x per fc1 output; the one lossless, single-file, no-.bss, no-barrier lever all three dialogues actually converge on once the source is read is the reduction-hoist (persistent 8-wide accumulator, scale folded per-block). D1's "kernel is scalar, widen the gather" premise is factually wrong; D3's Route-B engine reorder optimizes a path that is not in the measured baseline.


**Single next experiment (do this first, no code):**

> Run smolvlm2-profile on the BOARD against the isolated hot mul_mat_Q8_0 scalar-fallback tile (compute_block_dot_product_q8_0 path) and capture retired-inst/MAC, L2_EVICT_REQ/MAC, and TFMA_WAIT%. Falsifiable expectation: TFMA_WAIT~0 (the scalar path issues zero tensor ops), L2-miss/MAC<0.02 (int8 weights, L2-resident, reused across 1024 tokens x 4 frames), and a dominant epilogue/issue instruction fraction. If confirmed, proceed to the reduction-hoist; if TFMA_WAIT>30% or the read is evict/streaming-bound, STOP and re-scope to layout rather than writing the hoist. This one measurement makes or breaks both the hoist and the N-blocking follow-on and costs no code.


**Boldest unconstrained idea:**

> Take N-register-blocking to its logical end as a lossless weight-stationary micro-GEMM ON THE VPU: gather+convert each int8 weight K-block exactly ONCE into an 8-wide f32 register and stream it against a whole strip of N activation columns (N=4-8, spill-gated) held in N persistent accumulators. This attacks the true floor -- the issue rate of the int8 gather+convert, not the MAC -- amortizing it N-fold with ZERO dtype change, ZERO tensor engine, ZERO cross-shire barrier, ZERO static .bss, and full frozen-Q8_0 compliance. It is the one idea that compounds multiplicatively with the reduction-hoist for a potential combined 2-4x on the 76% kernel while staying inside every hard quality gate, and it sidesteps the entire tensor-engine family (int8 washed, Route B unproven, fp16 untried) that both prior ports and the int8 experiment showed to be the wrong bet for this workload.


## Ranked next moves


### 1. Board PMC baseline on the isolated hot scalar-fallback tile BEFORE any code change (smolvlm2-profile): capture retired-inst/MAC, L2_EVICT_REQ/MAC, TFMA_WAIT%, L1D load-stall on the compute_block_dot_product_q8_0 path (mul_mat_Q8_0.c:296-330).

- **EV:** Decides the whole plan; cheap (one profiling run, no build). Expectation: TFMA_WAIT~0 (scalar path issues zero tensor ops, structurally forced), L2-miss/MAC<0.02 (weights already int8, L2-resident, reused across 1024 tokens), high epilogue/issue instruction fraction. Confirms epilogue/issue-bound => hoist pays.

- **Risk:** Low. Only risk is it comes back L1D->VPU streaming/evict-bound, in which case STOP both hoist and N-blocking and re-scope to layout.

- **Why:** Every synthesis gated its GO on this measurement; the entire diagnosis is prior-based until it lands, and the hoist's follow-on (N-blocking) is contingent on the issue-bound-vs-evict-bound verdict this run produces.


### 2. Reduction-hoist in compute_block_dot_product_q8_0 + its call site (block_ops.h:24-75, mul_mat_Q8_0.c:317-321): carry ONE persistent 8-wide fp32 accumulator across all 24 K-blocks; hoist mask save/set (lines 28-29) and fbci init (line 30) above the K-loop; fold each block's distinct f16 d-scale as a broadcast fmul.ps on the converted int8 vector BEFORE accumulating (cannot defer to a single final scale as the current code does at line 74); exactly ONE 7-op horizontal reduce (lines 55-68) per output instead of 24. Local build-verify (max_abs + WikiText-2 PPL<=26.739) then board A/B (smolvlm2-ab-score, video_dog, gen=3).

- **EV:** ~1.3-2x on the kernel (removes ~23 of 24 horizontal reduces + 24 mask-dances + 24 fbci inits per fc1 output; instruction-count estimate ~1.35x, synthesis optimism to 2.5x). On a 76.3% kernel that is ~20-35% total device cycles, ~15-30x the >=1% gate, stacking on the landed cont_f32 -15.5%.

- **Risk:** Moderate: hoisting re-associates the float sum (per-block scale-then-add vs per-block reduce-then-scale), so it is not bit-identical -- MUST re-check max_abs locally and hold the ET-vs-CPU PPL-within-1% band. Numerically close; PPL gate is the guard.

- **Why:** Grounded in verified source: the epilogue is the per-block-orchestration floor the int8-wash pointed at, now confirmed on the shipping kernel (~25-30% productive-instruction fraction). One file, hits no .bss wall, needs no cross-shire barrier, weights stay frozen Q8_0. This is D1's scalar-widen and D2's reduction-hoist collapsed into the same concrete edit once the code is read.


### 3. N-register-blocking (start N=4), CONTINGENT on the PMC showing issue-bound + low-evict: gather+fcvt each weight K-block ONCE and fmadd it against N=2-4 activation columns into N persistent 8-wide accumulators, amortizing the int8-unpack (gather+convert) issue stream N-fold with zero dtype change.

- **EV:** Further ~1.5-2x on the amortizable issue stream (gather/convert are the real floor, not the MAC). Compounds multiplicatively with the hoist.

- **Risk:** f-reg spill on the in-order VPU (32 f-regs). N=4 likely fits, N=8 spills -- gate N by the compiled asm / measured spill. Buys nothing if the PMC verdict is evict/streaming-bound.

- **Why:** It is an ISSUE-stream reduction (fewer gather+fcvt), not a memory-traffic reduction, because the fc1 weight block is L2-resident -- so it is specifically the answer to the issue-bound branch the PMC run tests for. Both panels named it as the natural follow-on to the hoist.


### 4. DEPRIORITIZE Route B (dequant Q8_0->f32 -> fp32 matrix engine, mul_mat_Q8_0.c:142-294) and D3's weight-stationary pre-dequant-K-panel reorder. Park as fallback only.

- **EV:** Unproven; Route B is NOT in the measured baseline (the 76.3% is the scalar-fallback VPU kernel, not this engine path). It hit the .bss wall and issues ~4x more FMA dispatches plus per-K-block dequant + f0..f31 save/restore + engine_evict + WAIT.

- **Risk:** High opportunity cost: it must first BEAT the about-to-be-hoisted-and-N-blocked scalar kernel and may be strictly dominated by it. D3's reorder optimizes a path that is not the hot one.

- **Why:** D3's redundant-dequant-across-n-tiles observation (wblk depends on (mb,kb) not nb, re-dequanted every n-tile at 246-265) is real for Route B but only matters if Route B ever clears the raised bar. Revisit only after the scalar levers land and only if the PMC proves genuine fmadd-issue headroom the engine could absorb.


## Kill list — stop pursuing

- int8 opcode-3 TFMA on the vision GEMM: measured +8% cycle regression, reverted, root-caused (cut zero bound traffic since weights are already int8; swapped a dense branch-free vector loop for a serialized per-K-block engine chain). Do not revisit.
- D1's 'the kernel is scalar, widen the gather onto 512b/64 int8 lanes' framing: the kernel is already 8-wide vectorized (fgb.ps/fcvt.ps.pw/fmadd.ps, block_ops.h:42-44). There is no scalar gather to widen; the 64-int8-lane int32-accumulate variant just reinvents the dead int8 path.
- 4-frame cross-frame weight-stationary amortization: a proven phantom (ne12==1; each frame/sub-image is a separate graph launch, ~480 cont_f32 launches). Nothing to amortize across frames within a launch. Only ever a Route-B-internal amortizer, never a lever on the scalar kernel.
- Multi-accumulator N-strip-mine that spills the C tile to SCP: only ONE C accumulator tile is physically holdable (SCP 48x64B = A+B+C, tenc_loc=0). Any wide strip spills C to tenc_loc=1 forcing per-FMA DRAM read-modify-write (~384x C-traffic). VPU N-register-blocking in f-regs is the ONLY safe form.
- Re-grouping / widening Q8_0 scale groups to cut the per-32 epilogue count: weights are FROZEN Q8_0, re-quant/GPTQ/AWQ forbidden. The only lever against the per-block scale is making the epilogue cheap (the hoist), not re-grouping scales.
- Large static .bss weight/dequant panels: event-65535 load crash at backend init (12MB, 3.7MB, AND 384KB all hit it). Any buffer must stay per-hart-stack KB-scale (the ~1KB wblk streaming form).
- Route B (dequant->fp32 engine) as the LEAD lever: it is not in the measured baseline (the 76.3% is the scalar VPU kernel), hit the .bss wall, and must first beat the hoisted+N-blocked scalar kernel to matter. Fallback only.
- fp16 opcode-1 tensor path: hold as last-resort escalation ONLY if the hoisted+N-blocked kernel proves genuinely fmadd-issue-bound with headroom the fp32 engine could absorb. Not worth trying speculatively.


**Biggest remaining disagreement:** The dialogues factually contradict each other on whether the hot kernel is scalar (D1: "scalar-per-output int8-gather, widen it onto 512b/64 int8 lanes") or already 8-wide vectorized (D2: "8-lane fp32 MAC throttled by a per-block epilogue, hoist the reduction"). Source settles it decisively for D2: block_ops.h:42-44 is fgb.ps (8-wide int8 gather) / fcvt.ps.pw (int8->f32 vector) / fmadd.ps (8-wide) -- there is no scalar gather to widen, and D1's 64-int8-lane int32-accumulate idea would reinvent the measured-dead int8 path. The residual LIVE disagreement, settled only by the PMC run: whether the hoisted kernel lands issue-bound (=> N-blocking pays, the D1/D2 bet) or L1D-streaming/evict-bound (=> neither hoist-follow-on helps, re-scope to layout); and whether Route B retains any positive margin once the scalar kernel is hoisted+N-blocked, or is strictly dominated and worth abandoning.


## Per-topic syntheses


### D1 — What is the ACTUAL binding constraint on the 76%-hot mul_mat_Q8_0 vision GEMM -- and does the right next move follow from it, or is there a reframing that sidesteps the question?

*(dialogue: 7 turns, converged=False, confidence=0.82)*


**Verdict:** The binding constraint on the 76%-hot scalar mul_mat_Q8_0 vision GEMM is VPU integer ISSUE-RATE (retired-instruction-bound), not DRAM/L2 bandwidth and not sync/orchestration. The "MAC-bound ~130 MACs/byte" and "issue-bound" claims are the same diagnosis on two axes, not a contradiction: the high arithmetic intensity (fc1 re-derived at ~440 MACs/byte, weights reused across all 1024 tokens x 4 frames) means bandwidth is slack, so the compute is bound by how fast the in-order VPU can ISSUE the scalar int8 gather (~4-6 retired instructions per MAC at IPC~0.9). The "orchestration-bound" reading is REAL but applies to a DIFFERENT kernel — the failed int8-TFMA and the active Route-B engine paths, whose serialized per-K-block load/scale/wait chain is the engine path's floor. The scalar path issues zero tensor ops, so TFMA_WAIT is structurally ~0 there and sync cannot be its wall. Weights-already-int8 IS the decisive reframe: int8 TFMA (opcode-3) cuts zero bound weight traffic and only compresses an inner MAC that was never the floor — which is exactly why it regressed +8%.


**Decision:** GO on the scalar-widen (M1-adjacent): vectorize the Q8_0 int8 gather onto the 512b integer vector unit (64 int8 lanes) with per-lane int32 partial-sum accumulation, attacking the 76% directly with zero tensor_wait, zero evict, zero barrier, and bit-exact (PPL-safe) output. But GATE IT ON A BOARD PMC RUN FIRST: the entire diagnosis rests on priors until retired-inst/MAC, L2-miss/MAC, and TFMA_WAIT% are measured on the isolated hot tile. NO-GO / DEFER on Route B (dequant->f32 matrix engine) and the 4-frame N-collapse — the collapse is a conceded no-op on the scalar path (it only amortizes the engine path's per-tile tensor_load+dequant), and Route B must clear the raised bar set by the widened scalar kernel before it earns further effort.


**Rationale:** All three experts converged on issue-bound, and it is structurally forced, not merely averaged: the scalar fallback (verified at mul_mat_Q8_0.c:296-330) is nested n/m/kb loops calling compute_block_dot_product_q8_0 with NO tensor instructions, so TFMA_WAIT is necessarily ~0 and issue-rate is the only floor it can have. The intensity arithmetic (weights already int8, reused across N and 4 frames) puts L2-miss/MAC at floor level, killing the bandwidth hypothesis — and since the weights are ALREADY int8, int8 TFMA saves no bound traffic, cleanly explaining the measured +8% wash. The debate's own resolution (Nakamura's cost model N_blocks x (orch_fixed + MACs x inst_per_MAC / IPC)) shows orch_fixed is <2-3% of the scalar instruction stream, so amortizing it via frame-collapse divides an already-negligible term by 4 (<1%) — Sam and Nakamura both conceded this on the scalar path. That leaves inst_per_MAC as the dominant term, which is precisely what scalar-widen attacks. Vega's turn-3 honesty correction matters: cont_f32's 15.5% is the wrong anchor (it was a store-port-bound memory move), whereas the Q8_0 gather is genuine compute where cutting issued instructions cuts cycles near-proportionally, so 2-4x on the 76% is credible even after the int8->int32 widening tax. I verified QK8_0=32, which makes Sam's turn-4 sharpening the key open risk: the per-32 scale forces a reduce+rescale epilogue every 32 K-steps (24 per output tile), so the horizontal reduce is a metronome, not a once-per-768 noise term — this is the ceiling candidate that could plateau the win near ~2x, and only a board PMC delta settles it. Even the conservative ~2x is ~35% board improvement, ~30x the >=1% gate.


**Novel ideas surfaced:**
- Per-lane int32 partial-sum vectorization: hold a vector of int32 accumulators (one per lane) so gather->sign-extend->multiply->accumulate all run lane-parallel, leaving only the horizontal reduce serial (Vega turn 3) — the specific mechanism that keeps the widened MAC from serializing on the reduction
- The Q8_0-per-32-scale METRONOME (Sam turn 4, verified QK8_0=32): the per-32 scale forces reduce+convert+rescale+fma every 32 K-steps = 24 epilogues per 16x16x768 output tile, NOT once over 768 — this is the true ceiling candidate that may plateau the win near ~2x before L1D port contention bites
- cont_f32's 15.5% is the WRONG magnitude anchor (Vega turn 3): it was a store-port-bound memory move through the FP VRF, categorically different from a genuine compute loop, so the scalar-widen ceiling is set by issue-rate not store-port and the expected win is larger
- Vector loads REDUCE L1D port pressure (one 512b load feeds 64 lanes), so issue-rate binds BEFORE the load port does — inverting the usual worry that vectorization stalls on the port (Vega turn 3, rebutting Sam's flip condition)
- Reframing 4-frame parallelism as a GEMM-SHAPE lever (fuse frames into N->4096 against one resident dequant block) instead of a failed multi-hart partition — valid only for Route B, where it amortizes the real per-tile tensor_load+dequant; the trap to avoid is materializing a K x 4096 activation tile in SCP (stream activations past the resident weight block, each read once)
- Because weights are FROZEN Q8_0, the scale-group cannot be widened to reduce epilogue count; the only lever against the metronome is making the reduce+rescale epilogue cheap, not re-grouping scales (implication of the metronome + frozen-weights gate)


**Consensus:**
- Bandwidth is a red herring: weights are already int8 and reused across 1024 tokens x 4 frames, so L2-miss/MAC is floor-level and there is no DRAM traffic for int8 TFMA to cut
- The +8% int8-TFMA regression is explained: opcode-3 cut zero bound traffic and swapped a dense branch-free scalar loop for a serialized per-K-block engine chain (no double-buffer, 2 load IDs idle)
- 'MAC-bound ~130 MACs/byte' and 'issue-bound' are orthogonal axes of one cost equation, not competing diagnoses; 'orchestration-bound' describes the engine path's floor, not the scalar path's
- The decisive board signature is retired-inst/MAC (PMC18/19) + L2-miss/MAC + TFMA_WAIT%: issue-bound iff inst/MAC > 2-3, L2-miss/MAC < 0.02, TFMA_WAIT ~ 0; sync-bound iff TFMA_WAIT > 30%
- Scalar-widen onto the 512b integer vector unit is the correct FIRST lever (attacks the 76% directly, PPL-exact, no wait/evict/barrier); frame-collapse and Route B are second-order
- The 4-frame collapse is a no-op on the current scalar kernel (nothing to amortize) and only rescues Route B, where tensor_load+dequant is a genuine per-tile fixed cost


**Action items:**
- Board PMC baseline on the isolated hot mul_mat_Q8_0 tile BEFORE any change: capture retired-inst (PMC18), cycles (PMC19), IPC, L1D load-stall, and TFMA_WAIT_TENB. Confirm the falsifiable diagnosis: retired-inst/MAC ~4-6, L2-miss/MAC < 0.02, TFMA_WAIT ~ 0. If TFMA_WAIT > 30%, STOP and pivot to M1/M2 pipelining instead
- Implement scalar-widen on a single isolated 16x16x768 tile, single-hart (kernel dev loop): 512b int vector gather of Q8_0 quants, per-lane int32 partial sums in the private int VRF, horizontal-reduce + f32 rescale epilogue every 32 K-steps (one per Q8_0 block), fma into the f32 accumulator
- Local math gate via smolvlm2-build-verify: bit-close (max_abs) vs CPU reference at 1 hart / sys-emu — must stay exact since the int8 gather is lossless (PPL and exact-answer gates protected)
- Board A/B via smolvlm2-ab-score to prove >=1% device-cycle win; hold Sam's falsifiable prediction: retired-inst/MAC toward ~1 => 2-4x lands; plateau at inst/MAC ~2.5 => ~2x (still ~30x the gate, still ship). The PMC delta, not priors, decides whether the block epilogue is the ceiling
- Defer Route B and the 4-frame N-collapse. Only revisit the collapse as a Route-B amortizer, and only if Route B beats the widened-scalar bar after it lands. Reuse the existing f-reg save/restore machinery (mul_mat_Q8_0.c:246-289) is Route B's, keep it out of the scalar path


**Unresolved:**
- The entire issue-bound diagnosis is prior-based until the board PMC run lands — retired-inst/MAC, L2-miss/MAC, and TFMA_WAIT% on the hot tile have not been measured (int8-TFMA experiments profiled a different path)
- Whether the 24 per-tile reduce+rescale scale epilogues (the QK8_0=32 metronome) plateau the win near ~2x (inst/MAC ~2.5) or the epilogue stays cheap and 2-4x lands — Sam's held prediction, settled only by board
- Whether scalar-widen stalls on L1D load-port contention rather than issue rate (Sam's flip condition; Vega argues 512b loads reduce port pressure so issue binds first — untested)
- The int8->int32 widening tax roughly halves effective lane throughput (64 lanes -> ~effective 32); whether the residual margin after widening + per-block scale application still clears 2x
- Whether Route B, once the scalar bar is raised 2-4x, retains any positive margin at all — it may be strictly dominated by the widened scalar kernel and worth abandoning entirely


### D2 — Is moving the Q8_0 GEMM onto ANY tensor-engine path the right family (int8 washed, float Route B in flight, fp16 opcode-1 untried) -- or should we stay on the VPU and software-pipeline mul_mat_Q8_0 in place?

*(dialogue: 7 turns, converged=False, confidence=0.86)*


**Verdict:** Stay on the VPU and software-pipeline mul_mat_Q8_0 in place. This is the correct family — best-grounded in ET-SoC1 ground truth and the M0 board profile — not any tensor-engine path. The source (block_ops.h:24-75) confirms the shipping hot kernel is NOT scalar: it already runs an 8-lane fp32 MAC (fcvt.ps.pw + fmadd.ps, lines 43-44), but wraps it in a per-Q8_0-block epilogue — mask save/set (28-29), fbci init (30), gather-pattern reload (34), a 7-instruction fswizz/fadd horizontal reduction (55-68), mask restore (71) — that fires on EVERY block. For fc1 (K=768 = 24 blocks) that is 24 reductions + 24 mask-dances per output element against ~96 productive fmadd.ps, i.e. ~25-30% productive-instruction fraction. The caller then scalar-accumulates (mul_mat_Q8_0.c:317-321). This is exactly the PER-BLOCK-ORCHESTRATION floor the disputed int8-wash pointed to, now confirmed on the shipping kernel — settling the dispute in favor of orchestration-bound, not MAC-bound. The cheap lossless lever was skipped because both prior attempts reflexively read 'pipeline the chain' as 'change the dtype/engine' and jumped straight to int8 opcode-3 (washed +8%) and Route B (in flight), never reading the inner loop.


**Decision:** GO: implement the reduction-hoist (persistent 8-wide accumulator) as the M2 in-place software-pipeline, single file (block_ops.h + its call site), instrumented as the PMC fork. NO-GO / deprioritize the tensor-engine family as the primary lever: int8 opcode-3 is measured-dead; Route B (dequant->fp32 engine, mul_mat_Q8_0.c:14-294) stays only as a parallel fallback, not the lead; fp16 opcode-1 is not worth trying unless the hoist trace comes back issue-bound AND N-blocking is exhausted. Run the hoist A/B first — its own TL_OPS(15)/L2_EVICT_REQ(12)-vs-issue-stall trace IS the measurement that decides whether N-register-blocking follows.


**Rationale:** Weighing by ET-SoC1 ground truth rather than averaging: the VPU position is anchored in verifiable source and the M0 board profile (mul_mat_Q8_0 = 76.3% of 10.87B cyc), while the engine positions are either measured-dead (int8 +8% regression) or unproven and structurally disadvantaged (Route B issues ~4x more FMA dispatches, re-adds weight-dequant marshalling + f0..f31 save/restore + engine_evict + WAIT per K-block, and its whole rationale rests on a '130 MAC/byte => MAC-bound' assumption the source now contradicts — the kernel is orchestration/issue-bound, not MAC-bound). The hoist is lossless (bit-identical numerics, unlike anything touching quantization), hits no .bss wall, needs no cross-shire coherency barrier, and lands in one file. A ~1.5-2.5x cut on a kernel that is 76.3% of cycles is ~30%+ of total — already ~15-30x the >=1% gate, on top of the already-landed cont_f32 -15.5%. Vega correctly conceded the read and correctly sharpened the fork: because 2.5MB fc1 weights fit inside the 4MB shire L2, token-column 2+ is an L2 hit — so L2_EVICT will NOT scale with N, and if the hoisted kernel comes back evict-bound it is an L1D->VPU streaming/working-set problem that neither hoist nor N-blocking fixes. The panel reached genuine consensus on sequencing, which is the right, cheapest-first, measurement-driven plan.


**Novel ideas surfaced:**
- Reduction-hoist / persistent 8-wide accumulator: carry ONE fp32 accumulator across all K-blocks, hoist mask/fbci/gather-pattern setup above the K-loop, and do exactly one horizontal reduce per output element instead of one per block — eliminating ~24 reductions + 24 mask-dances per fc1 output.
- Per-block scale MUST be folded as a broadcast fmul.ps on the converted int8 vector BEFORE accumulating into the shared accumulator — because each Q8_0 block has a distinct f16 d scale, you cannot defer scaling to a single final reduction as the current per-block code does (this is the one numerical-correctness pitfall of the hoist).
- N-register-blocking on the VPU (Sam): gather+fcvt.ps.pw each weight K-block ONCE and fmadd it against N=2-4 activation columns into N persistent accumulators — amortizes the int8-unpack issue stream N-fold with no dtype change; gate N by measured f-reg spill (N=4 fits 32 f-regs, N=8 spills).
- Reframing the hoist A/B as its own PMC fork: the same one-file experiment both improves the kernel AND emits the TL_OPS/L2_EVICT-vs-issue trace that decides the next lever — no separate profiling milestone needed.
- Corrected N-blocking mechanism (Vega): it is an ISSUE-stream reduction (fewer gather+fcvt ops), NOT a memory-traffic reduction, because the fc1 weight block is L2-resident; therefore N-blocking is specifically the answer to the issue-bound branch and buys nothing in the evict-bound branch.
- fp16 opcode-1 tensor path remains the only entirely-untried engine option — reserved as an escalation ONLY if the hoisted kernel proves fmadd-issue-bound and N-blocking is exhausted.


**Consensus:**
- The shipping mul_mat_Q8_0 hot path is an 8-lane fp32 MAC (not scalar) throttled by a per-block epilogue: mask save/set, fbci init, gather reload, and a 7-op horizontal reduction firing every Q8_0 block; ~25-30% productive-instruction fraction — VERIFIED in block_ops.h:24-75.
- Stay on the VPU; the tensor-engine family (int8/fp16/Route-B) is chasing the wrong bottleneck for the primary lever.
- int8 opcode-3 is measured-dead (+8% cycle regression, reverted) and the wash is explained by orchestration overhead, not insufficient MAC density.
- Run the reduction-hoist FIRST, single file, lossless — and treat its A/B as the PMC diagnostic (TL_OPS(15)/L2_EVICT_REQ(12) vs issue-stall).
- 2.5MB fc1 weights fit in the 4MB shire L2, so token-column 2+ is an L2 hit; L2_EVICT will not scale with N and N-blocking cuts instructions (gather+fcvt issue), not DRAM bytes.
- N-register-blocking is contingent on a high-issue/low-evict trace verdict, not automatic; if the trace is evict-bound it is a layout/streaming problem and neither hoist-follow-on lever helps.


**Action items:**
- Implement the reduction-hoist in block_ops.h::compute_block_dot_product_q8_0 (and its call site mul_mat_Q8_0.c:317-321): one persistent 8-wide accumulator across all K-blocks, mask/fbci/gather-pattern setup hoisted above the K-loop, per-block d scale folded as broadcast fmul.ps on the converted int8 vector before accumulate, exactly one fswizz/fadd horizontal reduce per output.
- Local math gate first (smolvlm2-build-verify / test-backend-ops, 1 hart sys-emu): confirm bit-close/bit-identical vs CPU reference before any board time — the hoist reorders float accumulation so re-check max_abs and the PPL gate.
- Board A/B via smolvlm2-ab-score (paired-main, video_dog, gen=3): prove >=1% device_cmd_exec_dur win while the exact lowercase-alphanumeric answer, WikiText-2 PPL<=26.739, and ET-vs-CPU PPL-within-1% all hold, with ZERO vision CPU fallbacks.
- During that same board A/B, capture the PMC fork read: TL_OPS(15) / L2_EVICT_REQ(12) vs issue-stall. High-issue/low-evict => proceed to N-register-blocking (N gated by measured f-reg spill, start N=4). High-evict => STOP both levers; the residual is an L1D->VPU streaming/layout problem, re-scope.
- Keep Route B (mul_mat_Q8_0.c streaming-dequant path) parked as a fallback only; do NOT invest further unless the hoist+N-blocking ceiling is hit and the trace proves genuine fmadd-issue headroom the fp32 engine could absorb.
- Do NOT pursue int8 opcode-3 further (measured-dead); hold fp16 opcode-1 as a last-resort escalation contingent on an issue-bound verdict after N-blocking.


**Unresolved:**
- Whether the hoisted kernel lands issue-bound (Sam/Vega's bet, enabling N-blocking) or evict/streaming-bound — decided only by the board PMC trace, not yet run.
- Actual realized speedup magnitude of the hoist (estimates span ~1.5x to ~2.5x on the kernel); all comfortably clear the 1% gate but the exact number is unmeasured.
- Whether N=4 register-blocking truly fits within 32 f-regs alongside operands and the persistent accumulators without spill on this in-order VPU — must be confirmed from the compiled asm / measured spill.
- Whether fp16 opcode-1 would ever beat a fully-hoisted + N-blocked VPU kernel — untested and only relevant in the issue-bound-after-N-blocking corner case.
- The exact horizontal-reduction and scale-fold instruction sequence for the persistent accumulator that preserves bit-closeness within the ET-vs-CPU 1% PPL band (float re-association risk).


### D3 — GREENFIELD: unconstrained by anything tried and by the current premises, what single change to the device kernels would most reduce firmware cycles on this workload?

*(dialogue: 7 turns, converged=False, confidence=0.75)*


**Verdict:** The single highest-leverage device-kernel change is a weight-stationary reorder of the Q8_0 engine path in mul_mat_Q8_0.c: keep n innermost (one SCP-resident C accumulator tile), pre-dequant the full K-panel for each m-tile ONCE into an aligned DRAM/stack buffer, FENCE+evict it once, then stream all n_tiles reading transposed weights via tensor_load from that panel. This collapses the redundant weight-dequant + per-kb evict/WAIT_CACHEOPS barrier chain ~64x (fc1) within a single launch and eliminates the per-kb q8_save/restore_fregs dance. It is opcode-agnostic (stays fp32 dequant-exact, PPL-safe), keeps .bss KB-scale, needs no cross-shire publish, and touches only the modifiable kernel surface. It is NOT the 4-frame amortization all three experts opened with — that premise is a proven phantom (ne12==1).


**Decision:** GO — but measurement-gated. Step 1 is a board PMC to confirm dequant+evict is actually the floor; only if confirmed, implement the pre-dequant K-panel reorder. NO-GO on any N-strip-mine with multiple accumulators (spills force ~384x C-side DRAM RMW), and NO-GO on further int8/dtype experiments (measured-dead, +8%).


**Rationale:** The dialogue converged hard and I verified the convergence in source. All three experts independently located the same cost sink — wblk depends on (mb,kb) not nb yet is re-dequanted + engine_evict'd + WAIT_CACHEOPS'd inside every n-tile (confirmed at mul_mat_Q8_0.c:246-265). That is the exact per-block orchestration the briefing named as the disputed floor, now grounded in the live loop. The debate's real value was killing three wrong turns: (a) the cross-frame amortization premise (Sam's cont-launch-count trace: ~480 cont_f32 launches only closes if each frame/sub-image is a separate graph execution, so ne12==1 and the batch machinery at 229-237 is dead weight — no frame axis to hold weights stationary across); (b) N-strip-mining with 8 accumulators (only one C-tile is physically holdable — SCP is 48x64B = A(16)+B(16)+C(16); tenc_loc=0 confirmed at the fma call — so 7 accumulators spill to tenc_loc=1 with per-FMA DRAM read-modify-write); (c) the .bss wall fear (event-65535 is NOLOAD .bss in the load image, invisible to a runtime stack/DRAM panel). The surviving design — pre-dequant panel, n innermost, single SCP C — is the only weight-stationary form that avoids both the multi-accumulator spill and the load wall, and it exploits that tensor_load bypasses L1/L2 to read DRAM directly, so a panel-in-DRAM costs nothing extra to stream. I favor this over 'it's MAC-bound, fight the dtype' because int8 already washed (+8%) and both hardware experts named the same falsifiable flip condition (evicts/TL_OPS <5% of the kernel) that a single PMC settles — so the responsible call is to measure that ratio before writing the reorder, not to average the positions.


**Novel ideas surfaced:**
- The 4-frame weight-reuse premise that ALL THREE experts opened with is a phantom: cont_f32 launch count (~480) proves each frame/sub-image is a separate graph execution, ne12==1, so there is no frame axis inside the launch to amortize dequant across 4096 rows
- The redundant-dequant floor lives WITHIN a single launch (nb-independent wblk re-dequanted n_tiles times), not cross-launch — which is cleaner: no .bss persistence, no multi-hart seam
- tensor_load bypasses L1/L2 and reads DRAM directly, so 'weight-stationary' here buys dequant-compute + evict/barrier COUNT, not memory residency — a pre-dequanted panel in DRAM streams for free and the whole residency-pricing framing was mis-stated
- Only ONE C-accumulator tile is physically holdable (SCP 48 lines = A+B+C, and tenc_loc=0 confirmed), so any multi-wide N-strip-mine forces accumulator spill to tenc_loc=1 with per-FMA DRAM read-modify-write (~384x C-traffic) — killing that whole class of reorder
- Hoisting all scalar dequant out of the FMA phase eliminates the per-kb q8_save/restore_fregs entirely (FREG never live during accumulation) — a free secondary win, pending resolution of the SCP-vs-FREG accumulator question
- The event-65535 wall is NOLOAD .bss in the load image at backend init only; a 192KB runtime stack/DRAM panel is invisible to the loader, so the fc2 panel size is a stack-headroom vet, not a redesign blocker


**Consensus:**
- Target is the mul_mat_Q8_0.c engine path, not another dtype/int8 experiment
- The cost sink is redundant weight dequant + a per-kb engine_evict/WAIT_CACHEOPS barrier; wblk depends on (mb,kb), never nb (verified 246-265)
- Keep n innermost so C stays a single resident accumulator tile
- Pre-dequant the full K-panel per m-tile once into a DRAM/stack buffer and evict once
- Dequant stays f32-exact -> PPL-safe; weights remain frozen Q8_0; no cross-shire publish; .bss stays KB-scale
- The decisive gate is a board PMC: if dequant+evict is already <5% of the Q8_0 kernel, this lever is small and the real fight is M2 FMA-chain pipelining


**Action items:**
- Run smolvlm2-profile on board FIRST: attribute the 76.3% mul_mat_Q8_0 cycles into cacheop-wait/L2_EVICT_REQ/TL_OPS vs TFMA-wait. GATE: if dequant+evict <5%, abandon the reorder and pivot to M2 (software-pipeline the load->fma chain). This is the flip condition both hardware experts named.
- If confirmed the floor: implement pre-dequant-panel reorder in mul_mat_Q8_0.c — per m-tile, dequant the full K-panel (K/16 blocks of 16x16) into an aligned DRAM/stack buffer, FENCE + engine_evict once, then loop n_tiles doing tensor_load transpose-B from the panel; opcode-0 f32 FMA unchanged.
- Resolve the accumulator-bank contradiction (code comment line 247 says f0..f31 vs tenc_loc=0=SCP at the fma call) before relying on removing q8_save/restore_fregs; if C truly lives in SCP the FREG save/restore can be dropped for a free secondary win.
- Vet the fc2 192KB panel stack frame on board (event-65535 is .bss-only; confirm per-hart stack headroom for the runtime buffer).
- Local gate with smolvlm2-build-verify (bit-close max_abs vs CPU reference at 1 hart/sys-emu), then smolvlm2-ab-score paired-main A/B for a >=1% device_cmd_exec_dur win with PPL<=26.739 + exact lowercase-alphanumeric answer + zero vision CPU fallbacks.


**Unresolved:**
- Whether dequant+evict is actually the cycle floor vs TFMA_WAIT is unproven — no board PMC breakdown of the 76.3% kernel exists yet; this is the single measurement that makes or breaks the whole recommendation.
- fc2 192KB panel stack headroom on the board is unvetted (fc1 48KB is likely fine).
- The SCP-vs-FREG accumulator location has a live code-comment contradiction (line 247 vs tenc_loc=0), unresolved, and it determines whether the q8_save/restore elimination bonus is real.
- Expected magnitude of the win is unquantified: 64x fewer dequants/barriers within a launch translates to firmware cycles only if that orchestration is a large fraction of the kernel; could land anywhere from a modest few-percent to a large win.


---

## Full transcripts


### D1 — What is the ACTUAL binding constraint on the 76%-hot mul_mat_Q8_0 vision GEMM -- and does the right next move follow from it, or is there a reframing that sidesteps the question?

_The notes assert BOTH "MAC-bound ~130 MACs/byte" AND "orchestration-bound" -- opposite implications. Resolve it: name the decisive board signature (IPC, retired-inst/MAC, L2-miss/MAC, TFMA_WAIT) and threshold. Weights are ALREADY int8, so int8 TFMA does NOT cut weight DRAM traffic -- does that reframe the whole thing? Is bandwidth, issue-rate, or sync the real wall?_


#### [seed] Dr. Vega (ET-SoC1 hardware architect)

**Position: the wall is VPU issue-rate, not bandwidth and not sync — and that reframes everything.**

Weights are already int8 at ~130 MACs/byte, so the hot kernel is nowhere near the LPDDR wall; int8 TFMA cutting "weight DRAM traffic" is a non-argument — there is no traffic to cut. The 76.3% cost lives in `mul_mat_Q8_0.c`'s scalar tail (line 296+): a VPU scalar-per-output int8 gather, ~24.6k MAC *issues* per 16×16 tile. That is retired-instruction-bound. The decisive board signature: **retired-inst/MAC ≈ 1 with IPC ~0.9 and TFMA_WAIT_TENB ≈ 0** (PMC 18/19). If TFMA_WAIT dominated we'd be orchestration-bound; it can't, because the scalar path issues no tensor ops.

That explains the int8-TFMA +8% regression cleanly: it swapped a *dense, branch-free scalar loop* for the serialized per-K-block chain at lines 246–289 (dequant→evict→load0→load1→wait0→wait1→fma→wait_fma, **no double-buffer**, 2 load IDs idle). Orchestration is the *engine path's* floor; issue-rate is the *scalar path's* floor. Both said "pipeline first."

**Novel move not yet tried:** don't route to the engine at all — *widen the scalar kernel onto the 512b integer vector unit* (half from the private int VRF). Same win mechanism that made `cont_f32` land −15.5%: cut issue count 4–8× with zero tensor_wait, zero evict, zero barrier, PPL-exact. Threshold to flip me: a board run showing TFMA_WAIT_TENB > ~30% of cycles — then it's genuinely sync-bound and M2 pipelining wins instead.


#### [seed] Prof. Nakamura (applied-math / numerics)

## Position: it's ISSUE-rate bound, and the dtype debate is a red herring

Do the intensity arithmetic on fc1 (per frame): M·N·K = 3072·1024·768 ≈ 2.4G MACs; traffic = weights 3072·768·(1+ 1 scale/32B) ≈ 2.4MB + activations 1024·768·4 ≈ 3MB ≈ 5.4MB. That's **~440 MACs/byte** — even richer than the notes' 130, and weights are reused across all 1024 tokens ×4 frames. This kernel is **not** DRAM-bound. L2-miss/MAC should be ~0.

So "MAC-bound ~130" and "orchestration-bound" aren't contradictory — they describe different axes. It IS compute-dense (not bandwidth), but the scalar VPU gather in `mul_mat_Q8_0.c` spends ~4–6 retired instructions per MAC at IPC≈1, and the 24-Q8_0-block/tile serialized load→scale→wait chain dominates that instruction stream. **The wall is issue-rate + per-block orchestration, not bandwidth and not weight DRAM.** That's exactly why int8 TFMA washed: since weights are *already* int8, opcode-3 cuts zero weight traffic and only compresses the inner MAC — but the inner MAC wasn't the floor; the orchestration was. You swapped a cheap inner loop for an expensive one.

**Decisive board signature:** retired-inst/MAC and TFMA_WAIT%. If WAIT >30% → sync-bound (pipeline it, M1/M2). If WAIT low and inst/MAC >3 → issue-bound.

**My non-obvious lever, absent from results-so-far:** amortize orchestration by *widening the N-tile* — run all 4 frames' 4096 token-columns against one resident weight/dequant-block, so the fixed per-block load/scale/dequant cost is paid once per 4096 outputs instead of per small N-tile. This makes the 4-frame parallelism a *GEMM-shape* lever, not a multi-hart partition (which failed). Route B's per-block f-reg save/restore is *more* orchestration — it risks re-washing unless dequant is hoisted out of the N-loop.

**What flips me:** L2-miss/MAC materially >0.02 → reuse is broken, it's bandwidth after all, pivot to reuse-tiling. TFMA_WAIT dominant → sync-bound, pipeline beats everything.


#### [seed] Sam Okafor (kernel-optimization engineer)

Position: it's ISSUE-BOUND (scalar VPU issue rate), and the weights-already-int8 fact settles it — bandwidth is a red herring.

The decisive signature is retired-inst/MAC with L2-miss/MAC and TFMA_WAIT as the disambiguators. Threshold: if retired-inst/MAC > 2, L2-miss/MAC < 0.02, and TFMA_WAIT ≈ 0, it's issue-bound. The scalar `compute_block_dot_product_q8_0` (mul_mat_Q8_0.c:308-314) walks one output at a time — gather int8, sign-extend, multiply, accumulate, apply per-block scale: ~4-6 in-order instructions per MAC at IPC≈1. Each weight byte is reused across N≈1024 columns, so weight DRAM traffic is ~1 byte per 1024 MACs — L2-miss/MAC is floor-level. int8 TFMA cutting "weight bandwidth" cuts nothing that's bound. That's *why* it regressed +8%: it added 24-block-per-tile orchestration (load/fma/quant/store/WAIT ×24) without lifting MACs-per-issued-instruction enough. Orchestration-bound and issue-bound aren't contradictory — the scalar floor is issue, the failed fix hit an orchestration ceiling.

Follows: maximize MACs-per-instruction → Route B (float matrix engine, opcode-0) is directionally right because it retires a 16×16×16 tile per FMA with no activation-quant tax.

The idea NOT in results-so-far: **fuse the 4 frames into one GEMM (N→4096).** The frames share identical frozen weights; the kernel already loops i3 over ne13 (mul_mat_Q8_0.c:296). Collapsing that batch into the N-tiling amortizes the per-tile weight dequant/`tensor_load` 4× — attacking the orchestration ceiling that sank int8, and it's pure kernel-side reshape, no dispatch change.

Flips me: a board PMC run showing L2-miss/MAC > 0.05 or TFMA_WAIT-dominant on Route B.


#### [turn 1] Prof. Nakamura (applied-math / numerics)

> **Facilitator's challenge:** You and Sam both propose orchestration-amortization moves (widening N-tile vs. collapsing 4 frames into one GEMM), while Dr. Vega proposes scalar vectorization to attack issue-rate directly. Before anyone runs code: are these independent moves that stack (do 4-frame collapse, measure PMCs, then try vectorization if issue-rate is still ceiling), or are they competing diagnoses? Critically: *if* the 4-frame collapse successfully amortizes per-block cost 4×, does that make Dr. Vega's vectorization redundant, or is it the next sequential lever? And does widening N to 4096 columns risk breaking L2 reuse (larger per-tile footprint), or does weight reuse amortization actually hold?


**Answering the moderator directly, and Sam and Vega:**

These are orthogonal terms in the same cost equation, not competing diagnoses — but the arithmetic says they are *not* co-equal, and that reorders the plan. Write per-tile cost as `N_blocks × (orch_fixed + MACs_per_block · inst_per_MAC / IPC)`. Vega's vectorization attacks `inst_per_MAC`; the 4-frame/N-widen collapse attacks how often `orch_fixed` is paid. Different factors, so they stack.

But Amdahl caps the collapse. On the *current scalar* path, a 16×16 tile issues ~24.6k MAC-instructions; the per-block scale application across 24 Q8_0 blocks is a few hundred instructions — orch_fixed is already <2–3% of the scalar stream, and weights are already reused across all 1024 columns. Collapsing to 4096 extends amortization from /1024 to /4096: you're dividing an already-negligible term by 4. That's <1% on the scalar kernel. **So for the kernel we actually have, Vega's vectorization is the first lever — it attacks the 76% directly; the frame-collapse is second-order cleanup.** Where collapse genuinely pays is Route B, where `tensor_load`+dequant *is* a real per-tile fixed cost.

On L2: widening N does **not** break reuse, provided you keep the 256B weight block resident and *stream* activations past it — each activation is read exactly once regardless of N, weight L2-miss/MAC only drops. The reuse breaks only if someone materializes a K×4096 activation tile in the 3KB SCP; don't. So: not redundant, they stack — but vectorize first (dominant term), collapse-for-Route-B second. What flips the order: TFMA_WAIT>30%, then orchestration was the floor after all.


#### [turn 2] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** You frame the 4-frame collapse as "pure kernel-side reshape" independent of Route B, but activations differ per frame. How do you keep the per-frame weight block resident in L2 while streaming 4 different activation-column vectors through without either (a) materializing a K×4096 tile (breaking Nakamura's reuse math), or (b) paying the per-block dequant load four separate times anyway? Does your proposal actually require Route B, or is there a buffer layout that makes it work on scalar?


Nakamura, you've moved me — and it exposes a flaw in my own framing. I called the 4-frame collapse "pure kernel-side reshape independent of Route B," but the moderator's question is the right knife: on the *scalar* kernel there is no per-tile dequant to amortize. The scalar path (`compute_block_dot_product_q8_0`, mul_mat_Q8_0.c:308) reads the Q8_0 int8 weights and per-block scales *directly* — no `tensor_load`, no dequant epilogue. So `orch_fixed` is the few-hundred-instruction scale-application you costed at <2–3% of a 24.6k-issue tile. Extending the i3 batch loop (line 296) from /1024 to /4096 divides an already-negligible term by 4. That's your <1%. Conceded: on the kernel we actually have, collapse is a no-op.

Which means my proposal *does* implicitly require Route B — and that's the honest answer to the moderator. The buffer layout that "works" is: hold the dequanted 16×16 f32 weight tile resident (1KB, SCP-scale), stream 4096 activation columns past it one 16×K strip at a time, accumulate into the output tile. You never materialize K×4096 (that's the trap), each activation is read once, weight L2-miss/MAC only falls. But "resident *dequanted* tile" presupposes the dequant — so the collapse only buys anything precisely where there *is* a per-tile `tensor_load`+dequant fixed cost. On scalar there's nothing to hoist.

So the ordering flips to yours and Vega's: widen the scalar path onto the int VRF *first* — it attacks the 76% directly, zero wait, PPL-exact. Frame-collapse becomes the amortizer that could rescue Route B *if* Route B is even worth attempting after a 4–8× scalar speedup raises the bar. Flips me back: a board run showing scalar-widen stalls on L1D-port contention rather than issue rate.


#### [turn 3] Dr. Vega (ET-SoC1 hardware architect)

> **Facilitator's challenge:** You claim widening the scalar gather loop onto the int VRF cuts issue count 4–8×, with cont_f32's -15.5% board win as precedent. But 4–8× is an order of magnitude bolder than 15%. Are you claiming per-lane speedup equivalent to cont_f32 (so 4 lanes ≈ 60%, not 4–8×), or does the gather→sign-extend→multiply→accumulate chain parallelize categorically differently? Put it concretely: how many VRF lanes are available per hart, and does the latency chain break across lanes, or serialize?


Moderator, you caught a real sloppiness — and the honest answer is *smaller* than 4–8×, which actually strengthens the case.

First, cont_f32 is the **wrong magnitude anchor** and I shouldn't have leaned on it. That kernel was a memory move — fp32 through the 256b FP VRF, ~8 lanes — and it landed only 15.5% because it hit the L1D **store port**, not issue rate. Vectorizing a copy walls out fast. The scalar Q8_0 gather is categorically different: a genuine compute loop, retired-inst/MAC ~4–6 at IPC~0.9, so cutting *issued* instructions cuts cycles nearly proportionally. Different floor, different ceiling.

Lane count: the integer vector unit is 512b (half FP VRF, half private int VRF) = **64 int8 lanes**. You don't get 64×, and here's the chain honesty. gather→sign-extend→multiply are per-element independent — they break cleanly across lanes. The accumulate is the one dependency, a reduction. The trick that makes it *not* serialize: you hold a **vector of int32 partial sums, one per lane**, so the MAC is lane-parallel too; only the horizontal reduce is serial, and it's paid once per output over K=768 — amortized to noise.

The real taxes: int8→int32 **widening** (roughly halves effective throughput vs raw lanes), the per-Q8_0-block **scale application** (~24 blocks/tile, stays near-scalar), and activation L1D reads. Net I'll now claim **8–16 MACs/issued-instruction, ~2–4× board cycles on the 76%** — not 4–8×.

Sam — your L1D-port flip: vector loads *reduce* port pressure (one 512b load feeds 64 lanes), so issue-rate binds before the port does. Even the conservative 2× is ~35% board, 30× the gate. Prove it with retired-inst/MAC before and after.


#### [turn 4] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** You conceded that frame-collapse doesn't pay on scalar, and Vega's vectorization attacks the 76% directly — but you haven't committed to building the minimal int-VRF test that validates (or falsifies) his 8–16 MACs/inst claim. Can you design a single-tile Q8_0 gather-mac with lane-parallel int32 partials, run it on board with retired-inst/MAC and L1D-stall PMCs before/after, and report whether Vega's 2–4× translates or hits a ceiling earlier (register pressure, L1D port contention, load latency)?


Vega — I'm in, and I'll build it. But let me sharpen one hardware fact you glossed, because it's exactly where your 8–16 MACs/inst can bleed out.

You said the horizontal reduce is "paid once per output over K=768, amortized to noise." It isn't. Q8_0 carries a *per-32-element* scale. So the lane-parallel int32 partials can only run free for 32 K-steps before you're forced to reduce, convert to f32, multiply by that block's scale, and fma into the f32 accumulator — then zero the int32 vector and start the next block. That's **24 reduce+rescale epilogues per output tile**, cadence every 32, not one over 768. The reduce isn't the noise term; it's the metronome. That's the ceiling candidate I most expect to bite before L1D port contention does.

The test, concretely: single 16×16×768 tile, weights resident, stream activation strips, int32 partials in the private int VRF, block-scale epilogue every 32. Board PMCs before/after: retired-inst (18), cycles (19-ish), IPC, L1D load-stall counter, and TFMA_WAIT (must read ~0 — sanity that we're not accidentally on the engine). Metric = retired-inst/MAC and cycles on the isolated tile.

Prediction I'll hold myself to: if the block epilogue is cheap, retired-inst/MAC drops from ~4–6 toward ~1 and your 2–4× lands. If the 24 per-tile reduces dominate, we plateau near ~2× and inst/MAC sticks around 2.5 — still 30× the gate, still worth shipping, but it *reframes* whether widening the block (fewer, larger scale-groups) matters. Either way the PMC delta settles it, not our priors. I'll wire it via smolvlm2-build-verify for the math gate, then the board A/B. Building now.


### D2 — Is moving the Q8_0 GEMM onto ANY tensor-engine path the right family (int8 washed, float Route B in flight, fp16 opcode-1 untried) -- or should we stay on the VPU and software-pipeline mul_mat_Q8_0 in place?

_In-place VPU pipelining was named THE lever by both prior ports but NEVER landed here -- both attempts jumped to dtype changes. It is lossless, no .bss wall, no non-coherency barrier. Meanwhile Route B (float engine) issues ~4x more FMA dispatches than int8 and re-adds weight-dequant marshalling. And fp16 opcode-1 has never been tried. Which FAMILY has the best expected value, and why has the cheap lossless one been skipped?_


#### [seed] Dr. Vega (ET-SoC1 hardware architect)

Position staked below.

---

**Stay on the VPU. Software-pipeline `mul_mat_Q8_0` in place — but the real skipped lever isn't pipelining the scalar chain, it's that the hot kernel never used the vector unit at all.**

Look at the actual 76.3% path, `mul_mat_Q8_0.c:315-320`: `float sum = 0.0f; ... sum += compute_block_dot_product_q8_0(...)`. That is a **scalar** accumulate, one output per hart, a per-block dequant-and-multiply. Meanwhile the briefing (d) says this silicon has a **512b combined integer vector unit — 64 int8 lanes**. The Q8_0 block *is* 32 int8 quants + one f16 scale: the format is purpose-built for an int8×int8→int32 lane-parallel dot with a *single* f16 scale multiply per block at the end. We are running the most SIMD-friendly data layout in the codebase through a scalar loop.

That is why the cheap lossless option got skipped: both attempts read "pipeline the chain" as "feed the TFMA," so they jumped to dtype/engine changes (Route B re-adds evict + transpose-load + f0..f31 save/restore, `:250-267` — pure marshalling the VPU path never pays). But TFMA tile fill/orchestration is the very floor DnCNN/YOLO hit. The VPU vector path has **no** `tensor_wait`, no `.bss`, no cross-shire evict, no f-reg clobber.

**Family EV ranking:** VPU vectorized-dot >> fp16 opcode-1 > Route B. My new idea: rewrite `compute_block_dot_product_q8_0` as a 64-lane int8 vector MAC → int32 → one scale mul, then double-buffer two blocks. Expected: the int8 MAC savings that "washed" on TFMA cash out here because there's zero marshalling to eat them.

**What flips me:** a PMC trace showing the scalar dot is already memory-bound on `nb01` weight loads (TL_OPS/L2_EVICT dominating, not issue), so wider lanes stall on the same bytes — then vectorization buys nothing and fp16 double-buffering becomes the play.


#### [seed] Prof. Nakamura (applied-math / numerics)

## Position: stay on the VPU and software-pipeline in place. That family has the best EV, and the cheap lossless win was skipped because both attempts reflexively reached for a dtype change instead of reading the actual inner loop.

Look at the shipping scalar path, `block_ops.h:24` `compute_block_dot_product_q8_0`. The 8-wide SIMD is already there (`fmadd.ps`, 8 lanes). The waste is structural: it is called **once per Q8_0 block**, and each call pays a full epilogue — mask save/restore, `f10` re-init, gather-pattern reload, then a 6-instruction horizontal reduction (`fswizz/fadd/fswizz/fadd/fmvz`). For fc1 (K=768) that is **24 horizontal reductions + 24 function epilogues per output element**, versus 96 productive `fmadd.ps`. Productive-instruction fraction is roughly 25–30%.

The lever nobody landed: fold each block's `d` scale as a broadcast `fmul.ps` onto the converted int8 vector, so **one** 8-wide accumulator stays live across all 24 K-blocks and you do **one** horizontal reduction per output. Cost +4 `fmul`/block; saving ~6 (reduction) + ~8 (epilogue)/block. Net ~10 ops/block eliminated on the kernel that is **76.3% of cycles** — a plausible ~1.5–2x cut of its non-FMA issue stream. Lossless, no `.bss` wall, no cross-shire barrier.

Why not the engines: reuse is enormous (fc1 ≈ 9.66G MACs, weights 2.5MB reused over 4096 token-rows → ~130 MAC/byte, MAC/issue-bound, not DRAM-bound). Route B adds ~4x FMA dispatches + re-adds dequant marshalling; int8 already washed +8%.

**What flips me:** a board A/B of the reduction-hoist showing <1% — that would prove the epilogue wasn't the floor, and I'd pivot to fp16 opcode-1 as the untried engine.


#### [seed] Sam Okafor (kernel-optimization engineer)

## Position: stay on the VPU and fix the kernel that's actually running — the tensor-engine family is chasing the wrong bottleneck.

Read `block_ops.h::compute_block_dot_product_q8_0` (the scalar path, mul_mat_Q8_0.c:296–330 — this is the 76.3% kernel, since Route B isn't board-proven). The useful work per Q8_0 block is **four `fmadd.ps`**. Wrapped around them, *per block*: `mova.x.m`+`mov.m.x` mask save/set, `fbci` accumulator init, a `gather_pattern` reload, and a **~7-instruction `fswizz`/`fadd` horizontal reduction that runs every block** and returns a scalar the caller re-accumulates. That's ~4 useful of ~16 issued ops — **~25% VPU efficiency**. This is the "PER-BLOCK ORCHESTRATION" root cause of the int8 wash, on silicon, in the shipping kernel. It settles the disputed premise: not MAC-bound.

Two lossless levers, no .bss wall, no coherency barrier:
1. **Persistent 8-wide vector accumulator across all K_blocks**, one horizontal reduce per output — hoist mask/gather setup out of the loop. Fold per-block scale into the vector add. Alone ~2–2.5x this kernel ≈ **~30%+ of total cycles**, already 30x the gate.
2. **The idea not yet on the table — N register-blocking.** A (the int8 weight gather+`fcvt.ps.pw`) is re-executed for *every* activation column; gather+convert once, FMA against 2–4 B columns. On a 1024-token GEMM that's an N-fold redundant gather eliminated — raises arithmetic intensity *on the VPU*, no dtype change.

**What flips me:** if an A/B of lever-1 shows the per-block reduce/setup is already hidden under `fmadd.ps` latency (in-order, so I doubt it) and the kernel is genuinely `fmadd`-issue-bound, then int8 opcode-3's 4x MAC density is the only real headroom and I concede to the engine family. Cheap, lossless, one file — run it first.


#### [turn 1] Dr. Vega (ET-SoC1 hardware architect)

> **Facilitator's challenge:** Nakamura and Sam both reference `fmadd.ps` (8-wide SIMD) already in `compute_block_dot_product_q8_0`, contradicting your claim of a scalar path with no vector unit. Can you reconcile this against a read of `block_ops.h:24` — are you describing a truly scalar inner function, or does the existing code already vectorize within blocks but waste cycles on per-block epilogues (as they claim)? If it's already vectorized, would you then commit to running the PMC trace (TL_OPS/L2_EVICT vs. issue stalls) to determine if your 64-lane int8 rewrite will actually escape memory bottleneck, or if Nakamura's hoisted reduction is the cheaper first test?


You're right, and I'll take the correction cleanly. **Nakamura and Sam win the read.** `block_ops.h:24` is not scalar — lines 40-44 run `fcvt.ps.pw` + `fmadd.ps` 8-wide, four chunks a block. My "never used the vector unit" was wrong. What's actually there is an **8-lane fp32** MAC wrapped in exactly the epilogue they described: per-block `mova.x.m`/`mov.m.x` mask save-set (28-29), `fbci` init (30), `flw.ps` gather reload (34), and the 7-instruction `fswizz`/`fadd` horizontal reduction (55-68) that fires **every** block, then `mova.m.x` restore. The caller (`mul_mat_Q8_0.c:317-321`) then scalar-`sum +=` the reduced result. For K=768 that's 24 reductions + 24 mask dances per output against 96 productive `fmadd.ps`. Their ~25-30% productive fraction is real.

Where I hold ground: their fix keeps **8 fp32 lanes**; the 512b integer unit is **64 int8 lanes** — 8× width — and `fcvt.ps.pw` is pure conversion overhead the int8 path skips. That's a distinct, larger lever. **But it's downstream of measurement, and it re-imports activation-quant marshalling that washed +8%.** So I concede sequencing entirely: **run Nakamura's reduction-hoist first, and it IS my PMC trace.** Instrument that A/B with TL_OPS(15)/L2_EVICT_REQ(12) vs issue-stall. If the hoisted kernel still shows low TL_OPS and high issue, my 64-lane int8 has genuine headroom and we escalate. If it's `nb01` weight-load bound, wider lanes stall on the same bytes and I drop the int8 rewrite for good. One file, lossless, cheapest-first. I'm in.


#### [turn 2] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** You proposed both the reduction-hoist (hoisting mask/gather-reload outside the K-loop) and N register-blocking (FMA against 2–4 B columns per gather) as independent lossless levers. Vega and Nakamura are now betting opposite outcomes on the PMC trace of the hoisted kernel—Vega says it'll still be memory-bound on weight loads, Nakamura says the epilogue was the floor. Can you implement the reduction-hoist and A/B it with PMC instrumentation, and is N register-blocking independent of that hoisting, or does one block the other?"


Nakamura, Vega — yes, I'll implement it, and the sequencing you both landed on is right: reduction-hoist first, instrumented, because it's the same one file and it *is* the PMC trace. Concretely, that's one persistent 8-wide accumulator carried across all 24 K-blocks (fc1), scale folded in as a broadcast `fmul.ps` per block, mask/`fbci`/gather-pattern setup hoisted above the K-loop, and exactly one `fswizz`/`fadd` horizontal reduce per output instead of 24. I'll A/B it with TL_OPS(15)/L2_EVICT_REQ(12) against issue-stall, which is the exact fork you two are betting on.

But here's the thing that dissolves the bet, Vega: **N register-blocking isn't downstream of the trace outcome — it's the answer to your side of it.** The two levers are orthogonal axes. The hoist restructures the *K-loop epilogue*; N-blocking reuses the *weight gather+`fcvt.ps.pw`* across 2–4 activation columns. They compose cleanly: gather+convert one weight block once, `fmadd.ps` it into N persistent accumulators, N horizontal reduces at the end. Neither blocks the other.

And that composition is precisely why the trace doesn't fork the plan. If Nakamura's right and it's epilogue-bound, the hoist alone cashes out. If *you're* right and it comes back `nb01`-weight-load-bound, N=4 blocking cuts weight byte traffic and `fcvt` count by 4x — it directly attacks the memory bound you're predicting. Either outcome, both levers ship.

The one real coupling is register pressure: N accumulators = N×8 fp32 lanes live. N=4 plus operands fits inside 32 f-regs; N=8 doesn't. So I'll gate N by measured spill. Building the hoist now.


#### [turn 3] Dr. Vega (ET-SoC1 hardware architect)

> **Facilitator's challenge:** You predicted that if the hoist leaves us weight-load-bound (low TL_OPS, high L2_EVICT), then wider lanes stall on the same bytes. Sam just claimed N=4 blocking cuts weight traffic 4× by gathering each block once instead of N times. But is that reuse model sound? The weight block is only 2.5MB over 4096 tokens — that's ~640 bytes/token. If we're still memory-bound after the hoist, wouldn't the PMC trace already show those re-loads, and wouldn't simpler register allocation (to keep more weights live across N columns without spilling) be cheaper than restructuring the inner loop for N-blocking?


Sam, moderator — the moderator's instinct is right, and it actually sharpens my fork, so let me correct Sam's framing rather than his lever.

The "4× weight traffic" claim is the soft part. The 2.5MB fc1 weight block fits inside the 4MB shire L2. So on the second token-column those bytes are an L2 hit, not a DRAM refetch — TL_OPS(15)/L2_EVICT_REQ(12) will *not* scale 4× with N. If the hoisted kernel comes back memory-bound, N-blocking won't rescue it: the bound would be L1D→VPU streaming of a working set that doesn't fit in 4KB/hart, and N=4 doesn't shrink that footprint. So the moderator is exactly right — those re-loads would already be in the trace, and cutting DRAM traffic isn't what N-blocking does.

What N-blocking actually cuts is *instructions*: the weight gather + `fcvt.ps.pw` int8→fp32 unpack, ~4 ops/block × 24 blocks for fc1, re-executed per activation column. Amortize over N=4 and that VPU-issue overhead drops ~4×. That's an **issue-bound** win — which is my branch of the fork, not the memory branch. So Sam's "either outcome, both ship" is the one thing I'll push on: N-blocking is the answer to the issue-bound outcome *specifically*. If we come back evict-bound, it buys nothing.

And to the moderator's regalloc question: there is no cheaper allocation. You can't hold 2.5MB live in 32 f-regs; the only thing you *can* hold is one converted weight block across N accumulators — and that restructuring *is* N-blocking. Same mechanism, no shortcut.

So: run the hoist, read TL_OPS. High-issue/low-evict → N-block. High-evict → neither lever, and we're staring at a layout problem.


#### [turn 4] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** You said both levers 'compose cleanly' and 'either outcome, both ship.' But Vega just argued that if the hoist comes back eviction-bound (high L2_EVICT), then N-blocking won't help—because the 2.5MB fc1 weight block stays L2-resident on token-column 2+, so the bottleneck is L1D→VPU instruction issue (which N-blocking amortizes gather overhead for), not DRAM bytes. Is Vega's L2-residency model correct, and if yes, do you concede N-blocking is contingent on the PMC verdict, not automatic?"


Vega — yes, you're right, and I'll take the correction on the mechanism, because my "4× weight traffic" was sloppy. 2.5MB fc1 weights inside a 4MB shire L2 means token-column 2+ is an L2 hit, not a DRAM refetch. So L2_EVICT_REQ(12) will *not* scale with N, and if the hoisted kernel comes back evict-bound, that's an L1D→VPU streaming/working-set problem and N-blocking doesn't shrink the footprint. I conflated byte traffic with instruction count. Conceded.

What N-blocking actually amortizes is the `flw.ps` gather + `fcvt.ps.pw` int8→fp32 unpack — ~4 ops/block × 24 blocks for fc1, re-executed per activation column. That's a pure *issue-stream* reduction. So to the moderator's question directly: yes, N-blocking is contingent on the PMC verdict being high-issue/low-evict — it is not automatic, and I retract "either outcome, both ship."

But here's where I still push, and it's not hand-waving: the hoist itself almost certainly *lands* us in the issue-bound branch. These harts are in-order; once the per-block reduction and mask-dance are gone, the K-loop is a tight `fmadd.ps` dependency chain with the gather/`fcvt` issued inline against it. With 130 MAC/byte reuse and an L2-resident weight block, there's no DRAM wall to hit — the residual bound is issue, which is exactly N-blocking's target. So it's contingent in principle, but I'd bet the trace reads high-issue. If it comes back high-evict, I fold N-blocking entirely and we're looking at a layout/streaming problem, not a GEMM-restructuring one. Hoist first, read TL_OPS(15) vs issue-stall, then decide N.


### D3 — GREENFIELD: unconstrained by anything tried and by the current premises, what single change to the device kernels would most reduce firmware cycles on this workload?

_Question the premises. Is the GEMM even the right target vs cont/attention/fusion? Can the frozen-Q8_0 weights or the 4-frame parallelism be exploited in a way nobody tried? Can ops be fused so a whole tensor is never materialized (FlashAttention, LN->GELU->GEMM, im2col-free)? Is there a layout/residency trick that turns the 1024-token x 4-frame weight reuse into a win? Bring the highest-leverage idea you can defend, tried or not._


#### [seed] Dr. Vega (ET-SoC1 hardware architect)

## Position: the floor is redundant weight dequant, not the FMA — go weight-stationary.

The premise I reject is "the GEMM is one indivisible hot tile." Look at the live engine route in `mul_mat_Q8_0.c:228-284`: the loop is **output-tile-major** (`tile` over `m_tiles*n_tiles*batch`) with `kb` innermost. For every (m,n) tile it **re-dequants the same 16 weight rows** (lines 256-262), then pays `FENCE; engine_evict(wblk); WAIT_CACHEOPS` (263-265) **plus** an f-reg save/restore (248/267) around the float math. That per-K-block orchestration chain — exactly the cost sink the briefing names as the real floor — is billed **once per N-tile**. For fc1 (K=768, N=1024/frame) that's n_tiles=64 redundant dequant+evict+cacheop passes over identical frozen bytes, ×4 frames = the same weight block marshalled ~256×.

**Single change:** reorder to weight-stationary — dequant weight tile [mb,kb] **once**, hold `wblk` resident, stream all N activation tiles (and all 4 frames — same weights, `batch_idx` sharing) through it before eviction. The frozen-Q8_0 + 4-frame reuse nobody exploited: one dequant amortized over 4096 rows. This attacks the orchestration floor directly, is dequant-exact (PPL-safe), keeps `.bss` KB-scale (still one 1KB `wblk`), and needs no int8 marshalling.

**What flips me:** a board PMC where `L2_EVICT_REQ`/`TL_OPS` are already small and `TFMA_WAIT_TENB` dominates — then the evicts are noise and the lever is M2 pipelining the FMA chain, not loop order.


#### [seed] Prof. Nakamura (applied-math / numerics)

# Position: the target is right, but the win is residency, not dtype

Stop relitigating int8-vs-f32. Both washed because both re-pay the same hidden tax: **redundant weight dequant + a cache barrier per K-block**. Look at the *active* Route-B code already in `mul_mat_Q8_0.c` (lines 228–265). The tile loop is `(m_tile, n_tile)` outer, `kb` inner. The dequanted weight block `wblk` depends on `(mb, kb)` — **never on `nb`** — yet it is re-dequanted, `engine_evict`'d, and `WAIT_CACHEOPS`'d inside every n-tile.

Arithmetic on fc1 (M=1024, K=768, N=3072): m_tiles=64, n_tiles=192, k_blocks=48. Unique weight K-blocks = 64×48 = **3072**. Blocks actually dequanted = 64×192×48 = 590K **per frame** × 4 frames = **2.36M**. Redundancy factor = 192×4 = **768×**. That's ~604M float-muls of pure dequant *and* 2.36M serialized `evict → WAIT_CACHEOPS` barriers — the very "per-block orchestration" that the disputed root-cause note flagged, now quantified.

**Single highest-leverage change:** make the weight operand *stationary*. Reorder so `kb`/`nb` sweep with the dequanted `wblk` (and its SCP lines 16:31) held resident, streaming all 192 N-tiles — and all 4 frames — past each weight block. Dequant collapses 3072× (768× fewer barriers). This *is* the 4-frame-reuse lever the greenfield framing asked for, and it's orthogonal to opcode choice.

**What flips me:** if board profiling shows `wblk` already L1-resident (dequant/barrier <5% of the kernel's 76%), then it's genuinely MAC-bound and the engine dtype is the fight after all.


#### [seed] Sam Okafor (kernel-optimization engineer)

**Position: the highest-leverage single change is a weight-stationary loop reorder in `mul_mat_Q8_0.c`, not another dtype experiment.**

Look at the engine fast-path nest (mul_mat_Q8_0.c:228–269): the outer loop is `tile = global_id; tile < total_tiles` where `total_tiles = ne13*ne12*tiles_per_batch` (line 226). That means the batch dimension is *outside* the M-tile, and the weight quartet-repack/dequant at lines 256–262 is redone **per frame**. But the SigLIP weights are broadcast across all 4 frames (r2=ne12/ne02, r3 at 172–173) — the identical `src0` tile serves 4 frames × 1024 tokens = **4096 activation columns**. Today marshalling amortizes over one frame's N; it should amortize over 4096.

This is exactly what killed int8-TFMA: the dispute was "MAC-bound" vs "per-block orchestration." The reorder resolves it empirically for free — hoist the frame/batch loop *inside* the K-block dequant so each dequantized/repacked weight tile stays resident in L1 SCP (48×64B lines) and streams all 4 frames through before eviction. Marshalling cost drops ~4×. If int8-TFMA was orchestration-bound, this converts its +8% regression into a win; if MAC-bound, it's neutral and cheap to revert.

It touches only loop order in the modifiable surface, keeps weights frozen Q8_0, dequant stays f32-exact (PPL-safe), no new .bss, no cross-shire publish. **What flips me:** if profiling shows the batch loop already collapses (ne12==1, batching folded into N upstream), then reuse is already captured and I pivot to fusing `cont_f32`'s im2col out of the GEMM input instead.


#### [turn 1] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** You flagged ne12==1 as a blocker that would flip your position entirely — have you actually traced what the batching structure is at fc1 in the live SmolVLM2 workload? If batching is already folded upstream, there's no 4-frame reuse to amortize over and all three of you are optimizing a phantom.


I traced it. Answer below.

---

**Answering the moderator directly, and Nakamura/Vega:** you're right to force this, and the trace changes the story. I did not have it at turn 0 — I do now.

The tell is the cont launch count, not the GEMM. `EXPERIMENTS.md:139` pins it: 2 conts/attention-layer × 12 layers × ~18–20 passes ≈ 480 of the 496 `cont_f32` launches. That arithmetic only closes if each of video_dog's 4 frames × ~4–5 sub-image tiles is a **separate encoder pass** — a distinct graph execution. If the 4 frames rode a batch axis, the conts would batch too and you'd see ~96–120 launches, not 480. So at `fc1`, `src1` arrives as `[K=768, tokens=1024]` with **ne12==1**. The `batch_idx`/`r2`/`r3` machinery at `mul_mat_Q8_0.c:229–237` is dead weight on this workload.

**So concede: the "×4-frame reuse, one dequant amortized over 4096 rows" framing — mine, Vega's, and Nakamura's — is a phantom.** There is no frame axis inside the launch to hold weights stationary across.

But the lever survives, just smaller and *cleaner*. Redundant dequant lives **within a single launch**: `wblk` depends on `(mb, kb)`, never `nb`, yet `256–265` re-dequants + `engine_evict` + `WAIT_CACHEOPS` inside every one of the **n_tiles=64** token-tiles. That's a real 64× redundant dequant/barrier chain per launch — no cross-launch state needed, which is *better* for us (no .bss persistence, no seam).

The honest cost I now put on the table: true weight-stationary means holding the dequanted K-panel resident — 48KB (fc1) / 192KB (fc2) per hart, on **stack, not .bss**. 192KB stack frames are the real risk to vet before I commit. Reorder still wins; the sales pitch drops from 256× to 64×.


#### [turn 2] Dr. Vega (ET-SoC1 hardware architect)

> **Facilitator's challenge:** Sam's trace kills the 4-frame reuse — ne12==1 means each launch is solo. So redundancy is 64×, not 768×. But he's flagged 192KB stack per hart as a risk for weight-stationary dequant panels. What's Erbium's stack depth per hart, and will 192KB frame allocations spill into L2 or cause hart-to-hart contention? If stack is tight, do we reconsider the priority order — maybe fusing `cont_f32` im2col out of the GEMM input becomes higher-leverage than orchestration reorder?


## Dr. Vega — the 192KB panel is a self-inflicted wound; strip-mine and it vanishes

Sam, good trace — I concede the 4-frame reuse is a phantom, ne12==1 kills it. But your 192KB stack fear rests on a conflation I want to kill before we downgrade priority.

Hardware first: stack lives in DRAM at `0x8005801000`, not a fast tier. L1D is **4KB/hart**, SCP is **48×64B = 3KB, hart0-only**. So a 192KB dequant panel is *never* resident anywhere fast — and it doesn't need to be, because `tensor_load` **bypasses L1/L2 and reads DRAM directly**. "Weight-stationary" here buys you dequant-compute + evict/barrier *count*, not memory residency. You were pricing residency you can't and don't need to hold.

Where does 192KB even come from? Full weight-stationary across K forces you to hold `n_tiles=64` separate output accumulators (kb no longer innermost → TenC can't accumulate one tile). *That's* the 64KB–192KB, not the weight bytes. The fix is **N-strip-mining**: inner strip of ~8 n-tiles, kb middle. Dequant each `(mb,kb)` block once per strip → redundancy drops 64×→8× (most of the win), while live state = 1KB `wblk` + 8×1KB accumulators = **~9KB stack**. Tunable knob, KB-scale, no seam.

So no — don't reorder the priority. cont_f32 is already the M1 landing (−15.5%); re-profile before touching its im2col. The orchestration reorder stays #1, just executed as strip-mine, not full-panel.

**What flips me:** a board PMC showing `TL_OPS`/`L2_EVICT_REQ` already <5% — then dequant isn't the floor and we're back to pipelining the FMA chain.


#### [turn 3] Sam Okafor (kernel-optimization engineer)

> **Facilitator's challenge:** Vega's strip-mining proposal cuts the live state to ~9KB and preserves 8× redundancy reduction. Can you confirm whether 8 n-tile inner strips actually hold dequant redundancy to 64→8×, and whether 9KB is realistic for the dequant buffer + 8 tile accumulators, or is there a hidden cost in TenC's multi-tile wiring that breaks this math?


Traced it. Answering the moderator and Vega directly.

**Vega — your 9KB is the right number for the wrong resource, and the mis-accounting hides a hard wall.** The dequant-count math is fine: 8 n-tiles inner, kb middle, dequant each `(mb,kb)` once per strip → 64→8×. That part holds. But look at where TenC actually lives. Line 286: `tensor_store(0, 0, 3, ...)` reads `start_reg=0` — the store pulls the accumulator **out of the vector register file**, f0..f31. That's exactly why lines 248/267 `q8_save/restore_fregs` bracket the dequant: the dequant's scalar float math would clobber the live accumulator. f0..f31 in `.ps` = 32×8 = 256 floats = **one 16×16 tile, and that's the entire register file.** There is no multi-tile TenC wiring to hold 8 accumulators — there's one, and it's the FREG bank.

So an 8-wide strip needs 8 accumulators live across the 48 kb steps, but only one fits in registers. The other 7 spill to DRAM (`tenc_loc=1`) with a read-modify-write **per fma**: 8 strips × 48 kb × 8 n-tiles of 1KB C-traffic ≈ 384× the current C-side DRAM round-trips. That eats the dequant win whole. Your 9KB was priced as if the accumulators were free stack; they're register-file contention.

**The clean version keeps n innermost so C stays the single resident FREG tile — and pre-dequants the full K-panel once per m-tile into a 48KB (fc1)/192KB (fc2) DRAM/stack buffer, evicted once.** `tensor_load` reads B from DRAM anyway (your own point), so panel-in-DRAM costs nothing extra and gets the full 64× dequant+barrier cut with zero multi-accumulator problem. **What flips me:** if the 192KB fc2 panel blows the stack frame — that's the one number I still have to vet on-board.


#### [turn 4] Dr. Vega (ET-SoC1 hardware architect)

> **Facilitator's challenge:** Sam claims TenC accumulators are register-resident (f0..f31 = 256 floats = one tile exactly), with no multi-tile wiring—so 8 live accumulators force 7 into DRAM spillage. Is that hardware-correct? If it is, does strip-mining's dequant savings actually survive the C-side traffic explosion?


Verified against `tensor.h` and the live loop. Here's my turn.

---

**Dr. Vega — Sam, you've located the accumulator in the wrong physical bank, but the spill you're worried about is real anyway. Both matter.**

The mechanism first: `tensor_fma`'s `tenc_loc` param (tensor.h:469) is explicit — "0 = L1 scratchpad, 1 = memory." TenC accumulates in the **SCP**, not f0..f31. The `start_reg=0` you cite at the store is the `tenc2rf` copy-out — one tile, transiently, at store time only. During the 48 `kb` passes the accumulator never touches the register file; that's why `first_pass=(kb==0)` seeds it in SCP and the rest add into SCP. So "256 floats = the whole FREG = can't hold 8" mis-names the resource. The `q8_save/restore_fregs` dance protects the FREG only because scalar dequant runs *interleaved* — not because C lives there.

**But you're right in substance, and I'll take the correction on my "9KB stack" line.** The SCP is 48×64B = exactly one A-tile (16 lines) + B-transpose (16) + C (16). There is no room for 8 C-tiles in SCP any more than in FREG. An 8-wide strip forces 7 accumulators to `tenc_loc=1` — a genuine per-FMA DRAM read-modify-write. Your 384× C-traffic magnitude survives the resource rename. Strip-mining-as-I-framed-it is dead.

**So I converge on your pre-dequant panel** — n innermost, single SCP-resident C, weights dequanted once per m-tile into a DRAM buffer that `tensor_load` transpose-reads (DRAM-read is free, it bypasses L1/L2 regardless). Bonus you undersold: hoisting all scalar dequant out of the FMA phase **eliminates the per-kb `q8_save/restore_fregs` entirely** — the FREG is never live during accumulation.

**On your 192KB fc2 wall:** it won't trip event 65535 — that wall is NOLOAD `.bss` in the load image at backend init. A stack panel is runtime DRAM, invisible to the loader. The only real question is hart stack headroom, and that's a one-line board vet, not a redesign.
