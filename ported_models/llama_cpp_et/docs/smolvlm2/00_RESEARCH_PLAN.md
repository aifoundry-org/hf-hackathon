# SmolVLM2-500M-Video — ET-SoC1 optimization research plan

Entry point for the week-2 SmolVLM2 optimization track. This is an **optimization** track on the shared
`llama.cpp-et` runtime (a functional baseline already exists — commit `cc4049d` "Add ET vision kernels for
SmolVLM2"), not a fresh port. Read the numbered docs in order; this one is the index + sequenced plan.

| Doc | Contents |
|---|---|
| `01_ARCH_COMPUTE_PROFILE.md` | SmolVLM2 architecture + where the cycles go (verdict: vision-encoder-bound) |
| `02_OPTIMIZATION_CATALOG.md` | Strictly categorized technique catalog (A dataflow / B fusion / C math / D quant / E compression / F compiler), each cross-referenced to what we already tried on DnCNN/YOLO |
| `03_HARDWARE_NOTES.md` | ET-SoC1 "erbium" ground truth + the mandatory correctness guards |
| `04_KERNEL_STATE.md` | Audit of every modifiable kernel's current optimization state + ranked targets |
| `05_AGENT_ENVIRONMENT.md` | Agent roles, working-method contract, milestone ladder, prompt tips |
| `.claude/agents/smolvlm2-kernel-reviewer.md` | Adversarial reviewer spawned before each board gate |

## The situation in five facts

1. **Modifiable surface = ONLY** `…/llama.cpp-et/ggml/src/ggml-et/et-kernels/src/*` (the RISC-V ET device
   kernels). Backend `.cpp`, host runtime, CI, config, scorer are main-owned — a diff outside the kernel dir
   cannot land.
2. **Score = `device_cmd_exec_dur` firmware cycles** on the `video_dog` 4-frame case, **generation = 3
   tokens**. Beat paired-main mean **≥1%** (≤0.5% main drift, ≥0.25% wall). Since decode is 3 tokens, the
   score is **prefill + vision bound**.
3. **Hard quality gate**: exact lowercase-alphanumeric answer, **zero vision CPU fallbacks**, full LLM
   offload, WikiText-2 PPL ≤ 26.739 and ET-vs-CPU within 1%. → almost all value is **lossless dataflow +
   fusion**; weight re-quant / pruning / low-rank / linear-attention are **gate-forbidden**.
4. **Workload is strongly vision-encoder-bound** (~80% of MACs): SigLIP MLP fc1/fc2 ≈43% of all compute,
   attention QKV/O proj ≈22%, O(N²) attention ≈14%, over **4 embarrassingly-parallel frames** (no mask, no KV
   cache). Optimize the SigLIP GEMMs first.
5. **No GEMM kernel is well-optimized on the tensor engine.** Only `mul_mat_f32_matrix_engine.c` and `memops.c`
   touch the TFMA at all — and the matrix-engine GEMM is a **serialized, un-pipelined, single-tile-per-hart**
   chain. `mul_mat_f16.c` and `mul_mat_Q8_0.c` use **zero tensor engine** (naive VPU dot products). Whichever
   of these three the dominant vision GEMM lands on, it is leaving the matrix engine idle or under-fed.

## The one empirical unknown to resolve FIRST

**Which kernel does the dominant SigLIP GEMM actually dispatch to?** It depends on the stored weight dtype,
and the baseline mmproj is Q8_0:
- If **F16** → `mul_mat_f16.c` (no TFMA) — huge lever: get it onto fp16 TFMA (opcode 1) or F32→matrix-engine.
- If **Q8_0** → `mul_mat_Q8_0.c` (VPU int8, no TFMA) — lever: drive the int8 TFMA opcode-3 path (our proven
  DnCNN/YOLO territory), but recall yolo int8 hit **parity** when orchestration-bound, so pipeline first.
- If **F32** → `mul_mat_f32_matrix_engine.c` — the double-buffer/pipeline lever is the whole game.

M0 answers this on the board with `ET_PERF`/`GGML_ET_PROFILE` (which kernel fires + its cycle share) — do not
guess. The robust conclusion holds either way: **the top lever is getting the dominant vision GEMM onto a
well-pipelined tensor-engine path.**

## Sequenced plan (one lever per milestone; gate every step)

- **M0 — PROFILE (no edits).** Board PMC + `ET_PERF` cycle breakdown of `video_dog`. Resolve the dtype/kernel
  question above. Confirm vision dominance and classify the GEMM as load-latency-bound (→ pipeline) vs
  bandwidth-bound (→ int8/residency). Update `04_KERNEL_STATE.md` with measured numbers.
- **M1 — Double-buffer the TFMA GEMM** (`mul_mat_f32_matrix_engine.c`): 2 load IDs, issue tile N+1's load into
  the free SCP lines during tile N's FMA. `max_abs=0` sys-emu → reviewer → board A/B.
- **M2 — Full software-pipeline** the `load→fma→store` chain, single-hart (avoids DnCNN's SMT-producer
  silicon deadlock). The lever both prior ports named but never landed.
- **M3 — Route the dominant vision GEMM onto the tensor engine** per M0's dtype finding (fp16 TFMA, or F16/
  Q8_0→F32 pre-convert into the M1/M2 kernel, or int8 TFMA opcode-3). Hard PPL + answer re-check.
- **M4 — FlashAttention-style fused SCALE→SOFT_MAX** (online softmax, no materialized score row) + fuse GELU
  into the fc1 epilogue and NR-rsqrt the norms.
- **M5 — Cache-op batching + 4-frame multi-hart partition** across the encoder GEMMs (seam-pad, reviewer heavy
  on the seam axis — this is where silent board corruption lives).

Each milestone: implement → `max_abs=0` on sys-emu → `smolvlm2-kernel-reviewer` (BLOCK/PROCEED) → 3× clean
board gate + interleaved paired-main A/B + PPL + exact answer. Log LANDED/WASH/BLOCKED so the ladder stays
honest and dead levers aren't retried.

## Doctrine (hard-won on DnCNN + YOLO)

- **Measure on the board, not sys-emu** (sys-emu models zero cache/tensor cycle cost). Profile before
  optimizing; the bottleneck was orchestration, not MACs, on both prior ports — but SmolVLM2's *large* GEMMs
  may differ, so re-measure.
- **Divide-free device code** (no `fdiv`/`fsqrt`; kernels build `-O3` + a checker that *fails the build* on a
  trapping instruction — no `-Ofast` here, unlike DnCNN/YOLO; host-bake or NR reciprocals). **FREG-clobber
  barrier** around tensor ops. **64B-align + seam-pad** any multi-hart partition. **No `R_RISCV_64` pointer
  tables.** See `03_HARDWARE_NOTES.md`.
- **int8 is not a free win here.** yolo proved int8 = parity when the floor is per-tile orchestration; it only
  pays after the chain is pipelined and if the GEMM is genuinely MAC/bandwidth-bound. Sequence it after M1/M2.
