# SmolVLM2 optimization — agent working environment

How to drive this work with subagents. Mirrors the proven DnCNN/YOLO loop: a shared working-method
contract, a milestone ladder with explicit gates, and an adversarial reviewer spawned **before** every
board gate. The goal is to keep agents from the two failure modes that cost us days on prior tracks —
optimizing an unmeasured bottleneck, and shipping a silent-corruption change that only fails on the board.

## Roles (agents to spawn)

| Agent | Type | Job |
|---|---|---|
| **Profiler** | general-purpose | PMC/`ET_PERF` pass on the board; produce the real cycle breakdown before any opt. Owns `04_KERNEL_STATE.md` updates with measured numbers. |
| **Kernel author** | general-purpose (or fork of main loop) | Implement one milestone's kernel change under `et-kernels/src/`. One lever at a time. |
| **`smolvlm2-kernel-reviewer`** | dedicated agent (`.claude/agents/smolvlm2-kernel-reviewer.md`) | Adversarial red-team of the diff against the hardware traps + quality gate. Spawn BEFORE each board gate. Read-only; returns BLOCK/PROCEED. |
| **Oracle/verify** | general-purpose | Owns the host reference + max_abs bit-exactness check on sys-emu, and re-runs PPL + fixed-answer after any change. |

The parent (main loop) sequences milestones, reads the reviewer verdict, and only advances on PROCEED +
a clean board gate.

## The shared contract (every agent obeys)

1. **Modifiable surface is ONLY** `ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et/et-kernels/src/`.
   Any diff outside it cannot land (trusted overlay rejects). No config/scorer/runtime/CI edits.
2. **Isolate the kernel — don't develop against the whole model.** Build and test ONE kernel standalone
   against a tiny host oracle, loaded via `GGML_ET_KERNELS_PATH` (rebuild one `.elf`, not all of llama.cpp).
   Do NOT run the full SmolVLM2 forward each iteration. This is the DnCNN/YOLO loop and it's ~orders of
   magnitude faster to iterate.
3. **Single-hart first; multi-hart last.** Get `max_abs=0` at 1 hart before touching partition. The
   cacheline-seam-race class is invisible at 1 hart / sys-emu, so multi-hart (8) is introduced ONLY at the
   dedicated board seam gate, after the single-hart math is proven — never mixed into a math change.
4. **Don't sink time into sys-emu.** It models zero cache/tensor cycle cost → **it cannot show perf**, and it's
   ~2000–3400× slower than the board. Use it for exactly one thing: a fast single-hart `max_abs=0` correctness
   smoke on the isolated kernel. Everything perf — PMC, `ET_PERF` timing, A/B — is **board-only**; iterate
   perf on the board.
5. **One lever per milestone.** Implement → single-hart `max_abs=0` (sys-emu, quick) → reviewer → board gate
   (perf + PPL + answer). Revert cleanly if it's a WASH; log it so nobody retries a dead lever.
6. **Quality gate is non-negotiable**: exact `video_dog` answer, zero vision fallbacks, full offload,
   PPL ≤ 26.739 and ET-vs-CPU within 1%. A faster kernel that moves any of these is a FAIL.
7. **Correctness guards are not optional** (see `03_HARDWARE_NOTES.md`): no `fdiv`/`fsqrt`; FREG-clobber
   barrier; 64B alignment + seam-pad for multi-hart; no `R_RISCV_64` pointer tables; no float `/`/`sqrtf`
   (kernels build `-O3` + a build-time unimplemented-instruction checker — the trap fails the build here).
8. **Rigor for perf claims**: interleaved paired-main A/B, non-overlapping distributions, ≥1% mean win with
   ≤0.5% main drift. A single run is not evidence.

## Milestone ladder

Aim (from `01_ARCH_COMPUTE_PROFILE.md`): the workload is **vision-encoder-bound**; the SigLIP MLP + attention
GEMMs are ~65% of all compute and route to the FP32 matrix-engine kernel, which is currently a **naive,
un-pipelined single-tile-per-hart GEMM** (`mul_mat_f32_matrix_engine.c`). That kernel and attention are the
targets.

- **M0 — PROFILE.** Board PMC + `ET_PERF` breakdown of `video_dog`: cycles by op (im2col, the four MUL_MAT
  variants, NORM, UNARY/GELU, SOFT_MAX, RMS_NORM, ROPE, GLU, Q8_0 matmul). Confirm vision-encoder dominance
  empirically and identify whether the GEMM is load-latency-bound (→ pipelining) or bandwidth-bound (→ int8/
  residency). **No kernel edits.** Deliverable: measured table in `04_KERNEL_STATE.md`.
- **M1 — Double-buffered `tensor_load` in the GEMM** (top lever #2). Two load IDs; issue tile N+1's load
  during tile N's FMA. Bit-exact vs current kernel (`max_abs=0`), then board A/B.
- **M2 — Full software-pipeline the GEMM chain** (top lever #1). Steady-state `load→fma→(quant)→store`
  overlap, single-hart to avoid dncnn's SMT deadlock. Bit-exact, reviewer, board A/B.
- **M3 — FlashAttention-style fused SoftMax attention** (top lever #3). Online-softmax, no N×N score matrix,
  VPU exp/recip. Exact vs current; watch PPL. Board A/B.
- **M4 — Cache-op batching + 4-frame multi-hart partition** across the encoder GEMMs. Seam-pad; reviewer
  heavy on the seam axis. Board A/B (this is the class where silent corruption lives).
- **M5 — int8 / fp16 TFMA for the big SigLIP MLP GEMMs** (top lever #4/#10) — ONLY if M0 shows the GEMM is
  MAC/bandwidth-bound and only AFTER M1/M2 pipelining, since yolo proved int8 = parity when orchestration is
  the floor. Faithful affine-uint8 activations (pad-with-Z), on-device Q8_0 repack, bias fused into requant.
  Hard PPL/answer re-check.

Each milestone: implement → `max_abs=0` on sys-emu → `smolvlm2-kernel-reviewer` → 3× clean board gate +
paired A/B + PPL + answer. Log the result (LANDED/WASH/BLOCKED) so the ladder stays honest.

## Prompt-authoring tips (what worked on prior tracks)

- Give the kernel author the **exact target shape** from M0 (e.g. `[1024×768]·[768×3072]` fp32, batch 4) and
  the **exact field encodings** from `int8-tfma-encodings` — don't make it re-derive tensor CSR layout.
- Always point the reviewer at ground-truth `tensors.cpp`, not comments/memories — memories reflect the
  state when written; verify the file/flag still exists.
- Feed the reviewer's advice through `03_HARDWARE_NOTES.md`: the yolo reviewer once prescribed a *fatal* fix
  (forcing true division). Reviewers catch real bugs but can misprescribe on the no-divide axis.
- Keep the loop tight with `GGML_ET_KERNELS_PATH` (runtime-loaded ELF) for local iteration; only rebuild the
  host binary for the actual PR/board run.
