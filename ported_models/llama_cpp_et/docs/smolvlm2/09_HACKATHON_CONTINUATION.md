# 09 — Hackathon Continuation Summary (SmolVLM2 track)

**Dated 2026-07-18.** One-document catch-up for continuing this hackathon: where we are, what's proven
dead, what's open, and how to run everything. Read this first; then `08_STATE_HANDOFF.md` (kernel-level
detail), `EXPERIMENTS.md` (full measured ledger), `TFMA_VERDICT.md` + `EVENT65535_VERDICT.md` in
`local-artifacts/routeb-variants/` (freshest research).

## 1. The contest

- AIFoundry + OpenHW **CORE-ET model-porting hackathon**; submissions via GitHub PR to
  `aifoundry-org/hf-hackathon`, board CI runs on real **ET-SoC1** silicon.
- **Our target: `smolvlm2_500m_video`** (`SmolVLM2-500M-Video-Instruct`, frozen Q8_0 GGUF + Q8_0 mmproj,
  sha-pinned in `artifacts.json`). Optimization track on the shared `llama.cpp-et` runtime, not a fresh port.
- **Score = `device_cmd_exec_dur` firmware cycles** on the `video_dog` 4-frame case (gen = 3 tokens →
  prefill/vision-bound). Must beat paired-main by **≥1%**.
- **Hard quality gate:** exact lowercase answer (" Dog."), PPL ≤ 26.739, ET-vs-CPU PPL ≤ 1%, **zero vision
  CPU fallbacks**, full LLM offload. Weights frozen → almost all value is lossless dataflow/scheduling.
- **Modifiable surface = ONLY** `ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et/et-kernels/src/`
  (a git submodule). Backend `.cpp`, host runtime, CI are main-owned.

## 2. Scoreboard (board-verified)

| milestone | submodule commit | pmc_cycles | delta |
|---|---|---|---|
| M0 baseline | `cc4049d` | 10,817,973,417 | — |
| cont_f32 vectorize (flq2/fsq2) | `fb8cc4b` | 9,136,533,962 | −15.5% |
| scalar_hoist | `5413c722e` | 8,489,934,819 | −7.1% (−21.9% cum) |
| N=4 register-block | `fb0bb4f02` | 7,692,846,657 | −9.4% (−28.9% cum) |
| **N=8 reg-block + act-load pipeline (CURRENT HEAD)** | **`800daee36`** | **6,170,682,271** | **−19.8% (−43.0% cum)** |

Branch `smolvlm2-cont-vectorize`. Quality gate passes (PPL 22.2815, zero fallbacks).
`mul_mat_Q8_0` is still **~86% of cycles (~5.33B)** — the only kernel worth attacking.
Key mechanism learned (S2/S2b): the scalar loop is **load-use-stall-bound**, not instruction-count
bound — pipelined loads (2-reg round-robin) flipped N=8 from +10% to −22%.
The tensor engine for this GEMM is CLOSED (3 forms measured dead, incl. int8-TENC ~7x slower
2026-07-18 — board launches ~2048 harts which the scalar path uses and the engine cannot). It is already
8-lane VPU (`fgb.ps`→`fcvt.ps.pw`→`fmadd.ps`, dequant FREE inside the MAC), **feed/issue-bound
(~1 MAC/issued-instr)**, not MAC-bound.

## 3. Measured-dead (board evidence — do not retry these forms)

- **int8 TFMA opcode-3, per-tile (M2g):** +8% regression (192× redundant activation re-quant).
- **int8 barrier variant (shared pre-pass):** reached correctness at 1024 harts, then ~+15% slower perf
  proxy + `event 65535` crash in the profiled run. Marshaling > MAC savings.
- **fp32 Route B engine (v1e_base):** correct on board but **parity** (2.71s vs 2.73s scalar) — pays a
  separate dequant pass + per-K-block FREG save/restore that the scalar path fuses for free.
- **fp16 opcode-1:** TFMA_VERDICT Q3 — dominated (inherits fp32's FREG tax, adds f16 rounding). Don't build.

## 4. Open frontier (ranked by expected value)

1. **`event 65535` SOLVED (2026-07-18):** the int8-TENC (B)-hardening (fsq2 publish + evict +
   double-buffer + CLEAR_TENSOR_ERROR) eliminated it — engine kernels are scoreable now; that one
   is just slow. Original analysis: device `DEVICE_FW_ERROR` (SETTLED from source in
   `EVENT65535_VERDICT.md` — a kernel-triggered firmware fault, NOT trace-buffer, NOT .bss; host fall-through
   bug at `RuntimeImp.cpp:724-756` throws on the 0xFFFF sentinel). It only fires on heavy-engine kernels in
   the **profiled** scoring run. It is fixable from our surface → the engine is **not** structurally
   unscoreable. Diagnose WHICH firmware fault first.
2. **int8-TENC clean form** (TFMA_VERDICT Q2): opcode-3 accumulates in the TENC HW register file → no FREG
   save/restore, weights already int8 → ~60 MAC/instr theoretical vs scalar's ~1. The only engine form with
   real headroom over 7.66B. Risks: int8-activation PPL + the event-65535 wall.
3. **N-register blocking (scalar, N=4)** — gather+convert each weight K-block once, `fmadd` against 4
   activation columns; compounds with scalar_hoist (~1.5–2× plausible). Never built; the designated next
   SCALAR lever, no engine walls.
4. **Board PMC baseline on the hot tile** (inst/MAC, L2 evicts, TFMA_WAIT%) — never obtained; would settle
   issue-bound vs evict-bound definitively. Harness: `local-artifacts/run_mulmat_q8_hpm_board.sh`.
5. **Staged small patches** in `local-artifacts/routeb-variants/`: `norm_nr.patch`, `softmax_poly.patch`
   (tiny), `v1e_dbuf.patch` (Route B double-buffered, never board-run).
6. **Rigorous interleaved paired-main A/B** (`smolvlm2-ab-score` skill) to formalize the committed wins.
7. `mul_mat_f32_matrix_engine.c` double-buffering (2.72% kernel, technique transfer), `cont_f16.c` same
   flq2/fsq2 fix (off hot path).

## 5. Hardware traps (learned the hard way — full list in `03_HARDWARE_NOTES.md`)

- **Non-coherent across shires:** plain stores to cross-kernel-consumed tensors are invisible → use
  `flq2/fsq2`/`amoswapg` publishing + `evict` before `tensor_load` of scalar-written scratch. Sys-emu/1-hart
  CANNOT catch this (cont_f32 attempt #1 failed on board only).
- **Static `.bss` ~0:** large static arrays (≥384KB) crash kernel load with `event 65535`. Per-hart stack +
  runtime DRAM are fine.
- **SCP = 48 lines × 64B** — only one 16×16 C-accumulator tile is holdable; never strip-mine C.
- Forbidden ops (build fails): `fdiv.s`, `fsqrt.s`, `frsq.ps`, `fsin.ps`, long↔float `fcvt`. Use
  `et_fdiv`/NR/`frcp.ps`. Inline-asm constants in callee-saved `fsN` regs only.
- FREG clobber barrier (`f0`–`f31`) around tensor ops; `ua/ub` signedness wrapper bug (flag SET = unsigned).
- Cacheline seam race on multi-hart writes: keep `stride_bytes % 64 == 0`, single writer per line.
- **Stale-file trap:** a crashed board run leaves the previous score in place — ALWAYS check `scored_at`
  freshness before trusting any board number.
- Topology: `erbium-soc1sim` = 1 shire, 8 active harts (even); full multi-shire is policy-blocked for the
  scored harness, though the 32-shire board exists for experiments.

## 6. How to run things

| Task | Command / place |
|---|---|
| Local math gate (autonomous) | skill `smolvlm2-build-verify`; `ET_TOOLCHAIN=~/et-src/et ET_PLATFORM=~/et-src/et`, `test-backend-ops` max_abs=0 at 1 hart |
| Apply a candidate patch | `git -C $SUB checkout HEAD -- ggml/src/ggml-et/et-kernels/src/` then `git -C $SUB apply $P/<NAME>.patch` (SUB = submodule, P = `local-artifacts/routeb-variants`) |
| Board run (HUMAN only — agent cannot ssh) | `bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh` (board: `ivan@aifoundry2`, Tailscale) |
| Read the score | `.ci-work/smolvlm2-cont-candidate-output/score-smolvlm2_500m_video.json` → `scored_at` must be fresh; beat 8,489,934,819 |
| A/B diff | `python3 local-artifacts/smolvlm2_ab_diff.py <baseline-dir> .ci-work/smolvlm2-cont-candidate-output` |
| Reviewer before board | agent `smolvlm2-kernel-reviewer` (adversarial pre-gate) |
| Orient in a fresh session | `/smolvlm2` slash command; milestone commands `/smolvlm2-m0…m5`; loop discipline in `RUNBOOK.md` |

Restore after testing: `git -C $SUB checkout HEAD -- ggml/src/ggml-et/et-kernels/src/`.

## 7. Map of the repo (whole-folder context)

- `ported_models/llama_cpp_et/` — our track. `docs/smolvlm2/00–08` = plan/profile/catalog/hardware/kernel
  audit/agent contract/int8 design/expert debate/state handoff; `EXPERIMENTS.md` = append-only measured
  ledger; `RUNBOOK.md` = fresh-prompt flow. Submodule `src/llama.cpp-et` carries the kernels.
- `ported_models/yolo/` — int8 TFMA port (LANDED at parity→win via full VPU/multi-hart/weight-repack stack;
  1.484s leaderboard). Lessons transferred: orchestration-bound > MAC-bound, seam-race guards, A/B method.
  `local-artifacts/YOLO_INT8_TECHNIQUE_TRANSFER.md`.
- `ported_models/dncnn/` — earlier port; its experiments log seeded the catalog. Key lesson: int8 =
  marshalling-bound; cache-op batching landed −24%.
- `ported_models/ggonnx/` — ONNX runtime port (context only).
- `.claude/skills/` — smolvlm2 + yolo workflow skills (build-verify / board-gate / ab-score / profile /
  seam-check). `.claude/agents/` — kernel reviewer.
- `local-artifacts/` — all working files: board build/run scripts, A/B logs, debate raw output, staged
  patches + verdict docs (`routeb-variants/`), models, HPM harness, venv (`torch-venv`).
- `docs/` — hackathon-level guides: `ET_SOC1_QUICKSTART.md`, `SUBMISSION_GUIDE.md`, `BOARD_ACCESS.md`,
  `HF_REFERENCES.md`, opinionated porting playbooks (afonso/martin).
- Leaderboard (README): our SmolVLM2 row is the CI baseline 10.82B — the local −21.9% (8.49B) is
  board-verified locally but not yet submitted through the trusted CI gate.

## 8. Recommended next session

1. `/smolvlm2` to orient; confirm submodule HEAD = `fb0bb4f02` and the tree is clean.
2. **S3: deeper act-load pipeline (4 act regs)** in `dot_q8_0_row8_n8` — 9 f-regs free, same
   proven mechanism as S2b. Board vs 6,170,682,271.
3. **E1: double-buffer `mul_mat_f32_matrix_engine.c`** (~5% of the new total, w/e=2.52) —
   recon shapes, SCP-budget the double-buffer (TenB shadow?).
4. Small fry to bundle on any board run: `norm_nr.patch`, `softmax_poly.patch`
   (`local-artifacts/routeb-variants/`).
5. Log every outcome in `EXPERIMENTS.md` (LANDED/WASH/BLOCKED + WHY); a recorded WASH is never retried.
