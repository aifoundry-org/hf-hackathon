# 08 — SmolVLM2 State Handoff + Open TFMA Investigation

**Purpose:** a clean, neutral, self-contained snapshot for a FRESH agent (and the human) — because the
coordinating context accumulated errors (notably a wrong ".bss" diagnosis). Trust this doc + verify against
source/board; do NOT inherit prior conclusions as settled. Everything here is dated 2026-07-18.

Repo: `/home/marin/Projects/hakatonOpenhwAiFoundery`. Kernels (the ONLY modifiable surface):
`ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et/et-kernels/src/` (a git submodule, branch
`smolvlm2-cont-vectorize`). Board: `ivan@aifoundry2` (Tailscale, password; the HUMAN runs board scripts, an
agent cannot ssh). Score = `device_cmd_exec_dur` firmware cycles on `video_dog` (gen=3 tokens). Gate: exact
lowercase answer (" dog"), PPL ≤ 26.739, ET-vs-CPU PPL ≤ 1%, zero vision CPU fallbacks, beat paired-main ≥ 1%.

## 1. Scoreboard (board-verified, committed)
| milestone | commit | score (pmc_cycles) | delta | quality |
|---|---|---|---|---|
| M0 baseline | `cc4049d` | 10,817,973,417 | — | pass |
| cont_f32 vectorize | `fb8cc4b` | 9,136,533,962 | −15.5% | pass |
| **scalar_hoist (CURRENT)** | **`5413c722e`** | **8,489,934,819** | **−7.1%** (−21.9% cum) | pass, PPL 22.2815 |

`mul_mat_Q8_0` is STILL **89.7%** of exec cycles (7.66B), **compute-bound** (wait/exec ratio w/e=1.86). It is the
only kernel worth attacking. It is NOT scalar-per-output — it is already 8-lane vectorized
(`fgb.ps`→`fcvt.ps.pw`→`fmadd.ps`, `block_ops.h`); the int8→f32 dequant happens FREE inside the MAC.

## 2. What is DEAD (with board evidence — do not blindly re-run these exact forms)
- **int8 opcode-3 TFMA, "M2g" per-tile form:** +8% board regression (re-quantized each activation block 192×
  redundantly). **Caveat:** this was ONE specific (bad) structure — see §4 for the untested clean form.
- **int8 barrier variant:** crashed; a non-coherency barrier-release bug was fixed, then it hit `event 65535`.
  Earlier "correct at 1024 harts" claims were STALE benchmark files — likely never validated. Abandoned.
- **Route B fp32 engine, "v1e_base" (per-hart stack stream, 0 .bss):** LOADED + produced the CORRECT answer on
  board (" Dog.", 2.71s, 0 stream errors) → the engine WORKS and is NOT .bss-blocked. BUT: (a) **parity, not
  faster** — 2.71s ≈ 2.73s for the ORIGINAL scalar (engine gained nothing, and loses to scalar_hoist which is
  −7% vs original); (b) its **profiled scoring run crashed `event 65535`** → no cycle score. See §4/§5.

## 3. The .bss facts (correcting the coordinator's error)
- `event 65535` was WRONGLY attributed solely to ".bss too big." TWO distinct triggers exist:
  1. **Large STATIC `.bss` at BACKEND INIT** (`ExecutionContextCache`→`doMemcpyHostToDevice`): v1c (12 MB),
     v1d (384 KB) static weight arrays all crashed here. REAL — keep static `.bss` ~0.
  2. **PROFILED run + heavy tensor-engine kernel, even at `bss=0`:** v1e_base (0 static .bss, per-hart stack)
     crashed `event 65535` in the *profiled* (`GGML_ET_PROFILE`) run while its *non-profiled* correctness run
     succeeded. This is UNEXPLAINED and is the real scoring blocker for any engine kernel. NOT a .bss issue.
- The debate (doc 07) notes `event 65535` at load is "NOLOAD .bss in the load image" — so a per-hart STACK or
  runtime-DRAM buffer is invisible to the loader. Static arrays are the only .bss-load risk.

## 4. What we have NOT run (the open frontier — this is the real answer to "why can't the engine work")
We proved TWO engine forms don't win. We did NOT try these, and we lack the diagnostic to explain WHY:
1. **fp16 opcode-1 tensor path — NEVER TRIED.** Dequant Q8_0→f16 (half the SCP footprint of f32, ~2× tile
   throughput), acts→f16, f16×f16→f32 accumulate. Debate held it as last-resort; never attempted.
2. **int8 opcode-3 with the TENC hardware accumulator done CLEANLY.** KEY: research found opcode-3 (int8) has a
   separate `TENC` hardware accumulator + `tenc2rf` copy-to-FREG (`sw-sysemu/insns/tensors.cpp:1464-1465,
   1549-1551`) — this AVOIDS the f-reg-carry problem that forces fp32/fp16 to save/restore. The int8 "wash" was
   the redundant M2g quant + the crashed barrier, NOT a clean TENC-accumulate + double-buffered form. Untested.
3. **Board PMC baseline on the isolated hot tile** (retired-inst/MAC, L2_EVICT_REQ/MAC, TFMA_WAIT%,
   TIMA_OPS/TL_OPS). The debate gated EVERYTHING on this; the standalone HPM harness stream-errored and it was
   never obtained. This single measurement would classify the bottleneck definitively (issue-bound vs
   evict-bound vs orchestration) and is the honest prerequisite before more engine or scalar work.
4. **N-register-blocking (scalar, N=4)** — the debate's designated next SCALAR lever: gather+convert each weight
   K-block ONCE, `fmadd` against N=4 activation columns → amortize the gather+convert issue stream N-fold.
   Compounds with the hoist (potential further ~1.5-2×). NOT built. (An agent "WASH"ed this but analyzed the
   WRONG kernel — upstream `origin/et` `c1188785e`, K-quant/uberkernel — and the wrong shape m=768 not fc1
   m=3072; its verdict is INVALID for our kernel.)
5. **Root-cause of `event 65535` in the PROFILED engine run.** Never diagnosed. The non-profiled engine run
   works. Is the profiled run scoreable (fewer profiler events? a profiler flag? a uint16 EventId overflow from
   engine ops? — research said only ~18,606 host commands, under 65535, so it's likely NOT a simple count
   overflow — needs real investigation)? If solvable, we could actually SCORE an engine kernel.
6. **Stock `mul_mat_f32_matrix_engine.c` double-buffering (2 load IDs)** — the M1 lever, never done (2.72% kernel
   + the technique).
7. **v1e_dbuf (Route B double-buffered)** — staged patch, never board-run.
8. **Route B memory-accumulate variant** (per-K-block `tensor_store` partials + scalar-add, instead of FREG
   save/restore; uses opcode-0 first_pass=1) — never built. Avoids the save/restore but adds a store→read.
9. **Rigorous interleaved paired-main A/B** (`smolvlm2-ab-score` skill) — to make the committed wins official;
   only the candidate-vs-committed runner was used.
10. **softmax_poly board A/B** (deg-6 exp2 vs fexp; staged, lossless) — tiny stakes, never run.

## 5. Hardware constraints (verify in `~/et-src/et-platform/sw-sysemu/insns/tensors.cpp` + docs 03/07)
- **f-reg carry:** fp32 (opcode-0) & fp16 (opcode-1) FMA accumulate C in f0-f31 across K-blocks → any float
  work (dequant) between blocks clobbers it → forces FREG save/restore. ONLY int8 opcode-3 has a HW accumulator
  (TENC) that sidesteps this. (This asymmetry is why fp32 Route B pays a tax the scalar path doesn't.)
- **SCP = 48 lines × 64B.** Only ONE C-accumulator tile is holdable (A16+B16+C16). Multi-accumulator C spills to
  tenc_loc=1 → per-FMA DRAM read-modify-write (~384× C traffic) — do NOT strip-mine C.
- **Static `.bss` ~0** (large arrays crash load — §3). Per-hart STACK (min 4 KB/thread) & runtime DRAM are fine.
- Non-coherent across shires; 2 harts/minion share L1D; only EVEN harts are tensor-capable.
- Forbidden (build-checker fails): `fdiv.s`,`fsqrt.s`,`frsq.ps`,`fsin.ps`,long↔float `fcvt`. Persistent inline-asm
  constants across separate asm blocks must be callee-saved `fsN` (f8/f9/f18-f27), never caller-saved `ftN`.
- Weights are FROZEN Q8_0 (no re-quant/GPTQ/AWQ). dst reads that need cross-shire must publish (flq2/fsq2/amoswapg).

## 6. Artifact map
- Committed wins: submodule branch `smolvlm2-cont-vectorize` (`fb8cc4b` cont, `5413c722e` hoist).
- Staged patches: `local-artifacts/routeb-variants/*.patch` (scalar_hoist, v1e_base, v1e_dbuf, norm_nr,
  softmax_poly) + `MORNING_REPORT.md`, `RESEARCH_TFMA.md`, `v1e_sib_DESIGN.md`, `OVERNIGHT_PLAN.md`.
- Expert debate: `docs/smolvlm2/07_EXPERT_DEBATE.md` (+ `local-artifacts/smolvlm2_debate/`).
- Board runner (HUMAN runs): `bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh`.
- Local sys-emu gate: `ET_TOOLCHAIN=~/et-src/et ET_PLATFORM=~/et-src/et` + `test-backend-ops`.

---
## 7. FRESH-AGENT PROMPT — "Is the tensor engine (TFMA) genuinely a dead end, or did we do it wrong?"
Copy this to a fresh general-purpose agent (it should NOT trust the coordinator's conclusions):

```
Read-only-first investigation. Read docs/smolvlm2/08_STATE_HANDOFF.md, 07_EXPERT_DEBATE.md,
local-artifacts/routeb-variants/RESEARCH_TFMA.md, the kernels mul_mat_Q8_0.c + mul_mat_f32_matrix_engine.c, and
the emulator ground truth ~/et-src/et-platform/sw-sysemu/insns/tensors.cpp + packed_*.cpp. Assume NOTHING is
settled — two engine forms (int8 M2g, fp32 Route B v1e) failed, but fp16 opcode-1 and a clean int8-TENC form
were never tried, and we never got a board PMC. Answer, with file:line evidence and first-principles cycle math:

1. WHY is fp32 Route B parity with scalar? Quantify: scalar does int8->f32 dequant FREE inside fmadd
   (fcvt.ps.pw); the engine needs a separate dequant + FREG save/restore per K-block. Count the ops each way
   for one fc1 output tile (M=3072,N=1024,K=768). Is the engine's per-tile MAC throughput advantage real, and
   does the dequant+save/restore tax genuinely erase it, or was v1e's IMPLEMENTATION suboptimal (e.g. no
   double-buffer, save/restore too heavy)?
2. Does the int8 opcode-3 TENC hardware accumulator (tensors.cpp:1464-1551) let us dequant/feed WITHOUT the
   f-reg-carry tax that sank fp32? Design the cleanest int8-TENC + double-buffered kernel. Is it plausibly
   FASTER than the 7.66B scalar_hoist? (The old int8 wash was redundant-quant + a crashed barrier, NOT this.)
3. fp16 opcode-1: dequant Q8_0->f16 (half SCP, 2x tiles), acts->f16, f16xf16->f32. Trace the SCP budget +
   accuracy (PPL gate). Is it worth building?
4. ROOT-CAUSE event 65535 in the PROFILED engine run (v1e_base, bss=0, non-profiled run WORKED). Read the
   EventManager/profiler path (~/et-src/et-platform/esperanto-tools-libs/src/). Is the profiled engine run
   scoreable at all, or is scoring structurally impossible for a heavy-engine kernel? If we can't SCORE it, the
   engine is moot regardless of speed — resolve this FIRST.
5. VERDICT: is there ANY engine form with a credible path to beat 7.66B AND be scoreable? If yes, name the ONE
   to build + why. If no, give the DEFINITIVE reason (not a hand-wave) so we stop revisiting it.
Deliver to local-artifacts/routeb-variants/TFMA_VERDICT.md. No code changes, no commits, no board scripts.
```

## 8. TESTING PROMPT / commands (for the HUMAN to run board candidates)
Each candidate is a patch; apply ONE on the clean base, run, check freshness. Setup:
```bash
cd /home/marin/Projects/hakatonOpenhwAiFoundery
SUB=$PWD/ported_models/llama_cpp_et/src/llama.cpp-et; KD=ggml/src/ggml-et/et-kernels/src
P=$PWD/local-artifacts/routeb-variants
```
Run a candidate (replace <NAME>: scalar_hoist is already COMMITTED=HEAD so skip; test v1e_base / v1e_dbuf /
norm_nr / softmax_poly, or a NEW patch a fresh agent produces):
```bash
git -C "$SUB" checkout HEAD -- "$KD/"            # clean base = committed scalar_hoist
git -C "$SUB" apply "$P/<NAME>.patch"
git -C "$SUB" diff --stat -- "$KD/"             # sanity
bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh
```
Check the result (ALWAYS — stale-file trap is real):
```bash
OUT=$PWD/.ci-work/smolvlm2-cont-candidate-output
date '+now %H:%M'
python3 -c "import json;s=json.load(open('$OUT/score-smolvlm2_500m_video.json'));[print(f'  {k}: {s.get(k)}') for k in ('scored_at','passed','perplexity','pmc_cycles','vision_fallback_ops','note')]"
# scored_at MUST be within minutes (else the run crashed & left a stale score). pmc_cycles < 8,489,934,819 = win.
# If the [5/5] parser crashes on empty kernel_id.json, the run CRASHED — diagnose the board server log, don't trust the score.
```
Restore to the committed winner after testing: `git -C "$SUB" checkout HEAD -- "$KD/"`.
