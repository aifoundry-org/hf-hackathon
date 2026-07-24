# SmolVLM2 — fresh-prompt runbook

How to drive this optimization from a cold session. The whole environment is designed so you can open a
fresh Claude Code prompt, type one slash command, and be oriented + working the right milestone with the
right guards — no re-reading the whole corpus each time.

## What exists (the environment)

| Piece | Where | Purpose |
|---|---|---|
| Research corpus | `ported_models/llama_cpp_et/docs/smolvlm2/00–05` | plan, profile, catalog, hardware, kernel audit, agent method |
| Orient command | `/smolvlm2` | prints the milestone map + current state; does no work |
| Milestone commands | `/smolvlm2-m0 … /smolvlm2-m5` | run one milestone end-to-end |
| Build+math gate | skill `smolvlm2-build-verify` | build one isolated kernel + 1-hart sys-emu `max_abs=0` vs CPU |
| Profiler | skill `smolvlm2-profile` | `ET_PERF` / `GGML_ET_PROFILE` inventory + cycle attribution |
| Board gate | skill `smolvlm2-board-gate` | human-in-the-loop board correctness + seam gate |
| A/B score | skill `smolvlm2-ab-score` | rigorous paired-main firmware-cycle A/B |
| Reviewer | agent `smolvlm2-kernel-reviewer` | adversarial pre-gate red-team |

## Fresh-prompt flow (do this every new session)

1. **Type `/smolvlm2`.** It reads the plan, checks `git log` + the docs for what's landed, and tells you the
   next milestone. It does not start work.
2. **Type the recommended `/smolvlm2-mN`.** The command spawns a fresh subagent (clean context) that reads
   the shared contract (`05_AGENT_ENVIRONMENT.md`) + this milestone's section of `00_RESEARCH_PLAN.md` and
   executes it — build → single-hart `max_abs=0` → reviewer → gate.
3. **When a board step is reached**, the milestone invokes `smolvlm2-board-gate` / `smolvlm2-ab-score`,
   which hand YOU an exact command to run and say precisely what to paste back. Board runs are interview
   turns — nothing is fabricated.

## The core loop (one lever at a time)

```
pick ONE lever (from 02_OPTIMIZATION_CATALOG top-10)
  └─ edit ONE kernel under et-kernels/src/  (isolate; don't touch the model)
     └─ smolvlm2-build-verify   → single-hart max_abs=0 vs CPU (test-backend-ops), sys-emu
        └─ smolvlm2-kernel-reviewer  → BLOCK/PROCEED on the diff
           └─ smolvlm2-board-gate     → ≥3 clean board runs (only if multi-hart / layout touched)
              └─ smolvlm2-ab-score    → ≥1% paired-main firmware-cycle win, PPL + exact answer held
                 └─ log LANDED / WASH / BLOCKED in docs/smolvlm2/EXPERIMENTS.md
```

If WASH: revert cleanly, record WHY, never retry that lever (see the dead levers in `yolo-m30-int8-reaim`).

## Using `/loop` for the iterate-until-green cycles

`/loop` re-runs a prompt on an interval or self-paced — use it for the two genuinely repetitive waits:

- **Local build/math iteration** (autonomous, no human): `/loop /smolvlm2-m1` self-pends the milestone until
  `max_abs=0` + reviewer PROCEED, editing → building → checking each pass. Let it self-pace (no interval).
- **Board-gate polling** (human-in-the-loop): DON'T loop a board run — board runs need your paste-back. Loop
  only the *local* pre-validation so everything is staged the instant you have the board. Stop the loop
  (`/loop` with `stop`) once green.

Do **not** `/loop` a sys-emu perf read — sys-emu has zero cycle cost; perf only comes from the board via
`smolvlm2-ab-score`.

## First-run bootstrap (once, before M1)

- Clear disk (currently ~100% full) — build needs headroom.
- Build sys-emu once: `cmake -B build -DGGML_ET=ON -DGGML_ET_SYSEMU=ON && cmake --build build -j`.
- Confirm `GGML_ET_KERNELS_PATH` runtime kernel-loading works (rebuild one `.elf`, point the var at it).
- Run `/smolvlm2-m0` to get the `ET_PERF` kernel/shape/flops inventory and resolve the F16/Q8_0/F32 dtype
  of the dominant SigLIP GEMM — this picks the M1 target.

## Board facts to fill in on first access

- Board host + user (YOLO used `ivan@aifoundry2`; confirm the SmolVLM2 board via Discord `#community-lab`).
- Benchmark command (from the reference JSON):
  `BENCHMARK_DEVICE=soc1sim BOARD_BENCHMARK=1 .github/ci/scripts/run_model_benchmark.sh smolvlm2_500m_video`.
- The trusted CI gate (main → candidate → main, ≥1% paired improvement) is the authoritative score; local
  A/B is a fast proxy.
