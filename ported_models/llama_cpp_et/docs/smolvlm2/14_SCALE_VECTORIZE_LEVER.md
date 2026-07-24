# SmolVLM2 — Base-2 / Codegen review + the Q8_0 scale-vectorize lever

**Date:** 2026-07-19. **Audience:** a fresh agent picking up the optimization track.
**One-line:** the register-blocked MAC in `mul_mat_Q8_0` is at its floor, but the **fp16→f32 block-scale unpack** feeding it is on a slow scalar path when a **hardware vector convert already exists in the tree** — that is the one genuinely untried, contained lever on the 74%-of-board kernel. Everything else from this angle is already done or sub-1%.

Gate everything below the usual way (see §6): local `smolvlm2-build-verify` (max_abs) → `smolvlm2-kernel-reviewer` → board A/B (`smolvlm2-ab-score`, ≥1% interleaved, PPL ≤ 26.739 held). Board runs are scarce and human-in-the-loop — **one at a time** (see §6 gotchas).

---

## 1. LEVER A — vectorize the Q8_0 fp16 scale unpack with `fcvt.ps.f16` (PRIMARY)

### The problem
`mul_mat_Q8_0.c` converts every block's fp16 scale to f32 with a **scalar, branch-laden C function** in a pre-pass, at **5 call sites**:

```
mul_mat_Q8_0.c:87    scales[kb]  = fp16_to_fp32(q_row[kb].d);   // n8 / row8 path
mul_mat_Q8_0.c:171   scales[kb]  = fp16_to_fp32(q_row[kb].d);   // (another dot path)
mul_mat_Q8_0.c:318   scales[kb]  = fp16_to_fp32(q_row[kb].d);   // (another dot path)
mul_mat_Q8_0.c:865   scales0[kb] = fp16_to_fp32(q_row0[kb].d);  // M2xN4 tile path (the SCORED path)
mul_mat_Q8_0.c:866   scales1[kb] = fp16_to_fp32(q_row1[kb].d);  // M2xN4 tile path
```

`fp16_to_fp32` (in `math_fp.h`) is a manual bit-twiddle: it branches on exponent (zero/subnormal vs inf/nan vs normal) and, for subnormals, runs a **data-dependent `while`-loop** to normalize. That is ~8–10 instructions + branches per scale, executed `M × K_blocks` times per tile (see §2 for how often the tile itself repeats). On a kernel the profiler calls **compute-bound (w/e=2.75 — "cycles are in the GEMM math", exec not dispatch)**, this scalar prep is pure exec overhead.

### The hardware already does this in one instruction
`block_ops.h:100-104` (the f16 dot) already uses the HW vector fp16→f32 convert:

```asm
fgh.ps      f11, f31(%[a_p])   ; gather 8 fp16 half-words at base + f31-offset-pattern
fcvt.ps.f16 f11, f11           ; convert 8 fp16 -> 8 f32 in ONE vector instruction
fmadd.ps    f10, f11, f12, f10
```

`fcvt.ps.f16` converts 8 packed fp16 → 8 f32. It is correct for all fp16 inputs (including the subnormal/inf/nan the scalar path branches for), so it is **numerically equivalent-or-better** for Q8_0 scales — which are always *normal* fp16 (block δ = max|q|/127, never subnormal/inf/nan). Replacing the scalar loop with a HW convert is therefore expected **bit-neutral** → PPL-safe.

### Implementation
Replace each `for (kb) scales[kb] = fp16_to_fp32(q_row[kb].d);` loop with a gather+convert that produces 8 f32 scales per iteration.

**The one wrinkle — stride.** `block_q8_0` is `{ uint16_t d; int8_t qs[32]; }` = **34 bytes**, `d` at offset 0 (`quants.h:52-55`). So consecutive block scales are **34 bytes apart**, not contiguous. `fgh.ps` gathers half-words at `base + offset[i]`, where `offset[8]` is a pattern held in an f-register (see `block_ops.h:32-34,84` — the contiguous-fp16 pattern is `{0,2,4,6,8,10,12,14}`; there is a gather-config CSR at `0x7d3`, `block_ops.h:18`).

- **Option 1 (preferred): strided gather.** Set the offset pattern to `{0,34,68,102,136,170,204,238}` (8 blocks × 34 B) and `fgh.ps` directly from `&q_row[kb].d`, then `fcvt.ps.f16`, storing 8 f32 into `scales[kb..kb+7]`. Advance base by `8*34`. **First verify** `fgh.ps` + the `0x7d3` gather-config accept offsets up to 238 (the existing uses are ≤14). If yes, this is zero-copy.
- **Option 2 (fallback if the stride/offset doesn't fit): pack-then-convert.** Scalar-copy 8 `d` half-words into a contiguous `uint16_t dtmp[8]` (a plain load+store each — no branch, no loop, far cheaper than `fp16_to_fp32`), then `fgh.ps` with the contiguous `{0,2,…,14}` pattern + `fcvt.ps.f16`. Still replaces the branchy convert with a copy + one HW convert per 8.

Handle the `K_blocks % 8 != 0` tail with a scalar remainder (either the HW convert on a padded temp, or the existing `fp16_to_fp32` for the last <8). K_blocks for SigLIP fc1/fc2 = 768/32 = 24 and 3072/32 = 96 — both multiples of 8, so the common path is clean.

Do all 5 call sites, but the **M2xN4 path (865-866) is the scored one** (N≥1024 SigLIP GEMMs) — prioritize it; the n8/scalar paths (87/171/318) serve the unscored N=128 LLM shapes.

### Payoff / risk
- **Payoff:** unmeasured but mechanically sound — replaces ~8–10 branchy scalar instructions/scale with ~2 vector instructions/8 scales on the 74% compute-bound kernel. Whether it clears the ≥1% board gate depends on what fraction of exec the scale prep is (not separately profiled yet — a `GGML_ET_PROFILE` split of prep vs MAC would de-risk it before building).
- **Risk:** low. Expected bit-neutral (HW convert vs correct manual convert on normal fp16). Local `max_abs` must confirm 0 before the board. No layout/seam change in Option 1/2, so no seam gate needed for A alone.

---

## 2. LEVER B — kill the per-n-group scale-recompute redundancy (SECONDARY, structural)

The scale pre-pass sits **inside** the tile loop (`mul_mat_Q8_0.c:840`), where `tile` enumerates `(ng, mb)` = (n-group, m-block):

```c
for (int64_t tile = hart_id; tile < total_tiles; tile += stride_m) {
    ng = tile / m_blocks;  mb = tile - ng*m_blocks;  n0 = ng<<3;
    ... float scales0[..], scales1[..];
    for (mp<8) for (kb<K_blocks) scales0/1[kb] = fp16_to_fp32(...);   // <-- redone every tile
    ... dot_q8_0_m2n4(...)
}
```

The converted scales depend only on the **weight m-block `mb`**, not on the activation n-group `ng`. But the loop reconverts them for **every `(ng, mb)` tile** → the same weight scales are unpacked **N/8 times**. For fc1 `[1024×768]·[768×3072]`: N/8 = **384× redundant**.

**Fix:** restructure so a hart converts an `mb`'s scales once and reuses them across all its n-groups (e.g. mb-outer / ng-inner per hart, or a per-hart scale cache keyed by `mb`). This **changes the hart→tile work distribution**, so it is **seam-sensitive** → requires the multi-hart board seam gate (`smolvlm2-board-gate`), not just an A/B.

Do Lever A first (it makes each conversion cheap and is contained). Only pursue B if A lands and the profile still shows scale prep as a meaningful slice — B's win is bounded by how much of exec the prep is, same open question as A. If A already vectorizes the prep to near-free, B may not be worth the seam risk.

---

## 3. What's ALREADY optimal from this angle (do not touch)
- **Base-2 strength reduction in the hot MAC glue is done:** `kb<<5` (×32 block), `mb<<4`, `<<6`, `n0=ng<<3`, and unsigned `%8`→`&7`. No div/mul-by-constant to convert in the hot path.
- **`et_sinf` reciprocals baked to literals** (`math_fp.h`: `0.16666666…` etc. instead of `et_fdiv(1.0f,6.0f)`) — the hygienic reciprocal-hoist, already applied this session.
- The MAC itself is hand-written asm (`dot_q8_0_m2n4`, `dot_q8_0_row8_n8`) — the compiler doesn't generate it; register-blocking (N=4/N=8/M2xN4) is landed and exhausted (see `EXPERIMENTS.md`).

## 4. Minor notes (sub-1%, not worth a board slot)
- Signed `int64` `kb/TILE`, `%TILE`, `(kb/TILE)&1` in `mul_mat_f32_matrix_engine.c` emit sign-fixup shifts; unsigned or explicit `>>4`/`&15` would drop the fixup. But it's the index math of a 5.3% kernel — negligible.
- `im2col_f32.c:71-79` has genuine **runtime-divisor** integer divides (`/patch`, `/ow`, `/kw`, `/(kh*kw)`) the compiler can't strength-reduce; for the stride-16 non-overlapping patch embed the whole im2col is a reshape (see doc 11 §3/T4), but it's <0.5% of board — opportunistic only.

## 5. Why this reframes "the track is at its floor"
Docs 10–13 concluded the track is floored, and for the **MAC** that's correct (register-blocking spent, tensor engine closed 3×, softmax/norm micro-opts washed/sub-gate). But the review that produced doc 13 looked at *algorithms and the MAC*; it did **not** look at the **scale-prep codegen**. Lever A is on the same 74% kernel, targets exec cycles (what a compute-bound kernel is bound by), uses a HW op the codebase already ships, and is contained + likely bit-neutral. It is a better bet than the sub-1% PPL-gated hygiene batch (norm_nr, reciprocal hoists) because it targets the kernel that actually dominates. **Recommended order for the next session: (1) profile prep-vs-MAC split to size it, (2) Lever A, (3) Lever B only if A lands and prep still shows, (4) else formalize + submit.**

## 6. How to gate + board gotchas (hard-won)
- **Local math gate:** `smolvlm2-build-verify` — `ET_PLATFORM=~/et-src/et`, rebuild the kernel ELF, `test-backend-ops -b ET -o MUL_MAT -p type_a=q8_0,type_b=f32` (or the f32 filter for the engine kernel), require PASS / max_abs=0. Filter or it times out under sys-emu.
- **Board A/B:** `bash local-artifacts/run_when_free.sh local-artifacts/run_smolvlm2_cont_candidate_board.sh` builds from the working tree and scores. It is **candidate-only** (vs a hardcoded M0 baseline) — for keep/drop, measure candidate and main separately, or use `smolvlm2-ab-score` (interleaved main→candidate→main).
  - **ONE board run at a time.** Stacking retries this session cost ~1h: 4 chains fought over the single `/dev/et0_ops` → `Device or resource busy`, server crashes, `float(None)` scorer aborts. If a run fails, diagnose before re-firing: `ssh ivan@aifoundry2 'fuser -v /dev/et0_ops; pgrep -af run_smolvlm2_video_benchmark'`.
  - **Validate a run is real:** fresh `scored_at`, no `busy` / `Remote end closed` / `rc=-6` / `missing score` lines. Single-run noise floor ≈ 0.15% (two no-change mains differed by 0.14%) — treat any delta <~0.5% as noise; require the ≥1% gate.
- **WASH discipline:** anything that doesn't clear ≥1% is reverted and logged in `EXPERIMENTS.md` with WHY, never retried.
- **No commit/push without explicit user OK; no Co-Authored-By trailer.**

## 7. Key file references
- `ggml/src/ggml-et/et-kernels/src/mul_mat_Q8_0.c` — scale call sites 87/171/318/865/866; M2xN4 tile path ~840-905; block-dot asm ~95-150 (`dot_q8_0_m2n4` etc.).
- `ggml/src/ggml-et/et-kernels/src/block_ops.h:100-104` — the copy-ready `fgh.ps` + `fcvt.ps.f16` pattern; `:18` gather-config CSR `0x7d3`; `:32-34,84` offset-pattern setup.
- `ggml/src/ggml-et/et-kernels/src/math_fp.h` — `fp16_to_fp32` (the branchy scalar convert to replace); `fp32_to_fp16` uses `fcvt.f16.ps` (the forward HW op, for reference).
- `ggml/src/ggml-et/et-kernels/src/quants.h:52-55` — `block_q8_0` layout (d@0, 34-byte stride).
- `docs/smolvlm2/EXPERIMENTS.md` — ledger (what's landed/washed). `docs/smolvlm2/13_HANDOFF_2026-07-19.md` — the session handoff this doc extends.
