# FOSDEM 2026 "Zero to matmul with the ET-SoC-1" — distilled notes + what we can use

**Source:** FOSDEM 2026, AI Plumbers devroom, "Zero to matmul with the ET-SoC-1" (slide deck, 54 slides with speaker notes). PDF fetched 2026-07-19.
**Why it matters:** it is a full, worked fp32 matmul optimization ladder on *our* silicon (A0), by someone with the hardware manual, ending at the tensor engine at ~theoretical peak. Several of our standing assumptions are confirmed; a couple are challenged.

---

## 1. Confirmed hardware facts (verbatim from slides)

- **Cores:** 1093 RISC-V cores (RV64IMFC + custom extensions); after special/yield/firmware carve-outs, **1024 "minion" cores**. **650 MHz** (fixed for reproducibility). **8 vector lanes/core (custom SIMD, NOT RVV)**. Every FMA = 2 fp32 ops → **10.6 TFLOP/s fp32 theoretical**.
- **Harts:** **2048 harts** (2-way hyperthreading, 2 harts/core; `mhartid` 0–2047). The RISC-V **frontend issues only 1 instruction/cycle** — the two harts share it, so you can't get 2 IPC from scalar+vector on the two harts.
- **Memory hierarchy (default SRAM partition):** L0I$ 256×1K · L1I$ 128×32K · **L1D$ 1024×4K or 2048×½K** · L2$ 32×½M · **L2Scp 32×2½M** · L3$ 1×32M · DRAM 32G. Per-hart register files 2048×1¼K (**GPRs 32×64b/hart, Vectors 32×256b/hart**). **Up to 2465 caches, NO coherency. 64-byte cache lines.**
- **Non-coherency is real and bites:** two harts writing 4-byte results into the same 64B line each flush the *whole* line to L2, so the second flush clobbers the first (exactly our `int8-cacheline-seam-race` and `etsoc1-noncoherent-stores` notes). Fix = choose the cache per instruction: **standard RISC-V load/store → L1D$; custom "L" instr → nearest L2$; custom "G" instr → address-based (gives coherency appearance, at latency/bandwidth cost).**
- **L2Scp** = SRAM used as **plain addressable memory at L2$ latency** (one SRAM pool backs L2$/L3$/L2Scp with configurable partitioning). Every core has a nearest L2Scp; any core can reach any L2Scp (higher latency).
- **Tensor units** (three green boxes, each issues 1 instr/cycle, no PC/I$ — driven by the frontend via CSR writes):
  - **Tensor Compute** — the matrix engine. ~10 ops; one is **fp32 matmul-accumulate, 16×16 shape**. Uses the **Vector Unit**, so it runs **in parallel with the RISC-V frontend/Scalar Unit**. Accumulates **C onto hart 0's vector registers (f0–f31)**.
  - **Tensor L1 Load** — **A-type loads → L1Scp (48×512b = the 48-line scratchpad)**; **B-type loads → direct path into Tensor Compute, for values "used exactly once"** (don't occupy L1Scp).
  - **Tensor L2 Load** — loads into L2Scp; **issuable by hart 1** (the sibling) to prefetch while hart 0 computes.
  - **One scalar CSR write enqueues up to 512 tensor instructions, or up to 1 KiB of loads, asynchronously.**
  - **Tensor Compute + Tensor L1 Load are hart-0-of-core only.** Hart 1 returns immediately (or prefetches via Tensor L2 Load).
- **A0 silicon quirk:** the inner loop carries a seemingly-useless `fsgnjx.s f3, f1, f2` purely as a **hardware-bug workaround** for A0. (Matches our `etsoc1-asm-persistent-freg` / workaround lore.)

---

## 2. The optimization ladder (512³ fp32, one number per lever)

| Step | Lever | Result |
|---|---|---|
| 0 | Naive triple loop, all in DRAM, 1 hart | **7 MFLOP/s** |
| 1 | Fix non-coherency (`store_l`, custom "L") + parallelize outer loop over `mhartid` (512 harts) | 3.4 GFLOP/s |
| 2 | Use all **2048 harts** | 13.1 GFLOP/s |
| 3 | **8-wide SIMD** (`fp32x8`, `BCAST` a / `LD8` b / `ST8` c) | 104 GFLOP/s |
| 4 | Read operands from **L2Scp** instead of DRAM (pre-position a,b) | 312 GFLOP/s |
| 5 | **Register-block 2×16** (4× work/iter) → FMA share 14.3%→32% | 1.64 TFLOP/s |
| 6 | **Register-block 4×32** → FMA share 62.8%, "other" 5.8% | 2.94 TFLOP/s |
| 7 | **Tensor engine** (16×16 fp32 matmul-accum), naive `load→wait→FMA→wait` | 7.05 TFLOP/s |
| 8 | **Software-pipeline load↔FMA** (peel first load + last FMA; issue FMA(k) before load(k+1) so they run in parallel; one fewer `wait`) | **10.25 TFLOP/s** ✅ (~few % of the 10.6 peak) |

The FMA-percentage framing (steps 5–6) is exactly the metric our `mul_mat_Q8_0` register-blocking (N=8 / M2×N4) optimizes. Step 8's code toggles the double-buffer via `fma_cmd ^ 0x100` and clears first-pass via `& ~1ull`, with loop body `load_b · wait_a · fma · load_a` (FMA and next A-load issued together).

---

## 3. Cross-check against OUR kernels

- **`mul_mat_f32_matrix_engine.c` IS step 8.** Our `f2714ffe1` double-buffer + FMA-before-next-load reorder is precisely the 7.05→10.25 TFLOP/s technique (the `^0x100` buffer toggle = our `(kb/TILE)&1 ? 2*TILE : TILE`; first-pass clear = our `kb==0`). Confirmation the kernel is on the right design.
- **`mul_mat_Q8_0.c` is steps 3–6** (8-wide SIMD + register-blocking to raise FMA%). The talk validates that ladder as the correct scalar-SIMD path.
- **The `fsgnjx` A0 workaround, non-coherent 64B flush seam, and hart-0-only tensor ops** all match our existing memory notes.

## 4. What we can actually USE (new or reinforced levers)

1. ~~**B is a "used-once" operand → the B-type direct-to-compute path, which does NOT occupy L1Scp.** Our kernel currently double-buffers B in *general* SCP lines 16–47. If B streams via the B-type/TenB direct path instead (as the talk's design intends), **the 48 general lines free up for double-buffering A** — the exact "double-buffer A" lever from doc 10 Tier B, now independently confirmed as the hardware's intended usage. This is the strongest actionable item. (Ties to the HW-mining agent's TenB-bank finding.)~~
   **DISPROVEN 2026-07-19 (see EXPERIMENTS.md "FOSDEM lever E2").** The TenB direct path is PLAIN-load-only (`tensor_load_execute` forces `cmd=tload_cmd_load` when tenb set, tensors.cpp:484) — it cannot transpose. But our FMA-B operand `B[k][j]=src0[m0+j][k]` is a column of the row-major weight tile = a mandatory transpose (GGML mul_mat is the inner-product form `C=A1·A0ᵀ`; the talk's standard `C=A·B` has B naturally `[k][j]`, which is why B could go direct *there*). So B cannot leave SCP, and A cannot be double-buffered. Even A×2/B×1 only exposes the *expensive* transpose load instead of the cheap plain A load — the shipped assignment is already optimal. NOT actionable.
2. **Fewer waits via the peel.** The talk's step-8 loop has the FMA and next load issued together with **one** `wait` (wait_a). Audit our loop (`wait LOAD_1 · wait LOAD_0 · fma · prefetch B · wait FMA · prefetch A`) — if it carries an extra wait vs the peeled form, removing it is a direct latency win.
3. **L2Scp pre-positioning was a 3× step (104→312 GFLOP/s).** Confirm the SigLIP weights/activations are staged in L2Scp, not streamed from DRAM, for the engine path. Data placement, possibly backend-side.
4. **Hart-1 sibling prefetch via Tensor L2 Load** runs in parallel with hart-0 compute — a genuine extra execution resource we don't use (matches the HW-mining agent's finding).
   **ASSESSED 2026-07-19, NOT PURSUED (see EXPERIMENTS.md "FOSDEM lever E4").** Physically possible (et_tensor_load_l2scp CSR 0x85f; the idle odd harts are free) but board-only (sys-emu models zero cache cost, so it can't be disproven or proven locally), and poor risk/reward: the engine kernel is 5.2% of exec and already compute-bound (w/e=1.23, E1 hides most latency) → ~1% ceiling; it needs explicit L2Scp address management + sibling coordination + cross-hart non-coherency handling (the cont_f32/int8 footgun class) → highest complexity of any remaining lever; and it shares the mechanism of the S5 L2-staging WASH. The only physically-possible remaining lever, but not worth the slot over shipping.
5. **Scalar/tensor concurrency is free:** Tensor Compute uses the Vector Unit while the frontend runs scalar address/loop code on the Scalar Unit. Our per-tile index math (including the integer divides) can overlap tensor FMAs at no cost.
6. ~~**One CSR write enqueues up to 512 tensor instructions.** Our kernel issues one FMA per K-tile (K/16 = 48–192 iterations); batching the enqueue could cut frontend issue overhead.~~
   **DEAD 2026-07-19, already exploited (see EXPERIMENTS.md "FOSDEM lever E3").** That "512 instr / 1 KiB per CSR write" *is* the granularity we already issue at: one `tensor_fma` CSR write runs the full 16×16×16 tile (4096 MACs, emulator FMA loop `for k·for i·for j`), one `tensor_load` moves up to 16 lines = 1 KiB. The K/16 FMAs are the minimum count; the per-K-tile `tensor_wait`s are structural RAW (TenC accumulator f0–f31) + WAR (single A buffer), not un-batched issue. Nothing to batch.

## 5. The assumption this CHALLENGES (worth revisiting)

Our ledger records "route Q8_0 onto the tensor engine = measured dead 3×," with a root cause of a **~8-hart cap from the `.bss` execution-context load wall** (per-hart SCP staging scratch blew past ~12 MB). **But the talk runs the fp32 tensor engine on hart-0-of-every-core = ~1024 tensor harts and reaches 10.25 TFLOP/s** — because it stages operands in **L2Scp / the tensor L1Scp**, never in large `.bss` arrays. That suggests the 8-hart cap was an artifact of *how our int8 port staged data* (big `.bss`), not a fundamental limit of the engine. **This does not revive int8-on-engine** (the dtype-marshalling loss and the free-dequant-in-`fmadd` advantage still stand), but it does mean: *if* an engine path ever staged via L2Scp instead of `.bss`, the hart-count asymmetry that killed the three prior attempts might not apply. Low-confidence, but the ledger's "engine caps at 8 harts" should be re-read as "our int8 `.bss` staging capped at 8 harts," which is a narrower claim.

---

## 6. Net
The talk confirms our f32 engine kernel implements the peak technique, validates the Q8_0 register-blocking ladder, and hands us four reinforced levers (double-buffer A via the B-direct path, fewer waits, L2Scp staging, hart-1 prefetch) plus one assumption to re-examine (the `.bss`-vs-engine hart cap). It also open-sources the manuals + emulator (AIFoundry GitHub) and promises RTL "soon" — worth watching for the tensor-engine RTL.
