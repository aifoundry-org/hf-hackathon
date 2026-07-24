# ET-SoC1 "erbium" — hardware notes for the SmolVLM2 kernel work

Distilled from `local-artifacts/docs/ETSOC1_ARCH_REFERENCE.md` (emulator/ISA ground truth),
`docs/opinionated_porting_options/{martin,afonso}.md`, and our DnCNN/YOLO auto-memories. Every kernel
change must respect these. When a doc string and the emulator disagree, **the emulator wins**
(`~/et-src/et-platform/sw-sysemu/insns/tensors.cpp`).

## Topology (what the hackathon actually runs)
- Links the `erbium-soc1sim` target = **ET-SoC1 personality but ONE shire**: 8 minions, 16 harts, and the
  kernels drive **8 active harts** (thread-0 of each minion; odd harts skipped so no L1D-sharing siblings run
  concurrently). Kernel loads at DRAM base `0x8005801000`.
- Full chip is 32+ shires × 32 minions × 2 SMT harts, but you only get one shire here. Multi-shire is
  **policy/scope-blocked** for these tracks.
- 2 harts per minion share one pipeline **and L1D**; hart 1 helps only when hart 0 stalls (packing, prefetch,
  scalar post-proc). If hart 0 is compute-bound, the sibling just contends.

## Tensor engine (TFMA) — the money path
- CSR-driven: `tensor_load`(0x83f) / `tensor_fma`(0x801) / `tensor_quant`(0x806) / `tensor_store`(0x87f),
  each followed by a blocking `tensor_wait`. **Two load IDs (0/1) run concurrently → this is your
  double-buffer.**
- FMA opcode (bits[3:1]): **0 = fp32**, **1 = fp16×fp16→fp32**, 2 = illegal, **3 = int8×int8→int32** (IMA8A32,
  with the fused `tensor_quant` int32→uint8 chain). 4–7 illegal (trap).
- Tile unit is **16×16×16**. Field encodings are off-by-one and differ fp32 vs int8 — see
  `int8-tfma-encodings` memory and ARCH_REFERENCE §2.3. `a_num_rows = N-1`; store `cols = bytes/16 - 1`.
- **`ua`/`ub` signedness wrapper bug**: the SDK arg names are inverted vs the emulator. Flag SET ⇒
  operand treated **unsigned** (zero-extended). To make weights signed, pass `tenb_unsigned=false`; to make
  activations unsigned, pass `tena_unsigned=true`. (bit22 gates B/activations, bit21 gates A/weights.)
- **FREG clobber**: `tensor_fma`/`tensor_quant` with the copy-to-FREG pass overwrite f0..f31. Put
  `__asm__("":::"memory","f0",...,"f31")` between any tensor op and following scalar/VPU float code, or the
  compiler-kept-live values silently corrupt.
- int8 **zero-skip** is at 32-bit (4-byte-lane) granularity, not per-byte; it saves MAC work on aligned-zero
  tiles but **not** the fixed per-instruction issue/sync overhead.

## Scratchpad & memory
- **L1 SCP = 48 lines × 64 B.** Budget A weights / B activations / bias / scale lines within 48. A TenB
  "shadow" region extends at SCP[k+48] (`tenb_loc=1`).
- Three tiers: DRAM (large/slow) → shire-local L2/SCP (cooperation zone) → L1/minion SCP+regs (hot set).
  The whole game is keeping operands in SCP and feeding the TFMA. But **whether the wall is DRAM bandwidth or
  per-tile orchestration is workload-dependent — measure** (yolo/dncnn were orchestration-bound; SmolVLM2's
  large GEMMs may differ).
- **No cache coherence to lean on.** Data movement + sync are explicit parts of the algorithm. `evict` =
  writeback+invalidate; `flush` = writeback-only.
- **Cache ops are async + batchable** (≤31 outstanding; one strided evict covers ≤16×64B lines;
  `WAIT_CACHEOPS` is a single drain). Per-tile `FENCE+evict+WAIT` over-syncs — batch across tiles.
- **No in-kernel DMA** (PCIe DMA is host-runtime-only). Latency hiding = double-buffered `tensor_load` +
  `prefetch_va`/`lock_va` residency.
- **Cacheline seam race** (multi-hart): a per-hart row band whose stride isn't a whole number of 64B lines
  false-shares the boundary line on write-back → silent board corruption, invisible at 1 hart / sys-emu.
  Fix: pad so `stride_bytes % 64 == 0`, single-writer per line, `_Static_assert` it.

## Forbidden / trapping instructions (hard hardware limits)
- **No `fdiv.s`, `fsqrt.s`, `fcvt.{l,lu}.s`, `fcvt.s.{l,lu}`** — they trap (exc 30), can segfault the launcher,
  and the failure looks exactly like a bad mem-size/region config. `fcvt.w.s` (32-bit) is fine. The full
  build-rejected set (from `et-kernels/scripts/check_unimplemented_instructions.sh`): `fdiv.s fsqrt.s
  fcvt.l.s fcvt.lu.s fcvt.s.l fcvt.s.lu fdiv.pi fdivu.pi fremu.pi frem.pi fdiv.ps fsqrt.ps frsq.ps fsin.ps`.
- **⚠ DIFFERENT REGIME FROM DnCNN/YOLO.** The standalone DnCNN/YOLO ELFs built with `-Ofast` (which HID a
  `fdiv` by rewriting `a/b`→`a*(1/b)`), so there `-Ofast` was load-bearing and `-fno-unsafe-math-optimizations`
  was fatal. **The `llama_cpp_et` kernels build at `-O3`** (`et-kernels/CMakeLists.txt`) **with a post-build
  checker that FAILS the build if any trapping instruction is present.** So here the trap is caught at COMPILE
  time, not silently hidden — a stray float `/` or `sqrtf` breaks the build, it doesn't reach the board.
- **Design divide-free** anyway: host-bake reciprocals/rsqrt and multiply on-device, or Newton-Raphson / VPU
  `frcp.ps`. The sanctioned in-kernel helpers are `math_fp.h`'s `et_fdiv` / `et_powf` (already used by
  `norm_f32.c` / `rms_norm_f32.c`); note `et_powf(x,-0.5)` for rsqrt is transcendental (exp/log) → a perf
  candidate to replace with NR, not a correctness issue. Long→float cast workaround: `a=(float)(int)(b)`.
- **R_RISCV_64 pointer-table trap**: a table of `.rodata` pointers segfaults the board ELF loader before the
  kernel runs. Store indices into flat pools, not pointers.
- VPU has `fexp.ps`/`frcp.ps` (base-2 exp / reciprocal) — use for softmax/GELU/SiLU instead of scalar libm.

## PMC — measure before optimizing
- 6 programmable HPM counters/hart, hazard-safe via `HPM_SAFE_READ`: `CYCLES=1`, `TL_OPS=15`,
  `DCACHE_MISSES0/1=8/9`, `L2_EVICT_REQ=12`, `TFMA_WAIT_TENB=18`, `TIMA_OPS=19`. Live example
  `gp-sdk/device/tests/check_pmc/check_pmc.cc`.
- **The emulator models ZERO cycle cost for cache/tensor ops** — all timing is **board-only**. sys-emu is for
  plumbing/correctness (max_abs), never perf.
- Also `ET_PERF` and `GGML_ET_PROFILE` at the ggml level rank ops/kernels by time and reveal CPU fallbacks.

## Iteration tooling
- **`GGML_ET_KERNELS_PATH`** loads rebuilt kernel ELFs at runtime → fast local iteration without rebuilding
  the host binary. CI/PR builds still come from committed source only.
- `test-backend-ops` gives per-op correctness + perf checks (Path-A GGML route).
- Board is ~2000–3400× faster than sys-emu for these kernels — **iterate correctness on sys-emu (max_abs=0),
  iterate perf on the board.**
