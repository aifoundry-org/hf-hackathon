# DnCNN int8-TFMA kernel — design & optimizations

Optimizations in `src/dncnn_gen_int8.c` for the shipped `dncnn` model (`DNCNN_REAL=1`, `CH=64`,
`HIDDEN=18`, `IMG=64`, `ACTIVE_HARTS=8`): 20 layers = FP32 `conv_first` (1→64) + 18 int8 hidden
(64→64) + FP32 `conv_final` (64→1), residual `out = x + net(x)`. Board-measured ~221 ms per 64×64
tile at 8 harts. Everything below describes this 64-channel build.

## The int8 hidden convs run on the tensor engine (TFMA)

Each hidden conv is a sequence of tiles. A tile is a `16 OC × 16 spatial` int32 accumulator held in
the 32 FREGs, contracted over packed int8 activations × int8 weights, then requantized to uint8
in-register. The dispatch is `tensor_fma` opcode 3 (`fma_group`): **A = weights (signed), B =
activations (unsigned)**. `first` resets the int32 TENC, `last` copies it into the FREGs.

Three hardware field limits shape the whole design:
- `arows` (output channels per pass) is 4-bit and 16 OC × 16 spatial int32 = exactly 32 FREGs ⇒
  **≤16 OC per accumulation pass** (`OC_TILE=16`).
- `acols` (contraction) is 4-bit ⇒ **≤16 IC-quartets = 64 contraction elements** per dispatch. At
  64 channels one 3×3 tap is `64/4 = 16` quartets, which exactly fills this field.
- `P` (spatial tile = FMA bcols) is 2-bit ⇒ **≤16**.

Primitives (parameterized by the config dims): `pack_b_group` (quartet-interleaved activation
gather), `fma_group` (the `tensor_fma` CSR write), `requant_u8_bias` (fused int32→uint8 requant),
`store_u8_tile` (packed uint8 store), `conv_tile` (per-tile driver), `rearrange_hidden_weights`
(once-per-layer repack of the weight blob into group-major A layout). Every layout constant is a
`_Static_assert`, so a wrong value fails the build, not the board.

## Optimizations

### OC-tiling (4 tiles)
The int32 accumulator is OC × spatial FREGs, and `arows` caps it at 16 OC. `conv_tile` therefore
tiles the 64 output channels through the accumulator in slices of `OC_TILE=16` — **4 tiles** per
spatial tile — each reloading its 16-OC weight slice and re-running the FMA groups. This is what lets
a 64-channel layer run at all inside the 32-FREG / 4-bit `arows` limit.

### int8 tap dispatch
A 3×3 hidden conv is 9 taps. The kernel packs and dispatches them as `NGRP=9` FMA groups — one per
tap — because at 64 channels a single tap already fills the 64-element contraction field, leaving no
room to fold taps together. (The source keeps a general tap-fold path, `FOLD_TAPS =
min(K, 16/(CH/QUARTET))`, so the same code compiles narrower-channel builds where several taps share
one dispatch; at 64 channels `FOLD_TAPS=1`.) Regrouping never changes the result: the int32 TENC
accumulation is exact.

### Word-copy activation pack
`pack_b_group` gathers activation windows into the quartet-interleaved B layout. An IC-quartet is 4
contiguous 4-byte-aligned bytes on both the source and packed sides, so it moves as one aligned
`uint32` instead of 4 byte copies — quartering the instruction count in the kernel's hottest scalar
loop (there are `CH/QUARTET = 16` quartet-lines per tap). This relies on the channel stride being a
compile-time constant (see "Lessons").

### Batched cacheline evict
`conv_tile` packs **all 9** groups' B into the per-hart pack region first, then issues a single
`FENCE; evict; WAIT_CACHEOPS` for the whole ~9 KB/hart pack instead of one cacheop per group; the
requant output tile is likewise evicted once. Amortizes cacheop/fence latency across the tile.

## Multi-hart correctness (the load-bearing part)

### Row-banding at 8 harts
Each hart owns a horizontal band `row0 = IMG·hart/ACTIVE_HARTS … row1` — **8 rows** at 64×64 / 8
harts. Between layers a sync protocol keeps the 1-pixel conv halo coherent across band seams without
any shared write:
- `layer_publish`: fence + evict my band → barrier → hart 0 fills+evicts the whole-image halo →
  barrier, so my band is globally visible before the next layer reads across boundaries.
- `invalidate_read`: evict my read window (band ±1 halo row) right before reading, so neighbours'
  fresh rows come from DRAM.
- `fill_halo_band` / `bench_barrier`: replicate/zero the halo; cross-hart barrier.

### Cacheline seam invariant
Multi-hart int8 silently corrupted at band seams until the row stride was padded so **no 64 B cache
line is ever shared between two harts' bands**. The invariant is `PADW = roundup(IMG+2, 64/CH)`,
asserted `(PADW·CH) % 64 == 0`. At 64 channels this gives `PADW = 66` and `PADW·CH = 4224 = 66`
whole cache lines. If a row straddled a line boundary, two harts' partial write-backs to the shared
line would race and corrupt; the invariant makes every row an integer number of lines. Verified:
8-hart board output is bit-identical to 1-hart.

## Mixed precision + bias + residual

- **FP32 boundaries.** `conv_first` (`image/255`, zero-pad, +bias, ReLU, quantize) and `conv_final`
  (dequant, out_conv, +bias) stay full-precision scalar — the sensitive 1↔64-channel layers don't
  quantize well. They loop over the compile-time `CH` so the accumulate unrolls (see "Lessons").
- **Bias folded into the int32 accumulator.** Hidden biases fold via `tensor_quant INT32_ADD_COL`
  before dequant: `bias_i32[oc] = round(b_oc / (S_in·S_w))` on the SCP bias line, then
  `ADD_COL → INT32_TO_FP32 → MUL_COL(scale) → FP32_TO_INT32 → SATUINT8(relu+clip) → PACK`.
- **Residual.** `conv_final` computes `out = clip(round(255·(x + net(x))))`. deepinv's DnCNN predicts
  a correction added back to the input; without `+x` PSNR collapses (~4.8 dB).

## Memory map

Fixed harness anchors, then cumulative 64 KB-aligned working regions (all `_Static_assert`ed
non-overlapping and within the 16 MB launch window):

| region | offset | note |
|---|---|---|
| attestation slots | `0x0000` | ACTIVE_HARTS × 64 B |
| summary | `0x1000` | 16 × uint32 (CI `validate_dump`) |
| input | `0x2000` | uint8[64][64] (file-loaded) |
| output | `0x10000` | uint8[64][64] (gated region) |
| weights | `0x14000` | mixed FP32/int8 blob, ~673 KB (file-loaded) |
| qpad A / B | after weights | halo-padded activation ping-pong, 66×66×64 each |
| aw_tap | after qpad | rearranged hidden weights, NGRP×CH×64 |
| mvec / bvec | after aw_tap | per-OC requant scale / int32 bias |
| bpack | 64 KB-aligned | per-hart folded-B pack (batched evict), ~9 KB/hart |
| temp | after bpack | per-hart quant output, OC_TILE×P/hart |

SCP (48-line L1 scratchpad): A weights lines 0–15, B activations 16–31, bias line 32, scale line 33
— asserted `< 48`.

## Lessons (measure retired-inst, not wall)

On this platform the sys-emu `kernel_wait_s` wall is a **poor proxy for scalar cost** and is not
proportional to retired instructions. Judge instruction-count changes with the PMC `minstret`
retired-inst counter (readable in local sys-emu, no board needed) plus disassembly — the
`-DDNCNN_PMC` build wires this up (`pmc_probe.h`).

**A runtime dimension that merely equals a compile-time constant is still slower, because it blocks
unrolling and strength-reduction.** Two large regressions during development, both invisible in a
wall read and only exposed by retired-inst + disassembly:
- Passing the channel count as a runtime param turned the word-copy pack's compile-time `j·CH`
  stride into a per-quartet multiply.
- Passing the boundary channel dims as runtime params de-optimized the FP32 accumulate — the `fmul.s`
  count collapsed from fully unrolled down to a rolled loop.

That is why every hot loop bounds on the compile-time `CH`, with the runtime `NET[]` shape table used
only in the cold `main` dispatch, and why the source asserts `CH ∈ {16,32,64}`.

## Platform notes worth keeping

- `fdiv` hangs in U-mode here, so the FP32 boundary scales are compile-time constants **multiplied**,
  never divided; rounding uses inline `fcvt.w.s` with explicit modes (`rint_rne`, `round_half_up`) to
  match the reference without libm.
- The ±1 max_abs vs the PyTorch reference is torch-vs-scalar FP32 **accumulation-order rounding** in
  the FP32 boundaries — deterministic, and **not** a seam artifact (8-hart output is bit-identical to
  1-hart). Don't chase it as a multi-hart bug.
