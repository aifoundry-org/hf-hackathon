# DnCNN int8 TFMA kernel - port recipe

Agent-readable notes for porting the 5-layer, 16-channel DnCNN denoiser to an
int8 mixed-precision kernel on one shire of ET-SoC1, partitioned across 8 harts.
This is the kernel wired into the `dncnn` leaderboard entry
(`.github/ci/benchmark_config.json`).

- **Kernel source:** `ported_models/dncnn/src/dncnn3_tfma_int8.c`
- **Scales header:** `ported_models/dncnn/src/dncnn_int8_scales.h` (generated, committed)
- **Reference generators:** `scripts/gen_dncnn_inputs.py` (image + int8 weight blob),
  `scripts/gen_dncnn_int8.py` (int8 reference + scales)
- **Prior FP32 kernel:** `ported_models/dncnn/src/dncnn3_vpu_fp_argbuf.c` (kept for reference;
  its optimization journey is in `docs/optimizations.md`)
- **Result:** bit-exact (`max_abs=0`) vs the int8 reference at 1 hart (sys-emu) and 8
  harts (board); ~0.0107 s kernel wait on the board (~2.9x the FP32 winner's 0.0314 s).

## 1. Network and precision split

The DnCNN graph here is `conv_first -> 3 hidden convs -> conv_final`, all 3x3, with
16 channels in the hidden stack (the bench uses synthetic fixed-seed weights, not the
trained deepinv/dncnn weights - see `MODEL.md`). The kernel runs **mixed precision**:

```
conv_first + quantize (FP32) -> hidden1/2/3 (int8 TFMA) -> dequantize + conv_final (FP32)
```

Rationale: `conv_first`/`conv_final` are <1% of the network's MACs but the most
sensitive to quantization, so they stay scalar FP32; the 3 hidden layers are ~99% of
the MACs and run int8 on the tensor engine, which is where nearly all the speedup comes
from (int8 TFMA is the single biggest lever - see `docs/optimizations.md sec 1.1`).

## 2. int8 hidden layer on the tensor engine

Each hidden output tile is 16 OC x 16 spatial. Per tile (`conv_tile`):

1. **Pack activations** (`pack_b_tile`): gather the 3x3 window into the
   quartet-interleaved B layout - `CH/QUARTET = 4` scratchpad lines, 4 input channels
   per FMA step.
2. **9 taps** (`fma_tap` -> `tensor_fma`, opcode 3 = int8): one dispatch per `(ky,kx)`
   tap accumulates 16 OC x 16 IC x 16 spatial into the int32 TENC accumulator.
   `first_pass` resets TENC on tap 0; `tenc_loc` copies TENC -> float registers on tap 8.
3. **Requant** (`requant_u8` -> `tensor_quant`): fused int32 -> uint8 in the float
   registers - dequant x per-tensor scale, round-nearest-even, ReLU + clamp[0,255],
   pack. One CSR write does the whole chain.
4. **Scatter** the uint8 [OC][P] tile into a *padded* destination interior so the next
   layer's `pack_b_tile` has neighbours in place.

Weights are rearranged once per layer from the blob's `WH[l][oc][ic][k]` into the
FMA's tap-major A layout `aw_tap[t][oc][ic]` (`rearrange_hidden_weights`).

## 3. Quantization scheme (how the scales are derived)

`gen_dncnn_int8.py` calibrates per-tensor activation scales from the FP32 activation
ranges (`S[l] = act[l].max() / 255`) and folds them into compile-time multipliers that
the kernel bakes in (so the U-mode path never issues `fdiv`, which hangs on this
platform):

- `DNCNN_QUANT0 = 1/S[0]` - conv_first FP32 output -> uint8.
- `DNCNN_REQUANT[l] = S[l] - HIDDEN_SCALE / S[l+1]` - the folded per-layer requant
  multiplier used in `tensor_quant`.
- `DNCNN_DEQUANT3 = S[3]` - uint8 -> FP32 for conv_final.

These land in `dncnn_int8_scales.h`. The generator also emits `dncnn_reference_int8.npy`
(the exact bytes the kernel must produce) and prints the int8-vs-FP32 accuracy
(`max_abs=2`, the real quality gate - quantization must stay close to FP32).

## 4. Suggested build-up order (if reproducing from scratch)

The kernel was grown incrementally; each step is independently checkable against a
numpy reference, which is the fastest way to localize a tensor-engine encoding bug:

1. One int8 `tensor_fma` tap (16x16x16) -> verify one MAC tile vs numpy.
2. Add the fused `tensor_quant` requant chain -> verify int32 -> uint8.
3. One full hidden conv layer (9 taps + requant + padded scatter).
4. The full 5-layer network, single hart (quant/dequant as their own passes).
5. Multi-hart row-band partitioning + inter-layer sync.
6. Fuse conv_first+quantize and dequant+conv_final into single passes.

## 5. Multi-hart parallelization

The image is split into 8 contiguous row bands, one per hart. Between layers each hart
publishes its band and reads neighbours' boundary rows through DRAM:

- `layer_publish`: evict my band -> barrier -> hart 0 fills the whole 1-pixel halo and
  evicts the whole buffer -> barrier. (Two barriers: all bands must be in DRAM before
  hart 0 builds the halo; the halo must be in DRAM before anyone reads.)
- `invalidate_read`: right before a hart reads, invalidate its read window (band +/-1 row)
  so neighbours' fresh rows come from DRAM, not a stale L1 copy.

### Seam-race gotcha (int8-only, board-only)

int8 activations are uint8 NHWC, so a padded row is `PADW*CH` bytes. With the natural
`PADW = IMG_W+2 = 66`, a row is `66*16 = 1056` bytes = **16.5 cache lines** - not a
whole number, so a cache line straddles two rows, i.e. two harts' bands. Both harts
write that "seam" line back on evict; the per-band evict leaves it dirty-and-stale in
one hart's L1, and the halo fill then reads a neighbour's edge pixel stale and clobbers
it. It only appears on the board at 8 harts (timing-sensitive) and only in int8 (FP32's
`CH*4 = 64`-byte pixel made rows accidentally line-aligned).

**Fix:** pad the row *stride* to `PADW = 68` so `PADW*CH = 1088 = 17` whole lines ->
every band boundary is line-aligned and each line has a single writer. Locked with
`_Static_assert((PADW*CH) % 64 == 0)`; the same alignment rule is asserted for the
per-hart scratch (`HART_SCRATCH`).

## 6. Self-attesting summary (CI correctness gate)

Board CI does not diff against a golden image; it trusts a checksum the kernel writes.
Each hart writes a 64-byte slot with the byte-sum of its output band; hart 0 folds them
(`slot_checksum_sum`) and independently sums the whole output image (`output_sum`).
`score_results.py` reads the summary at `SUMMARY_OFFSET` (0x1000) and passes only when:

- `magic == 0xD3C11003`
- `done_count == active_harts`
- `output_sum == slot_checksum_sum`

The two sums agree iff every band was written correctly and the bands tile the image
exactly - which is also what makes the seam-race fix observable in CI.

## 7. Build

The kernel `#include "dncnn_int8_scales.h"` with quotes, so the compiler finds the
committed header next to the source - no `-I` needed. The one non-obvious flag is
`-fno-tree-loop-distribute-patterns` (passed via the model's `defines` in
`benchmark_config.json`): at `-O3` GCC would otherwise turn the halo copy loops into
`memcpy` calls, which don't exist in the `-nostdlib` freestanding link.

Regenerate the scales header (deterministic, fixed seed) after any quantization change:

```bash
python3 ported_models/dncnn/scripts/gen_dncnn_int8.py
cp local-artifacts/erbium_amp_probe/dncnn3-bench/dncnn_int8_scales.h ported_models/dncnn/src/
```

Build the board ELF exactly as CI does:

```bash
BENCHMARK_DEVICE=board bash .github/ci/scripts/build_leaderboard_elf.sh dncnn
```

## 8. Verify

Two independent checks on the post-run dump of the launch region:

- **Correctness:** compare the output image at `OUTPUT_OFFSET` (0x10000, 64x64 uint8)
  against `dncnn_reference_int8.npy` (emitted by `gen_dncnn_int8.py`). Require
  `max_abs == 0`. Per-row differences localize any residual seam error to a band edge.
- **CI gate:** read the 16x`uint32` summary at `SUMMARY_OFFSET` (0x1000) and confirm
  `magic == 0xD3C11003`, `done_count == active_harts`, and
  `output_sum == slot_checksum_sum` - the same three conditions `score_results.py`
  enforces on the board.

The seam race is timing-sensitive, so confirm the 8-hart board run over 3 consecutive
passes, not a single run.

## 9. Dead ends and platform gotchas worth knowing

- **`tensor_fma` operand signedness:** the tensors.h argument names for the two operands
  are swapped; A is signed / B unsigned. Getting this backwards produces plausible noise.
- **`fdiv` hangs in U-mode** - all scales are precomputed multiplies, never divides.
- **FREG clobber:** `tensor_fma(tenc_loc=1)` overwrites `f0..f31` but GCC can't see it
  through the CSR asm; a clobber barrier is required or live FP scalars get corrupted.
- **64-byte alignment:** every tensor-touched buffer must be line-aligned; `tensor_load`
  silently rounds the address down otherwise.
- **Seam-race masking:** `-DDNCNN_DUMP` and an extra sync barrier both *hid* the seam
  race - the 8-hart board "passed" with them but the production build failed. The race is
  invisible at 1 hart and in sys-emu; only the no-dump, multi-run, 8-hart board build
  shows it. Never trust a single pass.
