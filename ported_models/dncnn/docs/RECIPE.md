# `dncnn` (variant `dncnn20l64`) — validated-model reproduce recipe

The `dncnn` leaderboard model is the real trained deepinv DnCNN (`dncnn_sigma2_gray.pth`, σ=2
grayscale, 20 conv / 64 ch) ported to the int8 ET-SoC1 kernel. Canonical variant `dncnn20l64`
(8-hart, 64×64 tile). This recipe reproduces the five fully-validated-model pieces
(`docs/SUBMISSION_GUIDE.md`): provenance, host reference, board correctness, model quality, board
metric. It runs in CI exactly like `yolo` — the weights + input are committed assets, no operator
setup.

## 0. Environment

- Python venv with `torch` + `numpy` (blob generation), plus `deepinv==0.4.1` for the oracle.
- Pinned weights (not committed): `bash scripts/download_hf_refs.sh` →
  `local-artifacts/hf_refs/dncnn/dncnn_sigma2_gray.pth`
  (HF `deepinv/dncnn` @ `3bb1f2a95321781343331069776c3eba98707a56`, BSD-3-Clause).

## 1. Provenance

Model identity: `MODEL.md` / `docs/HF_REFERENCES.md` — repo `deepinv/dncnn`, revision `3bb1f2a9…`,
file `dncnn_sigma2_gray.pth`, BSD-3-Clause. Architecture (from the state_dict):
`DnCNN(in_channels=1, out_channels=1, depth=20, bias=True, nf=64)`, no BatchNorm, residual
`out = out_conv(h) + x`, input domain `[0,1]`.

## 2. Host reference (common runtime = PyTorch/deepinv)

```
python3 scripts/gen_dncnn_oracle.py
```

Runs `deepinv.models.DnCNN` on the pinned weights and writes the board accuracy-gate reference
`refs/dncnn20l64_reference.npy` (the torch FP32 output quantized to uint8) plus
`refs/dncnn20l64_baseline_fp32.npy`. It cross-checks the deepinv module against our F.conv2d/numpy
reconstruction (`max_abs=0`), proving the reconstruction faithful and fixing the residual sign.
**The numpy int8 forward is a debug tool only — the gate reference is the PyTorch oracle.**

## 3. Committed board assets (like yolo)

```
python3 scripts/gen_dncnn_real.py     # emits dncnn20l64_input.bin, dncnn20l64_weights.bin (+scales)
```

The two blobs are committed under `ported_models/dncnn/assets/dncnn/`. `prepare_benchmark_inputs.sh`
auto-copies every `ported_models/*/assets/` file the config references, so the board run finds
`dncnn/dncnn20l64_input.bin` @0x2000 and `dncnn/dncnn20l64_weights.bin` @0x14000 with **no operator
staging**. Deterministic sha256 anchors:

```
dncnn20l64_input.bin    e60d0f8812bef0adf80b756265e6c836cf48c886d44701507dba8e314943f45c
dncnn20l64_weights.bin  41bce48cb78dde8db0b03ebcc96559cefa775060fe47f6403bbffab7f8343a06
```

## 4. Build the ELF

CI builds it via `.github/ci/scripts/build_leaderboard_elf.sh dncnn` from the entry's
`build.defines`. To reproduce by hand (RISC-V gcc + the ET-SoC1 erbium platform includes/linker):

```
riscv64-unknown-elf-gcc -O3 -march=rv64imfc -mabi=lp64f -mcmodel=medany -nostdlib \
  -fno-zero-initialized-in-bss -fno-tree-loop-distribute-patterns -fno-strict-aliasing \
  -ffunction-sections -fdata-sections $ERBIUM_GCC_INCLUDE_FLAGS \
  -Wl,--gc-sections -Wl,--defsym=NUM_HARTS=16 \
  -DNUM_HARTS=16 -DACTIVE_HARTS=8 -DBENCH_THREAD0_ONLY=1 \
  -DDNCNN_CH=64u -DDNCNN_HIDDEN=18u -DDNCNN_IMG=64u -DDNCNN_REAL=1 \
  -I ported_models/dncnn/src -T "$ERBIUM_LD" -o dncnn20l64.elf \
  ported_models/dncnn/src/dncnn_gen_int8.c .github/ci/support/hart_report_crt.S "$ERBIUM_LAYOUT"
```

The board launcher then file-loads `zero2m`@`0x0`, input@`0x2000`, weights@`0x14000` and dumps the
output at `0x10000`. Kernel design and why these flags: `docs/OPTIMIZATIONS.md`.

## 5. Board correctness

CI applies the `uint8_npy` gate (`benchmark_config.json` → `models.dncnn`): the 64×64 dump region
`@0x10000` vs `refs/dncnn20l64_reference.npy`, `max_abs ≤ 2`, plus the self-attesting summary
(`dump_magic 0xD3C11003`, `done_count==active_harts`, `output_sum==slot_sum`). Reproduce against a
board dump:

```
python3 scripts/check_board_dump.py <board_dump.bin> 8 --emit board_out.bin
```

Board-verified: the kernel matches the PyTorch oracle at `max_abs=1` (3/3 board runs, ~0.221 s,
`output_sum==slot_sum`). The board is deterministic integer arithmetic, so this is stable on the
fixed gate input; `max_abs ≤ 2` is a 1-unit margin (the FP32 conv_first/conv_final rounding can
move a pixel by 1).

## 6. Model quality (accuracy degradation)

```
python3 scripts/eval_quality.py [--dump board_out.bin]
```

Writes `refs/dncnn20_quality.json` + `docs/quality_report.md`. At σ=2 on `[0,1]`: noisy PSNR 41.9 dB
→ FP32 oracle **51.0 dB / SSIM 0.998** (+9.1 dB denoise gain); the int8 path costs **~0.4 dB /
0.0002 SSIM**. Pass `--dump` after a board run to fill the `board_int8` row.
