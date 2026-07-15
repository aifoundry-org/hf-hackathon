# DnCNN Model Card

- Reference family: DnCNN denoising CNN (σ=2 grayscale).
- Hugging Face base: `deepinv/dncnn` at
  `3bb1f2a95321781343331069776c3eba98707a56`.
- Reference file: `dncnn_sigma2_gray.pth`.
- License: `bsd-3-clause`.
- Leaderboard model: `dncnn` — canonical variant `dncnn20l64` (int8-TFMA, 20 conv / 64 ch, 8-hart).
- Main source: `src/dncnn_gen_int8.c` (config-driven int8 mixed-precision kernel).
- Leaderboard manifest: `manifests/dncnn_variants.txt`.
- Key docs: `docs/RECIPE.md` (reproduce recipe), `docs/OPTIMIZATIONS.md` (kernel design +
  optimizations), `docs/quality_report.md`.

Weights are not committed; download the pinned Hugging Face reference with
`scripts/download_hf_refs.sh` and build ET-SoC1 ELFs locally from source. The board input + weight
blobs are committed as deterministic assets under `assets/dncnn/`.

## Fully-validated-model standard

The `dncnn` model (20 conv / 64 ch grayscale, no BatchNorm, residual `out = out_conv(h) + x`, input
domain `[0,1]`, σ=2) meets `docs/SUBMISSION_GUIDE.md`:

- **Provenance:** HF `deepinv/dncnn` @ `3bb1f2a95321781343331069776c3eba98707a56`,
  `dncnn_sigma2_gray.pth`, BSD-3-Clause. Weights fetched via `scripts/download_hf_refs.sh`, not committed.
- **Host reference (common runtime):** `scripts/gen_dncnn_oracle.py` runs the actual
  `deepinv.models.DnCNN` (deepinv==0.4.1 / torch) and emits the board accuracy-gate reference
  `refs/dncnn20l64_reference.npy`. The deepinv module matches our F.conv2d/numpy reconstruction
  bit-for-bit (`max_abs=0`), confirming the residual sign. The numpy int8 forward is a debug proxy
  only, never the gate.
- **Board correctness:** the CI `uint8_npy` gate compares the 64×64 board dump against the PyTorch
  oracle. Board-verified: the kernel matches the oracle at `max_abs=1` (3/3 board runs, ~0.221 s,
  `output_sum==slot_sum`); the gate is set to `max_abs ≤ 2` as a 1-unit margin. Plus the
  self-attesting summary (`dump_magic 0xD3C11003`). Tracked checker: `scripts/check_board_dump.py`.
- **Model quality (accuracy degradation):** `scripts/eval_quality.py` (see `docs/quality_report.md`):
  the FP32 oracle gives ~+9.1 dB denoise gain (PSNR 51.0 / SSIM 0.998); the int8 path costs only
  ~0.4 dB / 0.0002 SSIM.
- **Board metric:** ranked on `kernel_wait_s` (lower better).

Reproduce end-to-end: `docs/RECIPE.md`.
