# DnCNN Model Card

- Reference family: DnCNN denoising CNN.
- Hugging Face base: `deepinv/dncnn` at
  `3bb1f2a95321781343331069776c3eba98707a56`.
- Reference files: `dncnn_sigma2_gray.pth`, `dncnn_sigma2_color.pth`.
- License: `bsd-3-clause`.
- Main source: `src/dncnn3_tfma_int8.c` (int8 mixed-precision, the current `dncnn`
  leaderboard kernel). Prior FP32 kernel kept for reference: `src/dncnn3_vpu_fp_argbuf.c`.
- Leaderboard manifest: `manifests/int8_tfma_variants.txt` (canonical `int8_tfma_8hart`).
  FP32 sweep manifests `manifests/v3x_variants.txt` / `v3_100_variants.txt` remain for
  the reference kernel.
- Key docs: `docs/int8_tfma_recipe.md` (int8 port recipe), `docs/optimizations.md`
  (FP32 optimization journey).

Runtime blobs, generated dumps, and generated ELFs are not git files. Download
the pinned Hugging Face references with `scripts/download_hf_refs.sh` and build
ET-SoC1 ELFs locally from source.
