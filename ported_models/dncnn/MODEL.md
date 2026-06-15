# DnCNN Model Card

- Reference family: DnCNN denoising CNN.
- Hugging Face base: `deepinv/dncnn` at
  `3bb1f2a95321781343331069776c3eba98707a56`.
- Reference files: `dncnn_sigma2_gray.pth`, `dncnn_sigma2_color.pth`.
- Main source: `src/dncnn3_vpu_fp_argbuf.c`.
- Smoke manifest: `manifests/v3x_variants.txt`.
- Sweep manifest: `manifests/v3_100_variants.txt`.
- Key docs: `docs/optimizations.md`.

Runtime blobs, generated dumps, and generated ELFs are not git files. Download
the pinned Hugging Face references with `scripts/download_hf_refs.sh` and build
ET-SoC1 ELFs locally from source.
