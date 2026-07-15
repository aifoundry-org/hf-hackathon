# DnCNN Denoiser (`dncnn`)

Real deepinv DnCNN (`dncnn_sigma2_gray.pth`, σ=2 grayscale, 20 conv / 64 ch) ported to an int8-TFMA
ET-SoC1 kernel. Canonical leaderboard variant `dncnn20l64` (8-hart, 64×64 tile).

- Example denoised images (low-noise model): [`samples/`](samples/README.md)
- Model card + validated-model summary: [`MODEL.md`](MODEL.md)
- Reproduce end-to-end (download weights, run the PyTorch reference, assets, build, board gate, quality): [`docs/RECIPE.md`](docs/RECIPE.md)
- Kernel design & optimizations: [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)
- Quality report: [`docs/quality_report.md`](docs/quality_report.md)

Pinned weights are downloaded on demand with `scripts/download_hf_refs.sh` (HF `deepinv/dncnn`,
BSD-3-clause) and are not committed. The board input + weight blobs are committed as deterministic
assets under `assets/dncnn/`; the PyTorch accuracy-gate reference is `refs/dncnn20l64_reference.npy`.
Generated ELFs and board dumps are not committed.
