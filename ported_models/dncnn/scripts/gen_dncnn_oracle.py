"""Torch/deepinv host oracle for the REAL dncnn20l64 model -- the common-runtime baseline.

The board accuracy gate compares the ET-SoC1 int8 dump against THIS oracle's output:
`deepinv.models.DnCNN` run on the pinned `dncnn_sigma2_gray.pth`, quantized to uint8 pixels.
The numpy int8 forward in `gen_dncnn_real.py` is a DEBUG reimplementation only, never the gate.

Provenance: Hugging Face `deepinv/dncnn` @ 3bb1f2a95321781343331069776c3eba98707a56,
license BSD-3-Clause, file `dncnn_sigma2_gray.pth`. Pinned runtime: deepinv==0.4.1 on torch
(CPU). Architecture: DnCNN(in_channels=1, out_channels=1, depth=20, bias=True, nf=64), no
BatchNorm, residual `out = out_conv(h) + x` (deepinv ground truth), input domain [0,1].

Emits into ported_models/dncnn/refs/ :
  dncnn20l64_reference.npy      uint8[IMG,IMG]   board accuracy-gate reference (torch oracle)
  dncnn20l64_baseline_fp32.npy  float32[IMG,IMG] raw fp32 denoised output (for PSNR/SSIM)
  dncnn20l64_input.bin          uint8[IMG*IMG]   the exact gated input (== board input)

Run: python3 ported_models/dncnn/scripts/gen_dncnn_oracle.py            (64x64 board region)
     DNCNN_IMG=16 python3 ported_models/dncnn/scripts/gen_dncnn_oracle.py  (small local check)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_dncnn_real as G   # reuse make_input / load_weights / forward / PTH / IMG for exact parity

REPO = G.REPO
IMG = G.IMG
REFS = HERE.parent / "refs"


def build_oracle():
    """Instantiate the real deepinv DnCNN and load the pinned grayscale weights."""
    from deepinv.models import DnCNN
    model = DnCNN(in_channels=1, out_channels=1, depth=20, bias=True, nf=64, pretrained=None)
    sd = torch.load(G.PTH, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model


def oracle_forward(model, img_u8):
    """Run the deepinv module. Returns (fp32 denoised [H,W], uint8 pixel [H,W])."""
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)[None, None]   # [1,1,H,W] in [0,1]
    with torch.no_grad():
        y = model(x)[0, 0].numpy()                                       # out_conv(h)+x, deepinv residual
    pix = np.clip(np.round(255.0 * y), 0, 255).astype(np.uint8)
    return y.astype(np.float32), pix


def main():
    REFS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img = G.make_input(rng)                     # identical input to the board build (gen_dncnn_real)
    W, B = G.load_weights()

    model = build_oracle()
    y_fp32, pix_oracle = oracle_forward(model, img)

    # cross-check: the deepinv module vs the manual F.conv2d reconstruction in gen_dncnn_real.
    # Proves our reconstruction is faithful and confirms the residual sign end-to-end.
    pix_fp_recon, pix_i8_debug, _ = G.forward(img, W, B)
    d_recon = int(np.abs(pix_oracle.astype(int) - pix_fp_recon.astype(int)).max())
    d_i8 = int(np.abs(pix_oracle.astype(int) - pix_i8_debug.astype(int)).max())
    d_i8_mean = float(np.abs(pix_oracle.astype(int) - pix_i8_debug.astype(int)).mean())

    np.save(REFS / "dncnn20l64_reference.npy", pix_oracle)
    np.save(REFS / "dncnn20l64_baseline_fp32.npy", y_fp32)
    img.tofile(REFS / "dncnn20l64_input.bin")

    print(f"IMG={IMG}  refs -> {REFS}")
    print(f"deepinv-module vs manual-fp32-reconstruction: max_abs={d_recon}  (should be ~0)")
    print(f"int8-debug vs torch-oracle:                    max_abs={d_i8}  mean_abs={d_i8_mean:.3f}")
    print(f"oracle pixel range [{int(pix_oracle.min())},{int(pix_oracle.max())}]")


if __name__ == "__main__":
    main()
