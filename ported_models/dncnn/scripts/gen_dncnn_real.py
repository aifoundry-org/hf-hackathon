"""Export the REAL deepinv/dncnn (dncnn_sigma2_gray.pth) into the modular int8 kernel's layout.

20 conv / 64ch grayscale DnCNN, NO BatchNorm, residual (out = x + net(x)), input domain [0,1].
Mixed precision matching the kernel: FP32 conv_first(+quant) -> 18x int8 hidden (int32 bias folded
via tensor_quant INT32_ADD_COL) -> FP32 conv_final(+dequant) + residual add (deepinv out=net(x)+x).

Emits into local-artifacts/erbium_amp_probe/dncnn-20l64ch<img suffix>/ :
  dncnn3_input.bin          uint8[IMG*IMG]           test image (deterministic; demo swaps a real photo)
  dncnn_weights_real.bin    mixed blob (see layout)  W_in|b_in (fp32) | WH (int8) | bias_hid (i32) | W_out|b_out (fp32)
  dncnn_int8_scales_real.h  DNCNN_QUANT0 / DNCNN_REQUANT[18] / DNCNN_DEQUANT3
  dncnn_reference_int8.npy  uint8[IMG,IMG]           what the int8 kernel must output (bit-exact gate)
  dncnn_reference.npy       uint8[IMG,IMG]           fp32-path output (quality anchor)

Run: DNCNN_IMG=16 python3 ported_models/dncnn/scripts/gen_dncnn_real.py   (small = fast local gate)
     python3 ported_models/dncnn/scripts/gen_dncnn_real.py                 (64x64 board)

Requires torch (local-artifacts/torch-venv). Per-tensor int8 is validated to track fp32 to
max_abs=1 over all 20 layers, so no per-channel scales.
"""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

IMG   = int(os.environ.get("DNCNN_IMG", 64))
CH    = 64
HIDDEN = 18                     # conv_list.0..17
K     = 3
REPO  = Path(__file__).resolve().parents[3]
PTH   = REPO / "local-artifacts/hf_refs/dncnn/dncnn_sigma2_gray.pth"
SUF   = "" if IMG == 64 else f"_{IMG}"
OUT   = Path(os.environ["DNCNN_OUT"]) if os.environ.get("DNCNN_OUT") \
        else REPO / f"local-artifacts/erbium_amp_probe/dncnn-20l64ch{SUF}"


def load_weights():
    sd = torch.load(PTH, map_location="cpu", weights_only=True)
    W = [sd["in_conv.weight"].numpy()] + [sd[f"conv_list.{k}.weight"].numpy() for k in range(HIDDEN)] + [sd["out_conv.weight"].numpy()]
    B = [sd["in_conv.bias"].numpy()]   + [sd[f"conv_list.{k}.bias"].numpy()   for k in range(HIDDEN)] + [sd["out_conv.bias"].numpy()]
    return W, B


def make_input(rng):
    # deterministic structured test image in [0,1] + sigma=2/255 noise, quantized to uint8.
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    clean = (0.3 + 0.4 * xx / IMG).astype(np.float32)
    clean[IMG//4:IMG*5//8, IMG//4:IMG*5//8] = 0.85
    clean[:, IMG*13//16: IMG*13//16 + max(1, IMG//16)] = 0.1
    noisy = np.clip(clean + (2/255.0) * rng.standard_normal((IMG, IMG)).astype(np.float32), 0, 1)
    return np.clip(np.round(noisy * 255), 0, 255).astype(np.uint8)


def conv_fp(x, w, b):   # x [C,H,W], w [O,C,3,3], b [O] -> [O,H,W]
    return F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), padding=1)[0].numpy()


def relu(x):
    return np.maximum(x, 0.0)


def hidden_acc_i32(q_u8, wq_i32):
    """int8 hidden conv accumulator: q_u8 [64,H,W] uint8, wq_i32 [64,64,3,3] int32 -> int64 [64,H,W]."""
    xp = np.pad(q_u8.astype(np.int64), ((0, 0), (1, 1), (1, 1)), mode='constant')   # zero pad == torch pad
    acc = np.zeros((CH, IMG, IMG), dtype=np.int64)
    for oc in range(CH):
        a = np.zeros((IMG, IMG), dtype=np.int64)
        for ky in range(K):
            for kx in range(K):
                a += (xp[:, ky:ky+IMG, kx:kx+IMG] * wq_i32[oc, :, ky, kx][:, None, None]).sum(0)
        acc[oc] = a
    assert np.abs(acc).max() < 2**31, "int32 overflow in hidden acc"
    return acc


def forward(img_u8, W, B):
    x = (img_u8.astype(np.float32) / 255.0)[None]           # [1,H,W] in [0,1]
    # ---- fp32 reference (residual) ----
    h = relu(conv_fp(x, W[0], B[0]))
    acts = [h]
    for k in range(1, HIDDEN + 1):
        h = relu(conv_fp(h, W[k], B[k])); acts.append(h)
    net_fp = conv_fp(h, W[19], B[19])
    pix_fp = np.clip(np.round(255 * (x + net_fp)[0]), 0, 255).astype(np.uint8)

    # ---- int8 forward mirroring the kernel exactly ----
    h0 = relu(conv_fp(x, W[0], B[0]))                        # [64,H,W] fp32
    S = [float(h0.max()) / 255.0]
    q = np.clip(np.round(h0 / np.float32(S[0])), 0, 255).astype(np.uint8)
    WH_i8, BIAS_i32, M = [], [], []
    for k in range(1, HIDDEN + 1):
        w = W[k]
        Sw = float(np.abs(w).max()) / 127.0
        wq = np.clip(np.round(w / np.float32(Sw)), -127, 127).astype(np.int32)
        acc = hidden_acc_i32(q, wq)
        Sout = float(relu(acts[k]).max()) / 255.0
        Sin = S[-1]
        Mk = np.float32(Sin * Sw / Sout)
        bias_i32 = np.round(B[k] / (Sin * Sw)).astype(np.int32)     # per-OC, folded into acc
        q = np.clip(np.round(relu((acc + bias_i32[:, None, None]).astype(np.float64)) * Mk), 0, 255).astype(np.uint8)
        S.append(Sout); WH_i8.append(wq.astype(np.int8)); BIAS_i32.append(bias_i32); M.append(float(Mk))
    # conv_final fp32 + residual
    h18_deq = q.astype(np.float32) * np.float32(S[-1])
    net_i8 = conv_fp(h18_deq, W[19], B[19])
    pix_i8 = np.clip(np.round(255 * (x + net_i8)[0]), 0, 255).astype(np.uint8)
    return pix_fp, pix_i8, dict(S=S, M=M, WH_i8=WH_i8, BIAS_i32=BIAS_i32)


def write_blob(W, B, st, path):
    parts = [
        W[0].astype('<f4').ravel(), B[0].astype('<f4').ravel(),                       # in_conv fp32
        np.concatenate([w.ravel() for w in st['WH_i8']]).astype('<i1'),               # hidden int8
        np.concatenate([b.ravel() for b in st['BIAS_i32']]).astype('<i4'),            # hidden bias int32
        W[19].astype('<f4').ravel(), B[19].astype('<f4').ravel(),                     # out_conv fp32
    ]
    with open(path, "wb") as f:
        for p in parts:
            f.write(p.tobytes())
    return sum(p.nbytes for p in parts)


def write_scales_h(st, path):
    S, M = st['S'], st['M']
    req = ", ".join(f"{m:.9g}f" for m in M)
    with open(path, "w") as f:
        f.write("/* auto-generated by gen_dncnn_real.py -- REAL deepinv dncnn int8 scales */\n")
        f.write("#ifndef DNCNN_INT8_SCALES_H\n#define DNCNN_INT8_SCALES_H\n")
        f.write(f"static const float DNCNN_QUANT0     = {1.0/S[0]:.9g}f;  /* conv_first FP32 -> uint8 */\n")
        f.write(f"static const float DNCNN_REQUANT[{HIDDEN}] = {{ {req} }};\n")
        f.write(f"static const float DNCNN_DEQUANT3   = {S[HIDDEN]:.9g}f;  /* uint8 -> FP32 for conv_final */\n")
        f.write("#endif\n")


def psnr(a, b):
    m = np.mean((a.astype(float) - b.astype(float))**2)
    return 99.0 if m == 0 else 10*np.log10(255.0**2/m)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W, B = load_weights()
    img = make_input(rng)
    pix_fp, pix_i8, st = forward(img, W, B)

    img.tofile(OUT / "dncnn3_input.bin")
    nbytes = write_blob(W, B, st, OUT / "dncnn_weights_real.bin")
    # committed-asset names for the `dncnn` benchmark entry (input @0x2000, weights @0x14000).
    # Copy these into ported_models/dncnn/assets/dncnn/ ; CI auto-stages them. The torch oracle
    # reference (scripts/gen_dncnn_oracle.py) keys off this exact input.
    img.tofile(OUT / "dncnn20l64_input.bin")
    write_blob(W, B, st, OUT / "dncnn20l64_weights.bin")
    write_scales_h(st, OUT / "dncnn_int8_scales_real.h")
    np.save(OUT / "dncnn_reference_int8.npy", pix_i8)
    np.save(OUT / "dncnn_reference.npy", pix_fp)

    d = np.abs(pix_i8.astype(int) - pix_fp.astype(int))
    print(f"IMG={IMG} out={OUT.name}")
    print(f"blob bytes={nbytes}  (W_in {W[0].size*4} + b_in {B[0].size*4} + WH {sum(w.size for w in st['WH_i8'])} "
          f"+ bias_i32 {sum(b.size for b in st['BIAS_i32'])*4} + W_out {W[19].size*4} + b_out {B[19].size*4})")
    print(f"int8-vs-fp32: max_abs={d.max()} mean_abs={d.mean():.3f}   (per-tensor int8 quality gate)")
    print(f"S range [{min(st['S']):.4g},{max(st['S']):.4g}]  M range [{min(st['M']):.3g},{max(st['M']):.3g}]")


if __name__ == "__main__":
    main()
