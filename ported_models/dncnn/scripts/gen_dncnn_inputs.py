"""Generate a SYNTHETIC DnCNN test vector + numpy host reference.

Produces dncnn3_input.bin (64x64 uint8) and dncnn3_weights.bin (int8, in the
exact layout dncnn3_vpu_fp_argbuf.c expects), plus dncnn_reference.npy — the
expected output computed by replicating the kernel's math in numpy.

Note: these are synthetic random weights (fixed seed) shaped to exercise the
5-layer 16-channel bench kernel, NOT the trained deepinv/dncnn weights. Purpose
is a deterministic correctness gate: diff the emulator's output region against
dncnn_reference.npy (target max_abs <= 1).
"""
import numpy as np
from pathlib import Path

IMG_W = IMG_H = 64
CH    = 16                 # hidden channels
K     = 3                  # 3×3 conv
LAYERS = 5                 # conv_first + 3×conv_hidden + conv_final
FIRST_SCALE  = 1.0/128.0
HIDDEN_SCALE = 1.0/256.0
FINAL_SCALE  = 1.0/16.0

W0_BYTES = CH*K*K              # 144
WH_BYTES = 3*CH*CH*K*K         # 6912   (LAYERS-2)=3 hidden layers
WF_BYTES = CH*K*K              # 144
WEIGHTS_BYTES = W0_BYTES + WH_BYTES + WF_BYTES   # 7200
INPUT_BYTES   = IMG_W * IMG_H                    # 4096
assert WEIGHTS_BYTES == 7200 and INPUT_BYTES == 4096

# Anchor output dir to the script location so cwd doesn't matter.
REPO = Path(__file__).resolve().parents[3]       # scripts→dncnn→ported_models→repo root
OUT  = REPO / "local-artifacts/erbium_amp_probe/dncnn3-bench"

def make_input(rng):
    return rng.integers(0, 256, size=(IMG_H, IMG_W), dtype=np.uint8)

def make_weights(rng):          
    W0 = rng.integers(-8,  8,  size=(CH, K, K),       dtype=np.int8)   # first: input is ±128, small W0 keeps range sane
    WH = rng.integers(-64, 64, size=(3, CH, CH, K*K), dtype=np.int8)   # hidden: bigger, else 1/256 scale kills the signal
    WF = rng.integers(-8,  8,  size=(CH, K, K),       dtype=np.int8)   # final: small, keeps output inside 0..255 (no clip)
    return W0, WH, WF

def write_weights_bin(W0,WH,WF,path): 
    blob = np.concatenate([W0.ravel(), WH.ravel(), WF.ravel()])
    assert blob.dtype == np.int8, blob.dtype
    assert blob.nbytes == 7200, blob.nbytes      # 144 + 6912 + 144
    blob.tofile(path)

def relu(x): return np.maximum(x, 0.0)
def conv_first(img, W0):  
    x = img.astype(np.float32) - 128.0
    xp = np.pad(x, ((1,1),(1,1)), mode='edge')  # pad to (66,66)
    out = np.zeros((IMG_H, IMG_W, CH), dtype=np.float32)
    for oc in range(CH):
        acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        for ky in range(-1, 2):  
            for kx in range(-1, 2):
                window = xp[1+ky:1+ky+IMG_H, 1+kx:1+kx+IMG_W]
                acc += window * W0[oc, ky+1, kx+1]
        out[:,:,oc] = np.maximum(acc * FIRST_SCALE, 0.0)
    return out              

def conv_hidden(act, Wl):
    xp = np.pad(act, ((1,1),(1,1),(0,0)), mode='edge')  # pad to (66,66,CH)
    out = np.zeros((IMG_H, IMG_W, CH), dtype=np.float32)
    for oc in range(CH):
        acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        for ky in range(-1, 2):  
            for kx in range(-1, 2):
                k = (ky+1)*3 + (kx+1)
                window = xp[1+ky:1+ky+IMG_H, 1+kx:1+kx+IMG_W, :]
                acc += (window * Wl[oc, :, k]).sum(axis=2)
        out[:,:,oc] = np.maximum(acc * HIDDEN_SCALE, 0.0)
    return out

def conv_final(act, WF):        # -> (64,64) uint8
    xp = np.pad(act, ((1,1),(1,1),(0,0)), mode='edge')  # pad to (66,66,CH)
    acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for ky in range(-1, 2):  
        for kx in range(-1, 2):
           window = xp[1+ky:1+ky+IMG_H, 1+kx:1+kx+IMG_W, :]
           acc += (window * WF[:, ky+1, kx+1]).sum(axis=2)
    val = 128.0 + acc * FINAL_SCALE
    return np.clip(np.floor(val + 0.5), 0, 255).astype(np.uint8)     

def host_reference(img,W0,WH,WF):
    act = conv_first(img, W0)
    for l in range(3): act = conv_hidden(act, WH[l])
    return conv_final(act, WF)

def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    img = make_input(rng);  W0,WH,WF = make_weights(rng)
    img.tofile(OUT/"dncnn3_input.bin")
    write_weights_bin(W0,WH,WF, OUT/"dncnn3_weights.bin")
    ref = host_reference(img, W0, WH, WF)
    np.save(OUT / "dncnn_reference.npy", ref)
    print("input", (OUT/'dncnn3_input.bin').stat().st_size,
          "weights", (OUT/'dncnn3_weights.bin').stat().st_size)
    print("reference", ref.shape, ref.dtype,
          "min", int(ref.min()), "max", int(ref.max()),
          "std", round(float(ref.std()),1), "unique", len(np.unique(ref)))

if __name__ == "__main__": main()