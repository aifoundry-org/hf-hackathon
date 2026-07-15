#!/usr/bin/env python3
"""Check a dncnn20l64 board dump against the torch/deepinv oracle reference.

Reads the 64x64 denoised output region (@0x10000) from an ET-SoC1 dump and compares it byte-for-byte
against refs/dncnn20l64_reference.npy (the common-runtime torch oracle, built by gen_dncnn_oracle.py).
This is the tracked, reproducible version of the accuracy gate that CI's `uint8_npy` scorer applies;
it also localises any error by hart band seam. Passes when max_abs <= gate (default 2).

Usage: check_board_dump.py <dump.bin> [active_harts] [--gate N] [--emit out.bin]
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFS = HERE.parent / "refs"
IMG = 64
OFFSET = 0x10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("harts", nargs="?", type=int, default=8)
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--ref", default=str(REFS / "dncnn20l64_reference.npy"))
    ap.add_argument("--emit", help="write the extracted 64x64 output region here (for eval_quality --dump)")
    a = ap.parse_args()

    d = Path(a.dump).read_bytes()
    out = np.frombuffer(d[OFFSET:OFFSET + IMG * IMG], np.uint8).reshape(IMG, IMG)
    ref = np.load(a.ref).astype(int)
    per_row = np.abs(out.astype(int) - ref).max(1)
    mx = int(per_row.max())
    mean = float(np.abs(out.astype(int) - ref).mean())
    band = IMG // max(1, a.harts)

    print(f"max_abs={mx} mean_abs={mean:.3f} gate<={a.gate}  ({a.dump}, {a.harts} harts, vs {Path(a.ref).name})")
    if mx > a.gate:
        for y in range(IMG):
            if per_row[y]:
                seam = "SEAM" if (a.harts > 1 and y % band in (0, band - 1)) else ""
                print(f"  row {y:2d} hart {y // band} err {per_row[y]:3d} {seam}")
    if a.emit:
        out.tofile(a.emit)
        print(f"wrote {a.emit}")
    ok = mx <= a.gate
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
