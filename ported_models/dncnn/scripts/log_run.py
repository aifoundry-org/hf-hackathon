#!/usr/bin/env python3
"""Append one experiment row to the DnCNN journal.

Optionally computes the correctness (max_abs vs host reference) straight from a
sys-emu dump, so logging a run is a single command.

Examples
--------
# sys-emu run: compute correctness from the dump, no perf number yet
python3 ported_models/dncnn/scripts/log_run.py \
    --id v4_00_base --hypothesis "canonical baseline" \
    --flags "-DDNCNN_PASSES=1 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1" \
    --dump local-artifacts/dncnn_dump.bin \
    --ref  local-artifacts/erbium_amp_probe/dncnn3-bench/dncnn_reference.npy \
    --device sysemu --decision baseline --notes "first run"

# board run: correctness already known, log the perf number
python3 ported_models/dncnn/scripts/log_run.py \
    --id v4_01_tilewpack --hypothesis "tile weights into scratchpad" \
    --flags "+ -DDNCNN_TILE_WPACK=1" --maxabs 0 \
    --metric 1.90 --device board --decision promote --notes "1.28x"
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_JOURNAL = REPO / "local-artifacts/experiments.md"
OUT_OFFSET = 0x10000        # device offset where the DnCNN kernel writes its output
OUT_BYTES = 64 * 64         # 4096-pixel uint8 image
GATE = 1                    # max_abs LSB tolerance for a PASS (FP32 rounding ~±1)

TABLE_HEADER = (
    "| id | date | device | hypothesis | flags (Δ from base) | correctness "
    "| kernel_wait_s | Δ vs base | decision | notes / dead-ends |"
)


def correctness_from_dump(dump_path: Path, ref_path: Path) -> str:
    """Return a 'PASS/FAIL(max_abs=N)' string from an emulator dump vs reference."""
    if not dump_path.exists():
        return "NO-DUMP"
    dump = np.fromfile(dump_path, dtype=np.uint8)
    if dump.size < OUT_OFFSET + OUT_BYTES:
        return f"DUMP-TOO-SMALL({dump.size}B)"
    ref = np.load(ref_path)
    emu = dump[OUT_OFFSET:OUT_OFFSET + OUT_BYTES].reshape(64, 64)
    max_abs = int(np.abs(emu.astype(int) - ref.astype(int)).max())
    verdict = "PASS" if max_abs <= GATE else "FAIL"
    return f"{verdict}(max_abs={max_abs})"


def cell(s: str) -> str:
    """Make a value safe for a markdown table cell."""
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Append a row to the DnCNN experiment journal.")
    ap.add_argument("--id", required=True, help="variant id, e.g. v4_01_tilewpack")
    ap.add_argument("--hypothesis", required=True, help="what you changed and why")
    ap.add_argument("--flags", default="", help="build-flag diff from baseline")
    ap.add_argument("--device", choices=["sysemu", "board"], required=True)
    ap.add_argument("--decision", default="park",
                    choices=["baseline", "promote", "revert", "park"])
    ap.add_argument("--notes", default="", help="notes, incl. dead-ends")
    ap.add_argument("--metric", default="NA",
                    help="kernel_wait_s (board only; sysemu timing is not valid)")
    ap.add_argument("--speedup", default="", help="Δ vs base, e.g. 1.28x (optional)")
    # correctness: either compute from a dump, or supply --maxabs directly
    ap.add_argument("--dump", type=Path, help="sys-emu dump to score correctness from")
    ap.add_argument("--ref", type=Path,
                    default=REPO / "local-artifacts/erbium_amp_probe/dncnn3-bench/dncnn_reference.npy")
    ap.add_argument("--maxabs", type=int, help="known max_abs (skip dump scoring)")
    ap.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    args = ap.parse_args()

    if args.maxabs is not None:
        correctness = f"{'PASS' if args.maxabs <= GATE else 'FAIL'}(max_abs={args.maxabs})"
    elif args.dump is not None:
        correctness = correctness_from_dump(args.dump, args.ref)
    else:
        correctness = "n/a"

    # sys-emu timing is not a valid perf number; force NA regardless of --metric.
    metric = args.metric
    if args.device == "sysemu" and metric not in ("NA", "", None):
        print("note: sysemu timing is not a valid perf number; recording metric as NA",
              file=sys.stderr)
        metric = "NA"

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = "| " + " | ".join(cell(x) for x in [
        args.id, date, args.device, args.hypothesis, args.flags or "—",
        correctness, metric or "NA", args.speedup or "—", args.decision,
        args.notes or "—",
    ]) + " |"

    journal = args.journal
    if not journal.exists():
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(
            "# DnCNN Optimization Journal\n\n"
            "## Experiment log\n\n" + TABLE_HEADER + "\n"
            + "|" + "---|" * 10 + "\n"
        )
    with journal.open("a") as f:
        f.write(row + "\n")

    print("logged →", journal)
    print(TABLE_HEADER)
    print(row)
    if correctness.startswith("FAIL"):
        print("\n⚠ correctness FAILED — do not make a perf claim for this variant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
