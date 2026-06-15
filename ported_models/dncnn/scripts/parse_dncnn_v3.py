#!/usr/bin/env python3
import csv
import glob
import os
import re
import struct

DNCNN_MAGIC = 0xD3C11003
SUMMARY = struct.Struct("<16I")


def wait_seconds(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"Kernel wait seconds: ([0-9.]+)", text)
    return float(match.group(1)) if match else 0.0


def summary(path):
    with open(path, "rb") as f:
        data = f.read(0x1000 + SUMMARY.size)
    fields = SUMMARY.unpack_from(data, 0x1000)
    return {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "active_mask": fields[7],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
    }


def variants(root):
    path = os.path.join(root, "v3x_variants.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    root = os.environ.get("DNCNN_V3_ROOT", ".")
    rows = []
    for variant in variants(root):
        logs = sorted(glob.glob(os.path.join(root, f"run_{variant}_*.log")))
        dumps = sorted(glob.glob(os.path.join(root, f"dump_{variant}_*.bin")))
        if not logs or not dumps:
            continue
        s = summary(dumps[-1])
        wait = wait_seconds(logs[-1])
        ops = s["ops"]
        rows.append({
            "variant": variant,
            "wait_s": f"{wait:.6f}",
            "valid": int(s["magic"] == DNCNN_MAGIC and
                         s["done_count"] == s["active_harts"] and
                         s["output_sum"] == s["slot_sum"]),
            "passes": s["passes"],
            "harts": s["active_harts"],
            "mask": hex(s["active_mask"]),
            "output_sum": s["output_sum"],
            "slot_sum": s["slot_sum"],
            "ops": ops,
            "log": os.path.basename(logs[-1]),
            "dump": os.path.basename(dumps[-1]),
        })

    rows.sort(key=lambda r: float(r["wait_s"]))
    out = os.path.join(root, "v3x_results.tsv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
