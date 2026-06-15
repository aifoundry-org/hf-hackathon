#!/usr/bin/env python3
import csv
import glob
import os
import re
import struct

YOLO_MAGIC = 0x10500001
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
        "width": fields[3],
        "height": fields[4],
        "channels": fields[5],
        "blocks": fields[6],
        "active_mask": fields[7],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
        "head_channels": fields[13],
    }


def variants(root, variant_file):
    with open(os.path.join(root, variant_file)) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    root = os.environ.get("YOLO_ROOT", ".")
    variant_file = os.environ.get("YOLO_VARIANTS", "yolo_10_variants.txt")
    out_name = os.environ.get("YOLO_RESULTS", "yolo_10_results.tsv")
    rows = []
    for variant in variants(root, variant_file):
        logs = sorted(glob.glob(os.path.join(root, f"run_{variant}_*.log")))
        dumps = sorted(glob.glob(os.path.join(root, f"dump_{variant}_*.bin")))
        if not logs or not dumps:
            continue
        s = summary(dumps[-1])
        wait = wait_seconds(logs[-1])
        ops = s["ops"]
        valid = (
            s["magic"] == YOLO_MAGIC and
            s["done_count"] == s["active_harts"] and
            s["output_sum"] == s["slot_sum"]
        )
        rows.append({
            "variant": variant,
            "wait_s": f"{wait:.6f}",
            "valid": int(valid),
            "passes": s["passes"],
            "harts": s["active_harts"],
            "mask": hex(s["active_mask"]),
            "blocks": s["blocks"],
            "output_sum": s["output_sum"],
            "slot_sum": s["slot_sum"],
            "ops": ops,
            "log": os.path.basename(logs[-1]),
            "dump": os.path.basename(dumps[-1]),
        })

    rows.sort(key=lambda r: float(r["wait_s"]))
    out = os.path.join(root, out_name)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
