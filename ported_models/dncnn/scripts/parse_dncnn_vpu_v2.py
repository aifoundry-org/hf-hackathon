#!/usr/bin/env python3
import csv
import glob
import os
import re
import struct

DNCNN_MAGIC = 0xD3C11003
SUMMARY = struct.Struct("<16I")


def wait_seconds(path):
    with open(path, errors="ignore") as f:
        text = f.read()
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
        "layers": fields[6],
        "active_mask": fields[7],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_checksum_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
    }


def main():
    root = os.environ.get("DNCNN_VPU_V2_ROOT", ".")
    variants = [
        "vpuv2_00_baseline_sharedw",
        "vpuv2_01_acc3x3_sharedw",
        "vpuv2_02_acc3x3_sharedw_lastonly",
        "vpuv2_03_acc3x3_noshared",
        "vpuv2_04_acc3x3_p1_sharedw",
        "vpuv2_05_acc3x3_private_sharedw",
        "vpuv2_06_oc2_sharedw",
        "vpuv2_07_oc2_sharedw_lastonly",
        "vpuv2_08_oc2_accfinal_sharedw",
        "vpuv2_09_oc2_accfinal_lastonly",
        "vpuv2_scale_acc3x3_sharedw_1h",
        "vpuv2_scale_acc3x3_sharedw_2h",
        "vpuv2_scale_acc3x3_sharedw_4h",
        "vpuv2_scale_acc3x3_sharedw_8h",
        "vpuv2_scale_acc3x3_sharedw_16h",
        "vpuv2_scale_oc2_sharedw_1h",
        "vpuv2_scale_oc2_sharedw_2h",
        "vpuv2_scale_oc2_sharedw_4h",
        "vpuv2_scale_oc2_sharedw_8h",
        "vpuv2_scale_oc2_sharedw_16h",
        "vpuv2_scale_oc2_accfinal_1h",
        "vpuv2_scale_oc2_accfinal_2h",
        "vpuv2_scale_oc2_accfinal_4h",
        "vpuv2_scale_oc2_accfinal_8h",
        "vpuv2_scale_oc2_accfinal_16h",
    ]
    rows = []

    for variant in variants:
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
                         s["output_sum"] == s["slot_checksum_sum"]),
            "passes": s["passes"],
            "harts": s["active_harts"],
            "mask": hex(s["active_mask"]),
            "output_sum": s["output_sum"],
            "slot_sum": s["slot_checksum_sum"],
            "ops": ops,
            "log": os.path.basename(logs[-1]),
            "dump": os.path.basename(dumps[-1]),
        })

    out = os.path.join(root, "vpuv2_results.tsv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
