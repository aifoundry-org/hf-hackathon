#!/usr/bin/env python3
import csv
import glob
import os
import re
import struct
from collections import defaultdict

SNAP_DIRECT_MAGIC = 0x504D5344
SNAP_SYSCALL_MAGIC = 0x504D5359
DNCNN_MAGIC = 0xD3C11003
DIRECT_STRUCT = struct.Struct("<8I6Q")
SYSCALL_STRUCT = struct.Struct("<8IQ")
DNCNN_SUMMARY_STRUCT = struct.Struct("<16I")


def parse_wait(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"Kernel wait seconds: ([0-9.]+)", text)
    return float(match.group(1)) if match else 0.0


def delta_u64(after, before):
    return (after - before) & ((1 << 64) - 1)


def read_snapshot(path):
    with open(path, "rb") as f:
        data = f.read()
    direct = {}
    for h in range(16):
        fields = DIRECT_STRUCT.unpack_from(data, h * DIRECT_STRUCT.size)
        if fields[0] != SNAP_DIRECT_MAGIC:
            continue
        direct[fields[1]] = {
            "hart": fields[1],
            "minion": fields[2],
            "thread": fields[3],
            "hpm": fields[8:14],
        }
    syscalls = {}
    base = 0x1000
    for i in range(36):
        fields = SYSCALL_STRUCT.unpack_from(data, base + i * SYSCALL_STRUCT.size)
        if fields[0] != SNAP_SYSCALL_MAGIC:
            continue
        _, kind, block, pmc, _, _, _, _, value = fields
        syscalls[(kind, block, pmc)] = value
    return direct, syscalls


def parse_summary(path):
    with open(path, "rb") as f:
        data = f.read(0x1000 + DNCNN_SUMMARY_STRUCT.size)
    fields = DNCNN_SUMMARY_STRUCT.unpack_from(data, 0x1000)
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


def direct_metrics(before, after, active_harts):
    rows = []
    by_minion = defaultdict(dict)
    for h in range(active_harts):
        if h not in before or h not in after:
            continue
        d = [delta_u64(after[h]["hpm"][i], before[h]["hpm"][i]) for i in range(6)]
        rows.append((h, before[h]["minion"], before[h]["thread"], d))
        by_minion[before[h]["minion"]][before[h]["thread"]] = d

    dedup_inst = 0
    for threads in by_minion.values():
        if 0 in threads:
            dedup_inst += threads[0][1]
        if 1 in threads:
            dedup_inst += threads[1][2]

    return {
        "hpm_cycles_max": max((d[0] for _, _, _, d in rows), default=0),
        "hpm_cycles_sum": sum(d[0] for _, _, _, d in rows),
        "hpm_inst_dedup": dedup_inst,
        "hpm_l2_miss_sum": sum(d[3] for _, _, _, d in rows),
        "hpm_icache_req_sum": sum(d[4] for _, _, _, d in rows),
        "hpm_icache_etlink_sum": sum(d[5] for _, _, _, d in rows),
    }


def syscall_metrics(before, after):
    out = {}
    for label, kind in (("sc", 0), ("ms", 1)):
        for pmc_label, pmc in (("cycles", 0), ("reads", 1), ("writes", 2)):
            vals = [
                delta_u64(after[k], v)
                for k, v in before.items()
                if k[0] == kind and k[2] == pmc and k in after
            ]
            out[f"{label}_{pmc_label}_sum"] = sum(vals)
            out[f"{label}_{pmc_label}_max"] = max(vals) if vals else 0
    return out


def main():
    root = os.environ.get("DNCNN_VPU_V2_PMC_ROOT", ".")
    variants = [
        "vpuv2_00_baseline_sharedw",
        "vpuv2_01_acc3x3_sharedw",
        "vpuv2_06_oc2_sharedw",
    ]
    rows = []

    for variant in variants:
        before_path = os.path.join(root, f"pmc_before_{variant}.bin")
        after_path = os.path.join(root, f"pmc_after_{variant}.bin")
        dump_path = os.path.join(root, f"dump_pmc_dncnn_{variant}.bin")
        logs = sorted(glob.glob(os.path.join(root, f"run_pmc_dncnn_{variant}_*.log")))
        if not (os.path.exists(before_path) and os.path.exists(after_path) and
                os.path.exists(dump_path) and logs):
            continue

        before_direct, before_sys = read_snapshot(before_path)
        after_direct, after_sys = read_snapshot(after_path)
        summary = parse_summary(dump_path)
        wait = parse_wait(logs[-1])
        ops = summary["ops"]
        row = {
            "variant": variant,
            "wait_s": f"{wait:.6f}",
            "valid": int(summary["magic"] == DNCNN_MAGIC and
                         summary["active_harts"] == summary["done_count"] and
                         summary["output_sum"] == summary["slot_sum"]),
            "output_sum": summary["output_sum"],
            "ops": ops,
            "log": os.path.basename(logs[-1]),
        }
        row.update(direct_metrics(before_direct, after_direct, summary["active_harts"]))
        row.update(syscall_metrics(before_sys, after_sys))
        if ops:
            row["cycles_per_op_max"] = f"{row['hpm_cycles_max'] / ops:.9f}"
            row["inst_per_op_dedup"] = f"{row['hpm_inst_dedup'] / ops:.9f}"
            row["l2_miss_per_kop"] = f"{row['hpm_l2_miss_sum'] * 1000.0 / ops:.6f}"
            row["sc_reads_per_kop"] = f"{row['sc_reads_sum'] * 1000.0 / ops:.6f}"
            row["sc_writes_per_kop"] = f"{row['sc_writes_sum'] * 1000.0 / ops:.6f}"
            row["ms_reads_per_kop"] = f"{row['ms_reads_sum'] * 1000.0 / ops:.6f}"
            row["ms_writes_per_kop"] = f"{row['ms_writes_sum'] * 1000.0 / ops:.6f}"
        rows.append(row)

    out = os.path.join(root, "vpuv2_pmc_results.tsv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
