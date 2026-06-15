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

HPM = [
    "cycles",
    "retired_inst0",
    "retired_inst1",
    "l2_miss_req",
    "minion_icache_req",
    "icache_etlink_req",
]


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
        _, kind, block, pmc, shire, hart, _, _, value = fields
        syscalls[(kind, block, pmc)] = value
    return direct, syscalls


def delta_u64(after, before):
    return (after - before) & ((1 << 64) - 1)


def parse_wait(log_path):
    text = open(log_path, errors="ignore").read()
    m = re.search(r"Kernel wait seconds: ([0-9.]+)", text)
    return float(m.group(1)) if m else 0.0


def parse_dncnn_summary(path):
    with open(path, "rb") as f:
        data = f.read(0x1000 + DNCNN_SUMMARY_STRUCT.size)
    fields = DNCNN_SUMMARY_STRUCT.unpack_from(data, 0x1000)
    ops = fields[11] | (fields[12] << 32)
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
        "ops": ops,
    }


def active_harts(n):
    return list(range(n))


def direct_metrics(before, after, n):
    rows = []
    active = active_harts(n)
    by_minion = defaultdict(dict)
    for h in active:
        if h not in before or h not in after:
            continue
        b = before[h]["hpm"]
        a = after[h]["hpm"]
        d = [delta_u64(a[i], b[i]) for i in range(6)]
        rows.append((h, before[h]["minion"], before[h]["thread"], d))
        by_minion[before[h]["minion"]][before[h]["thread"]] = d

    max_cycles = max((d[0] for _, _, _, d in rows), default=0)
    sum_cycles = sum(d[0] for _, _, _, d in rows)

    dedup_inst = 0
    for threads in by_minion.values():
        if 0 in threads:
            dedup_inst += threads[0][1]
        if 1 in threads:
            dedup_inst += threads[1][2]

    return rows, {
        "hpm_cycles_max": max_cycles,
        "hpm_cycles_sum_active": sum_cycles,
        "hpm_inst_dedup": dedup_inst,
        "hpm_l2_miss_max": max((d[3] for _, _, _, d in rows), default=0),
        "hpm_l2_miss_sum_active": sum(d[3] for _, _, _, d in rows),
        "hpm_icache_req_max": max((d[4] for _, _, _, d in rows), default=0),
        "hpm_icache_req_sum_active": sum(d[4] for _, _, _, d in rows),
        "hpm_icache_etlink_max": max((d[5] for _, _, _, d in rows), default=0),
        "hpm_icache_etlink_sum_active": sum(d[5] for _, _, _, d in rows),
    }


def syscall_metrics(before, after):
    out = {}
    for kind_name, kind in (("sc", 0), ("ms", 1)):
        for pmc_name, pmc in (("cycles", 0), ("reads", 1), ("writes", 2)):
            vals = []
            for key, b in before.items():
                k, block, p = key
                if k == kind and p == pmc and key in after:
                    vals.append(delta_u64(after[key], b))
            out[f"{kind_name}_{pmc_name}_sum"] = sum(vals)
            out[f"{kind_name}_{pmc_name}_max"] = max(vals) if vals else 0
    return out


def main():
    root = os.environ.get("DNCNN_PMC_ROOT", ".")
    variants = [
        ("baseline_o2_full_evict", "dncnn3_bench"),
        ("window_o3", "dncnn3_bench_window"),
        ("window_no_prebar_o3", "dncnn3_bench_noprebar"),
        ("nhwc_no_prebar_o3", "dncnn3_bench_nhwc"),
        ("nhwc_interior_direct_o3", "dncnn3_bench_interior"),
        ("nhwc_all_channels_o3_rejected", "dncnn3_bench_allch"),
        ("best", "dncnn3_bench_best"),
        ("boundary_only_cacheops", "dncnn3_bench_boundary"),
        ("hidden_unroll", "dncnn3_bench_unroll"),
        ("skip_final_barrier", "dncnn3_bench_skip_finalbar"),
        ("no_layer_barriers_unsafe", "dncnn3_bench_nolayerbar"),
        ("fused_halo", "dncnn3_bench_fused_halo"),
        ("fused_halo_unroll", "dncnn3_bench_fused_halo_unroll"),
        ("model_ch8", "dncnn3_bench_ch8"),
        ("model_ch4", "dncnn3_bench_ch4"),
    ]
    hart_counts = [1, 2, 4, 8, 16]
    rows = []
    per_hart_rows = []
    for variant, prefix in variants:
        for harts in hart_counts:
            tag = f"{prefix}_{harts}h"
            before_path = os.path.join(root, f"pmc_before_{tag}.bin")
            after_path = os.path.join(root, f"pmc_after_{tag}.bin")
            dump_path = os.path.join(root, f"dump_pmc_dncnn_{tag}.bin")
            logs = sorted(glob.glob(os.path.join(root, f"run_pmc_dncnn_{tag}_*.log")))
            if not (os.path.exists(before_path) and os.path.exists(after_path) and logs):
                continue
            before_direct, before_sys = read_snapshot(before_path)
            after_direct, after_sys = read_snapshot(after_path)
            direct_rows, dm = direct_metrics(before_direct, after_direct, harts)
            sm = syscall_metrics(before_sys, after_sys)
            summary = parse_dncnn_summary(dump_path)
            wait = parse_wait(logs[-1])
            ops = summary["ops"]
            row = {
                "variant": variant,
                "tag": tag,
                "harts": harts,
                "kernel_wait_s": wait,
                "summary_ok": int(summary["magic"] == DNCNN_MAGIC),
                "done_count": summary["done_count"],
                "active_mask": hex(summary["active_mask"]),
                "output_sum": summary["output_sum"],
                "slot_checksum_sum": summary["slot_checksum_sum"],
                "ops": ops,
            }
            row.update(dm)
            row.update(sm)
            if ops:
                row["cycles_per_op_max"] = dm["hpm_cycles_max"] / ops
                row["inst_per_op_dedup"] = dm["hpm_inst_dedup"] / ops
                row["sc_reads_per_kop"] = sm["sc_reads_sum"] * 1000.0 / ops
                row["sc_writes_per_kop"] = sm["sc_writes_sum"] * 1000.0 / ops
                row["ms_reads_per_kop"] = sm["ms_reads_sum"] * 1000.0 / ops
                row["ms_writes_per_kop"] = sm["ms_writes_sum"] * 1000.0 / ops
            rows.append(row)
            for hart, minion, thread, vals in direct_rows:
                pr = {
                    "variant": variant,
                    "tag": tag,
                    "harts": harts,
                    "hart": hart,
                    "minion": minion,
                    "thread": thread,
                }
                for name, val in zip(HPM, vals):
                    pr[name] = val
                per_hart_rows.append(pr)

    os.makedirs(os.path.join(root, "parsed"), exist_ok=True)
    out = os.path.join(root, "parsed", "dncnn_pmc_results.tsv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    out_h = os.path.join(root, "parsed", "dncnn_pmc_per_hart.tsv")
    with open(out_h, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_hart_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(per_hart_rows)
    print(out)
    print(out_h)


if __name__ == "__main__":
    main()
