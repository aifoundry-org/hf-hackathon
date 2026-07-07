#!/usr/bin/env python3
"""
decode_pmc.py — turn a DnCNN -DDNCNN_PMC board dump into a performance report.

The instrumented kernel (pmc_probe.h) writes a fixed region at 0xC0000 holding,
per active hart, the six default minion HPM counters snapshotted at the start and
end of the 5-layer network, plus (on hart 0) the shire-cache (L2) and memshire
(DDR) PMCs. This decodes those deltas and derives IPC, MAC efficiency, L2 miss
rate and memory-traffic numbers.

Usage:
  decode_pmc.py <dump.bin> [--wall-seconds S] [--base 0xC0000] [--macs N]

Counter map (firmware defaults, device-minion-runtime mm_setup_default_pmcs):
  hpm3 minion cycles (minion-0 of a neigh only) | hpm4 retired inst t0 (all harts)
  hpm5 retired inst t1 (all harts)              | hpm6 L2 miss requests (all harts)
  hpm7 minion icache req (neigh-lead only)      | hpm8 icache etlink req (neigh-lead)
  SC PMCs: P0=L2 reads, P1=L2 writes (per bank) | MS PMCs: P0=mesh reads, P1=mesh writes
"""
import struct
import sys

PMC_MAGIC      = 0x504D4331
PMC_HART_MAGIC = 0x504D4348
PMC_AGG_MAGIC  = 0x504D4341
HPM_COUNT   = 6
MAX_HARTS   = 32
SC_BANKS    = 4
MS_COUNT    = 8
PMC_PER     = 3          # cycle, p0, p1
HART_REC    = 128        # bytes per hart record
HDR_BYTES   = 128
AGG_OFF     = HDR_BYTES + MAX_HARTS * HART_REC   # 4224
ERR         = (1 << 64) - 1                      # -1 sentinel

# ~29.5 MMAC for the 64x64x16 5-layer net (one pass): conv_first + 3 hidden + conv_final
DEFAULT_MACS = 64 * 64 * (16 * 9 + 3 * 16 * 16 * 9 + 16 * 9)

HPM_LABEL = ["cycles(hpm3)", "ret_t0(hpm4)", "ret_t1(hpm5)",
             "L2miss(hpm6)", "icache(hpm7)", "ietlink(hpm8)"]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def delta(end, start):
    """Wrap-aware 64-bit delta; ERR sentinels -> None."""
    if end == ERR or start == ERR:
        return None
    return (end - start) & ERR


def parse_hart(b, base, i):
    o = base + HDR_BYTES + i * HART_REC
    if u32(b, o) != PMC_HART_MAGIC:
        return None
    hstart = [u64(b, o + 16 + 8 * k) for k in range(HPM_COUNT)]
    hend = [u64(b, o + 16 + 8 * (HPM_COUNT + k)) for k in range(HPM_COUNT)]
    return {
        "hart_id": u32(b, o + 4),
        "minion_id": u32(b, o + 8),
        "thread_id": u32(b, o + 12),
        "d": [delta(hend[k], hstart[k]) for k in range(HPM_COUNT)],
    }


def parse_agg_block(b, o, n):
    """n blocks of PMC_PER u64 starting at o."""
    return [[u64(b, o + (blk * PMC_PER + p) * 8) for p in range(PMC_PER)]
            for blk in range(n)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    wall = None
    base = 0xC0000
    macs = DEFAULT_MACS
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--wall-seconds":
            wall = float(args[i + 1]); i += 2
        elif a == "--base":
            base = int(args[i + 1], 0); i += 2
        elif a == "--macs":
            macs = int(args[i + 1], 0); i += 2
        else:
            print(f"unknown arg {a}"); sys.exit(1)

    with open(path, "rb") as f:
        b = f.read()
    if len(b) < base + AGG_OFF + 600:
        print(f"ERROR: dump is {len(b)} bytes; need >= {base + AGG_OFF + 600}. "
              f"Re-run the board with a larger --dump_size (>= 0x{base + 0x2000:X}).")
        sys.exit(2)

    magic = u32(b, base)
    if magic != PMC_MAGIC:
        print(f"ERROR: no PMC region at 0x{base:X} (magic 0x{magic:08X} != 0x{PMC_MAGIC:08X}). "
              f"Was the ELF built with -DDNCNN_PMC and dumped past 0x{base:X}?")
        sys.exit(2)
    active = u32(b, base + 4)
    version = u32(b, base + 12)

    print(f"== DnCNN PMC report ==  (region 0x{base:X}, v{version}, active_harts={active})")
    print(f"   workload: {macs:,} MAC/pass" + (f"   wall={wall*1e3:.4f} ms" if wall else ""))
    print()

    # kernel stores at slot = compacted hart id (0..active-1); scan all slots to
    # stay correct even if that packing ever changes.
    harts = [h for h in (parse_hart(b, base, i) for i in range(MAX_HARTS)) if h]
    harts.sort(key=lambda h: h["hart_id"])

    # ---- per-hart minion counters ----
    print("  hart minion thr |   ret_inst_t0    L2_miss   ret_inst_t1  | (cyc/icache: hart-lead only)")
    print("  ---- ------ --- | ------------- ----------- ------------- |")
    tot_ret0 = tot_l2miss = 0
    hart0 = None
    for h in harts:
        d = h["d"]
        ret0 = d[1] or 0
        l2m = d[3] or 0
        ret1 = d[2] or 0
        tot_ret0 += ret0
        tot_l2miss += l2m
        if h["hart_id"] == 0:
            hart0 = h
        extra = ""
        if d[0]:  # cycles present -> neigh-lead hart
            extra = f"cyc={d[0]:,} icache={d[4] or 0:,} ietlink={d[5] or 0:,}"
        print(f"  {h['hart_id']:>4} {h['minion_id']:>6} {h['thread_id']:>3} | "
              f"{ret0:>13,} {l2m:>11,} {ret1:>13,} | {extra}")
    print(f"  {'TOTAL':>4} {'':>6} {'':>3} | {tot_ret0:>13,} {tot_l2miss:>11,} {'':>13} |")
    print()

    # ---- derived compute efficiency ----
    print("  -- derived --")
    if hart0 and hart0["d"][0]:
        cyc = hart0["d"][0]
        print(f"  hart0 cycles (region)      : {cyc:,}")
        if hart0["d"][1]:
            print(f"  hart0 IPC (thread 0)       : {hart0['d'][1] / cyc:.3f}")
        print(f"  MAC / hart0-cycle          : {macs / cyc:.2f}   "
              f"(peak int8 TFMA is many MAC/cyc; low => sync/overhead-bound)")
        if wall:
            freq = cyc / wall
            print(f"  implied clock              : {freq/1e9:.3f} GHz  (hart0 cycles / wall)")
    if tot_ret0:
        print(f"  MAC / retired-inst (all)   : {macs / tot_ret0:.2f}")
    if wall:
        print(f"  throughput                 : {macs / wall / 1e9:.3f} GMAC/s")
    print()

    # ---- shire-cache (L2) + memshire (DDR) ----
    ao = base + AGG_OFF
    if u32(b, ao) != PMC_AGG_MAGIC:
        print("  (no aggregate SC/MS record)")
        return
    shire = u32(b, ao + 4)
    sc_ok = u32(b, ao + 8)
    ms_ok = u32(b, ao + 12)
    body = ao + 16
    sc_start = parse_agg_block(b, body, SC_BANKS)
    sc_end   = parse_agg_block(b, body + SC_BANKS * PMC_PER * 8, SC_BANKS)
    ms_base  = body + 2 * SC_BANKS * PMC_PER * 8
    ms_start = parse_agg_block(b, ms_base, MS_COUNT)
    ms_end   = parse_agg_block(b, ms_base + MS_COUNT * PMC_PER * 8, MS_COUNT)

    print(f"  -- shire {shire} L2 cache (P0=reads P1=writes, per neigh bank) --")
    if not sc_ok:
        print("  SC counters UNAVAILABLE — the launcher firmware did not service "
              "SYSCALL_PMC_SC_SAMPLE (returned -1).")
    else:
        tr = tw = 0
        for nb in range(SC_BANKS):
            rd = delta(sc_end[nb][1], sc_start[nb][1])
            wr = delta(sc_end[nb][2], sc_start[nb][2])
            cy = delta(sc_end[nb][0], sc_start[nb][0])
            if rd:
                tr += rd
            if wr:
                tw += wr
            if (rd or wr):
                print(f"    bank {nb}: reads={rd or 0:>12,} writes={wr or 0:>12,} cyc={cy or 0:,}")
        print(f"    L2 total: reads={tr:,}  writes={tw:,}  (each ~= one 64B line access)")
        if macs:
            print(f"    L2 accesses / MAC : {(tr + tw) / macs:.4f}")

    print(f"  -- memshire DDR (P0=mesh reads P1=mesh writes, per shire) --")
    if not ms_ok:
        print("  MS counters UNAVAILABLE — the launcher firmware did not service "
              "SYSCALL_PMC_MS_SAMPLE (returned -1).")
    else:
        tr = tw = 0
        for ms in range(MS_COUNT):
            rd = delta(ms_end[ms][1], ms_start[ms][1])
            wr = delta(ms_end[ms][2], ms_start[ms][2])
            if rd:
                tr += rd
            if wr:
                tw += wr
            if (rd or wr):
                print(f"    ms {ms}: reads={rd or 0:>12,} writes={wr or 0:>12,}")
        print(f"    DDR total: reads={tr:,}  writes={tw:,}")


if __name__ == "__main__":
    main()
