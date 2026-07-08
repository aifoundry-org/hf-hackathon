# ET-SoC1 performance-counter harness (DnCNN)

A measurement environment for the int8 DnCNN kernel built on the chip's own
hardware performance monitors. It answers *where the time goes* - instruction
efficiency, L2 miss behaviour, and L2/DDR traffic - so the overhead-bound kernel
can be tuned against real counters instead of guesses.

## What the hardware exposes

The ETSoC-1 PMU (see `et-platform/et-common-libs/include/etsoc/drivers/pmu/pmu.h`
and `device-minion-runtime/.../MachineMinion/src/main.c: mm_setup_default_pmcs`)
has three tiers of counters. The firmware programs a fixed default event on each
at boot and leaves them free-running; U-mode kernels can read the minion HPMs
directly and sample the shire/memshire PMCs through a syscall.

### 1. Minion core counters - per hart, `csrr hpmcounterN` (U-mode)

This table is the **authoritative** hpm3-hpm8 event map. The mirrors in
`pmc_probe.h`, `decode_pmc.py` (docstring + `HPM_LABEL`) must be kept in sync
with it; if the firmware default event map changes, update it here first.

| CSR  | Default event (firmware)      | Scope                     |
|------|-------------------------------|---------------------------|
| hpm3 | `MINION_EVENT_CYCLES`         | minion-0 of a neigh only  |
| hpm4 | `RETIRED_INST0` (thread 0)    | every hart                |
| hpm5 | `RETIRED_INST1` (thread 1)    | every hart                |
| hpm6 | `L2_MISS_REQ`                 | every hart                |
| hpm7 | `MINION_ICACHE_REQ` (neigh)   | neigh-lead hart only      |
| hpm8 | `ICACHE_ETLINK_REQ` (neigh)   | neigh-lead hart only      |

Only hpm4/5/6 are meaningful on every active hart. hpm3 (cycles) and hpm7/8
(icache, neighborhood events) are enabled only on the neighborhood lead - in the
canonical 8-hart layout that is hart 0. Reading uses the RTLMIN-6496 work-around
(four back-to-back `csrr` inside a half cacheline).

The minion can count many more events by reprogramming `mhpmevent3..8` - the full
menu (dcache access/miss, `TFMA_WAIT_TENB`, `TIMA_OPS`, `TXFMA_INT_OPS`,
`TQUANT_INST`, `TREDUCE_INST`, coop load/store, PTW, ...) is in `pmu.h`. **But
`mhpmevent*` are M-mode CSRs and this backend exposes no syscall to set them, so
a U-mode kernel is limited to the six firmware defaults above.** Getting at
`TFMA_WAIT_TENB` (tensor-engine stall cycles - the number we'd most like for a
compute-bound analysis) would need a firmware change.

### 2. Shire-cache (L2) PMCs - per neighborhood bank, via syscall

`syscall(SYSCALL_PMC_SC_SAMPLE, shire, bank, pmc)` returns one counter:
`pmc 0` = cycles, `pmc 1` = P0 = **all L2 reads**, `pmc 2` = P1 = **all L2 writes**
(the firmware programs `PMU_SC_L2_READS`/`PMU_SC_L2_WRITES`). These are
shire-wide, so they capture the traffic of every hart on the shire - exactly the
aggregate L2 bandwidth we want.

### 3. Memshire (DDR) PMCs - per memory shire, via syscall

`syscall(SYSCALL_PMC_MS_SAMPLE, ms, pmc, 0)` for `ms` in 0..7:
`pmc 1` = P0 = **all mesh reads**, `pmc 2` = P1 = **all mesh writes** to that DDR
controller (`PMU_MS_QUAL_ALL_MESH_READS/WRITES`). This is off-chip memory traffic.

> The SC/MS syscalls are U-mode-threshold calls (#9, #10) serviced by firmware in
> M-mode. If the launcher's firmware build doesn't service them the call returns
> `-1`; the probe records that and the decoder prints "UNAVAILABLE" instead of
> garbage. The per-hart minion counters never depend on the syscall.

## How the harness works

- **`src/pmc_probe.h`** - header-only probe, entirely behind `-DDNCNN_PMC`. With
  the flag off it compiles to nothing, so the leaderboard ELF is unchanged
  (verified: `.text` byte-identical to the pristine kernel). With the flag on,
  `pmc_probe_begin/end` bracket the 5-layer network: every active hart snapshots
  hpm3-8 at entry/exit, hart 0 additionally samples all SC and MS PMCs. Deltas
  are written to a fixed DRAM region at **`0xC0000`** (in the unused
  `0x8A000..0xD0000` gap), each hart flushing only its own cache line.
- **`local-artifacts/build_pmc_probe.sh`** - builds `board/int8_pmc.elf`
  (`-DDNCNN_PMC`, canonical 8-hart flags) plus a no-probe `int8_plain.elf` to
  confirm the probe is a no-op.
- **`local-artifacts/run_pmc_board.sh`** - runs `int8_pmc.elf` on the board with
  a `--dump_size` large enough to include `0xC0000`, pulls the dump, and decodes.
- **`local-artifacts/decode_pmc.py`** - parses the region and derives IPC,
  MAC/cycle, MAC/retired-inst, GMAC/s (with `--wall-seconds`), L2 accesses/MAC,
  and DDR read/write totals.

## Running it

```bash
bash local-artifacts/build_pmc_probe.sh     # -> local-artifacts/board/int8_pmc.elf
bash local-artifacts/run_pmc_board.sh       # board run + report (needs board pw once)
# or decode an existing dump directly:
python3 local-artifacts/decode_pmc.py <dump.bin> --wall-seconds 0.0104
```

## Reading the numbers

- **MAC/cycle far below the TFMA peak** confirms the kernel is sync/overhead-bound
  (the working hypothesis): time is going to cache-op stalls and barriers, not the
  tensor engine.
- **L2 miss (hpm6) per hart** localizes which band eats memory stalls; a hart with
  disproportionate misses points at a seam or a non-resident operand.
- **L2 reads/writes vs DDR reads/writes** shows how much of the traffic the L2
  absorbs. Redundant evicts (e.g. hart-0's whole-buffer evict) show up as excess
  L2 writes without matching useful work.
- **IPC on hart 0** bounds how much is compute vs stall on the lead hart.

Cross-check the counter deltas against the leaderboard `kernel_wait_s`; the
implied clock (`hart0 cycles / wall`) should be stable run-to-run.

## Caveats

- Re-verify correctness (`max_abs=0`) and `kernel_wait_s` on the profile build -
  the probe adds a few cache ops and this kernel class is timing-sensitive at
  8 harts. The probe build is for *measurement*, never for submission.
- hpm3/7/8 are valid only on the neighborhood lead (hart 0 here); the decoder only
  reports them there.
- SC/MS counters are shire/DDR-wide, not per-hart; they attribute traffic to the
  whole shire, which for our single-shire run is the total.
