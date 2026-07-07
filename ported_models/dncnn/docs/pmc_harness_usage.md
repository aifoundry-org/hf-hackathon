# Using the DnCNN PMC harness

A step-by-step guide to profiling the int8 DnCNN kernel with the on-chip
performance counters. For *what each counter measures*, see
[`perf_counters.md`](perf_counters.md); this doc is the workflow.

The harness has two halves:

- **Committed** (portable): `src/pmc_probe.h` (the probe, behind `-DDNCNN_PMC`)
  and `scripts/decode_pmc.py` (the report generator).
- **Local** (environment-specific, gitignored under `local-artifacts/`):
  `build_pmc_probe.sh` and `run_pmc_board.sh`, which hard-code this box's
  toolchain paths and the `ivan@aifoundry2` board. Treat them as templates.

## Prerequisites

- The docker-wrapped ET toolchain at `~/et-src/et` (build must run inside the
  repo tree — `/tmp` is not mounted in the toolchain container).
- Board access over Tailscale (`ivan@aifoundry2`); announce on Discord before use.
- Input/weight blobs under `local-artifacts/erbium_amp_probe/dncnn3-bench/`.

## 1. Build

```bash
bash local-artifacts/build_pmc_probe.sh
```

Produces two ELFs in `local-artifacts/board/`:

- `int8_pmc.elf` — the profile build (`-DDNCNN_PMC`, canonical 8-hart flags).
- `int8_plain.elf` — no probe, used to confirm the probe is a no-op.

**Sanity check the probe is free when off** — its `.text` must match the
pristine kernel byte-for-byte:

```bash
# build the pristine kernel the same way, then:
diff <(riscv64-unknown-elf-objdump -d int8_plain.elf | sed -E 's/^\s*[0-9a-f]+:\s+//') \
     <(riscv64-unknown-elf-objdump -d pristine.elf   | sed -E 's/^\s*[0-9a-f]+:\s+//')
```

## 2. Run on the board

```bash
bash local-artifacts/run_pmc_board.sh      # asks for the board password once
```

This copies `int8_pmc.elf` + blobs to the board, runs the launcher with a
`--dump_size` large enough to include the PMC region at `0xC0000`, pulls the
dump to `local-artifacts/pmc_board_dump.bin`, parses `kernel_wait_s` from the
log, and prints the report. The launcher's `Kernel wait seconds` line and the
dump are genuine — only the ELF differs from the leaderboard build.

## 3. Read the report

```
  hart minion thr |   ret_inst_t0    L2_miss   ret_inst_t1  | (cyc/icache: hart-lead only)
     0      0   0 |    15,715,559     240,637             0 | cyc=5,962,792 icache=2,914,214 ietlink=49
  ...
  -- derived --
  hart0 IPC (thread 0)       : 2.636
  MAC / retired-inst (all)   : 0.23
  throughput                 : 2.748 GMAC/s
  -- shire 0 L2 cache --  ... L2 total: reads=406,554 writes=126,619
  -- memshire DDR --      ... DDR total: reads=37,500  writes=36,380
```

What to look at, and what it means:

| Signal | Reads as |
|--------|----------|
| **MAC / retired-inst** low (≪1) | scalar-instruction-bound — the marshalling around the tensor op dominates, not the MACs |
| **IPC** near issue width | harts are executing, not stalling/spinning (→ not sync-bound) |
| **per-hart spread** small | work is balanced; no straggler band to fix |
| **L2 miss (hpm6)** skewed on one hart | that band hits a seam or a non-resident operand |
| **DDR reads+writes** small | not memory-bound; data lives in cache |
| **icache req vs ietlink (misses)** | icache pressure; ietlink≈0 means the code fits |

Cross-check `hart0 cycles / wall` (the implied clock) run-to-run for stability.

## 4. Verify correctness of the profile build

The probe adds a few cache ops and this kernel races at band seams if sync is
wrong, so confirm the profile build is still bit-exact before trusting timing:

```bash
python3 local-artifacts/check_int8_seams.py local-artifacts/pmc_board_dump.bin 8   # expect max_abs=0
```

## 5. Decode an existing dump directly

```bash
python3 ported_models/dncnn/scripts/decode_pmc.py <dump.bin> --wall-seconds 0.0107
# --base 0xC0000   PMC region offset (default)
# --macs N         override the MAC/pass used for the ratios
```

## Iterating on the kernel

The intended loop: change the kernel → rebuild → run → compare the report. The
number to drive down is **retired instructions per hart** (equivalently, MAC /
retired-inst upward); watch that `kernel_wait_s` falls with it and `max_abs`
stays 0. The probe costs ~3% wall time, so compare probe-build to probe-build,
and take the leaderboard number from the plain build.

## Extending the probe

- **Per-layer granularity:** call `pmc_probe_begin/end` around a single layer
  instead of the whole network (add more region slots). Costs extra cache ops in
  the hot loop, so use sparingly.
- **Other minion events** (dcache, `TFMA_WAIT_TENB`, `TQUANT_INST`): these need
  `mhpmevent*` reprogrammed, which is M-mode with no syscall on this backend —
  it would require a firmware change. See `perf_counters.md`.
