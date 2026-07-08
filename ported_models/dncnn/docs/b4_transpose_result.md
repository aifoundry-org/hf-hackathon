# B4 - transposed-GEMM direct store: board result (rejected)

**Verdict: correct but a net regression (+8.4 % wall). Do not commit.** The change did exactly
what the instruction-count hypothesis predicted - it cut retired instructions ~12 % - but the
saving was overwhelmed by a 1.5-3x increase in memory traffic, so `kernel_wait_s` went *up*.
The experiment is a clean negative result that reframes the optimization plan (see sec 4).

Branch: `dncnn-b4-direct-store` - build: `int8_pmc.elf`, 8-hart canonical - board: `ivan@aifoundry2`
(`soc1sim`) - correctness: **`max_abs=0` (PASS)**.

## 1. What was changed

Transposed the hidden-layer GEMM so the accumulator is **spatial-major** (`C[spatial][OC]`)
instead of OC-major, which let us:

- **Store the tile straight into `padout`'s NHWC interior** with one strided `tensor_store` -
  deleting the per-tile `temp` buffer, its evict, and the 256-iteration scalar transpose scatter
  (the intended B4 win).
- **Pack activations as the plain row-major A operand** (16 contiguous IC bytes per spatial) -
  replacing the 9x256 scalar quartet-interleave with a cheap contiguous copy (the pack_B win).
- Move the quartet-interleave onto the **static weights** (now the B operand), built once per layer.

All FMA/quant field values were unchanged (`CH==P==OC==16`); only operand roles, the two
signedness flags, `MUL_COL->MUL_ROW`, and the pack/store layouts moved. Numerics verified identical.

## 2. Board numbers (both are PMC builds, so directly comparable)

| Metric | Baseline (OC-major) | Transposed | delta  |
|---|---:|---:|---:|
| **`kernel_wait_s` (wall)** | **10.7306 ms** | **11.6368 ms** | **+8.4 % (slower)** |
| retired-inst, total (8 harts) | 127,132,174 | 112,317,973 | **-11.7 %** |
| retired-inst / hart | ~15.9 M | ~14.06 M | -11.6 % |
| hart0 cycles | 5,962,792 | 6,481,341 | +8.7 % |
| hart0 IPC | 2.636 | 2.141 | -18.8 % |
| MAC / retired-inst | 0.23 | 0.26 | +13 % |
| throughput | 2.748 GMAC/s | 2.534 GMAC/s | -7.8 % |
| L2 miss, total | 1,932,621 | 2,627,363 | +35.9 % |
| L2 reads / writes | 406,554 / 126,619 | 666,134 / 193,977 | +63.9 % / +53.2 % |
| L2 accesses / MAC | 0.0181 | 0.0292 | +61 % |
| **DDR reads / writes** | 37,500 / 36,380 | 119,797 / 115,895 | **+3.19x / +3.19x** |

## 3. Why it got slower - the instruction win was real, the memory cost was bigger

The counters confirm the direction of the earlier reframe (instruction-bound) **and** reveal a
latent memory ceiling underneath it. Once ~12 % of instructions were removed, the harts did not
speed up - **IPC fell 2.64 -> 2.14**, i.e. they spent the freed time *stalling on memory*, not
retiring useful work. The kernel flipped from instruction-bound toward memory-bound, and the
transpose landed on the wrong side of that trade. Two mechanisms, in order of impact:

1. **Sparse A-operand eviction (dominant).** Activations moved from the **dense** 4-line quartet
   B-layout (256 B/tap, all 64 of 64 bytes/line used) to the **sparse** 16-line A-layout
   (1024 B/tap, only 16 of 64 bytes/line used). `tensor_load`'s source stride is 64-byte-granular,
   so the A operand *must* be one spatial per SCP line - it cannot be packed denser. Result: the
   per-tap activation evict quadrupled (256 B -> 1024 B), adding ~ 5.3 MB of write-back traffic over
   the run (768 tiles x 9 taps x +768 B). This is the bulk of the L2-write / DDR increase.
2. **Cache-bypassing direct store.** `tensor_store` writes `padout` straight to DRAM per tile;
   the old scalar scatter wrote through L1 and coalesced. This inflates DDR writes on top of (1).

Crucially, **the direct store (B4) and dense activation packing are mutually exclusive**: the store
can only be direct if the output is spatial-major, which forces activations to be the (sparse) A
operand. So the scalar scatter the old kernel paid for in *instructions* was actually buying
*memory efficiency* - the dense activation packing. The original orientation was memory-optimal.

## 4. Recommendation & what this changes

- **Revert / do not merge this transpose.** It is correct but a regression on the leaderboard
  metric. Keep the branch for the record.
- **The plan needs a memory-traffic axis, not just instruction count.** The "instruction-bound"
  reframe was half the picture: the kernel sits just under a memory ceiling, and cutting
  instructions exposes it. Rank the remaining levers by *bytes of cache/DRAM traffic removed*, not
  just instructions:
  - **B1 (batch the 9 per-tap evicts into one)** now matters for a second reason - fewer, larger
    evicts amortize better - but note it reduces evict *count/latency*, not total *bytes*, so on its
    own it will not undo a 4x byte increase. Measure, don't assume.
  - **Weight/activation residency (B7)** and anything that **keeps operands on-chip across taps**
    (avoiding re-evict) directly attack the byte traffic that bound this experiment.
  - A pure output-store win (deleting the scatter) is **not worth 4x activation eviction** - only
    revisit the direct store if activations can stay the *dense* operand, which the current
    tensor-engine layout does not allow.
- **One open sub-question:** how much of the +8 % is the bypass store vs the sparse evict? A quick
  A/B (keep the transpose but route the store through cache + a per-tile band evict, or shrink the
  apk evict to the used lines) would separate them - but given the sparse-evict math dominates,
  reverting is the higher-value move.

## 5. Status

The transpose kernel change was **reverted** (working tree restored to baseline) - the negative
result lives in this doc and in `experiments_log.md`, not in committed code. The full change is
described in sec 1; the deltas that killed it are sec 2-sec 3. If ever revisited, the board-run recipe is
`build_pmc_probe.sh` -> `run_pmc_board.sh` -> `check_int8_seams.py ... 8` (expect `max_abs=0`).
