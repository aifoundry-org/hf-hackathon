# Porting Models & Kernels to CORE-ET -- Hackathon Field Guide

> A practical guide for people who have never touched CORE-ET or ET-SoC-1 before and
> want to get a model, or a single hot kernel, running and then running fast
> without losing a day to footguns we already hit.
>
> This guide complements the repo quickstart
> [`ET_SOC1_QUICKSTART.md`](../ET_SOC1_QUICKSTART.md) and the upstream
> ET backend reference
> [`docs/backend/ET.md`](../../ported_models/llama_cpp_et/src/llama.cpp-et/docs/backend/ET.md).
> Read the quickstart for this repo's CI flow. Read ET.md for the exact
> toolchain install and CMake flags. Read this for the mental model, the
> footguns that cause silent corruption or hangs, and the performance playbook.

## Table Of Contents

1. [The 60-second mental model](#1-the-60-second-mental-model)
2. [Two porting paths](#2-two-porting-paths-pick-one)
3. [Getting started](#3-getting-started)
4. [The hardware in depth](#4-the-hardware-in-depth)
5. [Correctness footguns](#5-correctness-footguns)
6. [Performance playbook](#6-performance-playbook)
7. [Dead ends](#7-dead-ends)
8. [Debugging toolkit](#8-debugging-toolkit)
9. [Pre-flight checklist](#9-pre-flight-checklist)

## 0. TL;DR -- Read This Even If You Read Nothing Else

- **The chip is a 2D mesh of tiny RISC-V cores with a matrix engine bolted on.**
  Think "1000+ in-order cores sharing a slow main memory," not "a GPU."
- **There are two ways in.** Make GGML run your model by filling gaps in the
  existing backend, or write a standalone kernel for one hot operation. Most
  hackathon wins come from the GGML path plus optimizing one kernel.
- **Main memory is the bottleneck.** The whole game is keeping data in on-chip
  scratchpad and feeding the matrix engine. Bandwidth, not FLOPs, is the wall.
- **The worst bugs are silent.** Wrong synchronization or cache handling often
  gives plausible-looking wrong numbers, or hangs the chip.
- **Edit a kernel `.c`? Rebuild the host binary for CI.** Kernels are embedded
  into the checked-in framework build used by the board workflow. For local
  iteration, the ET backend can also load rebuilt kernel ELFs from
  `GGML_ET_KERNELS_PATH`.

## 1. The 60-Second Mental Model

ET-SoC-1 is a massively-manycore RISC-V accelerator. The hierarchy that matters
for porting:

```text
Chip
 `-- ~32 Shires           (a shire is a tile on a 2D mesh / NoC)
      `-- ~32 Minions     (a minion is a small in-order RISC-V core)
           `-- 2 Harts    (hardware threads per minion)
                |-- hart 0 -> has the matrix/tensor engine
                `-- hart 1 -> no matrix engine
```

Three facts should reshape how you write code:

1. **The two harts share one minion pipeline, and only diverge when the matrix
   engine is involved.** Both harts issue into the same execution pipeline, so
   two busy harts contend for the same functional units. Hart 1 is useful when
   hart 0 is stalled on memory or matrix-engine work. When hart 0 is already
   compute-bound, the second hart buys little.
2. **The shire is the unit of cooperation.** Cooperative loads, shire-local
   scratchpad, and cheap barriers operate within a shire. Crossing shires goes
   over the NoC and is much more expensive.
3. **There is no CPU-like cache coherence model to lean on.** Treat data
   movement and synchronization as explicit parts of the algorithm. If you get
   them wrong, the result may be a hang or silent numerical corruption.

If you internalize "slow shared DRAM, fast local scratchpad, explicit movement,
explicit sync, asymmetric harts," everything else follows.

## 2. Two Porting Paths: Pick One

### Path A -- Make GGML Run Your Model

This is the recommended path for most teams.

`llama.cpp`/GGML already has an ET backend. When a model runs, GGML walks the
compute graph op by op. For each op it asks the backend "can you do this one?"
If yes, the op runs on ET. If no, it falls back to CPU. That fallback is useful
for correctness, but slow, and it forces tensor movement across the host/device
boundary.

In this path, "porting a model" usually means finding the ops or shapes that
fall back to CPU and adding ET backend support for them.

- **Pro:** You get a whole working model runner for free.
- **Pro:** `test-backend-ops` gives per-op correctness and performance checks.
- **Con:** You must satisfy GGML's op semantics and dispatch contract.
- **Con:** A model can look like it "runs" while the important work is still on
  CPU unless you inspect logs and profiles.

### Path B -- Write A Straight Kernel

Skip GGML and write a baremetal RISC-V kernel for one operation: a matmul, an
attention step, a custom layer, or a model-specific fused block. In the
`llama.cpp-et` backend, kernels live under:

```text
ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et/et-kernels/src/
```

The repo's standalone smoke ports use committed source under each model's
`ported_models/<model>/src/` directory.

- **Pro:** Total control over data movement, synchronization, and layout.
- **Pro:** Best path for "how fast can op X go?" experiments.
- **Con:** You own the validation harness.
- **Con:** It is easy to hit silent-corruption footguns without GGML catching
  shape, dtype, and lifetime mistakes for you.

For a time-boxed hackathon, start with Path A to get a model running, then use
Path B for the single kernel that dominates runtime. Use `ET_PERF`,
`GGML_ET_PROFILE`, and board CI logs to find that kernel. Do not guess.

## 3. Getting Started

Build and toolchain details live in
[`ET_SOC1_QUICKSTART.md`](../ET_SOC1_QUICKSTART.md) and upstream
[`docs/backend/ET.md`](../../ported_models/llama_cpp_et/src/llama.cpp-et/docs/backend/ET.md).
The short version:

1. Install the ET toolchain and platform under `/opt/et`, or set
   `ET_TOOLCHAIN` and `ET_PLATFORM`.
2. Use the committed framework source under `ported_models/<framework>/src/`.
   CI builds from committed source only.
3. Put model/runtime artifacts in `artifacts.json`. Large generated blobs and
   downloaded references stay outside git.
4. Register board benchmarks in `.github/ci/benchmark_config.json`.
5. Use PR board CI to validate what changed on real silicon.

The repo has two model styles:

- Standalone kernel ports: `ported_models/yolo/`.
- Framework ports: `ported_models/llama_cpp_et/` (per-model GGUF configs live
  under its `benchmarks/`) and `ported_models/ggonnx/`.

For llama.cpp/LLM work, start from:

```text
ported_models/llama_cpp_et/artifacts.json
ported_models/llama_cpp_et/benchmarks/<model>.json
.github/ci/scripts/run_llama_server_benchmark.py
```

For a new model on an existing framework, the smallest useful PR usually adds:

1. An artifact entry for the model file.
2. A benchmark JSON that defines prompt, decode length, and PPL settings.
3. A `.github/ci/benchmark_config.json` row that points at the correct runner.

## 4. The Hardware In Depth

Think in three memory tiers:

- **DRAM/main memory:** large and slow. Avoid repeated streaming of the same
  tile.
- **L2 or shire-local scratchpad:** the cooperation zone. Tile into it, reuse
  it, and synchronize within the shire.
- **L1/minion-local scratchpad and registers:** the hot working set. Keep
  accumulators and matrix-engine operands here when possible.

The practical implication is that a naive CPU loop often scales badly even if it
uses many harts. The right question is usually not "how many harts are active?"
but "how many bytes are we moving per useful operation, and from where?"

Use hart 0 for matrix/tensor work. Use hart 1 for work that can overlap with
hart 0 stalls: packing, pointer arithmetic, prefetch-style movement, or scalar
post-processing. If hart 0 is not stalled, hart 1 may just contend for the same
minion resources.

Use shires as the first unit of decomposition. Split work so most reuse and
barriers are shire-local, then combine across shires only when necessary.

## 5. Correctness Footguns

### 5.0 Kernel Build Footgun

For CI and repo-authoritative framework builds, rebuild the host binary after
kernel source changes. The board workflow builds from the committed framework
source and runs those binaries.

For local llama.cpp-et development, upstream also supports runtime-loaded
kernel ELFs via `GGML_ET_KERNELS_PATH`. That is useful for fast iteration, but
do not rely on a private local kernel path as the source of truth for a PR.

### 5.1 CPU Fallback Hiding Missing ET Coverage

A model can produce correct tokens while key ops run on CPU. That is not a
successful ET port. Inspect backend logs, `ET_PERF` lines, and trace output.
For leaderboard rows, the benchmark runner should report enough notes to reveal
fallbacks, timeouts, artifact failures, and missing measurements.

### 5.2 Shape And Stride Mismatch

Most bad ports are not "wrong math"; they are wrong assumptions about layout.
Before optimizing, record:

- tensor shape
- stride
- dtype and quantization block format
- scale and zero point where relevant
- alignment assumptions
- exact slice or token position being tested

Compare a small deterministic host reference before scaling up.

### 5.3 Synchronization That Looks Almost Right

Missing or misplaced barriers may only fail under load. If a kernel uses
cooperative loads or shared scratchpad, make the producer/consumer boundary
obvious in code and test repeated runs. A single passing run is weak evidence.

### 5.4 Cache And Memory Visibility

Do not assume CPU-style coherence between host, DRAM, scratchpad, and minion
state. Make ownership and movement explicit. If a buffer crosses host/device or
shire boundaries, there should be an obvious transfer/sync point in the code or
runtime path.

### 5.5 Unsupported Instructions

The upstream ET backend documents instructions that are not implemented in
hardware/firmware yet. Avoid float division, trig, square root, and long-float
casts in kernels unless you are using known workarounds from the ET backend.
If a kernel build rejects an instruction, treat that as a kernel source problem,
not a CI flake.

### 5.6 Success Criteria Drift

Decide success before measuring:

- For a standalone kernel: output matches host reference within the declared
  tolerance, no hang, score JSON produced, and runtime is measured.
- For a model: expected output path runs on the board, required ET ops do not
  fall back unexpectedly, model metric is present, and PPL/correctness smoke is
  within the declared threshold.
- For leaderboard: the canonical benchmark variant passes in board CI and writes
  mergeable JSON under `data/`.

## 6. Performance Playbook

1. **Profile first.** Use `ET_PERF` and `GGML_ET_PROFILE` to rank kernels by
   time. Optimize the kernel that dominates the board run.
2. **Reduce DRAM traffic.** Tile into scratchpad, reuse tiles, and avoid
   rereading weights or activations.
3. **Use the matrix engine where it matters.** Hart 0 has the useful tensor
   path. Feeding it reliably is more important than adding scalar work.
4. **Pack for the hardware, not for convenience.** If a model format forces
   bad access patterns, add a one-time pack/transpose step and benchmark the
   end-to-end effect.
5. **Fuse only after correctness is boring.** Fusion hides bugs. First validate
   individual boundaries, then fuse producer/consumer pairs that save real
   memory traffic.
6. **Measure variance.** Run enough repetitions to distinguish a real speedup
   from board noise, thermal state, or queue effects.
7. **Keep PPL in the loop for LLMs.** Tokens/second without perplexity can hide
   broken logits, bad KV layout, or quantization mistakes.

Useful metrics:

| Metric | Why it matters |
|---|---|
| wall time | User-visible performance |
| kernel duration | Where time is spent |
| tokens/s | Primary LLM decode score |
| WikiText-2 raw PPL | LLM correctness sanity check |
| bytes/op | Whether the kernel is memory-bound |

## 7. Dead Ends

- Do not start by "using all cores." Start by making one boundary correct and
  measuring where time goes.
- Do not optimize a CPU fallback. Move the op to ET first.
- Do not commit private absolute paths or environment-variable source overrides.
  Public CI must build from committed framework source.
- Do not dump the entire board as a default debug strategy. Prefer targeted
  tensor taps, compact traces, score JSON, and kernel logs. Full dumps are hard
  to diff, can be huge, and often hide the actual failing boundary.
- Do not chase a large model before a tiny deterministic repro passes.
- Do not treat sys-emu timeout as proof the kernel is wrong. Sys-emu is useful
  for plumbing, not final performance truth.

## 8. Debugging Toolkit

Use these in roughly this order:

1. **CI logs and artifacts.** Every board run should leave `run.log`, score
   JSON, and benchmark notes. The PR comment is the summary, not the whole
   diagnosis.
2. **`ET_PERF` lines.** These identify op, kernel, tensor, shape, duration, and
   sometimes FLOPs.
3. **`GGML_ET_PROFILE`.** Produces runtime traces and a kernel map for deeper
   timeline inspection.
4. **Host reference taps.** Save small tensors at stable boundaries and compare
   `max_abs`, `mean_abs`, top-k agreement, or token logits.
5. **PPL for transformer models.** This repo uses WikiText-2 raw PPL for LLM
   board rows. Keep it enabled for leaderboard candidates.
6. **Selective memory dumps.** Dump named buffers, shapes, and ranges. Include
   dtype and layout in the dump metadata. Avoid raw board-wide dumps unless the
   failure is already narrowed to memory ownership or corruption.
7. **Repeated-run stress.** Hangs and sync bugs often need repeated execution
   to reproduce.

When a run fails, classify it before editing code:

| Failure | First thing to inspect |
|---|---|
| artifact setup failed | `artifacts.json`, sha256, local cache path, source path |
| CMake/build failed | committed framework source and toolchain path |
| timeout | last `ET_PERF` line, board log tail, input size |
| wrong output | first mismatching tensor boundary |
| missing metric | runner score JSON and parser |
| all skipped | board access, self-hosted runner, or workflow selection |

## 9. Pre-Flight Checklist

Before opening or merging a port PR:

- The source that will run on the board is committed under `ported_models/`.
- `artifacts.json` has stable URLs, revisions, sizes, hashes, and local cache
  names.
- The model is registered once in `.github/ci/benchmark_config.json`.
- The benchmark runner writes a score JSON with status, notes, and the primary
  metric.
- LLM rows include PPL settings unless there is a documented reason not to.
- The PR board workflow selects the changed model.
- Logs show the expected ET path, not a hidden CPU fallback.
- Any new third-party source has a clear license note.
- Generated artifacts, downloaded models, dumps, and board scratch files are
  not committed.
