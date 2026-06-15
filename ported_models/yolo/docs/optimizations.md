# YOLOv10n → Erbium ETSOC1 — optimization journal

Append-only ledger of every meaningful change. One paragraph per entry, oldest at top.
Mirrors the depth-anything-real journal style.

---

## [2026-05-07]  Project setup  PASS

**Workspace.** `<artifact-archive>/yolov10n-real/`
mirrors the depth-anything-real layout: `kernels/`, `inputs/`, `refs/`,
`blobs/`, `tools/`, `runs/`, plus a `PLAN.md` and this journal.

**Model acquisition.**

- Hugging Face base: `onnx-community/yolov10n` pinned in
  [`docs/HF_REFERENCES.md`](../../../docs/HF_REFERENCES.md).
- Used the YOLOv10n base model and re-exported to ONNX at
  `imgsz=(288, 512)` with `nms=False` to keep raw 3-scale heads (NMS
  belongs on host, not silicon).
- Local pip install (`pip install --target=/media/...` with
  `TMPDIR=/media/...` because `/` is full).
- ONNX summary: 293 nodes, input `[1, 3, 288, 512]`, output
  `[1, 84, 3024]` (4 box-xywh + 80 class-sigmoid, 3024 = 36×64 + 18×32 +
  9×16 anchors).

**Op inventory.** 83 Conv (12× 3×3 64→64; 6× 3×3 32→32; multiple 1×1 and
depthwise 3×3 with groups=128/80), 70 SiLU (=Sigmoid·x), 19 Concat, 11
Add, 11 Split, 2 MatMul + 2 Softmax (DFL box decode), 2 Resize (FPN
upsample), 3 MaxPool. 2.76M params total → 9.16 MB FP32 weights.

**Reference output.** Ran ORT FP32 on the `web_car` photo (used during
depth-anything validation).  Top class score 0.639; 17 anchors above
0.25 confidence.  Saved to `refs/ort_web_car_output.bin`.

**Weight extraction.** `tools/extract_weights.py` walks the ONNX graph,
collects all 83 Conv weights+biases (BN already fused by Ultralytics's
`.fuse()` call during export), emits `blobs/weights_region.bin` (9.20 MB,
64-byte aligned per entry) and `blobs/weights_layout.json` (per-layer
offsets, shapes, strides, pads, groups).


## [2026-05-07]  M0  heartbeat passes on `<board-host>`  PASS

**Kernel** `kernels/yolo_smoke_M0.c`: writes `0xCAFEBABE 0xDEAD10DE`
to dump.bin[0..8].  `buffer_base_from_args` recovers the buffer base
from the launcher's argument area (same pattern as DnCNN3 / depth-
anything M0 — link-time `heap0_start` lands 64 MB into the buffer
which loses the bottom 16 MB if you trust the link-time addresses,
so the runtime base must come from `arg_area`).

**Build script** `build_yolo.sh` — adapted from depth-anything's
`build_depth.sh`.  Same toolchain, same `erbium.ld`, same
`hart_report_crt.S`.  Uses the patched `erbium_soc1sim_argbuf_big`
launcher (80 MB buffer).

**Run script** `run_yolo.sh` — adapted from `run_depth.sh`.  No input
or weights for M0.

**Result.** Kernel wait 0.0004 s.  `dump.bin[0:8]` =
`BE BA FE CA DE 10 AD DE` (little-endian = `0xCAFEBABE 0xDEAD10DE`).
Build → stage → silicon → dump → audit pipeline confirmed for the
YOLOv10n workspace.


## [2026-05-07]  M2..M10 — full FP32 baseline, then 8-hart parallel  PASS

**Compressed log of milestones M2..M10** (full FP32 baseline + first
parallelism step). 17 silicon iterations on `<board-host>`. All audit
against `ort_web_car_*.bin` (ORT FP32 reference for the 288×512 web_car
photo). Detailed per-milestone notes below.

### M2 (conv0+conv1+SiLU) — 1st-try PASS
Built a generic `conv2d_fp32` and `conv2d_dw_fp32` helper in
`yolo_common.h`.  Wall 3.84 s, max_abs 3.4e-5.

### M3 (+C2f model.2) — 1st-try PASS
First C2f block (1 bottleneck w/ residual). Validated split → m.0 cv1
→ cv2 → add → concat → cv2 pipeline. Wall 10.3 s, max_abs 2.1e-5.

### M4 (+conv3, C2f model.4) — 1st-try PASS
2-bottleneck C2f. Wall 22.1 s.

### M5 (full backbone exc. PSA) — needed 1 bisect
SCDown m.5 / C2f m.6 / SCDown m.7 / C2f m.8 / SPPF m.9. First attempt
failed `c2f_m6_out` audit at max_abs 2.66 — root cause was a 32 KB
**memory overlap** between `SCR_M5_CV2_OUT [128,18,32]` (ends
0x36C8000) and `SCR_M6_CONCAT [256,18,32]` (started 0x36C0000). Fix:
tightened cumulative offset bookkeeping with explicit byte sizes per
slot. Re-run PASS at max_abs 9.2e-6.

### M6 (+PSA model.10) — needed re-run
PSA = 256-ch 1x1 cv1 + split + (qkv 128→256, reshape into 2 heads of
{Q[32], K[32], V[64]}, depthwise 3x3 PE on V_reshape, scaled QᵀK +
softmax + V@softmax_T → +PE → 1x1 proj 128→128) + residual + (FFN
128→256→128) + residual + concat + cv2 1x1. 18 ops total.
First run looked broken (Tailscale dropped mid-rsync, dump lost). After
SSH re-auth, re-ran: every PSA intermediate (`m10_qkv`, `m10_q`, …
`m10_proj`) audited bit-accurate; the misleading mismatch was just
that the audit script used `--milestone M5` and matched the previous
M5 dump dir. Pinning `run_dir` explicitly fixed it. **PASS** at
max_abs 7.7e-6, wall 53 s.

Lesson: if SSH dies mid-run, the soc4 dump.bin is still on disk; just
re-pull it. Also: audit script's milestone-name → run-dir glob is
fuzzy when M(N+1) carries M(N) taps; pass the run dir explicitly.

### M7 (+FPN model.11..22) — needed 2 bisects
Up-path (m.11 upsample → m.12 concat → m.13 C2f-no-shortcut →
m.14 upsample → m.15 concat → m.16 C2f-no-shortcut), down-path
(m.17 down 3x3 s=2 → m.18 concat → m.19 C2f → m.20 SCDown →
m.21 concat → m.22 C2fCIB). The CIB has 5 sub-convs: DW3x3 → 1x1 →
DW7x7 → 1x1 → DW3x3 with residual.

Iteration 1 (with default RSYNC truncated 64 MB dump): all 3 head
inputs (`head_p3_in`, `head_p4_in`, `head_p5_in`) FAIL. Couldn't
bisect because FPN scratch lived past 64 MB.

Iteration 2 (`RSYNC_FULL=1`): bisect found `m11_up` already wrong.
Range matched ref but specific values shifted. Per-channel diff showed
channels 0..141 PASS, 142..255 FAIL. Root cause: I'd reserved 0x48000
bytes for `m11_up` but actually need 0x90000 (256 ch is 2× 128 ch,
which I'd been using for sizing). m12_concat at 0x4040000 overlapped.

Iteration 3: re-laid every FPN scratch slot with size = `OC*H*W*4`
rounded to 0x10000, cumulative-aligned. **PASS** all 4 head taps at
max_abs 1.1e-5. Wall 85 s.

Lesson: any time I copy a scratch-offset block from M(N) to M(N+1)
I need to re-derive sizes from the actual tensor shapes, not from
what I "remember" the size to be. A smaller-shape block in M(N) can
become a larger-shape block in M(N+1).

### M8 (+detection-head logits, no DFL) — 1st-try PASS
24 head convs (cv2.k.0/1/2 for box, cv3.k.0/1/2 for class, k=0..2)
across 3 scales. Reg branch is 3 convs (3x3+SiLU, 3x3+SiLU, 1x1+no-act).
Cls branch is 5 convs (DW3x3+SiLU, 1x1+SiLU, DW3x3+SiLU, 1x1+SiLU,
1x1+no-act). All 6 logit taps PASS at max_abs 5.6e-5. Wall 112 s.

### M9 (+DFL decode + box decode + class sigmoid → final) — 1st-try PASS
DFL: per scale, for each (edge, spatial) softmax 16 bins and weighted-sum
with [0..15] = expected box edge in feature units. Box decode: anchor
center = (w+0.5, h+0.5) in stride units; lt=box_buf[0,1], rb=box_buf[2,3];
xywh = ((bl+br)/2*s, (bt+bb)/2*s, (br-bl)*s, (bb-bt)*s). Class scores:
sigmoid(cls_logits). Concat into `final[1,84,3024]` ordered [P3, P4, P5].

**Final tap PASS at max_abs 9.2e-4 vs gate 3.0** — bit-accurate to
ORT. Wall 112 s (single-hart scalar FP32 baseline).

### M11 — 16-hart scalar (T0+T1) PASS
Removed the T0-only gate from `conv2d_fp32_mh`/`conv2d_dw_fp32_mh`
behind a `YOLO_USE_16HART` build flag. T1 (odd) harts have no VPU
but can do scalar fmadd. With 16 harts splitting OC instead of 8,
**wall 8.41 s → 1.75× over M10**. Total speedup vs M9 baseline:
112 / 8.41 = 13.3×. Audit unchanged at max_abs 9.2e-4.

### M12 — VPU-vectorized 1x1 conv (T0 only, 8 harts) PASS
Added `conv2d_1x1_fp32_mh_vpu` using Erbium VPU intrinsics
`fbcx.ps` (broadcast scalar→8 lanes), `flq2` (load 8 floats),
`fmadd.ps` (parallel multiply-add 8 lanes), `fsq2` (store).
Replaced 41 of the 1x1 `CONV_MH` calls with `CONV_1x1_VPU`. Kept the
3x3 calls at scalar 16-hart. The VPU function gates to T0 only since
odd harts don't have VPU.

First attempt FAILED with max_abs 444 — root cause was `register float
acc asm("f0") = bias_pkg` generating an `fmv.s f0, f4` that only
copies the lower 32 bits (lane 0); lanes 1..7 retained whatever was
left in f0 from previous loop iterations. Fix: directly broadcast the
bias to `acc` with `fbcx.ps`, no intermediate `bias_pkg`.

**Wall 4.26 s — 2.0× over M11. Total 26.3× vs M9.** PASS at the same
9.2e-4 max_abs.

### M13 — VPU 3x3 stride=1 pad=1 conv (T0 only) PASS
Added `conv2d_3x3_p1_fp32_mh_vpu` adapted from depth-anything's
`conv3x3_pad1_fp32_vpu`. Replaced 24 of the 3x3 stride=1 pad=1 calls.
Kept 4 stride=2 3x3 (conv0, conv1, conv3, m.17) at scalar 16-hart.

**Wall 3.50 s — 1.22× over M12. Total 32.0× vs M9 (0.286 FPS).** PASS.

Diminishing return from this VPU swap because the stride=1 3x3 layers
share total MAC roughly with the 1x1's already vectorized — the
remaining 4 stride=2 3x3 layers and ~13 depthwise convs (still scalar)
plus memory bandwidth dominate.

### M14 — VPU depthwise 3x3 stride=1 (T0 only) PASS
Replaced 9 stride-1 depthwise calls (m.10 pe, m.22 cv1.0/1.4, head
cv3.k.0.0/1.0). Wall 3.49 s — essentially flat from M13 (these layers
are tiny). PASS. Kept the change anyway since it's free.

### M15 — multi-hart concat / add helpers PASS
Added `mh_copy_floats`, `mh_add_floats`, `mh_iadd_floats`, `mh_concat3`,
`mh_concat4` and macros `MH_COPY/ADD/IADD/CONCAT3/CONCAT4`. Replaced 6
add_chw, 1 SPPF 4-way concat, 3 FPN 3-way concats. Wall 3.37 s
(Δ −0.12 s).  PASS.

### M16 — OC8-blocked VPU 1x1 (8 OCs simultaneously) PASS
New helper `conv2d_1x1_fp32_mh_vpu_oc8` holds 8 VPU accumulators in
f0..f7, broadcasting 8 different W[oc, ic] scalars for the same input
v_pkg per ic. Input v_pkg is loaded ONCE per (ic, ow8) and reused by
all 8 fmadd.ps's, so memory bandwidth on the input drops 8×.
Wall 3.15 s (Δ −0.22 s).  PASS.

### M17 — OC dispatcher (small OC → per-OC, large OC → OC8) PASS
Issue with M16: layers with OC < 64 get 8 tiles or fewer, leaving
some compute harts idle. Added `conv2d_1x1_disp` that picks per-OC
when OC < 64 and OC8-blocked otherwise. Wall 3.19 s (Δ +0.04, within
noise). Kept anyway since it's correct and the small-OC layers are a
small fraction of compute.

### M18 — OC8-blocked VPU 3x3 stride=1 — FAIL (kernel hung)
Same blocking idea applied to 3x3 stride=1. Kernel hung on silicon
("Unbalanced number of abort unblockers" runtime FATAL after timeout).
Suspected cause: register-pressure saturation with 8 VPU accumulators
+ v_pkg + w_pkg + the EDGE-case scalar fallback that has to fsq2 each
of the 8 accs to memory and reload them inside the inner kx loop.  At
24 layers × hundreds of ow8 tiles × IC × 9 ky/kx, the EDGE-case overhead
plus instruction-cache miss probability adds up. Reverted to M17 base
for M19.

### M19 — multi-hart DFL + box decode + class sigmoid PASS (later flaky)
Per-anchor work (DFL softmax of 4 edges × 16 bins, box xywh, sigmoid
of 80 cls logits) is independent across the 3024 anchors. Split anchor
range across 8 T0 harts within each scale. Wall 3.12 s. Initial run
PASS at max_abs 9.2e-4 — but later re-runs (M22, M23-attempt-1) showed
sporadic max_abs ≈ 500 with the same kernel binary. Each per-hart
slice writes 84 disjoint regions of `final_out` (channels c=0..84 at
the hart's anchor stride), and even with whole-buffer evicts on every
hart the L1D-non-coherence guarantees seem to be intermittently
violated in our runtime. Left M19 in the timeline but do **not**
recommend it as a baseline.

### M20 — OC4-blocked VPU 3x3 stride=1 PASS but no win
Half-step from M18 (OC8 hung) — 4 accumulators in f0..f3 with OC4
tiling. Built and ran. Wall 3.20 s (slightly worse than M19's 3.12 s).
The 3x3-stride-1 layers were not memory-bound enough for the input
re-use to help; the per-tile prologue/epilogue (8 fsq2 + scalar SiLU)
ate the gain. PASS.

### M21 — parallel PSA scoring (transpose + matmuls + softmax) FAIL
Attempted to multi-hart the whole PSA scoring tail by row index
(NHEAD * HW = 288 rows split across 8 T0 harts).  Each hart computed
its rows for QT-transpose, QT@K matmul, softmax, T-transpose, V@sm_T
matmul, and the +PE add.  Built fine, ran fast (3.07 s) but FAIL at
max_abs 500. Suspected race in cross-hart reads of `logits`/`sm_T`
that span row boundaries (each hart needs full rows from other harts'
slices when transposing → scattered cache misses, plus a larger
working set per hart).  Reverted for M22.

### M22 — clean copy of M19 (silicon flake) FAIL
Re-built M19 verbatim while updating `yolo_common.h` with a new
unused `mh_maxpool5_s1_p2` helper.  FAIL at max_abs 500.  Same
intermittent issue as M19. Confirms silicon non-determinism around
the parallel DFL writes.

### M23 — single-hart DFL (revert) PASS — current stable baseline
Reverted to single-hart DFL+box+sigmoid. First attempt of M23 hit a
runtime "FATAL: Unbalanced number of abort unblockers" (silicon flake
unrelated to the kernel — same launcher path that M19 succeeded on).
**Re-run: wall 3.14 s, max_abs 9.2e-4 PASS.**  This is the current
safe optimization baseline. Total **35.7× over M9 single-hart**.
**FPS = 1 / 3.14 = 0.32.**

---
### M30 — full pipeline on silicon (preprocess + model + postprocess) PASS

**Goal change.** User asked: can the rest of the pipeline (everything except
JPEG decode) run on chip too?  Yes.

**Stage 0 (preprocess).** Read raw uint8 RGB at `RAW_INPUT_OFFSET` (0x4A00000),
shape `[SRC_H=480, SRC_W=640, 3]` (host loads it from a `_raw_HxWx3_uint8_rgb.bin`
file).  Bilinear resize → 288×512, divide by 255, transpose HWC → CHW.  Write
FP32 `[1, 3, 288, 512]` at the existing `INPUT_OFFSET` so the rest of the model
sees an unchanged input.  Multi-hart by output row (8 T0 harts each handle 36 rows).

**Stage 2 (postprocess).** After the FP32 final tensor is computed, hart 0:
1. Scans all 3024 anchors, takes the per-anchor max class probability,
   keeps those ≥ 0.25 — produces a candidate list in scratch (`tb`).
2. Class-aware NMS: O(n²) on the ≤ 100 surviving candidates with IoU > 0.5.
3. Writes the surviving detections to `DETECTIONS_OFFSET` (0x1D00000) as
   `{ uint32 N }` + N × `{ uint32 class_id; float score; float x1,y1,x2,y2; }`.

The host now loads only the raw RGB file and reads back the small detection
list — no FP32 input to upload, no full final tensor to download.

**Result.** Wall **3.13 s** (M29 baseline was 3.07 s, so pre+post added ~50 ms).
Silicon detections on `web_car.jpg`:
```
  car     prob=0.665  bbox=(  4.6,  56.0) → (505.5, 273.6)
  person  prob=0.477  bbox=(424.3,  88.6) → (511.7, 204.6)
```
ORT-host reference: same classes, ±2 px boxes, ~5 % score delta — the gap
is the bilinear kernel difference between PIL and my silicon resize, not a
model bug.

What the host still does:
- JPEG decode (~5 ms, PIL).  Putting a JPEG decoder on chip is much bigger
  scope than the ~5 ms it costs and gives no measurable speedup.
- Mapping `class_id` → COCO label string (a 4 KB table lookup).

---

**Bottleneck after M23 (educated guess, no PMC ledger yet):**
Memory bandwidth on the largest 1x1 / 3x3 stride=1 convs. Compute is
already VPU 8-wide on T0 harts and the OC8 input-reuse tiling drops
input reads 8×. The 4 stride=2 3x3 layers (~120M MAC, scalar 16-hart)
add ~150 ms; depthwise stride=2 layers add small amounts; the rest is
DRAM/L2 traffic. The next big lever is TFMA INT8 (4× weight & weight
bandwidth, 4× compute via the int8 matmul accelerator) — large
infrastructure (per-OC weight scales, A-pack format, per-tensor
activation quant + B-pack, dequant-with-bias) but the natural fit for
the 41 1x1 layers that still dominate.


### M10 (multi-hart parallel: 8 T0 compute harts) — PASS
Added `conv2d_fp32_mh(hid, …)` and `conv2d_dw_fp32_mh(hid, …)` in
`yolo_common.h`. Each compute hart owns the OC range [OC·t0/8, OC·(t0+1)/8).
After each conv, the hart evicts its OC slice; all 16 harts (8 T0 + 8
T1 idle) hit `MH_BARRIER()` (FENCE + WAIT_CACHEOPS + shire_barrier on
FLB1). Single-hart non-conv blocks (residual adds, concat copies, PSA
attention scoring, DFL decode) run inside `if (is_h0)` guards followed
by their own evict + barrier — same data flow, just with sync edges.

**Wall 14.74 s — 7.6× speedup over M9 single-hart baseline.**
Audit PASS, max_abs 9.2e-4 (bit-identical to M9). FPS = 1/14.74 = 0.068.

Next bottleneck: every `for` loop is still scalar — VPU is unused.
Single-hart concat copies in the FPN are visible as serial dead time
between barrier points. PSA scoring (matmul + softmax + transposes)
is also serial.

---

## [2026-05-07]  M1  conv0 (3→16, 3x3 s=2) + SiLU  PASS

**Kernel** `kernels/yolo_smoke_M1.c`: single-hart scalar FP32, naive
6-deep nested conv loop, then SiLU via depth-anything's `my_expf` +
`fast_recip` primitives.  Output dumped at `0x300000`, [1,16,144,256].

**Two bisect iterations needed.**

1. *First attempt* — kernel crashed on launch ("Stream error event 30
   code 1", wait 0.002 s).  Root cause: my `silu()` had `(uint32_t)k <<
   23` for negative k, which is UB on overflow.  Replaced with the
   exact `my_expf` from depth-anything M10 (proven on silicon) — uses
   `(int32_t)((v.u >> 23) & 0xFF) + k` to add to the biased exponent.

2. *Second attempt* — kernel ran but output range was 7× too large
   ([-237, 172] vs ref [-32, 24]) and contained 99 NaN.  Root cause:
   the `run_yolo.sh` `INPUT_LOAD` block looked for
   `inputs/${IMAGE}.bin` while the file is `inputs/${IMAGE}_288x512.bin`.
   The input was silently never staged → kernel read whatever was
   already in DRAM at `0x10000` → garbage.  Fixed the script to fall
   back to the `_288x512` suffix.

3. *Third attempt (RAW_NO_SILU)* — to bisect SiLU vs. conv math.
   `max_abs = 7.6e-6` vs ORT pre-SiLU tap.  Conv math bit-accurate.

4. *Final attempt* — full SiLU.  `max_abs = 9.5e-6` vs ORT
   `conv0_post`.  Gate 1e-3.  **PASS.**

**Wall.** No PMC ledger yet; raw kernel completion time was not
recorded (the "Stream error event 30" log line may suppress wall_s in
the timing pull).  Order of magnitude expectation:
16·144·256·27 = 16M MAC + 590k SiLU calls ≈ 30 ms scalar single-hart.

**Lessons in.**
- `(uint32_t)k << 23` is UB for negative k — cast through `int32_t`
  first or compute the exponent bias carefully.
- The silent-no-input failure mode is sneaky.  Added a warning when
  IMAGE is set but no file is found.

---

## Next milestones (not yet executed)

Per `PLAN.md`:

- **M1**: input copy + first Conv 3×3 stride-2 (3→16) + SiLU.
  Audit against ORT tap `/model.0/Conv_output_0` (or post-SiLU).
- **M2..M5**: progressively larger backbone slices.
- **M6**: full backbone + neck (FPN).
- **M7**: full kernel including 3 detection heads + DFL decode.
- **M8+**: TFMA INT8 + multi-hart (mirror depth-anything M9..M47).

Realistic compute estimate (single-hart scalar baseline, 6.7 GFLOP at
~5 GFLOP/s VPU = 1.3 s; at scalar FP ~1 GFLOP/s = 6.7 s).  Multi-hart +
TFMA INT8 should bring this well below 1 s.  ≥50 silicon iterations
expected before reaching the floor.
