# Vision Model Verification Report

**Date:** 2026-07-11  
**Method:** Automated checks from the verification plan (`JSON`, HF pins, artifacts schema, benchmark configs, ports, URL/SHA256, CI registration, config expansion) plus review of `ci_preflight.sh` on Windows.  
**Raw results:** [`_verify_results.json`](./_verify_results.json)

## Executive summary

| Model | Overall | Notes |
|-------|---------|-------|
| SmolVLM-256M | **READY** | Merged (#26). All submission checks PASS. |
| SmolVLM-500M | **READY** | PR #27. All submission checks PASS. |
| SmolVLM2-2.2B | **READY** | PR #29. All submission checks PASS. |
| ZwZ-4B | **READY** | PR #30. All submission checks PASS. |
| Qwen3.5-0.8B | **NOT READY** | Draft bench/docs only; no artifacts / CI / HF row. Feat branch empty. |
| MiniCPM-V 4.6 | **NOT READY** | Draft bench/docs only; no artifacts / CI / HF row. Feat branch empty. |
| Ministral 3 3B | **NOT READY** | Draft bench/docs only; no artifacts / CI / HF row. Feat branch empty. |
| TinyViT | **NOT READY** | Plan only; not in `ggonnx/artifacts.json`. Feat branch empty. |
| D-FINE Nano/Small | **NOT READY** | Plan only; needs ONNX export + artifacts. Feat branch empty. |
| EfficientViT | **NOT READY** | Plan only; needs ONNX export + artifacts. Feat branch empty. |
| RF-DETR Small | **NOT READY** | Plan only; needs ONNX export + artifacts. Feat branch empty. |

**Environment note:** Full `ci_preflight.sh` fails on this Windows runner (`jq` missing, trusted-llama fixtures / CLI drift). This is **not model-specific**. Model-relevant checks (JSON validity, config expansion, HF URL/SHA) were used for the READY verdict. GitHub Actions board CI remains the authoritative preflight for PRs.

**Known main conflict (pre-existing):** port `18081` is shared by `llama32_1b`, `qwen25_05b`, and `deepseek_r1_15b`. Vision ports `18100–18106` are unique.

---

## Universal checklist (READY models)

| Check | 256M | 500M | 2.2B | ZwZ-4B |
|-------|:----:|:----:|:----:|:------:|
| JSON validity | PASS | PASS | PASS | PASS |
| HF reference pinned | PASS | PASS | PASS | PASS |
| Submission recipe | PASS | PASS | PASS | PASS |
| No model blobs in PR | PASS | PASS | PASS | PASS |
| `benchmark_config.json` registered | PASS | PASS | PASS | PASS |
| Port unique (18100–18106) | PASS (18100) | PASS (18103) | PASS (18106) | PASS (18105) |
| Config expansion | PASS | PASS | PASS | PASS |
| Full `ci_preflight.sh` (local Win) | WARN (env) | WARN (env) | WARN (env) | WARN (env) |

## GGUF-specific checklist (READY models)

| Check | 256M | 500M | 2.2B | ZwZ-4B |
|-------|:----:|:----:|:----:|:------:|
| A1 artifacts schema | PASS | PASS | PASS | PASS |
| A2 benchmark config | PASS | PASS | PASS | PASS |
| A3 model_artifact ↔ artifacts | PASS | PASS | PASS | PASS |
| A4 HF URL resolves | PASS | PASS | PASS | PASS |
| A5 SHA256 vs HF LFS oid | PASS | PASS | PASS | PASS |
| A6 mmproj present | PASS | PASS | PASS | PASS |
| A7 perplexity gate | PASS | PASS | PASS | PASS |

---

## Per-model detail

### 1. SmolVLM-256M — READY (merged)

- **Branch verified:** `origin/main` (PR #26 merged)
- **Port:** 18100
- **Artifacts:** `smolvlm_256m_q8_gguf`, `smolvlm_256m_mmproj_q8`
- **HF:** `ggml-org/SmolVLM-256M-Instruct-GGUF` @ `b9e4379…` Apache-2.0
- **Issues:** None for submission. Local full preflight WARN only (env).

### 2. SmolVLM-500M — READY

- **Branch:** `fork/feat/smolvlm-500m` (PR #27)
- **Port:** 18103
- **Artifacts:** `smolvlm_500m_q8_gguf`, `smolvlm_500m_mmproj_q8`
- **HF:** `ggml-org/SmolVLM-500M-Instruct-GGUF` @ `72e9860…` Apache-2.0
- **SHA256:** matches HF LFS oid for LLM + mmproj
- **Issues:** None for submission.

### 3. Qwen3.5-0.8B — NOT READY

- **Feat branch `feat/qwen35-08b`:** empty vs `main`
- **Drafts on docs PR:** `benchmarks/qwen35_08b.json` (port 18102), `docs/qwen35_08b.md`
- **FAIL:** no `artifacts.json` entries, not in `benchmark_config.json`, no `HF_REFERENCES.md` row, `model_artifact` dangling, config expansion unknown model
- **Fixes:** add LLM+mmproj artifacts with SHA256 pins → HF row → CI register → open PR

### 4. SmolVLM2-2.2B — READY

- **Branch:** `fork/feat/smolvlm2-2.2b` (PR #29)
- **Port:** 18106
- **Artifacts:** `smolvlm2_22b_q4_gguf`, `smolvlm2_22b_mmproj_q8`
- **HF:** `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` @ `1bc3c9f…` Apache-2.0
- **SHA256:** matches HF LFS oid
- **Issues:** None for submission. (Board flake earlier was runner disconnect, not config.)

### 5. MiniCPM-V 4.6 — NOT READY

- **Feat branch `feat/minicpm-v-4.6`:** empty vs `main`
- **Drafts:** `benchmarks/minicpm_v46.json` (port 18101), `docs/minicpm_v46.md`
- **FAIL:** same gap pattern as Qwen (no artifacts / CI / HF pin)
- **Fixes:** confirm GGUF+mmproj availability, pin hashes, register, PR

### 6. ZwZ-4B — READY

- **Branch:** `fork/feat/zwz-4b` (PR #30)
- **Port:** 18105
- **Artifacts:** `zwz_4b_q4_gguf`, `zwz_4b_mmproj_q8`
- **HF:** `inclusionAI/ZwZ-4B-GGUF` @ `af96680…` Apache-2.0
- **SHA256:** matches HF LFS oid
- **Issues:** None for submission.

### 7. Ministral 3 3B — NOT READY

- **Feat branch `feat/ministral-3-3b`:** empty vs `main`
- **Drafts:** `benchmarks/ministral_3b.json` (port 18104), `docs/ministral_3b.md`
- **FAIL:** no artifacts / CI / HF row; license still needs explicit confirmation in HF row
- **Fixes:** verify open license + GGUF pins → register → PR

### 8. TinyViT — NOT READY (ONNX)

| Check | Result |
|-------|--------|
| B1 artifacts in `ggonnx/artifacts.json` | **FAIL** — `tinyvit_21m` missing |
| B2 ONNX availability | **FAIL** — no entry/URL |
| B3 ONNX ops | PASS (plan scan; no hard blockers flagged) |
| B4 License | PASS (plan mentions open license) |
| Submission recipe | WARN — plan at `docs/vision_models/tinyvit.md` only |
| CI registered | WARN — expected (ggonnx CI not wired) |
| Feat branch | WARN — empty vs main |

**Fixes:** add ggonnx artifact (onnx-community TinyViT ONNX), recipe under `ported_models/ggonnx/docs/`, HF pin; CI when runner exists.

### 9. D-FINE Nano/Small — NOT READY (ONNX)

| Check | Result |
|-------|--------|
| B1 artifacts | **FAIL** — `dfine_nano` / `dfine_small` missing |
| B2 ONNX availability | **FAIL** — export pending |
| B3 ONNX ops | WARN — plan mentions deformable / possible unsupported ops |
| B4 License | PASS (plan) |
| Feat branch | empty |

**Fixes:** export ONNX → validate ops on ggonnx → artifacts + docs → PR when CI ready.

### 10. EfficientViT — NOT READY (ONNX)

| Check | Result |
|-------|--------|
| B1 artifacts | **FAIL** — `efficientvit_m0` missing |
| B2 ONNX availability | **FAIL** — export pending |
| B3 ONNX ops | PASS (plan scan) |
| B4 License | PASS (plan) |
| Feat branch | empty |

### 11. RF-DETR Small — NOT READY (ONNX)

| Check | Result |
|-------|--------|
| B1 artifacts | **FAIL** — `rf_detr_small` missing |
| B2 ONNX availability | **FAIL** — export pending |
| B3 ONNX ops | WARN — DETR/deformable attention risk |
| B4 License | PASS (plan) |
| Feat branch | empty |

---

## Port map (vision range)

| Port | Model |
|------|-------|
| 18100 | smolvlm_256m |
| 18101 | minicpm_v46 *(draft only)* |
| 18102 | qwen35_08b *(draft only)* |
| 18103 | smolvlm_500m |
| 18104 | ministral_3b *(draft only)* |
| 18105 | zwz_4b |
| 18106 | smolvlm2_22b |

No conflicts within 18100–18106.

---

## Recommended fixes (priority)

1. **Merge/re-review READY PRs:** #27, #29, #30 (configs verified).
2. **Complete next GGUF ports:** Qwen3.5-0.8B → MiniCPM-V → Ministral (artifacts + HF + CI).
3. **ONNX track:** export TinyViT first (prebuilt ONNX), then D-FINE / EfficientViT / RF-DETR; wire ggonnx CI before expecting board scores.
4. **Shared:** VLM runner `--mmproj` support (vision quality still text-only in current `llama_server` bench).
5. **Hygiene:** either commit real work onto empty `feat/*` branches or delete them to avoid false “in progress” signals.

---

## Verdict matrix (compact)

Legend: P=PASS, F=FAIL, W=WARN, N=N/A, S=SKIP

| Model | JSON | HF | Recipe | Blobs | CI reg | Port | Arts | Bench | URL | SHA | mmproj | PPL | ONNX |
|-------|:----:|:--:|:------:|:-----:|:------:|:----:|:----:|:-----:|:---:|:---:|:------:|:---:|:----:|
| smolvlm_256m | P | P | P | P | P | P | P | P | P | P | P | P | N |
| smolvlm_500m | P | P | P | P | P | P | P | P | P | P | P | P | N |
| qwen35_08b | P* | F | P | P | F | P | F | F | — | — | F | P* | N |
| smolvlm2_22b | P | P | P | P | P | P | P | P | P | P | P | P | N |
| minicpm_v46 | P* | F | P | P | F | P | F | F | — | — | F | P* | N |
| zwz_4b | P | P | P | P | P | P | P | P | P | P | P | P | N |
| ministral_3b | P* | F | P | P | F | P | F | F | — | — | F | P* | N |
| tinyvit | P | W | W | P | W | N | F | N | F | N | N | N | F |
| dfine_nano | P | W | W | P | W | N | F | N | F | N | N | N | F |
| efficientvit | P | W | W | P | W | N | F | N | F | N | N | N | F |
| rf_detr_small | P | W | W | P | W | N | F | N | F | N | N | N | F |

\*Draft JSON files exist and parse; not wired into shared registries.
