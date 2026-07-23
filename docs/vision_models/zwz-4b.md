# Porting Plan: ZwZ-4B

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Model ID:** `zwz_4b` · **Port:** 18105
**Status:** Vision harness wired — PR open (#30); depends on identity fix (#112). Prefer ET after #73 is green.

## Model Overview

| Property | Value |
|----------|-------|
| Parameters | **4.41 B** LLM (`n_vocab` 151936) + Qwen3-VL vision encoder / `qwen3vl_merger` |
| Architecture | `qwen3vl` — Qwen3 text decoder (36 blocks, emb 2560) + vision (24 layers, emb 1024, projection_dim **2560**) |
| Identity schema | Locked to #112 (`metadata_key_prefix`, nested `parameter_count`/`vocabulary`, empty `general.name` OK) |
| License | Apache-2.0 |
| HF base | `inclusionAI/ZwZ-4B` |
| GGUF repo | `inclusionAI/ZwZ-4B-GGUF` |
| GGUF revision | `af96680eb7d0978e4844e8395cb6fd727f0a1d84` |

## GGUF files

| File | Size | SHA256 |
|------|------|--------|
| `ZwZ-4B-Q4_K_M.gguf` | 2,716,069,408 B | `0c1e633a…c79d33` |
| `mmproj-ZwZ-4B-Q8_0.gguf` | 450,828,416 B | `ba2f4e1b…872f6c` |
| **Total** | **~3.17 GB** | |

## Why this model

- Fine-grained perception VLM (OCR / counting / small objects) on the proven `qwen3vl` family.
- Official GGUF ships both LLM and mmproj — no local conversion.
- Q4_K_M + Q8_0 mmproj keeps deployment under board DRAM while still exercising real vision.

## Porting steps (done)

1. Pinned HF refs in `docs/HF_REFERENCES.md`.
2. Added `zwz_4b_q4_gguf` + `zwz_4b_mmproj_q8` to `ported_models/llama_cpp_et/artifacts.json`.
3. Rewrote `ported_models/llama_cpp_et/benchmarks/zwz_4b.json` onto `smolvlm2_video` (port **18105**).
4. Registered `zwz_4b` in `.github/ci/benchmark_config.json` (kept `smolvlm_500m`).
5. Added `.github/ci/reference/zwz_4b.json` (#112 identity schema + COCO oracle).
6. Updated submission recipe `ported_models/llama_cpp_et/docs/zwz_4b.md`.

## Risks / follow-ups

| Risk | Mitigation |
|------|------------|
| Identity gate hard-coded `llama.*` keys | Companion PR #112 generalizes to `{arch}.*` + nested schema |
| PPL first-run baseline not yet host-measured | Loose `max_ppl=100` until host smoke fills `first_run_*` |
| ET vision kernel coverage / fallbacks | Same `require_zero_vision_fallbacks` gate as SmolVLM / Qwen3-VL-2B |
| Sequencing vs #73 | Do not request ET until #112 merged + host smoke green; prefer after #73 ET-green |
