# Porting Plan: Qwen3-VL-2B-Instruct

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Model ID:** `qwen3vl_2b` · **Port:** 18108
**Status:** Vision harness wired — PR open (#73); depends on identity fix (#112)

> Chosen as the small-Qwen VLM after the originally-planned **Qwen3.5-0.8B** was
> found to publish **no `mmproj`** on `ggml-org` (LLM-only GGUFs), which blocks
> the VLM path. Qwen3-VL-2B-Instruct ships both the LLM and mmproj GGUFs and uses
> the same `qwen3vl` architecture family already proven by ZwZ-4B and Qwen3-8B.

## Model Overview

| Property | Value |
|----------|-------|
| Parameters | **1.72 B** LLM (`n_vocab` 151936) + native Qwen3-VL vision encoder |
| Architecture | `qwen3vl` — Qwen3 text decoder + vision encoder + `qwen3vl_merger` |
| Identity schema | Locked to #112 (`metadata_key_prefix`, nested `parameter_count`/`vocabulary`, empty `general.name` OK) |
| License | Apache-2.0 |
| HF base | `Qwen/Qwen3-VL-2B-Instruct` |
| GGUF repo | `ggml-org/Qwen3-VL-2B-Instruct-GGUF` |
| Revision | `ea6a11058182570be6436b9a2e4ee7f7b49f908d` |

## GGUF files

| File | Size | SHA256 |
|------|------|--------|
| `Qwen3-VL-2B-Instruct-Q8_0.gguf` | 1,834,427,296 B | `b7802e29…813c39` |
| `mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf` | 445,053,056 B | `69066c8f…ca9e7b` |
| **Total** | **~2.28 GB** | |

## Why this model

- **Real VLM with ready mmproj** — no local conversion needed; both files pinned verbatim.
- **Proven architecture** — `qwen3vl` matches ZwZ-4B (PR #30); Qwen3 decoder matches Qwen3-8B (merged, PR #11).
- **Fits budget** — ~2.28 GB total, well under the ~10 GiB GGUF limit.
- **Apache-2.0** — clean license.

## Porting steps (done)

1. Pinned HF refs in `docs/HF_REFERENCES.md`.
2. Added `qwen3vl_2b_q8_gguf` + `qwen3vl_2b_mmproj_q8` to `ported_models/llama_cpp_et/artifacts.json`.
3. Created `ported_models/llama_cpp_et/benchmarks/qwen3vl_2b.json` (port 18108).
4. Registered `qwen3vl_2b` in `.github/ci/benchmark_config.json`.
5. Wrote submission recipe `ported_models/llama_cpp_et/docs/qwen3vl_2b.md`.
6. Switched board path to main-owned `smolvlm2_video` + `.github/ci/reference/qwen3vl_2b.json` (COCO oracle, `pmc_cycles`).
7. Opened companion [#112](https://github.com/aifoundry-org/hf-hackathon/pull/112) so identity accepts `qwen3vl.*` GGUF keys / new schema.
8. Aligned `.github/ci/reference/qwen3vl_2b.json` architecture to the #112 identity schema; board path stays on `smolvlm2_video` (not text-only).

## Risks / follow-ups

| Risk | Mitigation |
|------|------------|
| Identity gate hard-coded `llama.*` keys | Companion PR #112 generalizes to `{arch}.*` |
| ET vision kernel coverage / fallbacks | Qwen3-VL vision graph uses `CONCAT` / `ROPE` / `UPSCALE` (m-RoPE + merger), unlike idefics3’s `IM2COL`/`NORM`/`UNARY` path that SmolVLM ET already covers. Board run 30007183737 hit CPU fallbacks and invalid `pmc_cycles`; needs matching ET kernels in `llama.cpp-et` (same class of gap fixed for SmolVLM in `cc4049d`). |
| Early EOS on performance decode | Contract declares `performance.ignore_eos: true` (and `llama_server.ignore_eos`) for 3 fixed tokens, but **main's `smolvlm2_video` runner does not yet read those keys** — only `make_request(..., ignore_eos=)` exists. Board run 30007183737 stopped at 2/3 tokens. **Maintainer request (@AFOliveira):** wire `performance.ignore_eos` / `llama_server.ignore_eos` in the protected runner (participant PRs cannot edit it). |
| Port clash with `smolvlm2_500m_video` (18107) | Qwen uses **18108** |
