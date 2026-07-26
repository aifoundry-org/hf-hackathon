# Porting Plan: SmolVLM2-2.2B-Instruct

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)  
**Priority:** Best SmolVLM quality; larger DRAM footprint  
**Status:** Redesigned to `smolvlm2_video` + COCO oracle pattern (#27 / `smolvlm_500m`)

---

## Model Overview

| Property | Value |
|----------|-------|
| **LLM backbone** | SmolLM2-1.7B-Instruct (`llama`, 1.81 B language params in GGUF loader) |
| **Vision encoder** | SigLIP-class (n_embd 1152, 27 layers, patch 14, image 384) |
| **Projector** | `idefics3` (projection_dim 2048) |
| **Architecture** | Idefics3-based SmolVLM2 |
| **License** | Apache-2.0 |
| **HF Repo** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4` |

## GGUF pins (`ggml-org/SmolVLM2-2.2B-Instruct-GGUF` @ `1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91`)

| Artifact | File | SHA256 | Bytes |
|----------|------|--------|-------|
| `smolvlm2_22b_q4_gguf` | `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` | `0cf76814555b8665149075b74ab6b5c1d428ea1d3d01c1918c12012e8d7c9f58` | 1112602656 |
| `smolvlm2_22b_mmproj_q8` | `mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf` | `ae07ea1facd07dd3230c4483b63e8cda96c6944ad2481f33d531f79e892dd024` | 592523200 |

## Board contract

- Runner: `smolvlm2_video` (not plain `llama_server`)
- Metric: `pmc_cycles` (lower is better)
- Port: **18106**
- Prompt: `<|im_start|>User: {media_markers}\n{question}<end_of_utterance>\nAssistant:`
- `processor_image_size`: **384**
- COCO oracle + order pair (same fixtures as `smolvlm_500m`)
- Full offload + zero vision fallbacks required

See `.github/ci/reference/smolvlm2_22b.json` and
`ported_models/llama_cpp_et/docs/smolvlm2_22b.md` for fingerprint provenance
and remaining TBD items (PPL measure, maintainer `head_count_kv` gate for GQA=1).
