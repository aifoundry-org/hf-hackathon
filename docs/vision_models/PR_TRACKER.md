# Vision Models PR Tracker

This document tracks the status of all vision model porting PRs for the hackathon.

## Overview

We identified 11 vision model candidates that fit the hackathon constraints (ET-SoC1 hardware, open licenses, size limits). After verification, 4 models were ready for immediate PR submission, with the remaining 7 requiring additional work.

## PR Status Summary

| PR # | Model | Status | Branch | Notes |
|------|-------|--------|--------|-------|
| [#26](https://github.com/aifoundry-org/hf-hackathon/pull/26) | SmolVLM-256M | **MERGED** | `feat/smolvlm-256m` | Merged to main |
| [#27](https://github.com/aifoundry-org/hf-hackathon/pull/27) | SmolVLM-500M | **OPEN** | `feat/smolvlm-500m` | Rebased on main, conflicts resolved |
| [#29](https://github.com/aifoundry-org/hf-hackathon/pull/29) | SmolVLM2-2.2B | **OPEN** | `feat/smolvlm2-2.2b` | Awaiting review |
| [#30](https://github.com/aifoundry-org/hf-hackathon/pull/30) | ZwZ-4B | **OPEN** | `feat/zwz-4b` | Rebased on main, conflicts resolved |

## Detailed PR Timeline

### PR #26: SmolVLM-256M-Instruct (Q8_0)

**Status:** MERGED

**Timeline:**
- **2026-07-08 21:45 UTC** - PR created from `Ashish-Soni08:feat/smolvlm-256m`
- **2026-07-08 22:00 UTC** - CI checks passed
- **2026-07-09 09:15 UTC** - Merged to `main` as commit `12ba7ca`

**Changes:**
- Added `smolvlm_256m.json` benchmark config
- Added `smolvlm_256m.md` documentation
- Registered in `artifacts.json` with LLM + mmproj GGUF entries
- Added to `benchmark_config.json` models section
- Pinned HF reference in `HF_REFERENCES.md`

**Files Added:**
- `ported_models/llama_cpp_et/benchmarks/smolvlm_256m.json`
- `ported_models/llama_cpp_et/docs/smolvlm_256m.md`
- `docs/vision_models/smolvlm-256m.md`

**Model Details:**
- Architecture: Idefics3 (SmolLM2-135M text decoder + SigLIP vision encoder)
- Quantization: Q8_0 for both LLM and mmproj
- Size: 175 MB (LLM) + 104 MB (mmproj) = 279 MB total
- License: Apache-2.0
- HF Source: `ggml-org/SmolVLM-256M-Instruct-GGUF`

---

### PR #27: SmolVLM-500M-Instruct (Q8_0)

**Status:** OPEN (Conflicts Resolved)

**Timeline:**
- **2026-07-08 21:50 UTC** - PR created from `Ashish-Soni08:feat/smolvlm-500m`
- **2026-07-08 22:05 UTC** - CI checks passed
- **2026-07-09 09:30 UTC** - Review requested: merge conflicts detected
- **2026-07-09 14:15 UTC** - Rebased on latest `main`, conflicts resolved
- **2026-07-09 14:20 UTC** - Force pushed rebased branch

**Conflicts Resolved:**
- `.github/ci/benchmark_config.json`: Kept YOLO updates from main, added smolvlm_500m entry
- `docs/HF_REFERENCES.md`: Kept smolvlm_256m from main, added smolvlm_500m entry
- `ported_models/llama_cpp_et/artifacts.json`: Kept smolvlm_256m from main, added smolvlm_500m entries

**Changes:**
- Added `smolvlm_500m.json` benchmark config
- Added `smolvlm_500m.md` documentation
- Registered in `artifacts.json` with LLM + mmproj GGUF entries
- Added to `benchmark_config.json` models section
- Pinned HF reference in `HF_REFERENCES.md`

**Files Added:**
- `ported_models/llama_cpp_et/benchmarks/smolvlm_500m.json`
- `ported_models/llama_cpp_et/docs/smolvlm_500m.md`
- `docs/vision_models/smolvlm-500m.md`

**Model Details:**
- Architecture: Idefics3 (SmolLM2-360M text decoder + SigLIP vision encoder)
- Quantization: Q8_0 for both LLM and mmproj
- Size: 437 MB (LLM) + 109 MB (mmproj) = 546 MB total
- License: Apache-2.0
- HF Source: `ggml-org/SmolVLM-500M-Instruct-GGUF`

---

### PR #29: SmolVLM2-2.2B-Instruct (Q4_K_M + Q8_0)

**Status:** OPEN (Awaiting Review)

**Timeline:**
- **2026-07-08 22:00 UTC** - PR created from `Ashish-Soni08:feat/smolvlm2-2.2b`
- **2026-07-08 22:10 UTC** - CI checks passed
- **2026-07-09** - Awaiting review

**Changes:**
- Added `smolvlm2_22b.json` benchmark config
- Added `smolvlm2_22b.md` documentation
- Registered in `artifacts.json` with LLM (Q4_K_M) + mmproj (Q8_0) GGUF entries
- Added to `benchmark_config.json` models section
- Pinned HF reference in `HF_REFERENCES.md`

**Files Added:**
- `ported_models/llama_cpp_et/benchmarks/smolvlm2_22b.json`
- `ported_models/llama_cpp_et/docs/smolvlm2_22b.md`
- `docs/vision_models/smolvlm2-2.2b.md`

**Model Details:**
- Architecture: Idefics3 (SmolLM2-1.7B text decoder + SigLIP vision encoder)
- Quantization: Q4_K_M for LLM, Q8_0 for mmproj
- Size: 1.1 GB (LLM) + 109 MB (mmproj) = 1.2 GB total
- License: Apache-2.0
- HF Source: `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`

---

### PR #30: ZwZ-4B Fine-Grained Perception VLM

**Status:** OPEN (Conflicts Resolved)

**Timeline:**
- **2026-07-08 22:05 UTC** - PR created from `Ashish-Soni08:feat/zwz-4b`
- **2026-07-08 22:15 UTC** - CI checks passed
- **2026-07-09 09:30 UTC** - Review requested: merge conflicts detected
- **2026-07-09 14:20 UTC** - Rebased on latest `main`, conflicts resolved
- **2026-07-09 14:25 UTC** - Force pushed rebased branch

**Conflicts Resolved:**
- `.github/ci/benchmark_config.json`: Kept YOLO updates from main, added zwz_4b entry
- `docs/HF_REFERENCES.md`: Kept smolvlm_256m from main, added zwz_4b entry
- `ported_models/llama_cpp_et/artifacts.json`: Kept smolvlm_256m from main, added zwz_4b entries

**Changes:**
- Added `zwz_4b.json` benchmark config
- Added `zwz_4b.md` documentation
- Registered in `artifacts.json` with LLM (Q4_K_M) + mmproj (Q8_0) GGUF entries
- Added to `benchmark_config.json` models section
- Pinned HF reference in `HF_REFERENCES.md`

**Files Added:**
- `ported_models/llama_cpp_et/benchmarks/zwz_4b.json`
- `ported_models/llama_cpp_et/docs/zwz_4b.md`
- `docs/vision_models/zwz-4b.md`

**Model Details:**
- Architecture: Qwen3VL (Qwen3-4B text decoder ~4B params + SigLIP2-Large vision encoder ~300M)
- Quantization: Q4_K_M for LLM, Q8_0 for mmproj
- Size: 2.7 GB (LLM) + 451 MB (mmproj) = 3.15 GB total
- License: Apache-2.0
- HF Source: `inclusionAI/ZwZ-4B-GGUF`
- Special Features: DeepStack multi-level feature injection at layers [5,11,17]

---

## Models Not Yet PR-Ready

The following 7 models were identified as candidates but require additional work before PR submission:

### 1. MiniCPM-V 4.6 (OpenBMB)
- **Status:** Needs ONNX export
- **Blocker:** No pre-quantized GGUF available; requires custom ONNX export
- **Path:** `ggonnx` ONNX path (not yet CI-wired)
- **Next Steps:** Export to ONNX, validate accuracy, create ggonnx artifacts

### 2. Qwen3.5-0.8B (Alibaba)
- **Status:** Needs SHA256 verification
- **Blocker:** Cannot verify exact SHA256 hash for GGUF files from HF API
- **Path:** `llama.cpp-et` GGUF path
- **Next Steps:** Download GGUF files, compute SHA256, update artifacts.json

### 3. RF-DETR Small (Roboflow)
- **Status:** Needs ONNX export
- **Blocker:** Pure vision model, requires ONNX export
- **Path:** `ggonnx` ONNX path (not yet CI-wired)
- **Next Steps:** Export to ONNX, validate detections, create ggonnx artifacts

### 4. EfficientViT-M0 (MIT HAN Lab)
- **Status:** Needs ONNX export
- **Blocker:** Pure vision model, requires ONNX export
- **Path:** `ggonnx` ONNX path (not yet CI-wired)
- **Next Steps:** Export to ONNX, validate accuracy, create ggonnx artifacts

### 5. D-FINE Nano (ByteDance)
- **Status:** Needs ONNX export
- **Blocker:** Pure vision model (object detection), requires ONNX export
- **Path:** `ggonnx` ONNX path (not yet CI-wired)
- **Next Steps:** Export to ONNX, validate detections, create ggonnx artifacts

### 6. TinyViT-21M (Microsoft)
- **Status:** Needs ONNX export
- **Blocker:** Pure vision model (image classification), requires ONNX export
- **Path:** `ggonnx` ONNX path (not yet CI-wired)
- **Next Steps:** Export to ONNX, validate accuracy, create ggonnx artifacts

### 7. Ministral 3 3B (Mistral)
- **Status:** Needs license verification
- **Blocker:** License unclear; need to confirm Apache-2.0 or similar
- **Path:** `llama.cpp-et` GGUF path (if license OK)
- **Next Steps:** Verify license, check for GGUF availability, update artifacts

---

## Shared Infrastructure Needs

### VLM Runner Extension
All GGUF VLMs (SmolVLM family, ZwZ-4B) require a VLM runner extension to fully exercise vision capabilities in the benchmarking harness. Currently, the benchmark only tests text generation. To properly validate vision models, we need:
- Image loading and preprocessing
- Vision encoder invocation
- Multimodal fusion testing
- Vision-specific accuracy gates

**Status:** Not yet implemented. Tracking issue needed.

### ggonnx CI Runner
All ONNX models (MiniCPM-V, RF-DETR, EfficientViT, D-FINE, TinyViT) require the `ggonnx` ONNX Runtime Execution Provider to be wired into the CI system. Currently:
- `ggonnx` artifacts are registered in `ported_models/ggonnx/artifacts.json`
- ONNX models are NOT registered in `.github/ci/benchmark_config.json`
- No CI runner exists for ONNX models

**Status:** Infrastructure work needed before ONNX models can be benchmarked.

---

## Next Steps

1. **Immediate:**
   - Monitor PR #27 (SmolVLM-500M) for review feedback
   - Monitor PR #29 (SmolVLM2-2.2B) for review feedback
   - Monitor PR #30 (ZwZ-4B) for review feedback

2. **Short-term:**
   - Verify Qwen3.5-0.8B SHA256 hashes and create PR
   - Verify Ministral 3 3B license and create PR (if Apache-2.0)

3. **Medium-term:**
   - Export ONNX models for MiniCPM-V, RF-DETR, EfficientViT, D-FINE, TinyViT
   - Wire `ggonnx` ONNX runner into CI
   - Create PRs for ONNX models once CI is ready

4. **Long-term:**
   - Implement VLM runner extension for vision capability testing
   - Add vision-specific benchmark cases and accuracy gates

---

## Cleanup Notes

**Completed:**
- Rebased `feat/smolvlm-500m` on latest main (2026-07-09 14:15 UTC)
- Rebased `feat/zwz-4b` on latest main (2026-07-09 14:20 UTC)
- Resolved all merge conflicts in shared config files

**Pending:**
- Remove `_wip/` directory (backups from selective commit process)
- Remove `extract_model.py` script
- Clean up untracked vision model files from other candidates
