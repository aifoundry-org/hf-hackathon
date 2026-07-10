# Porting Plan: SmolVLM-500M-Instruct

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #4 — Sweet spot of quality vs size, reuses SmolLM2-360M
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 500M total |
| **LLM backbone** | **SmolLM2-360M-Instruct** (already ported!) |
| **Vision encoder** | SigLIP ~93M params |
| **Projector** | MLP (2 layers, pixel-shuffle 3×3) |
| **Architecture** | Idefics3-based |
| **Input patches** | 512×512, 64 visual tokens/image |
| **Context** | 8,192 tokens |
| **License** | **Apache-2.0** |
| **HF Repo** | `HuggingFaceTB/SmolVLM-500M-Instruct` |

## GGUF Files (from `ggml-org/SmolVLM-500M-Instruct-GGUF`)

| File | Size |
|------|------|
| `SmolVLM-500M-Instruct-Q8_0.gguf` | 437 MB |
| `SmolVLM-500M-Instruct-F16.gguf` | 820 MB |
| `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` | 109 MB |
| `mmproj-SmolVLM-500M-Instruct-F16.gguf` | 199 MB |
| **Total (Q8_0 + mmproj Q8_0)** | **546 MB** |
| **Total (F16 + mmproj F16)** | **1.02 GB** |

## Benchmarks

| Benchmark | SmolVLM 256M | **SmolVLM 500M** | SmolVLM 2.2B |
|-----------|-------------|------------------|-------------|
| OCRBench | 52.6 | **61.0** | 65.5 |
| DocVQA Val | 58.3 | **70.5** | 79.7 |
| TextVQA Val | 49.9 | **60.5** | 72.1 |
| MMMU | 28.3 | **33.7** | 38.3 |
| ScienceQA | 73.6 | **79.7** | 84.5 |
| AI2D | 47.0 | **59.5** | 64.0 |

The jump from 256M → 500M is substantial: +8.4 OCRBench, +12.2 DocVQA, +12.5 AI2D.

---

## Porting Steps

Same pattern as SmolVLM-256M. Key differences:

- **Artifact IDs**: `smolvlm_500m_q8_gguf`, `smolvlm_500m_mmproj_q8`
- **HF Base**: `ggml-org/SmolVLM-500M-Instruct-GGUF`
- **LLM backbone**: SmolLM2-360M-Instruct (already ported)
- **Vision encoder**: Same SigLIP ~93M as 256M variant (same mmproj size)

### Benchmark Config Notes

- `gpu_layers`: 99
- Q8_0 is safe here (500M is large enough that quantization impact is moderate)
- Can also test F16 since total is only ~1 GB

## Infrastructure Reuse — HIGH

- **SmolLM2-360M** is already ported in the hackathon
- Same `idefics3` projector type as 256M
- Only the vision encoder + projector are new
- The 256M and 500M share the same SigLIP ~93M encoder — only the projector MLP weights differ slightly

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Q8 quantization quality at 500M | **Low-Medium** | 500M is above the danger threshold; Q8_0 should be fine. F16 available as fallback |
| Shared `idefics3` arch support needed | **Medium** | Same risk as 256M; solve once, both work |

## Strategy: Port 256M + 500M Together

Since both use the same SigLIP encoder and `idefics3` projector type, and both reuse already-ported SmolLM2 backbones, these two can share a single PR:

1. Add both to `artifacts.json`
2. Create two benchmark configs
3. Both register in `benchmark_config.json`
4. Single submission recipe doc covering both sizes

**However**, the hackathon guideline says "one model per PR." So plan for two PRs, but the infrastructure work (runner extension for mmproj) is shared.
