# Porting Plan: SmolVLM2-2.2B-Instruct

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #9 — Best SmolVLM quality but larger footprint
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 2.2B total |
| **LLM backbone** | SmolLM2-1.7B-Instruct |
| **Vision encoder** | SigLIP ~400M params (shape-optimized) |
| **Projector** | MLP (2 layers, pixel-shuffle 3×3) |
| **Architecture** | Idefics3-based |
| **Input patches** | 384×384, 81 visual tokens/image |
| **Context** | 16,384 tokens (v1), 8,192 (v2) |
| **License** | **Apache-2.0** |
| **HF Repo** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |

## GGUF Files (from `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`)

| File | Size |
|------|------|
| `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` | 1.11 GB |
| `SmolVLM2-2.2B-Instruct-Q8_0.gguf` | 1.93 GB |
| `SmolVLM2-2.2B-Instruct-F16.gguf` | 3.63 GB |
| `mmproj-Q8_0.gguf` | ~370 MB (est.) |
| `mmproj-F16.gguf` | ~720 MB (est.) |
| **Total (Q4_K_M + mmproj Q8_0)** | **~1.5 GB** |
| **Total (Q8_0 + mmproj F16)** | **~2.65 GB** |

## Benchmarks

| Benchmark | SmolVLM 256M | SmolVLM 500M | **SmolVLM 2.2B** | SmolVLM2 2.2B |
|-----------|-------------|-------------|-----------------|--------------|
| OCRBench | 52.6 | 61.0 | 65.5 | **72.9** |
| DocVQA Val | 58.3 | 70.5 | 79.7 | **79.98** |
| TextVQA Val | 49.9 | 60.5 | 72.1 | **73.21** |
| MMMU | 28.3 | 33.7 | 38.3 | **42.0** |
| MathVista | 35.9 | 40.1 | 43.9 | **51.5** |
| ScienceQA | 73.6 | 79.7 | 84.5 | **90.0** |

## Video Support (SmolVLM2 only)

SmolVLM**2** (the "2" variants) adds video understanding:

| Benchmark | SmolVLM2 256M | SmolVLM2 500M | SmolVLM2 2.2B |
|-----------|--------------|--------------|--------------|
| Video-MME | 33.7 | 42.2 | **52.1** |
| MLVU | 40.6 | 47.3 | **55.2** |
| MVBench | 32.7 | 39.73 | **46.27** |

Video works by feeding image frame sequences (no separate video decoder in llama.cpp).

---

## Porting Steps

Same pattern as SmolVLM-256M/500M with these adjustments:

- **Artifact IDs**: `smolvlm2_22b_q8_gguf`, `smolvlm2_22b_mmproj_f16`
- **HF Base**: `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`
- **Recommended quantization**: Q4_K_M (1.11 GB) + mmproj Q8_0 (~370 MB) = ~1.5 GB total
- **LLM backbone**: SmolLM2-1.7B — check if this is ported; if not, port SmolLM2-1.7B first

### Benchmark Config Notes

- `ctx_size`: 4096 (model supports 16K but reduce for board DRAM)
- `gpu_layers`: 99
- `ready_timeout_s`: 900 (larger model needs more load time)

## Infrastructure Reuse — MEDIUM

- SmolLM2-1.7B may or may not be ported yet (check `artifacts.json`)
- If not ported, need to add SmolLM2-1.7B text-only benchmark first
- Same `idefics3` projector type — infrastructure from 256M/500M ports applies

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Larger model (~2.65 GB at Q8) | **Medium** | Use Q4_K_M to reduce to ~1.5 GB |
| SmolLM2-1.7B may not be ported yet | **Medium** | Port SmolLM2-1.7B text-only first as a prerequisite |
| Board DRAM may be tight | **Medium** | Start with Q4_K_M, verify it runs, then try Q8 |
