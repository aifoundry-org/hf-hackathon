# ZwZ-4B — Porting Plan

## Model Overview

| Property | Value |
|----------|-------|
| **HF Repo** | `inclusionAI/ZwZ-4B-GGUF` |
| **Params** | ~4.7B total |
| **Architecture** | `qwen3vl` (Qwen3 text decoder + SigLIP2-Large + DeepStack projector) |
| **License** | Apache-2.0 |
| **LLM file** | `ZwZ-4B-Q4_K_M.gguf` (~2.72 GB) |
| **mmproj file** | `mmproj-ZwZ-4B-Q8_0.gguf` (~451 MB) |
| **Total deployment** | ~3.32 GB |

## Why ZwZ-4B

- Fine-grained perception VLM optimized for OCR, counting, and small object detection
- Region-to-image distillation for enhanced visual understanding
- Same `qwen3vl` architecture as Qwen3-VL — infrastructure already proven with Qwen3-8B port
- Q4_K_M quantization keeps total deployment under board DRAM constraints

## GGUF Details

### LLM: `ZwZ-4B-Q4_K_M.gguf`

| Property | Value |
|----------|-------|
| Source | `inclusionAI/ZwZ-4B-GGUF` |
| Revision | `af96680eb7d0978e4844e8395cb6fd727f0a1d84` |
| Quantization | Q4_K_M |
| Size | 2,716,069,408 bytes (~2.72 GB) |
| SHA256 | `0c1e633a544708ff3d52060f764ddeccba8a98535f223c40361fddeb59c79d33` |

### mmproj: `mmproj-ZwZ-4B-Q8_0.gguf`

| Property | Value |
|----------|-------|
| Source | `inclusionAI/ZwZ-4B-GGUF` |
| Revision | `af96680eb7d0978e4844e8395cb6fd727f0a1d84` |
| Quantization | Q8_0 |
| Size | 450,828,416 bytes (~451 MB) |
| SHA256 | `ba2f4e1b792ce56bbfc78a322e99fd0efbbbe8ea23479894851ab7a060872f6c` |
| Vision encoder | SigLIP2-Large (~300M params, 24 layers, hidden 1024, patch 16) |

## llama.cpp Support

Fully supported as `qwen3vl` architecture. Same backend as Qwen3-8B and Qwen3-VL-4B.

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Port | 18105 |
| device | ET |
| gpu_layers | 99 |
| ctx_size | 4096 |
| batch_size | 512 |
| ubatch_size | 256 |
| ready_timeout_s | 900 |
| request_timeout_s | 1200 |
| API | completion |
| Perplexity | enabled (WikiText-2, ctx 128, 4 chunks) |

## Infrastructure Reuse

- Same `llama-server` binary and ET backend as Qwen3-8B
- Same WikiText-2 perplexity harness
- Same board deployment workflow
- Same `qwen3vl` architecture support in llama.cpp
- Vision projector loaded via `--mmproj` flag

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Q4_K_M quality degradation | Low | Q8_0 total would be ~5.29 GB (exceeds board DRAM); Q4_K_M is the only viable option |
| Novel model (not upstream Qwen3) | Medium | Architecture is identical `qwen3vl`; only weights differ |
| Vision not benchmarked | Info | Text-only benchmark; vision via mmproj needs runner extension |

## Porting Steps

1. Add `zwz_4b_q4_gguf` and `zwz_4b_mmproj_q8` to `artifacts.json`
2. Create `benchmarks/zwz_4b.json` with port 18105
3. Add `zwz_4b` key to `benchmark_config.json`
4. Add HF reference row to `HF_REFERENCES.md`
5. Create recipe doc at `docs/zwz_4b.md`
6. Run `ci_preflight.sh` to validate
7. Open PR
