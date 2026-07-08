# SmolVLM2-2.2B-Instruct Q4_K_M — llama.cpp-et board benchmark

## The flagship VLM

SmolVLM2-2.2B-Instruct is the largest variant in the SmolVLM family and the
first with native video support (frame sequences, no separate video decoder).
It uses the Idefics3 architecture with a significantly larger SigLIP vision
encoder (~400M params, shape-optimized compared to the ~93M encoder in the
256M/500M variants) and an MLP projector with pixel-shuffle 3×3. The LLM
backbone is SmolLM2-1.7B-Instruct, already proven on ET-SoC1 as
`smollm2_17b_q8_gguf`.

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [ggml-org/SmolVLM2-2.2B-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF) |
| Revision | `1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91` |
| LLM file | `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` (~1.04 GiB / 1,112,602,656 bytes) |
| mmproj file | `mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf` (~565 MiB / 592,523,200 bytes) |
| Total Q4_K_M + Q8_0 mmproj | ~1.59 GiB (1,705,125,856 bytes) |
| License | Apache-2.0 |
| Export step | None (upstream GGUF used as-is) |

## Architecture

- **Architecture family**: Idefics3 (projector type `idefics3`)
- **Vision encoder**: SigLIP ~400M parameters (shape-optimized, larger than 256M/500M variants)
- **LLM backbone**: SmolLM2-1.7B-Instruct (HuggingFace transformer decoder, already ported)
- **Projector**: MLP (2 layers) with pixel-shuffle 3×3, bridging SigLIP vision tokens to LLM embedding space
- **Input patches**: 384×384, 81 visual tokens per image
- **Context**: 8,192 tokens (v2)
- **Video support**: Frame sequences (no separate video decoder module)
- **llama.cpp support**: PR [#13050](https://github.com/ggerganov/llama.cpp/pull/13050) merged April 22, 2025

## ET backend settings

Extends the SmolVLM family pattern with higher context and longer timeouts for
the larger model:
`device=ET`, `gpu_layers=99`, completion API, `ctx_size=4096`.

| Parameter | Value |
|-----------|-------|
| device | ET |
| gpu_layers | 99 |
| ctx_size | 4096 |
| batch_size | 256 |
| ubatch_size | 128 |
| port | 18106 |
| ready_timeout_s | 300 |
| request_timeout_s | 600 |

## Total deployment size

At Q4_K_M (LLM) + Q8_0 (mmproj) the full VLM stack weighs approximately
1.59 GiB — tight but feasible on the ET-SoC1 board DRAM. The Q4_K_M
quantization was chosen specifically to keep the total under the board's
memory budget while maintaining acceptable quality for the 1.7B backbone.

## Benchmarks

SmolVLM2-2.2B achieves the best scores in the SmolVLM family:

| Benchmark | Score |
|-----------|-------|
| OCRBench | 72.9 |
| DocVQA | 79.98 |
| ScienceQA | 90.0 |
| Video-MME | 52.1 |

These represent a significant quality jump over the 500M variant and make it
the most capable VLM in the SmolVLM lineup for document understanding, OCR,
visual question answering, and video comprehension.

## Prerequisites

- **SmolLM2-1.7B text-only benchmark**: The LLM backbone (`smollm2_17b_q8_gguf`)
  is already registered in artifacts.json. A text-only benchmark for
  SmolLM2-1.7B should exist (`smollm2_17b` in benchmark_config.json) to
  validate the backbone before running the full VLM benchmark.

## Infrastructure reuse

This port reuses infrastructure already established for the SmolLM2-1.7B row:
- Same llama-server binary and ET backend
- Same WikiText-2 perplexity harness
- Same board deployment workflow
- Vision projector loaded via `--mmproj` flag in llama-server
- Same Idefics3 architecture support as SmolVLM 256M/500M

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `smolvlm2_22b_q4_gguf` and `smolvlm2_22b_mmproj_q8` artifacts
- `ported_models/llama_cpp_et/benchmarks/smolvlm2_22b.json` — board runner config
- `.github/ci/benchmark_config.json` — `smolvlm2_22b` model key
- `docs/HF_REFERENCES.md` — HuggingFace reference row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models smolvlm2_22b --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llamaserver_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Same LLM backbone: `benchmarks/smollm2_17b.json`
- Smaller variants: `benchmarks/smolvlm_256m.json`, `benchmarks/smolvlm_500m.json`
- llama.cpp Idefics3 support: PR #13050
