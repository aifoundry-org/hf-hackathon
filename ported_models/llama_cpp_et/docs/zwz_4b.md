# ZwZ-4B — llama.cpp-et board benchmark

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [inclusionAI/ZwZ-4B-GGUF](https://huggingface.co/inclusionAI/ZwZ-4B-GGUF) |
| Revision | `af96680eb7d0978e4844e8395cb6fd727f0a1d84` |
| LLM file | `ZwZ-4B-Q4_K_M.gguf` (~2.72 GB / 2,716,069,408 bytes) |
| mmproj file | `mmproj-ZwZ-4B-Q8_0.gguf` (~451 MB / 450,828,416 bytes) |
| Total | ~3.32 GB (Q4_K_M + Q8_0 mmproj) |
| License | Apache-2.0 |
| Export step | None (official GGUF with quantized mmproj) |

## Architecture

ZwZ-4B is a fine-grained perception vision-language model with three components:

| Component | Model | Details |
|-----------|-------|---------|
| LLM | Qwen3 text decoder | ~4B params, 36 layers, hidden 2560, 32Q/8KV heads, SwiGLU, tied embeddings |
| Vision encoder | SigLIP2-Large | ~300M params, 24 layers, hidden 1024, patch 16, temporal patch 2 |
| Projector | DeepStack injection | Multi-level feature injection at layers [5,11,17], interleaved MRoPE |

**Total parameters**: ~4.7B

**Architecture family**: `qwen3vl` — same as Qwen3-VL, sharing infrastructure with the existing Qwen3-8B port.

**llama.cpp support**: Fully supported as `qwen3vl` architecture.

## ET backend settings

Mirrors the `qwen3_8b` configuration (same `qwen3vl` architecture family):
`device=ET`, `gpu_layers=99`, completion API, `ctx_size=4096`.

| Parameter | Value |
|-----------|-------|
| device | ET |
| gpu_layers | 99 |
| ctx_size | 4096 |
| batch_size | 512 |
| ubatch_size | 256 |
| port | 18105 |
| ready_timeout_s | 900 |
| request_timeout_s | 1200 |
| flash_attn | false |

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `zwz_4b_q4_gguf` and `zwz_4b_mmproj_q8` artifacts
- `ported_models/llama_cpp_et/benchmarks/zwz_4b.json` — board runner config
- `.github/ci/benchmark_config.json` — `zwz_4b` model key
- `docs/HF_REFERENCES.md` — HuggingFace reference row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models zwz_4b --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llama_server_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Similar architecture: `benchmarks/qwen3_8b.json`
- Official HF repo: https://huggingface.co/inclusionAI/ZwZ-4B
- Official GGUF: https://huggingface.co/inclusionAI/ZwZ-4B-GGUF
