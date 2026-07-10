# Ministral 3 3B Instruct Q8_0 — llama.cpp-et board benchmark

## First Mistral-family model in the hackathon

Ministral 3 3B is a vision-language model (VLM) by Mistral AI, ported to the
ET-SoC1 board. It uses the `mistral3` architecture combining a 3.4B parameter
LLM decoder with a 410M parameter Pixtral ViT vision encoder (frozen from
Mistral Small 3.1). This is the first Mistral-family model in the hackathon.

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo (original) | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) |
| Repo (GGUF) | [mistralai/Ministral-3-3B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF) |
| Revision | `eb599d408350ea2bb60452cb86be7c7b2fc28227` |
| LLM file | `Ministral-3-3B-Instruct-2512-Q8_0.gguf` (3.65 GB) |
| mmproj file | `Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf` (842 MB) |
| License | Apache-2.0 |
| Export step | None (official GGUF from Mistral used as-is) |

## Architecture

| Component | Details |
|-----------|---------|
| **LLM** | 3.4B params, 26 layers, hidden 3072, 32Q/8KV heads, SwiGLU, tied embeddings, 131K vocab |
| **Context** | 256K tokens (YaRN rope scaling) |
| **Vision encoder** | Pixtral ViT, 410M params, patch 16, 2D RoPE, frozen from Mistral Small 3.1 |
| **Projector** | Retrained per-size projection |
| **Total params** | ~3.8B |
| **Architecture** | `mistral3` in llama.cpp |

## llama.cpp Support

- **PR [#17644](https://github.com/ggerganov/llama.cpp/pull/17644)** merged Dec 1, 2025 (collaboration with Mistral AI)
- **PR [#17945](https://github.com/ggerganov/llama.cpp/pull/17945)** fixed `attn_factor` for correct attention scaling
- Architecture: `mistral3` (native VLM support in llama.cpp)

## ET backend settings

| Parameter | Value |
|-----------|-------|
| device | ET |
| gpu_layers | 99 |
| ctx_size | 2048 (model supports 256K, reduced for board DRAM) |
| batch_size | 256 |
| ubatch_size | 128 |
| port | 18104 |
| ready_timeout_s | 600 |
| request_timeout_s | 900 |

## Deployment size

| Quantization | LLM | mmproj | Total |
|-------------|-----|--------|-------|
| Q4_K_M + BF16 mmproj | 2.15 GB | 842 MB | ~2.99 GB |
| **Q8_0 + BF16 mmproj** | **3.65 GB** | **842 MB** | **~4.49 GB** |
| BF16 + BF16 mmproj | 6.87 GB | 842 MB | ~7.71 GB |

The Q8_0 + BF16 mmproj combination (~4.49 GB) is used for this benchmark as the
near-lossless option. A Q4_K_M variant (~2.99 GB total) is available from the
official Mistral GGUF repo for tighter DRAM budgets.

## Benchmarks (upstream)

| Benchmark | Ministral 3 3B |
|-----------|---------------|
| MM MTBench | 7.83 |
| Arena Hard | 0.305 |
| MATH | 0.830 |

## Notes

- **Official BF16 mmproj only** (842 MB): Mistral provides only BF16 quantization
  for the vision projector. Community quants (Q4_K_M, Q8_0 mmproj) are available
  from [unsloth](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF).
- **llama-cpp-python**: Not relevant for this hackathon (Ministral 3 support is
  not yet available in llama-cpp-python).
- **Known issues**: The `attn_factor` fix in llama.cpp PR #17945 is required for
  correct attention scaling. Ensure the ET backend includes this fix.
- **Vision capability**: The benchmark exercises text-only generation via the
  completion API. Vision inference via `--mmproj` is available but not exercised
  by the standard text harness.

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `ministral_3b_q8_gguf` and `ministral_3b_mmproj_bf16` artifacts
- `ported_models/llama_cpp_et/benchmarks/ministral_3b.json` — board runner config
- `.github/ci/benchmark_config.json` — `ministral_3b` model key
- `docs/HF_REFERENCES.md` — HuggingFace reference row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models ministral_3b --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llamaserver_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- llama.cpp mistral3 support: PR #17644, attn_factor fix PR #17945
- Similar VLM ports: `benchmarks/minicpm_v46.json`, `benchmarks/smolvlm_256m.json`
