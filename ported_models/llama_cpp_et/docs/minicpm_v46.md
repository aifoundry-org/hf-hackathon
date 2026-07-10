# MiniCPM-V 4.6 Q8_0 — llama.cpp-et board benchmark

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [openbmb/MiniCPM-V-4.6-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf) |
| Revision | `78e02f066e9819a60573b78a4275df8a0c27f698` |
| Files | `MiniCPM-V-4_6-Q8_0.gguf` (LLM), `mmproj-model-f16.gguf` (vision projector) |
| License | apache-2.0 |
| Export step | None (upstream GGUF used as-is) |

## Architecture

MiniCPM-V 4.6 is a vision-language model (VLM) with three components:

| Component | Model | Details |
|-----------|-------|---------|
| LLM | Qwen3.5-0.8B | 24 layers, hidden 1024, **hybrid GatedDeltaNet + attention** |
| Vision encoder | SigLIP2-400M | 27 layers, hidden 1152 |
| Projector | LLaVA-UHD v4 merger | Window-attn ViT + DownsampleMLP, 4× spatial merge |

The LLM backbone uses a novel hybrid GatedDeltaNet architecture — a mixture of
linear (DeltaNet-style SSM) and standard transformer attention layers. This is
the first VLM with this architecture class ported to the ET-SoC1 board.

## Deployment size

| Artifact | Size |
|----------|------|
| LLM Q8_0 GGUF | 811,591,616 bytes (~774 MB) |
| Vision projector F16 GGUF | 1,108,746,944 bytes (~1.06 GB) |
| **Total** | **~1.83 GB** |

## llama.cpp support

MiniCPM-V 4.6 (`MiniCPMV4_6ForConditionalGeneration`) is supported via
[llama.cpp PR #22529](https://github.com/ggml-org/llama.cpp/pull/22529),
merged May 6, 2026.

## ET backend settings

Based on Qwen3 family benchmarks and VLM-specific considerations:
`device=ET`, `gpu_layers=99`, completion API, `ctx_size=2048`,
`batch_size=256`, `ubatch_size=128`. Extended timeouts for the ~1.83 GB
combined weight load.

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `minicpm_v46_q8_gguf` and `minicpm_v46_mmproj_f16` artifacts
- `ported_models/llama_cpp_et/benchmarks/minicpm_v46.json` — board runner config
- `.github/ci/benchmark_config.json` — `minicpm_v46` model key
- `docs/HF_REFERENCES.md` — showcase row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models minicpm_v46 --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llama_server_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Similar benchmark: `benchmarks/qwen3_8b.json`
- llama.cpp PR: https://github.com/ggml-org/llama.cpp/pull/22529
