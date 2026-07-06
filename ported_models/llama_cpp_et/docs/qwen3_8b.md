# Qwen3-8B Q8_0 — llama.cpp-et board benchmark

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [ggml-org/Qwen3-8B-GGUF](https://huggingface.co/ggml-org/Qwen3-8B-GGUF) |
| Revision | `2473489dc243ccaffb4ce569c55bf1df66b2088f` |
| File | `Qwen3-8B-Q8_0.gguf` |
| License | apache-2.0 |
| Export step | None (upstream Q8_0 GGUF used as-is) |

## Files added

- `ported_models/llama_cpp_et/artifacts.json` — `qwen3_8b_q8_gguf` artifact
- `ported_models/llama_cpp_et/benchmarks/qwen3_8b.json` — board runner config
- `.github/ci/benchmark_config.json` — `qwen3_8b` model key

## ET backend settings

Mirrors existing Qwen3 rows (`qwen3_4b`) and the 8B Llama row (`llama31_8b`):
`device=ET`, `gpu_layers=99`, completion API, longer timeouts for the ~8.7 GiB
weight file.

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models qwen3_8b --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llama_server_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Similar Qwen3 benchmark: `benchmarks/qwen3_4b.json`
- Similar 8B benchmark: `benchmarks/llama31_8b.json`
