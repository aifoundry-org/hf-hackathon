# PR Timeline — MiniCPM-V 4.6

| Field | Value |
|-------|-------|
| **PR** | *Not opened yet* |
| **Branch** | *(planned)* `feat/minicpm-v-4.6` |
| **Status** | **WIP** — plan + draft benchmark/docs exist; not CI-registered |
| **Path** | llama.cpp-et GGUF (preferred) / ggonnx fallback |
| **Plan** | [../minicpm-v-4.6.md](../minicpm-v-4.6.md) |

## Timeline

| When | Event |
|------|-------|
| 2026-07 | Candidate selected; detailed porting plan written |
| 2026-07 | Verification: needs GGUF/mmproj confirmation or ONNX export |
| 2026-07 | Draft files: `ported_models/llama_cpp_et/benchmarks/minicpm_v46.json`, `docs/minicpm_v46.md` |

## Blockers

- Confirm official GGUF + mmproj availability and SHA256 pins
- Or complete ONNX export if staying on ggonnx path (CI not wired yet)

## Next

1. Pin HF revision + hashes in `artifacts.json`
2. Register in `benchmark_config.json`
3. Open PR once preflight passes
