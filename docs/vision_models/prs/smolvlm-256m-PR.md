# PR Timeline — SmolVLM-256M-Instruct

| Field | Value |
|-------|-------|
| **PR** | [#26](https://github.com/aifoundry-org/hf-hackathon/pull/26) |
| **Branch** | `Ashish-Soni08:feat/smolvlm-256m` |
| **Status** | **MERGED** |
| **Path** | llama.cpp-et GGUF (LLM Q8_0 + mmproj Q8_0) |
| **Plan** | [../smolvlm-256m.md](../smolvlm-256m.md) |

## Timeline

| When (UTC) | Event |
|------------|-------|
| 2026-07-08 | Porting plan written; artifacts/benchmark/docs prepared |
| 2026-07-08 ~21:55 | PR #26 opened |
| 2026-07-08 ~22:00 | CI / board checks passed |
| 2026-07-09 | Merged to `main` (`2ac1420` / `12ba7ca`) |

## What landed

- `ported_models/llama_cpp_et/benchmarks/smolvlm_256m.json`
- `ported_models/llama_cpp_et/docs/smolvlm_256m.md`
- `docs/vision_models/smolvlm-256m.md`
- Entries in `artifacts.json`, `benchmark_config.json`, `HF_REFERENCES.md`

## Model snapshot

- Arch: Idefics3 (SmolLM2-135M + SigLIP)
- Size: ~279 MB total (Q8_0)
- License: Apache-2.0
- HF: `ggml-org/SmolVLM-256M-Instruct-GGUF`

## Notes

First vision VLM PR in this series. Became the merge-base that later PRs (#27, #29, #30) conflicted with on shared config files.
