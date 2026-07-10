# PR Timeline — ZwZ-4B

| Field | Value |
|-------|-------|
| **PR** | [#30](https://github.com/aifoundry-org/hf-hackathon/pull/30) |
| **Branch** | `Ashish-Soni08:feat/zwz-4b` |
| **Status** | **OPEN** — mergeable; awaiting re-review after conflict fix |
| **Path** | llama.cpp-et GGUF (LLM Q4_K_M + mmproj Q8_0) |
| **Plan** | [../zwz-4b.md](../zwz-4b.md) |

## Timeline

| When (UTC) | Event |
|------------|-------|
| 2026-07-08 | Porting plan; files recreated after WIP extract hiccup |
| 2026-07-08 ~22:05 | PR #30 opened |
| 2026-07-08 | Initial CI passed |
| 2026-07-09 | Review: **CHANGES_REQUESTED** — “please fix conflicts” (@AFOliveira) |
| 2026-07-09 | Rebased onto `main` (post–PR #26 + YOLO); conflicts resolved |
| 2026-07-11 | Rebased again onto latest `main`; force-pushed; comment posted |

## Conflicts resolved

- `.github/ci/benchmark_config.json` — kept main YOLO + added `zwz_4b`
- `docs/HF_REFERENCES.md` — kept `smolvlm_256m` + added `zwz_4b`
- `ported_models/llama_cpp_et/artifacts.json` — kept 256M + added ZwZ LLM/mmproj

## What the PR adds

- `ported_models/llama_cpp_et/benchmarks/zwz_4b.json`
- `ported_models/llama_cpp_et/docs/zwz_4b.md`
- `docs/vision_models/zwz-4b.md`
- Registry rows in artifacts / CI config / HF references

## Model snapshot

- Arch: Qwen3VL (Qwen3 ~4B + SigLIP2-Large, DeepStack layers [5,11,17])
- Size: ~3.15 GB total (Q4_K_M LLM + Q8_0 mmproj)
- License: Apache-2.0
- HF: `inclusionAI/ZwZ-4B-GGUF`

## Open follow-ups

- Reviewer re-approval after conflict resolution
- Shared: VLM runner extension for vision quality gates
