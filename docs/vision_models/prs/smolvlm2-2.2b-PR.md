# PR Timeline — SmolVLM2-2.2B-Instruct

| Field | Value |
|-------|-------|
| **PR** | [#29](https://github.com/aifoundry-org/hf-hackathon/pull/29) |
| **Branch** | `Ashish-Soni08:feat/smolvlm2-2.2b` |
| **Status** | **OPEN** — mergeable after rebase |
| **Path** | llama.cpp-et GGUF (LLM Q4_K_M + mmproj Q8_0) |
| **Plan** | [../smolvlm2-2.2b.md](../smolvlm2-2.2b.md) |

## Timeline

| When (UTC) | Event |
|------------|-------|
| 2026-07-08 | Porting plan + selective extract from multi-model WIP |
| 2026-07-08 ~22:00 | PR #29 opened |
| 2026-07-08 | CI / board (`smolvlm2_22b`) passed |
| 2026-07-09–11 | Became **CONFLICTING** vs `main` (PR #26 + YOLO + later merges) |
| 2026-07-11 | Rebased onto latest `main`; conflicts resolved; force-pushed |

## Conflicts resolved

Same shared-file pattern as other VLM PRs:

- `.github/ci/benchmark_config.json`
- `docs/HF_REFERENCES.md`
- `ported_models/llama_cpp_et/artifacts.json`

Resolution: take current `main`, add only `smolvlm2_22b` / `smolvlm2_22b_*` entries.

## What the PR adds

- `ported_models/llama_cpp_et/benchmarks/smolvlm2_22b.json`
- `ported_models/llama_cpp_et/docs/smolvlm2_22b.md`
- `docs/vision_models/smolvlm2-2.2b.md`
- Registry rows in artifacts / CI config / HF references

## Model snapshot

- Arch: Idefics3 (SmolLM2-1.7B + SigLIP)
- Size: ~1.2 GB total (Q4_K_M LLM + Q8_0 mmproj)
- License: Apache-2.0
- HF: `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`

## Open follow-ups

- Review / merge
- Shared: VLM runner extension for vision quality gates
