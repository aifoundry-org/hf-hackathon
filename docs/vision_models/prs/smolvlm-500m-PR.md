# PR Timeline — SmolVLM-500M-Instruct

| Field | Value |
|-------|-------|
| **PR** | [#27](https://github.com/aifoundry-org/hf-hackathon/pull/27) |
| **Branch** | `Ashish-Soni08:feat/smolvlm-500m` |
| **Status** | **OPEN** — mergeable; awaiting re-review after conflict fix |
| **Path** | llama.cpp-et GGUF (LLM Q8_0 + mmproj Q8_0) |
| **Plan** | [../smolvlm-500m.md](../smolvlm-500m.md) |

## Timeline

| When (UTC) | Event |
|------------|-------|
| 2026-07-08 | Porting plan + selective extract from multi-model WIP |
| 2026-07-08 ~21:58 | PR #27 opened |
| 2026-07-08 | CI / board (`smolvlm_500m`) passed |
| 2026-07-09 | Review: **CHANGES_REQUESTED** — “please resolve the conflicts” (@AFOliveira) |
| 2026-07-09 | Rebased onto `main` (post–PR #26 + YOLO updates); conflicts in shared configs resolved |
| 2026-07-11 | Rebased again onto latest `main`; force-pushed; comment posted that conflicts are fixed |

## Conflicts resolved

Shared files touched by PR #26 / YOLO work:

- `.github/ci/benchmark_config.json` — kept main YOLO defines + added `smolvlm_500m`
- `docs/HF_REFERENCES.md` — kept `smolvlm_256m` + added `smolvlm_500m`
- `ported_models/llama_cpp_et/artifacts.json` — kept 256M entries + added 500M LLM/mmproj

## What the PR adds

- `ported_models/llama_cpp_et/benchmarks/smolvlm_500m.json`
- `ported_models/llama_cpp_et/docs/smolvlm_500m.md`
- `docs/vision_models/smolvlm-500m.md`
- Registry rows in artifacts / CI config / HF references

## Model snapshot

- Arch: Idefics3 (SmolLM2-360M + SigLIP)
- Size: ~546 MB total (Q8_0)
- License: Apache-2.0
- HF: `ggml-org/SmolVLM-500M-Instruct-GGUF`

## Open follow-ups

- Reviewer re-approval after conflict resolution
- Shared: VLM runner extension for real image+text benches (text-only today)
