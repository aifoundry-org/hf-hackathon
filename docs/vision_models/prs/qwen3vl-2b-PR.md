# PR Timeline — Qwen3-VL-2B-Instruct

| Field | Value |
|-------|-------|
| **PR** | [#73](https://github.com/aifoundry-org/hf-hackathon/pull/73) |
| **Branch** | `Ashish-Soni08:feat/qwen3vl-2b` |
| **Status** | **OPEN** — mergeable; CI running |
| **Path** | llama.cpp-et GGUF (LLM Q8_0 + mmproj Q8_0) |
| **Plan** | [../qwen3vl-2b.md](../qwen3vl-2b.md) |

## Timeline

| When (UTC) | Event |
|------------|-------|
| 2026-07-11 | Planned Qwen3.5-0.8B found to have **no mmproj** on `ggml-org` → blocked as VLM |
| 2026-07-11 | Selected Qwen3-VL-2B-Instruct instead (real VLM, mmproj ready, `qwen3vl` arch) |
| 2026-07-11 | Branch `feat/qwen3vl-2b` off `main`; artifacts/bench/CI/HF/docs added |
| 2026-07-11 | Verified: JSON, cross-ref, port 18107 unique, URLs 302, SHA256 = HF oid, config expands |
| 2026-07-11 | PR #73 opened |

## What the PR adds

- `ported_models/llama_cpp_et/artifacts.json` — `qwen3vl_2b_q8_gguf` + `qwen3vl_2b_mmproj_q8`
- `ported_models/llama_cpp_et/benchmarks/qwen3vl_2b.json` (port 18107)
- `ported_models/llama_cpp_et/docs/qwen3vl_2b.md` (recipe)
- `docs/vision_models/qwen3vl-2b.md` (plan)
- Registry rows in CI config + HF references

## Model snapshot

- Arch: `qwen3vl` (Qwen3 ~2B decoder + native Qwen3-VL vision encoder + merger)
- Size: ~2.28 GB total (Q8_0 LLM + Q8_0 mmproj)
- License: Apache-2.0
- HF: `ggml-org/Qwen3-VL-2B-Instruct-GGUF` @ `ea6a110`
- SHA256 — LLM `b7802e29…813c39`, mmproj `69066c8f…ca9e7b`

## Open follow-ups

- CI board result + review
- Shared: VLM runner extension for `--mmproj` vision quality gate
