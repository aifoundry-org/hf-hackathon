# PR Timeline — Qwen3.5-0.8B

| Field | Value |
|-------|-------|
| **PR** | *Not opened — superseded* |
| **Branch** | *(planned)* `feat/qwen35-0.8b` |
| **Status** | **SUPERSEDED** by [Qwen3-VL-2B (#73)](./qwen3vl-2b-PR.md) — no mmproj published |
| **Path** | llama.cpp-et GGUF |
| **Plan** | [../qwen35-0.8b.md](../qwen35-0.8b.md) |

> **Blocked as a VLM (2026-07-11):** `ggml-org/Qwen3.5-0.8B-GGUF` publishes only
> LLM GGUFs (`BF16` / `Q4_0` / `Q8_0`) and **no vision `mmproj`**, so the model
> cannot run through the VLM path. The Qwen small-VLM slot was filled by
> **Qwen3-VL-2B-Instruct** ([#73](https://github.com/aifoundry-org/hf-hackathon/pull/73)),
> which ships both LLM and mmproj GGUFs. Qwen3.5-0.8B remains a candidate for a
> future **text-only** LLM port (Q8_0, ~812 MB, verified SHA256
> `75526add…43bb3d`).

## Timeline

| When | Event |
|------|-------|
| 2026-07 | Candidate selected; porting plan written |
| 2026-07 | Verification: GGUF path OK; need exact SHA256 from downloaded files |
| 2026-07 | Draft: `benchmarks/qwen35_08b.json`, `docs/qwen35_08b.md` |

## Blockers

- Download GGUF + mmproj, compute SHA256, pin revision

## Next

1. Fill `artifacts.json` hashes
2. CI register + preflight
3. Open PR
