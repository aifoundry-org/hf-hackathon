# PR Timeline — Qwen3.5-0.8B

| Field | Value |
|-------|-------|
| **PR** | *Not opened yet* |
| **Branch** | *(planned)* `feat/qwen35-0.8b` |
| **Status** | **WIP** — plan + draft files; SHA256 verification pending |
| **Path** | llama.cpp-et GGUF |
| **Plan** | [../qwen35-0.8b.md](../qwen35-0.8b.md) |

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
