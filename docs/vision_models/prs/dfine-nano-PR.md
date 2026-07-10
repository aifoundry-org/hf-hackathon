# PR Timeline — D-FINE Nano / Small

| Field | Value |
|-------|-------|
| **PR** | *Not opened yet* |
| **Branch** | *(planned)* `feat/dfine-nano` |
| **Status** | **WIP** — plan + ggonnx notes; waiting on ggonnx CI |
| **Path** | ggonnx ONNX |
| **Plan** | [../dfine-nano.md](../dfine-nano.md) |

## Timeline

| When | Event |
|------|-------|
| 2026-07 | Candidate selected; ONNX porting plan written |
| 2026-07 | Draft notes: `ported_models/ggonnx/docs/dfine_nano.md` |

## Blockers

- ggonnx runner not CI-wired yet
- ONNX export + detection accuracy gate

## Next

1. Export/validate ONNX
2. Register in `ported_models/ggonnx/artifacts.json`
3. Open PR when CI path exists (or docs-only registration if accepted)
