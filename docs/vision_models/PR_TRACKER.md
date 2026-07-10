# Vision Models — PR Tracker

Index of per-model PR timelines. Each model has its own `prs/<model>-PR.md` with status, conflict history, and next steps.

**Last updated:** 2026-07-11

## Submitted PRs

| Model | PR | Status | Timeline |
|-------|-----|--------|----------|
| SmolVLM-256M | [#26](https://github.com/aifoundry-org/hf-hackathon/pull/26) | **MERGED** | [prs/smolvlm-256m-PR.md](./prs/smolvlm-256m-PR.md) |
| SmolVLM-500M | [#27](https://github.com/aifoundry-org/hf-hackathon/pull/27) | OPEN (mergeable; re-review) | [prs/smolvlm-500m-PR.md](./prs/smolvlm-500m-PR.md) |
| SmolVLM2-2.2B | [#29](https://github.com/aifoundry-org/hf-hackathon/pull/29) | OPEN (mergeable) | [prs/smolvlm2-2.2b-PR.md](./prs/smolvlm2-2.2b-PR.md) |
| ZwZ-4B | [#30](https://github.com/aifoundry-org/hf-hackathon/pull/30) | OPEN (mergeable; re-review) | [prs/zwz-4b-PR.md](./prs/zwz-4b-PR.md) |

## Not yet submitted

| Model | Path | Timeline |
|-------|------|----------|
| MiniCPM-V 4.6 | GGUF / ONNX | [prs/minicpm-v-4.6-PR.md](./prs/minicpm-v-4.6-PR.md) |
| Qwen3.5-0.8B | GGUF | [prs/qwen35-0.8b-PR.md](./prs/qwen35-0.8b-PR.md) |
| Ministral 3 3B | GGUF | [prs/ministral-3-3b-PR.md](./prs/ministral-3-3b-PR.md) |
| D-FINE Nano | ggonnx | [prs/dfine-nano-PR.md](./prs/dfine-nano-PR.md) |
| RF-DETR Small | ggonnx | [prs/rf-detr-small-PR.md](./prs/rf-detr-small-PR.md) |
| EfficientViT | ggonnx | [prs/efficientvit-PR.md](./prs/efficientvit-PR.md) |
| TinyViT | ggonnx | [prs/tinyvit-PR.md](./prs/tinyvit-PR.md) |

## Conflict cleanup (2026-07-09 → 2026-07-11)

All open VLM PRs conflicted on shared files after #26 merged and YOLO CI configs changed:

- `.github/ci/benchmark_config.json`
- `docs/HF_REFERENCES.md`
- `ported_models/llama_cpp_et/artifacts.json`

**Done:**
- Rebased `feat/smolvlm-500m`, `feat/smolvlm2-2.2b`, `feat/zwz-4b` onto latest `main`
- Kept main’s YOLO + SmolVLM-256M entries; added only each PR’s model keys
- Force-pushed; commented on #27 / #29 / #30 that conflicts are resolved

**Still needed from reviewers:** re-approve #27 and #30 (still marked CHANGES_REQUESTED).

## Shared infrastructure

| Need | Why | Status |
|------|-----|--------|
| VLM runner (image + mmproj) | Current benches are text-only for VLMs | Not started |
| ggonnx CI runner | ONNX vision models cannot score yet | Not started |

## Related docs

- [README.md](./README.md) — candidate index + porting paths
- [VERIFICATION_REPORT.md](./VERIFICATION_REPORT.md) — hackathon constraint check
- Plans: `./<model>.md` next to this file
