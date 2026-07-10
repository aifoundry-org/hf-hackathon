# Vision Model Porting Plans — ET-SoC1 Hackathon

Master index of porting plans for vision models targeting the AI Foundry + OpenHW CORE-ET hackathon on ET-SoC1 RISC-V silicon.

**PR status & timelines:** [PR_TRACKER.md](./PR_TRACKER.md) · per-model: [`prs/`](./prs/)

---

## Two Porting Paths

| Path | Mechanism | Benchmark Metric | Status in Repo |
|------|-----------|-----------------|----------------|
| **GGUF (llama.cpp-et)** | VLM: LLM GGUF + mmproj vision projector GGUF, run via `llama-server` / `llama-mtmd-cli` | Decode tokens/s (LLM) + vision quality | Active, CI-wired |
| **ONNX (ggonnx)** | Vision model ONNX → GGML/ET execution provider | Kernel wait time (seconds) | Vendored, **not yet CI-wired** |

---

## Candidates (All Apache-2.0)

### GGUF VLM Path — Multimodal (text + image → text)

| # | Model | Params | LLM + mmproj (Q8_0) | Plan | PR timeline |
|---|-------|--------|----------------------|------|-------------|
| 1 | **MiniCPM-V 4.6** | 1.3B | ~1.7 GB | [minicpm-v-4.6.md](./minicpm-v-4.6.md) | [PR](./prs/minicpm-v-4.6-PR.md) |
| 2 | **SmolVLM-256M-Instruct** | 256M | ~279 MB | [smolvlm-256m.md](./smolvlm-256m.md) | [PR #26](./prs/smolvlm-256m-PR.md) |
| 3 | **SmolVLM-500M-Instruct** | 500M | ~546 MB | [smolvlm-500m.md](./smolvlm-500m.md) | [PR #27](./prs/smolvlm-500m-PR.md) |
| 4 | **SmolVLM2-2.2B-Instruct** | 2.2B | ~2.3 GB | [smolvlm2-2.2b.md](./smolvlm2-2.2b.md) | [PR #29](./prs/smolvlm2-2.2b-PR.md) |
| 5 | **Qwen3.5-0.8B** | 0.8B LLM + 0.675B ViT | ~1.0 GB | [qwen35-0.8b.md](./qwen35-0.8b.md) | [PR](./prs/qwen35-0.8b-PR.md) |
| 6 | **Ministral 3 3B** | 3.8B | ~4.5 GB (Q8) / ~3 GB (Q4) | [ministral-3-3b.md](./ministral-3-3b.md) | [PR](./prs/ministral-3-3b-PR.md) |
| 7 | **ZwZ-4B** | 4B | ~5.3 GB (Q8) / ~3.3 GB (Q4) | [zwz-4b.md](./zwz-4b.md) | [PR #30](./prs/zwz-4b-PR.md) |

### ONNX Path — Pure Vision (ggonnx bridge)

| # | Model | Params | ONNX Size | Plan | PR timeline |
|---|-------|--------|-----------|------|-------------|
| 8 | **D-FINE Nano** | 3.79M | ~15 MB | [dfine-nano.md](./dfine-nano.md) | [PR](./prs/dfine-nano-PR.md) |
| 9 | **D-FINE Small** | 10.33M | ~40 MB | [dfine-nano.md](./dfine-nano.md) (shared) | [PR](./prs/dfine-nano-PR.md) |
| 10 | **RF-DETR Small** | 32.1M | ~120 MB | [rf-detr-small.md](./rf-detr-small.md) | [PR](./prs/rf-detr-small-PR.md) |
| 11 | **TinyViT** | 21M | ~80 MB | [tinyvit.md](./tinyvit.md) | [PR](./prs/tinyvit-PR.md) |
| 12 | **EfficientViT** | Varies | ~30–100 MB | [efficientvit.md](./efficientvit.md) | [PR](./prs/efficientvit-PR.md) |

---

## Priority Ranking

1. **MiniCPM-V 4.6** — Best VLM quality/size ratio, already deployed on iOS/Android
2. **Qwen3.5-0.8B** — Natively multimodal, Qwen ecosystem reuse (existing Qwen3-8B port)
3. **SmolVLM-256M** — Tiny, reuses SmolLM2-135M (already ported), fastest to implement
4. **SmolVLM-500M** — Sweet spot, reuses SmolLM2-360M (already ported)
5. **D-FINE Nano** — Easiest ONNX port, fills detection gap in ggonnx
6. **TinyViT** — Easiest ggonnx port, already ONNX with INT8
7. **Ministral 3 3B** — Good quality, Mistral ecosystem diversity
8. **ZwZ-4B** — Strong perception, Qwen3-VL architecture reuse
9. **SmolVLM2-2.2B** — Best SmolVLM quality but larger
10. **RF-DETR Small** — Modern DETR detection architecture
11. **EfficientViT** — MIT Han Lab, classification + segmentation

---

## Submission Template (from docs/SUBMISSION_GUIDE.md)

For each model, a PR should include:

```
1. ported_models/llama_cpp_et/artifacts.json    # model + mmproj entries
2. ported_models/llama_cpp_et/benchmarks/<id>.json  # runner config
3. .github/ci/benchmark_config.json             # register model
4. docs/HF_REFERENCES.md                        # pin HF reference
5. ported_models/llama_cpp_et/docs/<id>.md      # submission recipe
6. bash .github/ci/scripts/ci_preflight.sh      # validate
7. Open PR at github.com/aifoundry-org/hf-hackathon
```

For ONNX models:

```
1. ported_models/ggonnx/artifacts.json          # ONNX model entry
2. (Future: ggonnx runner + CI registration)
3. docs/HF_REFERENCES.md                        # pin HF reference
4. ported_models/ggonnx/docs/<id>.md            # porting notes
```

---

## Board CI Quality Gates (LLMs/VLMs)

- Completion tokens ≥ 32
- `"OK"` in decode output (standard benchmark prompt)
- WikiText-2 PPL in [1, 1000] (or [1, 100] for LFM)
- `"using device ET"` in logs
- Full GPU offload (`gpu_layers=99`)

---

## nvidia/LocateAnything-3B — DISQUALIFIED

| Reason | Detail |
|--------|--------|
| License | NVIDIA Non-Commercial Research License only |
| No GGUF | No llama.cpp support, no mmproj |
| No ONNX | Custom Eagle architecture |
| Components | Qwen2.5-3B-Instruct (Qwen Research License, not Apache) + MoonViT (MIT) |

Not suitable for hackathon submission despite interesting visual grounding capabilities.
