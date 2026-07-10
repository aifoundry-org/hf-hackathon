# Vision Model Port — Verification Report

Generated against hackathon constraints from `docs/SUBMISSION_GUIDE.md`,
`ported_models/llama_cpp_et/README.md`, and `.github/ci/scripts/ci_preflight.sh`.

## Universal Checklist (all 11 models)

| # | Check | SmolVLM-256M | MiniCPM-V | Qwen3.5-0.8B | SmolVLM-500M | ZwZ-4B | SmolVLM2-2.2B | Ministral-3B | TinyViT | D-FINE | RF-DETR | EfficientViT |
|---|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `artifacts.json` entries present | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 2 | Source repo + revision pinned | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3 | License recorded (Apache-2.0) | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 4 | `HF_REFERENCES.md` row added | PASS | PASS | PASS (2 rows) | PASS | PASS | PASS | PASS | PASS | PASS (2 rows) | PASS | PASS (2 rows) |
| 5 | Recipe `.md` doc created | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 6 | No model blobs committed | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 7 | No `data/*.json` edits | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## GGUF Models — Detailed Checks

| # | Check | SmolVLM-256M | MiniCPM-V | Qwen3.5-0.8B | SmolVLM-500M | ZwZ-4B | SmolVLM2-2.2B | Ministral-3B |
|---|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | `benchmark_config.json` key | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 9 | Benchmark JSON valid | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 10 | Port unique | 18100 | 18101 | 18102 | 18103 | 18105 | 18106 | 18104 |
| 11 | `device=ET`, `gpu_layers=99` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 12 | `min_completion_tokens>=32` | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 13 | Perplexity enabled | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 14 | mmproj artifact registered | PASS (Q8) | PASS (F16) | PASS (BF16) | PASS (Q8) | PASS (Q8) | PASS (Q8) | PASS (BF16) |
| 15 | Total size < 10 GiB | 279 MB | 1.83 GB | 1.0 GB | 521 MB | 3.32 GB | 1.59 GiB | 4.49 GB |
| 16 | SHA256 present | PASS | PASS | LLM only | PASS | PASS | PASS | PASS |
| 17 | Architecture proven on ET | SmolLM2-135M | GatedDeltaNet(new) | GatedDeltaNet(new) | SmolLM2-360M | qwen3vl | SmolLM2-1.7B | mistral3(new) |
| 18 | Completion API | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

### GGUF Issues Found

| Model | Severity | Issue |
|-------|----------|-------|
| Qwen3.5-0.8B | WARN | **Missing SHA256** for `qwen35_08b_q8_gguf` — artifacts.json has `size_bytes` but no `sha256` field. Must verify with `sha256sum` after download and add the hash. |
| Qwen3.5-0.8B | WARN | **mmproj from different repo** (unsloth vs ggml-org) — two separate HF repos pinned. Documented in HF_REFERENCES.md (2 rows), but increases supply-chain surface. |
| MiniCPM-V 4.6 | WARN | **Novel architecture** — GatedDeltaNet + attention hybrid. PR #22529 merged May 2026; ET backend not yet validated with this arch. Board run will be the first test. |
| Qwen3.5-0.8B | WARN | **Novel architecture** — Same GatedDeltaNet concern. Issue #20072 noted for potential throughput bug on hybrid models. |
| Ministral 3 3B | WARN | **First Mistral-family** — `mistral3` arch. Requires `attn_factor` fix (PR #17945). ET backend must include this fix. |
| Ministral 3 3B | INFO | **BF16 mmproj only** — No quantized mmproj available from official Mistral. Community quants (unsloth) could reduce the 842 MB mmproj footprint. |
| SmolVLM2-2.2B | INFO | **Q4_K_M LLM** — Quality degradation from Q8_0. Q8_0 would be ~2.6 GB (still fits), but Q4_K_M chosen for DRAM headroom. |
| ZwZ-4B | INFO | **Q4_K_M LLM** — Same Q4 trade-off. Q8_0 total would be ~5.29 GB (exceeds board DRAM). |
| All GGUF | INFO | **Vision not benchmarked** — All benchmarks use text-only completion API. Vision inference via `--mmproj` is documented but not exercised by the standard harness. A VLM runner extension is needed. |

## ONNX Models — Detailed Checks

| # | Check | TinyViT | D-FINE Nano/Small | RF-DETR Small | EfficientViT M0+SAM |
|---|-------|:---:|:---:|:---:|:---:|
| 19 | `ggonnx/artifacts.json` entries | PASS | PASS (2 entries) | PASS | PASS (2 entries) |
| 20 | `benchmark_config.json` key | N/A (correct) | N/A (correct) | N/A (correct) | N/A (correct) |
| 21 | ONNX file available | Pre-exported | Export required | Export required | Export required |
| 22 | Source URL in artifacts | PASS | PASS | PASS | PASS |
| 23 | Validation method set | `compare_outputs_to_onnxruntime_cpu` | `compare_outputs_to_onnxruntime_cpu` | `compare_outputs_to_onnxruntime_cpu` | `compare_outputs_to_onnxruntime_cpu` |
| 24 | ggonnx ops risk | Low | Medium | **High** | Low-Medium |
| 25 | Recipe doc complete | PASS | PASS | PASS | PASS |

### ONNX Issues Found

| Model | Severity | Issue |
|-------|----------|-------|
| All ONNX | WARN | **ggonnx runner not wired to CI** — No `benchmark_config.json` registration. Board execution blocked on ggonnx CI runner being built. All ONNX models are preparatory only. |
| RF-DETR Small | WARN | **Deformable attention ops** — Uses `GridSample` (opset 16+) for multi-scale deformable cross-attention. May not be supported by ggonnx ET backend. Op decomposition or graph patching needed. |
| D-FINE Nano/Small | WARN | **ONNX export untested** — Keras `.h5` format. `optimum-cli` may not support D-FINE directly. Custom export script likely needed. |
| EfficientViT SAM | WARN | **Two-component model** — Image encoder + mask decoder must be exported separately. Decoder has complex prompt-based inputs. First segmentation model in ggonnx. |
| TinyViT | INFO | **384x384 input** — Larger than the standard 224x224 used by existing ggonnx classifiers. Verify ggonnx can handle the input size and 184 MB ONNX file. |
| RF-DETR Small | INFO | **128.5 MB ONNX** — Largest ONNX model. Consider FP16/INT8 quantization post-export to reduce footprint. |

## Port Uniqueness Audit

All 7 GGUF models use unique ports. No conflicts with existing 17 models:

| Existing ports | 18080-18094 |
|---------------|-------------|
| New VLM ports | 18100-18106 |
| Conflicts | **None** |

Note: Ports 18081 is reused by 3 existing models (`llama32_1b`, `qwen25_05b`, `deepseek_r1_15b`) — this is a pre-existing issue, not introduced by the vision model ports.

## Deployment Size Summary

| Model | LLM | mmproj/ONNX | Total | Fits 10 GiB? |
|-------|-----|-------------|-------|:---:|
| SmolVLM-256M | 175 MB | 104 MB | **279 MB** | Yes |
| SmolVLM-500M | 417 MB | 104 MB | **521 MB** | Yes |
| Qwen3.5-0.8B | 812 MB | 207 MB | **1.0 GB** | Yes |
| MiniCPM-V 4.6 | 774 MB | 1.06 GB | **1.83 GB** | Yes |
| SmolVLM2-2.2B | 1.04 GiB | 565 MiB | **1.59 GiB** | Yes |
| ZwZ-4B | 2.72 GB | 451 MB | **3.32 GB** | Yes |
| Ministral 3 3B | 3.65 GB | 842 MB | **4.49 GB** | Yes |
| TinyViT 21M | — | 184.5 MB | **184.5 MB** | Yes |
| D-FINE Nano | — | 16.7 MB | **16.7 MB** | Yes |
| D-FINE Small | — | 43.1 MB | **43.1 MB** | Yes |
| RF-DETR Small | — | 128.5 MB | **128.5 MB** | Yes |
| EfficientViT M0 | — | 29.4 MB | **29.4 MB** | Yes |
| EfficientViT SAM L2 | — | 134 MB | **134 MB** | Yes |

## Verdict Summary

| Model | PR-Ready? | Blocker | Action Needed |
|-------|:---------:|---------|---------------|
| **SmolVLM-256M** | **Yes** | None | Ready for PR |
| **SmolVLM-500M** | **Yes** | None | Ready for PR |
| **SmolVLM2-2.2B** | **Yes** | None | Ready for PR |
| **ZwZ-4B** | **Yes** | None | Ready for PR |
| **MiniCPM-V 4.6** | **Yes** | Novel arch risk | Board run will validate GatedDeltaNet on ET |
| **Qwen3.5-0.8B** | **Almost** | Missing SHA256 | Add `sha256` for `qwen35_08b_q8_gguf` before PR |
| **Ministral 3 3B** | **Almost** | attn_factor fix | Verify ET fork includes PR #17945 |
| **TinyViT** | **Prep** | ggonnx CI runner | Preparatory only; download + validate ONNX when runner ready |
| **D-FINE** | **Prep** | ggonnx CI runner + ONNX export | Export ONNX from Keras, then validate |
| **RF-DETR** | **Prep** | ggonnx CI runner + GridSample risk | Export ONNX, check deformable attention ops |
| **EfficientViT** | **Prep** | ggonnx CI runner + ONNX export | Export from PyTorch, validate M0 first |

## Shared Prerequisites (Blocking All)

1. **VLM Runner Extension** — The `llama_server` runner does not pass `--mmproj` to `llama-server`. All 7 GGUF VLMs benchmark text-only. Vision capability needs a runner update to pass mmproj and image inputs.

2. **ggonnx CI Runner** — No runner exists for ONNX models in `.github/ci/scripts/`. All 4 ONNX models (6 artifacts) are registered in `ggonnx/artifacts.json` but cannot be benchmarked until a runner, benchmark config entries, and CI wiring are built.
