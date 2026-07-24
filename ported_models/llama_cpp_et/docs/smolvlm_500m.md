# SmolVLM-500M-Instruct — llama.cpp-et vision board benchmark

## Overview

SmolVLM-500M-Instruct is an Idefics3 VLM (SigLIP vision encoder + SmolLM2-360M
backbone). This port exercises **real vision** on ET-SoC1: it loads the pinned
`mmproj`, runs pinned COCO image fixtures, and requires a visual-answer/oracle
gate via the same `smolvlm2_video` harness used by `smolvlm2_500m_video`.

## Hugging Face base

| Field | Value |
|-------|-------|
| Base repo | [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) @ `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| GGUF repo | [ggml-org/SmolVLM-500M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF) |
| GGUF revision | `72e986006ef53e37cdd3f6d4241c90b0f01df376` |
| LLM file | `SmolVLM-500M-Instruct-Q8_0.gguf` (436,806,912 B) |
| mmproj file | `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` (108,783,360 B) |
| License | Apache-2.0 |

## Vision correctness

| Case | Fixture(s) | Expected |
|------|------------|----------|
| `coco_cat` (CI + perf) | COCO `000000524280.jpg` | cat / tabby |
| `coco_giraffes` | COCO `000000296969.jpg` | giraffe(s) |
| order pair | cat↔giraffes | second-image animal must flip |

Fixtures reuse the pinned artifacts already on main:
`smolvlm2_coco_cat_jpg`, `smolvlm2_coco_giraffes_jpg`.

Reference contract: `.github/ci/reference/smolvlm_500m.json`.

## ET settings

| Parameter | Value |
|-----------|-------|
| runner | `smolvlm2_video` |
| device | ET |
| gpu_layers | 99 |
| mmproj_artifact | `smolvlm_500m_mmproj_q8` |
| require_full_offload | true |
| require_zero_vision_fallbacks | true |
| primary metric | `pmc_cycles` (lower is better) |
| port | 18103 |

## Files

- `ported_models/llama_cpp_et/artifacts.json` — LLM + mmproj (already present)
- `ported_models/llama_cpp_et/benchmarks/smolvlm_500m.json` — multimodal board config
- `.github/ci/reference/smolvlm_500m.json` — vision/oracle contract
- `.github/ci/benchmark_config.json` — `smolvlm_500m` registration
- `docs/HF_REFERENCES.md` — HF pin

## Verify

```bash
python -m json.tool ported_models/llama_cpp_et/benchmarks/smolvlm_500m.json >/dev/null
python -m json.tool .github/ci/reference/smolvlm_500m.json >/dev/null
python .github/ci/scripts/benchmark_config_helpers.py --target board --models smolvlm_500m --format space
```

Feasibility (CPU): pinned GGUFs correctly answer "Cat" on the COCO cat fixture via `--mmproj`.
