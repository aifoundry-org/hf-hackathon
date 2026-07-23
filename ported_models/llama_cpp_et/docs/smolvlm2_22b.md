# SmolVLM2-2.2B-Instruct — llama.cpp-et vision board benchmark

## Overview

SmolVLM2-2.2B-Instruct is the largest SmolVLM family variant (Idefics3 + larger
SigLIP vision encoder + SmolLM2-1.7B backbone). This port exercises **real
vision** on ET-SoC1: it loads the pinned `mmproj`, runs pinned COCO image
fixtures, and requires a visual-answer/oracle gate via the same
`smolvlm2_video` harness used by `smolvlm_500m` / `smolvlm2_500m_video`.

Pinned quantization is **Q4_K_M LLM + Q8_0 mmproj** (~1.59 GiB total) to fit
board DRAM.

## Hugging Face base

| Field | Value |
|-------|-------|
| Base repo | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) @ `482adb537c021c86670beed01cd58990d01e72e4` |
| GGUF repo | [ggml-org/SmolVLM2-2.2B-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF) |
| GGUF revision | `1bc3c9f74ceafd4c8d4411cc9cf188bba3798f91` |
| LLM file | `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` (1,112,602,656 B) |
| mmproj file | `mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf` (592,523,200 B) |
| License | Apache-2.0 |

## Vision correctness

| Case | Fixture(s) | Expected |
|------|------------|----------|
| `coco_cat` (CI + perf) | COCO `000000524280.jpg` | cat / tabby |
| `coco_giraffes` | COCO `000000296969.jpg` | giraffe(s) |
| order pair | cat↔giraffes | second-image animal must flip |

Fixtures reuse the pinned artifacts already on main:
`smolvlm2_coco_cat_jpg`, `smolvlm2_coco_giraffes_jpg`.

Reference contract: `.github/ci/reference/smolvlm2_22b.json`.

Prompt template (SmolVLM / #27 pattern):

```text
<|im_start|>User: {media_markers}
{question}<end_of_utterance>
Assistant:
```

`processor_image_size` / vision `image_size` = **384** (not 512).

## Loader identity fingerprints

Grounded in llama.cpp loader output for the ggml-org GGUF pair (language Q8_0
log + Q8_0 mmproj; Q4_K_M language GGUF shares the same architecture metadata /
tensor count):

| Field | Value |
|-------|-------|
| language `general.architecture` | `llama` |
| language `general.name` | `SmolVLM2 2.2B Instruct` |
| language params | loader prints `1.81 B` → contract `parameter_count_millions: 1810.0` |
| language tensor count | 219 |
| language block / emb / ff / heads / kv / vocab | 24 / 2048 / 8192 / 32 / 32 / 49280 |
| vision projector | `idefics3` |
| vision n_tensors / n_embd / n_head / n_ff / n_layer | 438 / 1152 / 16 / 4304 / 27 |
| vision projection_dim / image_size / patch_size | 2048 / 384 / 14 |

### Known identity-gate gaps (honest TBD)

1. **`parameter_count_millions` unit**: `smolvlm2_video` identity regex expects
   `model params = X.XX M`, but this GGUF's loader prints `model params = 1.81 B`.
   Contract stores the million-equivalent (`1810.0`) and does **not** invent
   HF's 2246.8M total. A main-owned runner tweak is needed for B-scale llama
   models (this PR does not edit protected runners).
2. **`llama.attention.head_count_kv`**: `print_info` reports `n_head_kv = 32`,
   but published KV dumps omit that metadata key when GQA=1. Confirm with local
   GGUF dump; identity regex may fail until then.
3. **WikiText PPL**: `maximum_perplexity` / `max_ppl` are consistent placeholders
   (`80.0` / `96.0`) until a host measure on the pinned Q4_K_M artifact.

## ET settings

| Parameter | Value |
|-----------|-------|
| runner | `smolvlm2_video` |
| device | ET |
| gpu_layers | 99 |
| mmproj_artifact | `smolvlm2_22b_mmproj_q8` |
| require_full_offload | true |
| require_zero_vision_fallbacks | true |
| primary metric | `pmc_cycles` (lower is better) |
| port | **18106** |

## Files

- `ported_models/llama_cpp_et/artifacts.json` — `smolvlm2_22b_q4_gguf` + `smolvlm2_22b_mmproj_q8`
- `ported_models/llama_cpp_et/benchmarks/smolvlm2_22b.json` — multimodal board config
- `.github/ci/reference/smolvlm2_22b.json` — vision/oracle contract
- `.github/ci/benchmark_config.json` — `smolvlm2_22b` registration
- `docs/HF_REFERENCES.md` — HF pin

## Verify

```bash
python -m json.tool ported_models/llama_cpp_et/benchmarks/smolvlm2_22b.json >/dev/null
python -m json.tool .github/ci/reference/smolvlm2_22b.json >/dev/null
python .github/ci/scripts/benchmark_config_helpers.py --target board --models smolvlm2_22b --format space
```

Host COCO cat smoke (when llama-server + pinned GGUFs are available): expect a
one-word `cat` / `tabby` answer on fixture `000000524280.jpg`.
