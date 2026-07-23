# Qwen3-VL-2B-Instruct — llama.cpp-et Submission Recipe

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Model ID:** `qwen3vl_2b`
**Port:** 18108
**License:** Apache-2.0

## Source

| Component | Value |
|-----------|-------|
| HF base | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` |
| GGUF repo | `ggml-org/Qwen3-VL-2B-Instruct-GGUF` @ `ea6a11058182570be6436b9a2e4ee7f7b49f908d` |
| LLM file | `Qwen3-VL-2B-Instruct-Q8_0.gguf` (1,834,427,296 B) |
| mmproj file | `mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf` (445,053,056 B) |

SHA256:
- LLM: `b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39`
- mmproj: `69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b`

## Architecture

- `qwen3vl` — Qwen3 text decoder (**1.72 B** params, `n_vocab` 151936) + native Qwen3-VL vision encoder + `qwen3vl_merger`.
- Identity contract (must match #112): `metadata_key_prefix=qwen3vl`, `require_model_name=false`, nested `parameter_count` / `vocabulary` (no `parameter_count_millions`).
- Language: 28 blocks, emb 2048, ff 6144, 16/8 heads, 310 tensors.
- Vision: 24 layers, emb 1024, ff 4096, 16 heads, image 768 / patch 16, projection_dim 2048, 316 tensors.
- Same family as **ZwZ-4B** (#30); Qwen3 decoder path shared with **Qwen3-8B** (#11).
- Footprint ~2.28 GB total — within the ~10 GiB GGUF budget.

## Benchmark (vision)

- Runner: main-owned **`smolvlm2_video`** (same harness as `smolvlm2_500m_video` / SmolVLM-500M).
- Loads `--mmproj`, pinned COCO cat/giraffe fixtures, ET visual-answer/oracle + order-pair gate.
- Contract: `.github/ci/reference/qwen3vl_2b.json`
- Metric: `pmc_cycles` (firmware cycles, lower better).
- Prompt: Qwen chat template (`<|im_start|>user` / `<|im_end|>`).
- Extra: `--image-min-tokens 1024` (needed for reliable COCO answers offline).
- Port **18108** (avoids clash with `smolvlm2_500m_video` on 18107).
- PPL gate: WikiText-2, max 16.812 (20% over first-run 14.01).

## Dependency

Board identity for non-`llama` arches requires [#112](https://github.com/aifoundry-org/hf-hackathon/pull/112) (`{arch}.*` GGUF key prefix + optional `general.name` / vocab). Merge that before expecting #73 board CI to pass identity.

## Quantization notes

- Q8_0 for both LLM and mmproj — pinned verbatim from `ggml-org`; no local requantization.

## Reproduce

```bash
python .github/ci/scripts/benchmark_config_helpers.py --target board --models qwen3vl_2b --format space
```
