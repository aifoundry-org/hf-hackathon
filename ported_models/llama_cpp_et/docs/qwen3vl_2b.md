# Qwen3-VL-2B-Instruct — llama.cpp-et Submission Recipe

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Model ID:** `qwen3vl_2b`
**Port:** 18107
**License:** Apache-2.0

## Source

| Component | Value |
|-----------|-------|
| HF repo | `ggml-org/Qwen3-VL-2B-Instruct-GGUF` |
| Revision | `ea6a11058182570be6436b9a2e4ee7f7b49f908d` |
| LLM file | `Qwen3-VL-2B-Instruct-Q8_0.gguf` (1,834,427,296 B) |
| mmproj file | `mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf` (445,053,056 B) |
| Upstream base | `Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0) |

SHA256:
- LLM: `b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39`
- mmproj: `69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b`

## Architecture

- `qwen3vl` — Qwen3 text decoder (~2B) + native Qwen3-VL vision encoder + merger projector.
- Same architecture family as **ZwZ-4B** (PR #30) and shares the Qwen3 decoder with **Qwen3-8B** (already merged, PR #11), so the ET backend path is proven.
- Total on-board footprint: ~2.28 GB (Q8_0 LLM + Q8_0 mmproj), comfortably within the ~10 GiB GGUF budget.

## Benchmark

- Runner: `llama_server`, `device: ET`, `gpu_layers: 99`, `ctx_size: 4096`.
- Primary metric: decode tokens/s (higher is better).
- Perplexity gate: WikiText-2 raw, PPL in [1.0, 1000.0], 4 chunks.
- Standard "OK" repetition prompt with `min_completion_tokens: 32`.

## Quantization notes

- Q8_0 for both LLM and mmproj — no local requantization; files pulled verbatim from the pinned `ggml-org` GGUF repo.

## Vision caveat

The current `llama_server` board runner exercises the text decode path (and perplexity) only; it does not yet drive `--mmproj` image input. The mmproj is registered and pinned so that full vision benchmarking works as soon as the shared VLM runner extension lands. This matches the handling of the other VLM ports (SmolVLM family, ZwZ-4B).

## Reproduce

```bash
# resolve + hash-check both files
python .github/ci/scripts/... # standard artifact fetch by env QWEN3VL_2B_MODEL_PATH / QWEN3VL_2B_MMPROJ_PATH
# expand board config
python .github/ci/scripts/benchmark_config_helpers.py --target board --models qwen3vl_2b --format space
```
