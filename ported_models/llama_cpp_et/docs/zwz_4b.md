# ZwZ-4B — llama.cpp-et Submission Recipe

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Model ID:** `zwz_4b`
**Port:** 18105
**License:** Apache-2.0

## Source

| Component | Value |
|-----------|-------|
| HF base | `inclusionAI/ZwZ-4B` @ `e43981c19904b4ee4ec48efc21c9a77bf4e7838c` |
| GGUF repo | `inclusionAI/ZwZ-4B-GGUF` @ `af96680eb7d0978e4844e8395cb6fd727f0a1d84` |
| LLM file | `ZwZ-4B-Q4_K_M.gguf` (2,716,069,408 B) |
| mmproj file | `mmproj-ZwZ-4B-Q8_0.gguf` (450,828,416 B) |

SHA256:
- LLM: `0c1e633a544708ff3d52060f764ddeccba8a98535f223c40361fddeb59c79d33`
- mmproj: `ba2f4e1b792ce56bbfc78a322e99fd0efbbbe8ea23479894851ab7a060872f6c`

## Architecture

- `qwen3vl` — Qwen3 text decoder (**4.41 B** params from GGUF tensor element sum, `n_vocab` 151936) + Qwen3-VL vision encoder + `qwen3vl_merger`.
- Identity contract (#112 on main): `metadata_key_prefix=qwen3vl`, `require_model_name=false`, nested `parameter_count` / `vocabulary` (no `parameter_count_millions`).
- Language: 36 blocks, emb 2560, ff 9728, 32/8 heads, **399** tensors.
- Vision: 24 layers, emb 1024, ff 4096, 16 heads, image 768 / patch 16, projection_dim **2560**, **316** tensors.
- Same family as **Qwen3-VL-2B** (#73); Qwen3 decoder path shared with **Qwen3-8B** (#11).
- Footprint ~3.17 GB total (Q4_K_M + Q8_0 mmproj) — within the ~10 GiB GGUF budget.

## Benchmark (vision)

- Runner: main-owned **`smolvlm2_video`** (same harness as `smolvlm2_500m_video` / SmolVLM-500M / Qwen3-VL-2B).
- Loads `--mmproj`, pinned COCO cat/giraffe fixtures, ET visual-answer/oracle + order-pair gate.
- Contract: `.github/ci/reference/zwz_4b.json`
- Metric: `pmc_cycles` (firmware cycles, lower better).
- Prompt: Qwen chat template with `<|im_end|>`:
  `<|im_start|>user\n{media_markers}\n{question}<|im_end|>\n<|im_start|>assistant\n`
- Extra: `--image-min-tokens 1024` (needed for reliable COCO answers offline).
- Performance: `ignore_eos=true` so the board emits all 3 contracted fixed tokens (Qwen chat models otherwise stop early after the one-word answer).
- Port **18105**.
- PPL gate: WikiText-2, loose `max_ppl=100` (`first_run_perplexity=83.33`, ×1.2) until host smoke refines baselines.

## Dependency / sequencing

1. [#112](https://github.com/aifoundry-org/hf-hackathon/pull/112) merged on main (`{arch}.*` + nested schema).
2. Host CPU smoke green (identity + COCO correctness).
3. Prefer waiting for [#73](https://github.com/aifoundry-org/hf-hackathon/pull/73) ET-green before requesting ZwZ ET (shared `qwen3vl` vision path).

## Quantization notes

- Q4_K_M LLM + Q8_0 mmproj — pinned verbatim from `inclusionAI/ZwZ-4B-GGUF`; no local requantization.
- Q8_0 LLM would exceed comfortable board DRAM headroom for this size class.

## Reproduce

```bash
python .github/ci/scripts/benchmark_config_helpers.py --target board --models zwz_4b --format space
```
