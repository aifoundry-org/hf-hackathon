# Porting Plan: MiniCPM-V 4.6

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #1 — Best VLM quality/size ratio
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 1.3B total (0.8B LLM + 0.4B ViT + projector) |
| **LLM backbone** | Qwen3.5-0.8B (24 layers, hidden 1024, hybrid GatedDeltaNet + attention) |
| **Vision encoder** | SigLIP2-400M (27 layers, hidden 1152) |
| **Projector** | LLaVA-UHD v4 merger (window-attn ViT + DownsampleMLP, 4x spatial merge) |
| **Architecture** | `MiniCPMV4_6ForConditionalGeneration` |
| **Context** | 256K tokens |
| **License** | **Apache-2.0** |
| **HF Repo** | `openbmb/MiniCPM-V-4.6` |

## GGUF Files (from `openbmb/MiniCPM-V-4.6-gguf`)

| File | Size | Use Case |
|------|------|----------|
| `MiniCPM-V-4.6-Q8_0.gguf` | 812 MB | **Primary** — near-lossless |
| `MiniCPM-V-4.6-Q4_K_M.gguf` | 529 MB | Compact fallback |
| `mmproj-model-f16.gguf` | ~800-900 MB (est.) | **Required** — vision projector, keep F16 |
| **Total (Q8_0 + mmproj F16)** | **~1.7 GB** | |

## llama.cpp Support

- **Fully supported** as of release b9049 (May 6, 2026, PR #22529)
- Architecture: `minicpm-v-4.6` (custom, not a simple Qwen3-VL variant)
- Conversion: standard `convert_hf_to_gguf.py` (two-step: LLM then `--mmproj`)
- Tools: `llama-mtmd-cli`, `llama-server --mmproj`

## Benchmarks

| Benchmark | MiniCPM-V 4.6 | Qwen3.5-0.8B (text-only) |
|-----------|---------------|--------------------------|
| OCRBench | 74.5–79.1 | — |
| DocVQA | Competitive with 7B models | — |
| Intelligence Index | 13 | 10 |
| Token throughput | 1.5x–2.4x vs Qwen3.5-0.8B | baseline |

---

## Porting Steps

### Step 1: Pin HuggingFace References

```
HF Base:      openbmb/MiniCPM-V-4.6-gguf (or ggml-org/MiniCPM-V-4.6-GGUF)
Revision:     <pin commit SHA from files tab>
Files:        MiniCPM-V-4.6-Q8_0.gguf + mmproj-model-f16.gguf
License:      apache-2.0
```

### Step 2: Add to `artifacts.json`

```json
"minicpm_v46_q8_gguf": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "MiniCPM-V-4.6-Q8_0",
  "filename": "MiniCPM-V-4.6-Q8_0.gguf",
  "env": "MINICPM_V46_MODEL_PATH",
  "source": {
    "type": "huggingface",
    "repo": "openbmb/MiniCPM-V-4.6-gguf",
    "revision": "<SHA>",
    "filename": "MiniCPM-V-4.6-Q8_0.gguf",
    "url": "https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf/resolve/<SHA>/MiniCPM-V-4.6-Q8_0.gguf"
  },
  "local_cache": "local-artifacts/models/minicpm_v46/MiniCPM-V-4.6-Q8_0.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 812000000,
  "note": "MiniCPM-V 4.6 LLM Q8_0. Pairs with mmproj-model-f16.gguf."
},
"minicpm_v46_mmproj_f16": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "MiniCPM-V-4.6-mmproj-F16",
  "filename": "mmproj-model-f16.gguf",
  "env": "MINICPM_V46_MMPROJ_PATH",
  "source": {
    "type": "huggingface",
    "repo": "openbmb/MiniCPM-V-4.6-gguf",
    "revision": "<SHA>",
    "filename": "mmproj-model-f16.gguf",
    "url": "https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf/resolve/<SHA>/mmproj-model-f16.gguf"
  },
  "local_cache": "local-artifacts/models/minicpm_v46/mmproj-model-f16.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 850000000,
  "note": "MiniCPM-V 4.6 vision projector (SigLIP2-400M + merger). Must remain F16."
}
```

### Step 3: Create Benchmark Config

File: `ported_models/llama_cpp_et/benchmarks/minicpm_v46.json`

Key differences from text-only configs:
- `mmproj_artifact`: reference to mmproj entry
- `api`: `"chat"` (not `"completion"`) for VLM
- `prompt`: vision+text prompt (image path + question)
- PPL gate: may need to be **disabled** or adjusted (WikiText-2 PPL is for text-only models; VLMs may not produce meaningful perplexity on text corpus)
- `gpu_layers`: 99 (full offload — check if ET backend supports `minicpm-v-4.6` architecture)

### Step 4: Register in `benchmark_config.json`

```json
"minicpm_v46": {
  "config": "ported_models/llama_cpp_et/benchmarks/minicpm_v46.json"
}
```

### Step 5: Update `docs/HF_REFERENCES.md`

```markdown
| `minicpm_v46` | `openbmb/MiniCPM-V-4.6-gguf` | `<SHA>` | `apache-2.0` | `MiniCPM-V-4.6-Q8_0.gguf`, `mmproj-model-f16.gguf` |
```

### Step 6: Add Submission Recipe

File: `ported_models/llama_cpp_et/docs/minicpm_v46.md`

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ET backend may not support `minicpm-v-4.6` architecture | **Critical** — model won't run on silicon | Check llama.cpp-et fork's supported arch list; may need to add arch support to ET backend |
| Hybrid GatedDeltaNet + attention may not work with ET VPU/TFMA | **High** — decode may fail or produce garbage | Test Modal proxy first to verify GGUF works; board test is the true gate |
| mmproj F16 (~850 MB) + LLM Q8_0 (~812 MB) = ~1.7 GB total | **Medium** — may strain board DRAM | Try Q4_K_M (529 MB) + mmproj to reduce to ~1.4 GB |
| VLM benchmark prompt (image + text) requires new runner support | **Medium** — current runner is text-only completion | Need to extend llama_server runner to support `--mmproj` and image inputs |
| WikiText-2 PPL gate may not apply to VLMs | **Low** — may need to disable PPL check | Discuss with maintainers; VLMs are evaluated on vision benchmarks |

## Pre-Flight Validation (on x86/Modal)

```bash
# 1. Download GGUF files
huggingface-cli download openbmb/MiniCPM-V-4.6-gguf \
  MiniCPM-V-4.6-Q8_0.gguf mmproj-model-f16.gguf

# 2. Test with llama.cpp (x86 CPU first)
./llama-mtmd-cli \
  -m MiniCPM-V-4.6-Q8_0.gguf \
  --mmproj mmproj-model-f16.gguf \
  --image test.jpg \
  -p "Describe this image." \
  -n 128

# 3. Modal proxy test (if available)
# Verify completion tokens ≥ 32, output quality
```

## Infrastructure Reuse

- **SmolLM2 LLM layer**: MiniCPM-V 4.6 uses Qwen3.5-0.8B (NOT SmolLM2), so no direct LLM reuse. However, the SmolLM2 porting experience applies.
- **Qwen3-8B infrastructure**: The board CI pipeline, Modal deployment, and benchmark config format are directly reusable.
- **New**: The mmproj (vision projector) handling is new and requires extending the runner.

## Comparison to Other Candidates

| vs MiniCPM-V 4.6 | Advantage | Disadvantage |
|-------------------|-----------|--------------|
| Qwen3.5-0.8B | Same Qwen ecosystem, smaller (~1.0 GB) | Hybrid DeltaNet architecture — may have more ET backend issues |
| SmolVLM-256M | Much smaller (279 MB), reuses SmolLM2 | Lower quality (52.6 OCRBench vs 74.5) |
| Ministral 3 3B | Larger, stronger text | Much larger (~4.5 GB Q8), different arch |
| ZwZ-4B | Best fine-grained perception | Larger (~5.3 GB Q8), Qwen3-VL arch |
