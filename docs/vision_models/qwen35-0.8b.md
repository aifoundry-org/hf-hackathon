# Porting Plan: Qwen3.5-0.8B (Native Multimodal)

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #2 — Natively multimodal, Qwen ecosystem reuse (Qwen3-8B already ported)
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | ~1.44B total (752M LLM + 675M ViT + 10M adapter) |
| **LLM backbone** | Qwen3.5-0.8B (24 layers, hybrid GatedDeltaNet + attention) |
| **Vision encoder** | 27-layer ViT, hidden 1152, 16 heads, patch 16×16 |
| **Projector** | 4608 → 1024 (spatial merge 2×2) |
| **Architecture** | `qwen35` — hybrid Gated DeltaNet + Full Attention |
| **Context** | 262,144 tokens (256K) |
| **Vocabulary** | 248,320 tokens |
| **License** | **Apache-2.0** |
| **HF Repo** | `Qwen/Qwen3.5-0.8B` |
| **Key feature** | Natively multimodal (early fusion) — no separate VL variant needed |

## Architecture Detail: Hybrid GatedDeltaNet

This is a **novel hybrid architecture** — not a standard transformer:

- 6 × (3 × (Gated DeltaNet linear attention → FFN) → 1 × (Gated Attention → FFN))
- GatedDeltaNet: 16 linear attention heads for V, 16 for QK, head dim 128
- Gated Attention: 8 Q heads, 2 KV heads, head dim 256, RoPE dim 64
- Only 25% of layers use full attention; 75% use linear attention (DeltaNet)

This means **75% of inference uses linear attention** which is O(n) instead of O(n²) — potentially much faster on RISC-V.

## GGUF Files

| File | Size | Notes |
|------|------|-------|
| `Qwen3.5-0.8B-Q8_0.gguf` | ~812 MB | LLM weights |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | ~533 MB | Compact |
| `mmproj-BF16.gguf` | ~198 MB | Vision projector (keep BF16) |
| `mmproj-F32.gguf` | ~402 MB | Highest precision |
| **Total (Q8_0 + BF16 mmproj)** | **~1.0 GB** | |
| **Total (Q4_K_M + BF16 mmproj)** | **~731 MB** | |

Source repos:
- LLM GGUF: `ggml-org/Qwen3.5-0.8B-GGUF` or community (bartowski, etc.)
- mmproj: Convert via `convert_hf_to_gguf.py --mmproj` or download from community repos

## llama.cpp Support

- **Fully supported** since PR #19468 (merged Feb 2026)
- Architecture: `qwen35`
- MTP (multi-token prediction) supported
- Uses `llama-mtmd-cli` / `llama-server --mmproj`

### Known llama.cpp Issues
1. **Throughput bug** (#20072): ~192 tok/s on some hardware for dense 0.8B (expected ~300). Workaround PR exists.
2. **Thinking loops**: 0.8B prone to infinite thinking loops — use `enable_thinking: false`

## Benchmarks

| Benchmark | Qwen3.5-0.8B (thinking / non-thinking) |
|-----------|----------------------------------------|
| MMMU | 49.0 / 47.4 |
| MMMU-Pro | 31.2 / 31.4 |
| MathVista | 62.2 / 58.6 |
| RealWorldQA | 63.4 / 61.6 |
| MMStar | 58.3 / 55.9 |
| MMBench v1.1 | 69.9 / 68.0 |
| AI2D | 69.9 / 68.7 |
| OCRBench | 74.5 / 79.1 |

Competitive with Qwen2.5-VL-3B on many benchmarks at 1/4 the size.

---

## Porting Steps

### Step 1: Pin HuggingFace References

```
HF Base:      Qwen/Qwen3.5-0.8B (original) + community GGUF repo
Revision:     <pin commit SHA>
Files:        Qwen3.5-0.8B-Q8_0.gguf + mmproj-BF16.gguf
License:      apache-2.0
```

### Step 2: Add to `artifacts.json`

```json
"qwen35_08b_q8_gguf": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "Qwen3.5-0.8B-Q8_0",
  "filename": "Qwen3.5-0.8B-Q8_0.gguf",
  "env": "QWEN35_08B_MODEL_PATH",
  "source": {
    "type": "huggingface",
    "repo": "ggml-org/Qwen3.5-0.8B-GGUF",
    "revision": "<SHA>",
    "filename": "Qwen3.5-0.8B-Q8_0.gguf",
    "url": "..."
  },
  "local_cache": "local-artifacts/models/qwen35_08b/Qwen3.5-0.8B-Q8_0.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 812000000,
  "note": "Qwen3.5-0.8B natively multimodal LLM Q8_0. Hybrid GatedDeltaNet arch."
},
"qwen35_08b_mmproj_bf16": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "Qwen3.5-0.8B-mmproj-BF16",
  "filename": "mmproj-BF16.gguf",
  "env": "QWEN35_08B_MMPROJ_PATH",
  "source": {
    "type": "huggingface",
    "repo": "Qwen/Qwen3.5-0.8B",
    "revision": "<SHA>",
    "filename": "mmproj-BF16.gguf",
    "url": "..."
  },
  "local_cache": "local-artifacts/models/qwen35_08b/mmproj-BF16.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 198000000,
  "note": "27-layer ViT projector. Keep BF16 for quality."
}
```

### Step 3: Create Benchmark Config

File: `ported_models/llama_cpp_et/benchmarks/qwen35_08b.json`

Key settings:
- `gpu_layers`: 99
- `ctx_size`: 4096 (model supports 256K but reduce)
- `flash_attn`: false (probably not supported on ET)
- **Disable thinking mode** in prompt to avoid loops

### Step 4-6: Register, HF refs, recipe — standard pattern

---

## Infrastructure Reuse — HIGH

- **Qwen3-8B** is already ported and passing on the board (PR #11)
- Same Qwen ecosystem, same `artifacts.json` pattern
- Same CI pipeline, Modal deployment, board runner
- The `qwen35` architecture is new to llama.cpp-et but should be supported if Qwen3 is
- Same vocabulary and tokenizer as Qwen3-8B

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `qwen35` hybrid GatedDeltaNet architecture may not be in llama.cpp-et fork | **Critical** | Check ET fork's supported architectures; may need to merge upstream llama.cpp support |
| GatedDeltaNet linear attention uses different ops than standard attention | **High** | ET backend may not have optimized kernels for DeltaNet ops; test on Modal first |
| Thinking loops on 0.8B | **Medium** | Disable thinking mode in benchmark config |
| mmproj source may not have pre-built GGUF | **Low** | Convert locally with `convert_hf_to_gguf.py --mmproj` |
| Novel architecture = unknown ET backend behavior | **High** | This is both a risk and an opportunity — first to port hybrid attention to RISC-V |

## Why This Is Worth the Risk

- **Smallest natively multimodal model**: ~1.0 GB total (Q8_0 + BF16 mmproj)
- **Hybrid architecture**: Linear attention (75% of layers) is O(n) — could be dramatically faster on RISC-V than full attention models
- **Qwen ecosystem**: Shares tokenizer, vocabulary, and infrastructure with Qwen3-8B
- **256K context**: Generous for a model this size
- **Strong benchmarks**: Competitive with Qwen2.5-VL-3B at 1/4 the size

## Fallback: Qwen2.5-VL-3B-Instruct

If `qwen35` architecture doesn't work on ET backend:

| Property | Qwen3.5-0.8B | Qwen2.5-VL-3B-Instruct |
|----------|-------------|----------------------|
| Architecture | `qwen35` (hybrid) | Standard transformer |
| Total size (Q8+mmproj) | ~1.0 GB | ~4.3 GB |
| mmproj size | ~198 MB (BF16) | ~1.25 GB (F16) |
| DocVQA | — | 93.9 |
| llama.cpp arch | `qwen35` | `qwen2vl` (well-supported) |

Qwen2.5-VL-3B uses a standard transformer architecture that's more likely to work on ET backend, but at 4x the size.
