# Porting Plan: Ministral 3 3B

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #7 — Good quality, Mistral ecosystem diversity
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | ~3.8B total (3.4B LLM + 0.4B ViT) |
| **LLM backbone** | Ministral3 decoder (26 layers, hidden 3072, 32Q/8KV heads, SwiGLU) |
| **Vision encoder** | Pixtral ViT, 410M params, frozen from Mistral Small 3.1 |
| **Projector** | Retrained per-size projection (not the original ViT projection) |
| **Architecture** | `mistral3` in llama.cpp |
| **Context** | 256K tokens (YaRN extended) |
| **License** | **Apache-2.0** |
| **HF Repo** | `mistralai/Ministral-3-3B-Instruct-2512` |

## GGUF Files (from official `mistralai/Ministral-3-3B-Instruct-2512-GGUF`)

| File | Size |
|------|------|
| `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2.15 GB |
| `Ministral-3-3B-Instruct-2512-Q8_0.gguf` | 3.65 GB |
| `Ministral-3-3B-Instruct-2512-BF16.gguf` | 6.87 GB |
| `Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf` | **842 MB** |
| **Total (Q4_K_M + BF16 mmproj)** | **~2.99 GB** |
| **Total (Q8_0 + BF16 mmproj)** | **~4.49 GB** |

Note: Mistral only ships BF16 mmproj. Community quants available from unsloth/noctrex.

## llama.cpp Support

- **Supported** since PR #17644 (merged Dec 1, 2025, collab with Mistral)
- Architecture: `mistral3`
- Follow-up PR #17945 fixed `attn_factor` for mistral3 graphs
- Compatible with `llama-mtmd-cli`, `llama-server --mmproj`

## Benchmarks

| Benchmark | Ministral 3 3B | Qwen3-VL-4B | Gemma3-4B |
|-----------|---------------|-------------|-----------|
| MM MTBench (vision) | 7.83 | 8.01 | 5.23 |
| Arena Hard | 0.305 | 0.438 | 0.318 |
| WildBench | 56.8 | 56.8 | 49.1 |
| MATH Maj@1 | 0.830 | 0.900 | 0.759 |

Reasoning variant scores:
- AIME25: 0.721, AIME24: 0.775, GPQA Diamond: 0.534

---

## Porting Steps

### Step 1: Pin References

```
HF Base:      mistralai/Ministral-3-3B-Instruct-2512-GGUF
Revision:     <pin commit SHA>
Files:        Ministral-3-3B-Instruct-2512-Q8_0.gguf + Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf
License:      apache-2.0
```

### Step 2: Add to `artifacts.json`

```json
"ministral_3b_q8_gguf": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "Ministral-3-3B-Q8_0",
  "filename": "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
  "env": "MINISTRAL_3B_MODEL_PATH",
  "source": {
    "type": "huggingface",
    "repo": "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
    "revision": "<SHA>",
    "filename": "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
    "url": "..."
  },
  "local_cache": "local-artifacts/models/ministral_3b/Ministral-3-3B-Instruct-2512-Q8_0.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 3650000000,
  "note": "Ministral 3 3B LLM Q8_0. Architecture: mistral3."
},
"ministral_3b_mmproj_bf16": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "Ministral-3-3B-mmproj-BF16",
  "filename": "Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
  "env": "MINISTRAL_3B_MMPROJ_PATH",
  "source": { "type": "huggingface", "repo": "...", "revision": "<SHA>", "filename": "...", "url": "..." },
  "local_cache": "local-artifacts/models/ministral_3b/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 842000000,
  "note": "Pixtral ViT 410M + projector. BF16 only from official source."
}
```

### Step 3: Benchmark Config

Key differences:
- `gpu_layers`: 99
- `ctx_size`: 4096 (model supports 256K but reduce)
- `ready_timeout_s`: 900 (larger model)
- `port`: unique port (e.g., 18098)

### Step 4-6: Standard registration pattern

---

## Infrastructure Reuse — MEDIUM

- No existing Mistral models in the hackathon (new ecosystem)
- Board CI, Modal, and benchmark config format are reusable
- `mistral3` architecture needs ET backend verification
- mmproj (842 MB) is proportionally large relative to LLM

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| `mistral3` architecture may not be in llama.cpp-et fork | **Critical** | Check ET fork supported archs; Mistral is a major family so likely supported |
| Total size ~4.5 GB (Q8) may be tight for board | **Medium** | Use Q4_K_M (~3 GB total) as primary |
| Only BF16 mmproj available from official source | **Low** | 842 MB is manageable; community quants exist |
| `llama-cpp-python` is dead for Ministral 3 | **Low** | Not needed for hackathon (use llama-server directly) |

## Why Include This

- **Mistral ecosystem diversity**: First Mistral-family model on the leaderboard
- **256K context**: Matches Qwen3.5-0.8B
- **Strong reasoning**: AIME25 0.721 with reasoning variant
- **Native function calling**: Good for agentic demos
- **Pixtral ViT**: Different vision encoder architecture (not SigLIP)
