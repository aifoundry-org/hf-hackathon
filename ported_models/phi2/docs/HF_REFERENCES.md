# Phi-1.5 — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `microsoft/phi-1_5` |
| License | `mit` |
| Architecture | `PhiForCausalLM` (24L, hidden 2048, FFN 8192, 32 heads, head_dim 64, vocab 51200, partial rotary dim 32, LayerNorm+bias, GELU, parallel attn+FFN, ctx 2048) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `mradermacher/phi-1_5-GGUF` |
| Revision | `fb3f2464a557228caf113b4d6ca7bebb2dc6c08c` |
| Filename | `phi-1_5.Q8_0.gguf` |
| sha256 | `d44836d19a3203c1f5137965cd7244ceddd69bebe42075e2d5979795f4f36ba7` |
| Size | `1510471040` bytes |
| URL | `https://huggingface.co/mradermacher/phi-1_5-GGUF/resolve/fb3f2464a557228caf113b4d6ca7bebb2dc6c08c/phi-1_5.Q8_0.gguf` |

Direct Q8_0 conversion of Phi-1.5 — no custom quantization, packing, or shape change. The
sha256 above is the Hugging Face LFS object id of the pinned revision; verify with
`sha256sum` after download.
