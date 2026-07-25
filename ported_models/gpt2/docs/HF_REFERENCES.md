# GPT-2 (124M) — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `openai-community/gpt2` |
| License | `mit` |
| Architecture | `GPT2LMHeadModel` (12L, hidden 768, FFN 3072, 12 heads, vocab 50257, learned positions, LayerNorm+bias, GELU, ctx 1024) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `mradermacher/gpt2-GGUF` |
| Revision | `0cda0c2b1459ccd32256c6ddde9d230934112c1c` |
| Filename | `gpt2.Q8_0.gguf` |
| sha256 | `9ab5d3c0b9ac838651c2bfd2db2d5b75d40077562557ccd23fca9569bdc2eee0` |
| Size | `177669376` bytes |
| URL | `https://huggingface.co/mradermacher/gpt2-GGUF/resolve/0cda0c2b1459ccd32256c6ddde9d230934112c1c/gpt2.Q8_0.gguf` |

Direct Q8_0 conversion of GPT-2 124M — no custom quantization, packing, or shape change. The
sha256 above is the Hugging Face LFS object id of the pinned revision; verify with
`sha256sum` after download.
