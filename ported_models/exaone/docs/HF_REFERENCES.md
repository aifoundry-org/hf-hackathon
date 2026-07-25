# EXAONE-3.5-2.4B-Instruct — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` |
| License | EXAONE AI Model License Agreement (non-commercial / research); HF tag `other` |
| Architecture | `ExaoneForCausalLM` 3.5 (30L, hidden 2560, FFN 7168, 32 heads / 8 KV GQA, head_dim 80, vocab 102400, RoPE θ=1000000, SwiGLU) |

**License caveat:** unlike the Apache-2.0 ports in this set, EXAONE is non-commercial. It is
used only as a board benchmark artifact downloaded at CI time; weights are not committed.
Flagged for maintainer decision on whether the track admits non-commercial licenses.

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF` (official LG AI release) |
| Revision | `142acae803a41c206e8d0fa978c6102c748911bb` |
| Filename | `EXAONE-3.5-2.4B-Instruct-Q8_0.gguf` |
| sha256 | `464d3b40dabdc0fb0d1c05c84d51372bc7da44e038708e6924dd2bd4c9128a35` |
| Size | `2838845952` bytes |
| URL | `https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF/resolve/142acae803a41c206e8d0fa978c6102c748911bb/EXAONE-3.5-2.4B-Instruct-Q8_0.gguf` |

Direct official Q8_0 conversion — no custom quantization, packing, or shape change. The
sha256 above is the Hugging Face LFS object id of the pinned revision; verify with
`sha256sum` after download.
