# Phi-3-mini-4k-Instruct — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `microsoft/Phi-3-mini-4k-instruct` |
| License | `mit` |
| Architecture | `Phi3ForCausalLM` (32L, hidden 3072, FFN 8192, 32 heads / 32 KV, head_dim 96, vocab 32064, RoPE θ=10000, RMSNorm, SwiGLU, fused QKV + gate/up, ctx 4096) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `bartowski/Phi-3-mini-4k-instruct-GGUF` |
| Revision | `e1447f6da0be91f91683c5d19f938d4f51122d88` |
| Filename | `Phi-3-mini-4k-instruct-Q8_0.gguf` |
| sha256 | `0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47` |
| Size | `4061221376` bytes |
| URL | `https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/e1447f6da0be91f91683c5d19f938d4f51122d88/Phi-3-mini-4k-instruct-Q8_0.gguf` |

Direct Q8_0 conversion of Phi-3-mini-4k-Instruct — no custom quantization, packing, or shape
change. The Microsoft GGUF repo ships only fp16/q4, so the Q8_0 is pinned from the widely-used
bartowski conversion. The sha256 above is the Hugging Face LFS object id of the pinned
revision; verify with `sha256sum` after download.
