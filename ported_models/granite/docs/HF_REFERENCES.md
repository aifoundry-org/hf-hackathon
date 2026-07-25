# Granite-3.0-2B-Instruct — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `ibm-granite/granite-3.0-2b-instruct` |
| License | `apache-2.0` |
| Architecture | `GraniteForCausalLM` (40L, hidden 2048, FFN 8192, 32 heads / 8 KV GQA, vocab 49155, RoPE θ=10000, SwiGLU + scalar multipliers) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `lmstudio-community/granite-3.0-2b-instruct-GGUF` |
| Revision | `0f35cb534c61d4f1ea9a8e266efc522db70dc2fa` |
| Filename | `granite-3.0-2b-instruct-Q8_0.gguf` |
| sha256 | `41f268169c7f0ab6758d0a51f497d9e55af0226bc71723e0a99a291b08e2ebda` |
| Size | `2801068896` bytes |
| URL | `https://huggingface.co/lmstudio-community/granite-3.0-2b-instruct-GGUF/resolve/0f35cb534c61d4f1ea9a8e266efc522db70dc2fa/granite-3.0-2b-instruct-Q8_0.gguf` |

Direct Q8_0 conversion of Granite-3.0-2B-Instruct — no custom quantization, packing, or
shape change. The sha256 above is the Hugging Face LFS object id of the pinned revision;
verify with `sha256sum` after download.
