# OLMo-2-0425-1B — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `allenai/OLMo-2-0425-1B-Instruct` |
| License | `apache-2.0` |
| Architecture | `OLMo2ForCausalLM` (16L, hidden 2048, FFN 8192, 16 heads / 16 KV, vocab 100352, RoPE θ=500000) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `allenai/OLMo-2-0425-1B-Instruct-GGUF` |
| Revision | `62f8c199538474c3e33ed5d7e0580abd66686a27` |
| Filename | `OLMo-2-0425-1B-Instruct-Q8_0.gguf` |
| sha256 | `4a180742923d27d2434e53062b21f508257a5ee15abda398a3db13385e33d6e6` |
| Size | `1581667488` bytes |
| URL | `https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct-GGUF/resolve/62f8c199538474c3e33ed5d7e0580abd66686a27/OLMo-2-0425-1B-Instruct-Q8_0.gguf` |

The Q8_0 GGUF is a direct AllenAI release — no custom quantization, packing, or shape
change was performed. The sha256 above is the Hugging Face LFS object id of the pinned
revision; verify with `sha256sum` after download.
