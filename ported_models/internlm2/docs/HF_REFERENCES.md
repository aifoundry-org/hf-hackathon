# InternLM2.5-1.8B-Chat — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `internlm/internlm2_5-1_8b-chat` |
| License | `apache-2.0` |
| Architecture | `InternLM2ForCausalLM` (24L, hidden 2048, FFN 8192, 16 heads / 8 KV GQA, vocab 92544, RoPE θ=1000000, bias=false) |

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `internlm/internlm2_5-1_8b-chat-gguf` |
| Revision | `916410ad25d03d5dee11b451fe2b6e0353913b64` |
| Filename | `internlm2_5-1_8b-chat-q8_0.gguf` |
| sha256 | `8526cc24717fcab32b20540c546f8c23a6ea3ff40b86f421a0cd060c8123e8b2` |
| Size | `2009613056` bytes |
| URL | `https://huggingface.co/internlm/internlm2_5-1_8b-chat-gguf/resolve/916410ad25d03d5dee11b451fe2b6e0353913b64/internlm2_5-1_8b-chat-q8_0.gguf` |

The Q8_0 GGUF is a direct InternLM release — no custom quantization, packing, or shape
change was performed. The sha256 above is the Hugging Face LFS object id of the pinned
revision; verify with `sha256sum` after download.
