# Gemma-2B-it (v1) — Hugging Face provenance

Pinned so the model identity is reproducible (per `docs/HF_REFERENCES.md` submission rule).

## Base model (weights)

| Field | Value |
|-------|-------|
| Repo | `google/gemma-2b-it` |
| License | `gemma` (Gemma Terms of Use) |
| Architecture | `GemmaForCausalLM` v1 (18L, hidden 2048, FFN 16384, 8 heads / 1 KV MQA, head_dim 256, vocab 256000, RoPE θ=10000, GeGLU) |

The base model is gated on Hugging Face (Gemma license acceptance). The benchmark artifact
below is an ungated third-party Q8_0 conversion.

## Benchmark artifact (Q8_0 GGUF)

| Field | Value |
|-------|-------|
| Repo | `MaziyarPanahi/gemma-2b-it-GGUF` |
| Revision | `72164ae6fc4003cecf37bc07f3a825b8a20b8cbb` |
| Filename | `gemma-2b-it.Q8_0.gguf` |
| sha256 | `dae8a0a75dda3553c06af737adfff0c003ed1b6393932964cce72bb4f0fd41f6` |
| Size | `2669070080` bytes |
| URL | `https://huggingface.co/MaziyarPanahi/gemma-2b-it-GGUF/resolve/72164ae6fc4003cecf37bc07f3a825b8a20b8cbb/gemma-2b-it.Q8_0.gguf` |

Direct Q8_0 conversion of Gemma-2B-it — no custom quantization, packing, or shape change.
The sha256 above is the Hugging Face LFS object id of the pinned revision; verify with
`sha256sum` after download.
