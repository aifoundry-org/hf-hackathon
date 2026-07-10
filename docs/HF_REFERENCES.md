# Hugging Face Model References

Use Hugging Face model repos as the base reference for hackathon ports. Pin the
repo, revision, and file in your PR so the model identity is reproducible.

Downloaded model packages, board dumps, and exported ELFs stay out of git. A
small fixed validation fixture may be committed when its public source, source
hash, deterministic transform, and committed hash are all recorded. If a port
needs derived artifacts, document the conversion command from the pinned
Hugging Face base model.

## Current showcase bases

| Showcase row | Hugging Face base | Revision | License tag | File(s) |
|--------------|-------------------|----------|-------------|---------|
| `yolo` | `kadirnar/yolov10n` | `9fa42234fbcdb13b78fa57ebaac6c50e6dd2eb21` | `agpl-3.0` | `yolov10n.pt` |
| `lfm25` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `047e06635fbe71469926b35ea414537245218200` | `other` | `LFM2.5-1.2B-Instruct-Q8_0.gguf` |
| `llama32_1b` | `lmstudio-community/Llama-3.2-1B-Instruct-GGUF` | `199151125cf15a129ab3b548b26afeed976df066` | `llama3.2` | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| `gemma3n_e2b` | `ggml-org/gemma-3n-E2B-it-GGUF` | `989cffaba23976934324f5e3abfabe31b30eb73b` | `gemma` | `gemma-3n-E2B-it-Q8_0.gguf` |
| `tinyllama11b` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `52e7645ba7c309695bec7ac98f4f005b139cf465` | `apache-2.0` | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` |
| `rwkv7_15b` | `Mungert/rwkv7-1.5B-world-GGUF` | `c8a25c8d349fdf76837c68bfc73c0b953f41b3ce` | `apache-2.0` | `rwkv7-1.5B-world-q8_0.gguf` |
| `qwen3_8b` | `ggml-org/Qwen3-8B-GGUF` | `2473489dc243ccaffb4ce569c55bf1df66b2088f` | `apache-2.0` | `Qwen3-8B-Q8_0.gguf` |

## Submission rule

For new models or updates to existing models:

1. Start from a Hugging Face model repo whenever the model family exists there.
2. Pin the exact revision and filename(s).
3. Record the upstream model license.
4. Document any export, quantization, packing, preprocessing, or shape changes.
| `smolvlm_256m` | `ggml-org/SmolVLM-256M-Instruct-GGUF` | `b9e4379657e1450d04d02eec8e345667265b0a00` | `apache-2.0` | `SmolVLM-256M-Instruct-Q8_0.gguf`, `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` |
