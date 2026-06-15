# Whisper Reference Model

Hugging Face refs are downloaded on demand with
`scripts/download_hf_refs.sh`; do not commit downloaded model packages or
generated metadata JSON. The current base is `openai/whisper-tiny.en` pinned in
[`docs/HF_REFERENCES.md`](../../docs/HF_REFERENCES.md).

This folder includes both the compact benchmark kernels and the larger
Whisper real-model kernel notes. The port is layer/block oriented: encoder
matmuls, layernorm, scalar ops, decoder token steps, cache layout, and PMC
ledgers are iterated independently.

Start with `docs/optimizations.md`.
