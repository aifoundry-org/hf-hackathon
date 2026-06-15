# Whisper Model Card

- Reference family: Whisper Tiny English.
- Hugging Face base: `openai/whisper-tiny.en` at
  `87c7102498dcde7456f24cfd30239ca606ed9063`.
- Reference files: `model.safetensors`, tokenizer, processor, and config files.
- Benchmark source: `src/whisper_transformer_vpu_argbuf.c`.
- Real-model sources: `src/whisper_*_argbuf.c`.
- Smoke manifest: `manifests/whisper_10_variants.txt`.
- Sweep manifest: `manifests/whisper_100_variants.txt`.
- Key docs: `docs/optimizations.md`.

Whisper should be ported by audited blocks: encoder matmul, layernorm,
attention score/value, decoder token steps, cache layout, and logits/argmax.
