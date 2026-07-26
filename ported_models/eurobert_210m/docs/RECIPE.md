# EuroBERT-210m -- Conversion Gap (No Claim Filed)

## Summary

Attempted to add `EuroBERT/EuroBERT-210m` (210M-parameter multilingual
encoder, `arch = eurobert` in llama.cpp) to the `llama.cpp-et` framework.
**No pre-made GGUF exists anywhere on Hugging Face for this model, and
self-conversion fails** because the only public checkpoint declares
`architectures: ["EuroBertForMaskedLM"]`, a Hugging Face model class the
vendored `convert_hf_to_gguf.py` does not recognize. No model-ports claim
is filed.

## Model Reference

- **Source**: `EuroBERT/EuroBERT-210m` (Hugging Face), revision
  `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e`
- **License**: Apache-2.0
- No GGUF quantization exists at any third-party or official repo
  (searched Hugging Face broadly) -- self-conversion was the only path.

## Root cause: confirmed, not speculative

The vendored `convert_hf_to_gguf.py` does have EuroBERT support:

```python
@ModelBase.register("EuroBertModel", "JinaEmbeddingsV5Model")
class EuroBertModel(TextModel):
    model_arch = gguf.MODEL_ARCH.EUROBERT
```

But the registration only recognizes the HF class names `EuroBertModel`
and `JinaEmbeddingsV5Model` -- not `EuroBertForMaskedLM`, which is what
`EuroBERT/EuroBERT-210m`'s own `config.json` actually declares
(`"architectures": ["EuroBertForMaskedLM"]`, confirmed by downloading the
real config). Running the converter against the real downloaded
safetensors fails immediately:

```
INFO:hf-to-gguf:Loading model: eurobert-210m
INFO:hf-to-gguf:Model architecture: EuroBertForMaskedLM
ERROR:hf-to-gguf:Model EuroBertForMaskedLM is not supported
```

## A real alternate path checked and ruled out

The converter's registration also covers `JinaEmbeddingsV5Model` (an
alias mapping to the same `eurobert` arch) -- worth checking whether a
real, public Jina Embeddings v5 model uses that exact class and could
substitute. `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF`
(official, pre-made Q8_0 GGUF) exists and loads, but its
`general.architecture` reports **`qwen3`**, not `eurobert` -- that
specific model is a Qwen3-backbone embedding fine-tune, not built on the
`EuroBertModel`/`JinaEmbeddingsV5Model` class path. `qwen3` is already a
seed identity on this board, so this checkpoint would not create a new
identity even though it loads successfully. Checked live, not assumed.

## Why no claim is filed

Neither the base `EuroBERT/EuroBERT-210m` checkpoint (wrong HF class,
converter rejects it) nor the one available Jina-v5 GGUF (right class
family in principle, but resolves to the already-claimed `qwen3` arch in
practice) actually produces a working, genuinely-new `eurobert` port.

## What would need to change

Either `EuroBERT/EuroBERT-210m`'s `config.json`/model class would need to
be adapted to the recognized `EuroBertModel` class name (a checkpoint
change, not something we control), the vendored converter would need a
new registration for `EuroBertForMaskedLM` specifically (a submodule
change, out of scope for a model-port PR), or a genuine
`JinaEmbeddingsV5Model`-class checkpoint would need to be found publicly
(none located this round).
