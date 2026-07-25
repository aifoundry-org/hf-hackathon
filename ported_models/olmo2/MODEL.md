# OLMo-2-0425-1B Model Card

- Reference family: **OLMo 2** (AllenAI), decoder-only transformer. New `llama.cpp`
  execution family `LLM_ARCH_OLMO2` — distinct from the seed `llama`/`qwen`/`smollm2`
  families already registered for the model-ports track.
- Hugging Face base (weights): `allenai/OLMo-2-0425-1B-Instruct`, `apache-2.0`.
- Pinned GGUF artifact: `allenai/OLMo-2-0425-1B-Instruct-GGUF` at
  `62f8c199538474c3e33ed5d7e0580abd66686a27`, file
  `OLMo-2-0425-1B-Instruct-Q8_0.gguf`
  (sha256 `4a180742923d27d2434e53062b21f508257a5ee15abda398a3db13385e33d6e6`,
  1,581,667,488 bytes).
- Benchmark model id: `olmo2_1b`. Runner: `llama_server` (shared `llama.cpp-et`).
- Key docs: `docs/RECIPE.md` (reproduce recipe), `docs/HF_REFERENCES.md` (provenance),
  `docs/proposed_identity_entry.json` + `docs/proposed_reference_contract.json`
  (maintainer stage-1 registration inputs).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (from `config.json`)

`OLMo2ForCausalLM`: 16 layers, hidden 2048, FFN 8192, 16 attention heads, 16 KV heads
(head_dim 128), vocab 100352, RoPE theta 500000, RMSNorm eps 1e-6, context 4096.

OLMo-2 is llama-shaped (RMSNorm + rotary attention + gated SwiGLU FFN) with two
distinguishing traits handled by `src/models/olmo2.cpp` in the committed framework:
**post-norm** placement (normalization applied to the attention/FFN outputs) and
**QK-RMSNorm** (RMSNorm on the query and key projections). No attention/FFN bias, no
learned positional embeddings, no partial rotary.

## Why this port passes the ET board smoke

The model-ports smoke rejects any host/CPU fallback: every op in the graph must have a
real ET-SoC1 kernel. OLMo-2's op set is a subset of what the seed models already prove
on the board:

| Op | ET kernel | Seed-proven by |
|----|-----------|----------------|
| `RMS_NORM` (incl. QK-RMSNorm) | `et-kernels/src/rms_norm_f32.c` | llama32_1b, qwen25_05b, smollm2 |
| `ROPE` (NEOX, full, n_dims=128) | `et-kernels/src/rope_f32.c` | all seed llama/qwen |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | all seed models |
| `GLU` (SwiGLU) | `et-kernels/src/glu_f32.c` | all seed llama/qwen |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` | ggml-et | all seed models |

QK-RMSNorm and post-norm add only extra `RMS_NORM`/`ADD` nodes — no new op types. There
are zero implemented-but-unproven ops in the graph (no LayerNorm, no unary GELU, no
partial rotary), which is why OLMo-2 was chosen over falcon/gpt2/stablelm.

## Model-ports track compliance

- New standalone root `ported_models/olmo2/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/olmo2.json`.
- New execution family `olmo2` — not in `baseline_port_roots`, not a variant of any
  registered family.
- One benchmark entry `olmo2` added to `.github/ci/benchmark_config.json` (the single
  writable exception), pointing at `ported_models/olmo2/benchmarks/olmo2.json`.
