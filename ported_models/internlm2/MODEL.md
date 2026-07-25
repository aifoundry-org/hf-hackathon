# InternLM2.5-1.8B-Chat Model Card

- Reference family: **InternLM2** (Shanghai AI Lab), decoder-only transformer. New
  `llama.cpp` execution family `LLM_ARCH_INTERNLM2` — distinct from the seed
  `llama`/`qwen`/`smollm2` families already registered for the model-ports track.
- Hugging Face base (weights): `internlm/internlm2_5-1_8b-chat`, `apache-2.0`.
- Pinned GGUF artifact: `internlm/internlm2_5-1_8b-chat-gguf` at
  `916410ad25d03d5dee11b451fe2b6e0353913b64`, file `internlm2_5-1_8b-chat-q8_0.gguf`
  (sha256 `8526cc24717fcab32b20540c546f8c23a6ea3ff40b86f421a0cd060c8123e8b2`,
  2,009,613,056 bytes).
- Benchmark model id: `internlm2_1_8b`. Runner: `llama_server` (shared `llama.cpp-et`).
- Key docs: `docs/RECIPE.md` (reproduce recipe), `docs/HF_REFERENCES.md` (provenance),
  `docs/proposed_identity_entry.json` + `docs/proposed_reference_contract.json`
  (maintainer stage-1 registration inputs).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (from `config.json`)

`InternLM2ForCausalLM`: 24 layers, hidden 2048, FFN 8192, 16 attention heads, 8 KV heads
(GQA, head_dim 128), vocab 92544, RoPE theta 1000000, RMSNorm eps 1e-5, `bias: false`,
context 32768.

InternLM2 is llama-shaped (RMSNorm + rotary attention + gated SwiGLU FFN) with grouped-
query attention, handled by `src/models/internlm2.cpp` in the committed framework. With
`bias: false` there are no attention/FFN bias terms; no LayerNorm, no learned positional
embeddings, no partial rotary.

## Why this port passes the ET board smoke

The model-ports smoke rejects any host/CPU fallback: every op in the graph must have a
real ET-SoC1 kernel. InternLM2's op set is a subset of what the seed models already prove
on the board:

| Op | ET kernel | Seed-proven by |
|----|-----------|----------------|
| `RMS_NORM` | `et-kernels/src/rms_norm_f32.c` | llama32_1b, qwen25_05b, smollm2 |
| `ROPE` (NEOX, full, n_dims=128) | `et-kernels/src/rope_f32.c` | all seed llama/qwen |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | all seed models |
| `GLU` (SwiGLU) | `et-kernels/src/glu_f32.c` | all seed llama/qwen |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` | ggml-et | all seed models |

GQA (8 KV heads) reuses the same `MUL_MAT`/`SOFT_MAX` path the seed Llama-3.2 and Qwen2.5
GQA graphs already run. There are zero implemented-but-unproven ops in the graph (no
LayerNorm, no unary activation, no partial rotary), which is why InternLM2 ranks alongside
OLMo-2 as a lowest-risk new family.

## Model-ports track compliance

- New standalone root `ported_models/internlm2/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/internlm2.json`.
- New execution family `internlm2` — not in `baseline_port_roots`, not a variant of any
  registered family.
- One benchmark entry `internlm2_1_8b` added to `.github/ci/benchmark_config.json` (the
  single writable exception), pointing at `ported_models/internlm2/benchmarks/internlm2.json`.
