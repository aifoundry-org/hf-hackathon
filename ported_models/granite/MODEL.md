# Granite-3.0-2B-Instruct Model Card

- Reference family: **Granite 3.0** (IBM), decoder-only transformer. New `llama.cpp`
  execution family `LLM_ARCH_GRANITE` — not a seed/registered family.
- Hugging Face base (weights): `ibm-granite/granite-3.0-2b-instruct`, `apache-2.0`.
- Pinned GGUF artifact: `lmstudio-community/granite-3.0-2b-instruct-GGUF` at
  `0f35cb534c61d4f1ea9a8e266efc522db70dc2fa`, file
  `granite-3.0-2b-instruct-Q8_0.gguf`
  (sha256 `41f268169c7f0ab6758d0a51f497d9e55af0226bc71723e0a99a291b08e2ebda`,
  2,801,068,896 bytes).
- Benchmark model id: `granite_2b`. Runner: `llama_server` (shared `llama.cpp-et`).
- Key docs: `docs/RECIPE.md`, `docs/HF_REFERENCES.md`,
  `docs/proposed_identity_entry.json` + `docs/proposed_reference_contract.json`.

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (from `config.json`)

`GraniteForCausalLM`: 40 layers, hidden 2048, FFN 8192, 32 attention heads, 8 KV heads
(GQA, head_dim 64), vocab 49155, RoPE theta 10000, RMSNorm eps 1e-5, context 4096.

Granite 3.0 is llama-shaped (RMSNorm + rotary attention + gated **SwiGLU** FFN + GQA) with
four scalar multipliers applied as elementwise `SCALE`: embedding (×12.0), residual
(×0.22), attention (×0.015625), and logits (÷8.0). Handled by `src/models/granite.cpp`. No
LayerNorm, no bias, no ALiBi, no partial rotary.

## Op coverage on ET (confidence: MEDIUM-HIGH)

| Op | ET kernel | Status |
|----|-----------|--------|
| `RMS_NORM` | `et-kernels/src/rms_norm_f32.c` | seed-proven |
| `ROPE` (NEOX, n_dims=64) | `et-kernels/src/rope_f32.c` | seed-proven |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `GLU` (**SwiGLU**) | `et-kernels/src/glu_f32.c` | seed-proven |
| `SCALE` (4 scalar multipliers) | ggml-et | kernel present, **not seed-exercised** |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` | ggml-et | seed-proven |

Granite is stronger than the Gemma port: its FFN is **SwiGLU** (seed-proven), so the only
non-seed-exercised op is the trivial elementwise `SCALE`. GQA (8 KV heads) and head_dim 64
stay inside the seed-proven MatMul/RoPE paths. Fallback risk is low.

## Model-ports track compliance

- New standalone root `ported_models/granite/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/granite.json`.
- New execution family `granite` — not in `baseline_port_roots`, not a variant of any
  registered family.
- One benchmark entry `granite_2b` added to `.github/ci/benchmark_config.json`.
