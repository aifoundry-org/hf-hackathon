# GPT-2 (124M) Model Card

- Reference family: **GPT-2** (OpenAI), decoder-only transformer. New `llama.cpp`
  execution family `LLM_ARCH_GPT2` — not a seed/registered family.
- Hugging Face base (weights): `openai-community/gpt2`, `mit`.
- Pinned GGUF artifact: `mradermacher/gpt2-GGUF` at
  `0cda0c2b1459ccd32256c6ddde9d230934112c1c`, file `gpt2.Q8_0.gguf`
  (sha256 `9ab5d3c0b9ac838651c2bfd2db2d5b75d40077562557ccd23fca9569bdc2eee0`,
  177,669,376 bytes).
- Benchmark model id: `gpt2`. Runner: `llama_server` (shared `llama.cpp-et`).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (GPT-2 124M)

12 layers, hidden 768, FFN 3072, 12 attention heads (MHA, head_dim 64), vocab 50257,
**learned positional embeddings** (no RoPE), LayerNorm (with bias), GELU FFN, context 1024.

GPT-2 is the classic pre-LLaMA architecture — handled by `src/models/gpt2.cpp`. Positions
come from a learned embedding table (`ggml_get_rows(pos_embd)` + `ggml_add`), not rotary.
This is the smallest port in the set (~178 MB Q8_0), so its board run is near-instant.

> **Note:** the benchmark config uses `ctx_size: 1024` (GPT-2's trained context length), not
> the 2048 used by the RoPE models.

## Op coverage on ET (confidence: MEDIUM-HIGH)

| Op | ET kernel | Status |
|----|-----------|--------|
| `GET_ROWS` (token + position embeddings) | ggml-et | seed-proven |
| `MUL_MAT` (Q8_0 × F32; fused QKV, attn, FFN) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `NORM` (LayerNorm + bias, 2×/layer) | `et-kernels/src/norm_f32.c` | kernel present, **not seed-exercised** |
| `UNARY` (GELU) | `et-kernels/src/unary_f32.c` | kernel present, **not seed-exercised** |
| `SOFT_MAX`, `ADD` (biases + residual), `MUL`, `CONT` | ggml-et | seed-proven |

GPT-2 has **no rotary embedding**, which removes the RoPE edge cases entirely; positions are
a plain `GET_ROWS` + `ADD`. The two non-seed-exercised ops — `NORM` (LayerNorm) and `UNARY`
GELU — both have real ET kernels (`norm_f32.c`, `unary_f32.c`) and pass the `supports_op`
gate. Fallback risk is low; combined with the tiny model size this is a fast, high-confidence
board run.

## Model-ports track compliance

- New standalone root `ported_models/gpt2/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/gpt2.json`.
- New execution family `gpt2` — not in `baseline_port_roots`, not a variant of any
  registered family.
- One benchmark entry `gpt2` added to `.github/ci/benchmark_config.json`.
