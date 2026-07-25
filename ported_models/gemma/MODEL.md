# Gemma-2B-it (v1) Model Card

- Reference family: **Gemma v1** (Google), decoder-only transformer. New `llama.cpp`
  execution family `LLM_ARCH_GEMMA` — distinct from the seed `gemma3n` family and from
  the `gemma2`/`gemma3` families other participants are porting.
- Hugging Face base (weights): `google/gemma-2b-it`, `gemma` license.
- Pinned GGUF artifact: `MaziyarPanahi/gemma-2b-it-GGUF` at
  `72164ae6fc4003cecf37bc07f3a825b8a20b8cbb`, file `gemma-2b-it.Q8_0.gguf`
  (sha256 `dae8a0a75dda3553c06af737adfff0c003ed1b6393932964cce72bb4f0fd41f6`,
  2,669,070,080 bytes). Ungated third-party Q8_0 conversion of the Gemma-licensed base.
- Benchmark model id: `gemma_2b`. Runner: `llama_server` (shared `llama.cpp-et`).
- Key docs: `docs/RECIPE.md`, `docs/HF_REFERENCES.md`,
  `docs/proposed_identity_entry.json` + `docs/proposed_reference_contract.json`.

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (Gemma-2B v1)

18 layers, hidden 2048, FFN 16384, 8 attention heads, 1 KV head (MQA, head_dim 256),
vocab 256000, RoPE theta 10000, RMSNorm eps 1e-6, context 8192.

Gemma v1 is RMSNorm + rotary attention + **GeGLU** FFN, handled by
`src/models/gemma.cpp`. Two Gemma-specific traits, both elementwise `SCALE`: the input
embedding is scaled by `sqrt(hidden_size)` and the query by `1/sqrt(head_dim)`. No
LayerNorm, no bias, no ALiBi.

## Op coverage on ET (confidence: MEDIUM-HIGH)

Every op in the graph has a real ET-SoC1 kernel — verified against the backend, not
assumed:

| Op | ET kernel | Status |
|----|-----------|--------|
| `RMS_NORM` | `et-kernels/src/rms_norm_f32.c` | seed-proven (llama/qwen/smollm2) |
| `ROPE` (NEOX, n_dims=256) | `et-kernels/src/rope_f32.c` | seed-proven; n_dims=256 is at the kernel limit (≤256, %16==0) |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `GLU` (**GeGLU**) | `et-kernels/src/glu_f32.c` (`GGML_GLU_OP_GEGLU`, `block_geglu`) | kernel present, **not seed-exercised** |
| `SCALE` (embed + query scale) | ggml-et | kernel present, **not seed-exercised** |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` | ggml-et | seed-proven |

Unlike OLMo-2 / InternLM2 (whose graphs are a strict subset of the *seed-exercised* set),
Gemma introduces two ops — GeGLU and SCALE — that are implemented in the ET kernels and
pass the `supports_op` gate but are not hit by any seed model. Both are simple elementwise
paths (`glu_f32.c` explicitly handles `GGML_GLU_OP_GEGLU`), so the fallback risk is low but
non-zero. MQA (1 KV head) and head_dim 256 both stay within the seed-proven MatMul/RoPE
paths. This is the reason Gemma is ranked below the two RMSNorm-SwiGLU ports and should be
board-smoked before it is relied on.

## Model-ports track compliance

- New standalone root `ported_models/gemma/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/gemma.json`.
- New execution family `gemma` — not in `baseline_port_roots`, not a variant of the seed
  `gemma3n` family (different architecture + `general.architecture=gemma` vs `gemma3n`).
- One benchmark entry `gemma_2b` added to `.github/ci/benchmark_config.json`.
