# EXAONE-3.5-2.4B-Instruct Model Card

- Reference family: **EXAONE 3.5** (LG AI Research), decoder-only transformer. New
  `llama.cpp` execution family `LLM_ARCH_EXAONE` — not a seed/registered family.
- Hugging Face base (weights): `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`.
- Pinned GGUF artifact: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF` (official) at
  `142acae803a41c206e8d0fa978c6102c748911bb`, file
  `EXAONE-3.5-2.4B-Instruct-Q8_0.gguf`
  (sha256 `464d3b40dabdc0fb0d1c05c84d51372bc7da44e038708e6924dd2bd4c9128a35`,
  2,838,845,952 bytes).
- Benchmark model id: `exaone_2_4b`. Runner: `llama_server` (shared `llama.cpp-et`).

**License note:** EXAONE is released under the **EXAONE AI Model License Agreement
(non-commercial / research)**, not a permissive OSI license (HF tag `license:other`). It is
used here only as a board benchmark artifact downloaded at CI time; weights are not
committed or redistributed. This is a more restrictive license than the other ports in this
set (Apache-2.0 / Gemma) — flagged for maintainer review; if the track requires permissive
licensing, prefer the Apache-2.0 ports (olmo2, internlm2, granite).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (from `config.json`)

`ExaoneForCausalLM` (3.5): 30 layers, hidden 2560, FFN 7168, 32 attention heads, 8 KV heads
(GQA, head_dim 80), vocab 102400, RoPE theta 1000000, RMSNorm eps 1e-5, context 32768.

EXAONE 3.5 is a llama-shaped model (RMSNorm + rotary attention + gated **SwiGLU** FFN + GQA),
handled by `src/models/exaone.cpp`. No bias, no LayerNorm, no ALiBi, no partial rotary.

## Op coverage on ET (confidence: HIGH)

| Op | ET kernel | Status |
|----|-----------|--------|
| `RMS_NORM` | `et-kernels/src/rms_norm_f32.c` | seed-proven |
| `ROPE` (NEOX, n_dims=80) | `et-kernels/src/rope_f32.c` | seed-proven |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `GLU` (SwiGLU) | `et-kernels/src/glu_f32.c` | seed-proven |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` | ggml-et | seed-proven |

EXAONE's op set is a strict subset of the seed-exercised set (like OLMo-2 and InternLM2) —
**zero implemented-but-unproven ops.** GQA (8 KV heads) and head_dim 80 stay inside the
seed-proven MatMul/RoPE paths. Board-pass odds are high; the only open question is licensing,
not correctness.

## Model-ports track compliance

- New standalone root `ported_models/exaone/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/exaone.json`.
- New execution family `exaone` — not in `baseline_port_roots`, not a variant of any
  registered family.
- One benchmark entry `exaone_2_4b` added to `.github/ci/benchmark_config.json`.
