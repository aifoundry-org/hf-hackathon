# Phi-3-mini-4k-Instruct Model Card

- Reference family: **Phi-3** (Microsoft), decoder-only transformer. New `llama.cpp`
  execution family `LLM_ARCH_PHI3` — not a seed/registered family, and distinct from `phi2`.
- Hugging Face base (weights): `microsoft/Phi-3-mini-4k-instruct`, `mit`.
- Pinned GGUF artifact: `bartowski/Phi-3-mini-4k-instruct-GGUF` at
  `e1447f6da0be91f91683c5d19f938d4f51122d88`, file
  `Phi-3-mini-4k-instruct-Q8_0.gguf`
  (sha256 `0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47`,
  4,061,221,376 bytes).
- Benchmark model id: `phi3_mini`. Runner: `llama_server` (shared `llama.cpp-et`).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (Phi-3-mini-4k)

32 layers, hidden 3072, FFN 8192, 32 attention heads, 32 KV heads (head_dim 96), vocab 32064,
RoPE theta 10000, RMSNorm, gated **SwiGLU** FFN, context 4096.

Phi-3 is a llama-shaped model (RMSNorm + rotary attention + SwiGLU) that fuses the QKV
projection (`wqkv`) and the gate/up projection into single matrices, split by view/reshape.
Handled by `src/models/phi3.cpp`. No LayerNorm, no bias, no ALiBi, no partial rotary. (The
4k-instruct variant uses standard RoPE — no long-rope scaling.)

## Op coverage on ET (confidence: MED-HIGH)

| Op | ET kernel | Status |
|----|-----------|--------|
| `RMS_NORM` | `et-kernels/src/rms_norm_f32.c` | seed-proven |
| `ROPE` (NEOX, full, n_dims=96) | `et-kernels/src/rope_f32.c` | seed-proven |
| `MUL_MAT` (Q8_0 × F32; fused QKV / gate-up) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `GLU` (SwiGLU) | `et-kernels/src/glu_f32.c` | seed-proven |
| `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT`, `VIEW` (QKV/gate-up split) | ggml-et | seed-proven |

Phi-3's op set is a subset of the seed-exercised set — the fused QKV and gate/up matrices are
split with `VIEW`/`CONT` (metadata + copy, both proven), so there are **no
implemented-but-unproven ops**. At 4 GB it is the largest port in the set; the ET board runs
larger models (the seed leaderboard includes 8B), so size is not expected to be a constraint.

## Model-ports track compliance

- New standalone root `ported_models/phi3/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/phi3.json`.
- New execution family `phi3` — not in `baseline_port_roots`, distinct from `phi2`.
- One benchmark entry `phi3_mini` added to `.github/ci/benchmark_config.json`.
