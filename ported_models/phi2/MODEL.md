# Phi-1.5 Model Card

- Reference family: **Phi-1.5 / Phi-2** (Microsoft), decoder-only transformer. New
  `llama.cpp` execution family `LLM_ARCH_PHI2` — not a seed/registered family.
- Hugging Face base (weights): `microsoft/phi-1_5`, `mit`.
- Pinned GGUF artifact: `mradermacher/phi-1_5-GGUF` at
  `fb3f2464a557228caf113b4d6ca7bebb2dc6c08c`, file `phi-1_5.Q8_0.gguf`
  (sha256 `d44836d19a3203c1f5137965cd7244ceddd69bebe42075e2d5979795f4f36ba7`,
  1,510,471,040 bytes).
- Benchmark model id: `phi1_5`. Runner: `llama_server` (shared `llama.cpp-et`).

Weights are not committed. The board CI downloads the pinned Q8_0 GGUF by url + sha256
(`artifacts.json`).

## Architecture (Phi-1.5)

24 layers, hidden 2048, FFN 8192, 32 attention heads (head_dim 64), vocab 51200,
**partial rotary** (rotary dim 32 = 0.5 × head_dim), LayerNorm (with bias), GELU FFN,
**parallel attention + FFN** (single pre-LayerNorm per block), context 2048.

Phi-1.5 is the Phi-2 architecture — handled by `src/models/phi2.cpp`. Like GPT-J, each block
runs attention and FFN in parallel off one LayerNorm and sums both into the residual.

## Op coverage on ET (confidence: MEDIUM)

| Op | ET kernel | Status |
|----|-----------|--------|
| `ROPE` (NEOX, **partial**, n_rot=32) | `et-kernels/src/rope_f32.c` | kernel copies the un-rotated tail → partial rotary supported (n_rot=32, %16==0) |
| `MUL_MAT` (Q8_0 × F32) | `et-kernels/src/mul_mat_*.c` | seed-proven |
| `NORM` (LayerNorm + bias) | `et-kernels/src/norm_f32.c` | kernel present, **not seed-exercised** |
| `UNARY` (GELU) | `et-kernels/src/unary_f32.c` | kernel present, **not seed-exercised** |
| `SOFT_MAX`, `GET_ROWS`, `ADD` (biases + parallel residual), `MUL`, `CONT` | ggml-et | seed-proven |

This is the most-unproven-op port in the set (partial rotary + `NORM` LayerNorm +
`UNARY` GELU are all implemented in the ET kernels but not exercised by any seed model), hence
MEDIUM rather than MED-HIGH. All three paths pass `supports_op` and have clean F32 kernels
(`rope_f32.c` explicitly handles the partial-rotary tail). At 1.5 GB it runs quickly. This
port is a candidate that should be board-smoked before it is relied on.

## Model-ports track compliance

- New standalone root `ported_models/phi2/`; only regular files added beneath it.
- Claim: `ported_models/submissions/model_ports/phi2.json`.
- New execution family `phi2` — not in `baseline_port_roots`, distinct from `phi3`.
- One benchmark entry `phi1_5` added to `.github/ci/benchmark_config.json`.
