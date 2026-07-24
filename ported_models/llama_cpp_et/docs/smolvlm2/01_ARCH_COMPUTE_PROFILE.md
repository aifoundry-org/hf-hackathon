# SmolVLM2-500M-Video — Architecture & Compute Profile (ET-SoC1 port)

Target: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `7b375e1b73b11138ff12fe22c8f2822d8fe03467`,
run through the pinned `llama.cpp-et` runtime (`cc4049d…`, commit "Add ET vision kernels for SmolVLM2").
Numbers below are grounded in the pinned `config.json`, `.github/ci/reference/smolvlm2_500m_video.json`,
and the SmolVLM (arXiv 2504.05299) / Idefics3 (2408.12637) / SigLIP (2303.15343) papers.

## 1. Architecture

| Block | Field | Value |
|---|---|---|
| **Vision (SigLIP-B/16)** | layers | 12 |
| | hidden / heads / head_dim | 768 / 12 / 64 |
| | MLP intermediate | 3072 (4×) |
| | activation | `gelu_pytorch_tanh` (tanh-approx GELU) |
| | patch / image | 16 / 512² → **32×32 = 1024 patches per frame** |
| | norm / attention | LayerNorm (pre-norm), bidirectional, **no CLS, no KV cache** |
| **Connector (idefics3)** | type | pixel-shuffle (r=4 → 16× token compression) + 1 Linear (no bias) |
| | tokens/frame after shuffle | 1024/16 = **64** |
| | projector matmul | 12288 → 960 (768·r² → LLM hidden) |
| **LLM (SmolLM2-360M, llama arch)** | layers | 32 |
| | hidden / intermediate | 960 / 2560 |
| | heads / KV-heads | 15 / 5 (GQA 3:1, head_dim 64) |
| | activation / norm | SwiGLU (SiLU) / RMSNorm (eps 1e-5) |
| | RoPE θ / vocab / max pos | 100000 / 49280 / 8192 |
| | embeddings | tied lm_head↔input embedding |

## 2. The scored workload (`video_dog`)

- 4 frames, each one 512² tile → **64 visual tokens/frame → 256 image tokens** + a few structural
  tokens. Token count scales **linearly** with frames (no cross-frame pooling).
- Prefill ≈ 256 image tokens + chat template + question ≈ **~290–320 tokens**; **generation = 3 tokens**
  (from the reference JSON). Decode is therefore negligible — **the score is prefill + vision bound.**
- Metric: `device_cmd_exec_dur` firmware cycles, median of 1 measured request, ≥4000 kernel launches;
  candidate must beat paired-main mean by **≥1%** with **≤0.5% main drift** and **≥0.25% wall** corroboration.

## 3. Where the cycles go (MAC estimate, prefill ~300 tok)

| Stage | MACs | Share |
|---|---:|---:|
| **(a) Vision encoder × 4 frames** | ~427 G | **~80%** |
| (b) Projector (pixel-shuffle + Linear) | ~3 G | ~0.6% |
| (c) LLM prefill (~300 tok) | ~100 G | ~18% |
| (d) LLM decode (3 tok) | ~1 G | ~0.2% |

**Inside the vision encoder (the dominant 80%), per 4 frames:**
- **SigLIP MLP fc1/fc2 (768↔3072): ~232 G — largest single class, ~43% of the WHOLE workload.**
- Attention QKV + O projections: ~116 G (~22%).
- Attention scores/context O(N²=1024²), bidirectional: ~77 G (~14%).
- Patch-embed conv (stride-16 non-overlapping = pure im2col-GEMM): ~2.4 G.

## 4. Verdict → optimization aim

**Strongly VISION-ENCODER-BOUND.** The SigLIP forward is ~4× the LLM prefill compute. The three biggest
levers, in order:

1. **SigLIP MLP GEMMs (fc1/fc2)** — 43% of all compute. This is a dense `[1024×768]·[768×3072]` (and back)
   GEMM, batched over 4 independent frames. Saturating the tensor engine here is the whole game.
2. **Attention QKV/O projection GEMMs** — another 22%, same GEMM shape family.
3. **O(N²) self-attention** — 14%; a FlashAttention-style fused pass avoids materializing the 1024×1024
   score matrix.

The 4 frames are **embarrassingly parallel** (batch of 4, no causal mask, no KV cache, no cross-frame
dependency) — ideal for multi-hart partition. The LLM path (RMSNorm/RoPE/SwiGLU/Q8_0 matmul) is real but
secondary; decode is ~nothing at 3 generated tokens.

See `02_OPTIMIZATION_CATALOG.md` for techniques, `04_KERNEL_STATE.md` for the current kernel implementations
these ops route to, and `00_RESEARCH_PLAN.md` for the sequenced plan.
