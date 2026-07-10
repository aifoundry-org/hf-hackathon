# Qwen3.5-0.8B Q8_0 — llama.cpp-et board benchmark

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo (LLM) | [ggml-org/Qwen3.5-0.8B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) |
| Revision | `9447f74101aeb4e93621884dfa36ee8effb8831b` |
| File (LLM) | `Qwen3.5-0.8B-Q8_0.gguf` |
| Repo (mmproj) | [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) |
| Revision | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` |
| File (mmproj) | `mmproj-BF16.gguf` |
| License | apache-2.0 |
| Export step | None (upstream Q8_0 GGUF used as-is for LLM; BF16 mmproj used as-is) |

## Architecture

**Natively multimodal model** — no separate VL variant needed. Qwen3.5 uses early fusion to integrate vision and language in a single model.

### LLM backbone
- **Architecture**: `qwen35` — Hybrid Gated DeltaNet + Full Attention
- **Parameters**: 752M (LLM only)
- **Layers**: 24 total
  - 18 layers use Gated DeltaNet (linear attention, ~75%)
  - 6 layers use full attention (~25%)
- **Hidden size**: 1024
- **Context**: 262,144 tokens (256K native, reduced to 4096 for board)
- **Vocabulary**: 248,320 tokens (shares Qwen3 vocabulary)

### Vision encoder
- **Architecture**: 27-layer Vision Transformer
- **Parameters**: ~675M
- **Hidden size**: 1152
- **Heads**: 16
- **Patch size**: 16×16

### Projector
- **Input**: 4608 (1152 hidden × 4 from 2×2 spatial merge)
- **Output**: 1024 (LLM hidden size)
- **Parameters**: ~10M

### Total deployment size
- **LLM Q8_0**: 811,843,488 bytes (~812 MB)
- **mmproj BF16**: 207,346,528 bytes (~207 MB)
- **Total**: ~1.0 GB

## Files added

- `ported_models/llama_cpp_et/artifacts.json` — `qwen35_08b_q8_gguf` and `qwen35_08b_mmproj_bf16` artifacts
- `ported_models/llama_cpp_et/benchmarks/qwen35_08b.json` — board runner config
- `.github/ci/benchmark_config.json` — `qwen35_08b` model key
- `docs/HF_REFERENCES.md` — Hugging Face reference rows

## ET backend settings

Reuses proven Qwen3 settings (mirrors `qwen3_8b` and `qwen3_4b`):
- `device=ET`, `gpu_layers=99`
- `ctx_size=4096` (reduced from 256K native for board memory constraints)
- `batch_size=256`, `ubatch_size=128`
- Completion API (text-only benchmark; vision via mmproj is an additional feature not exercised here)
- Longer timeouts: `ready_timeout_s=300`, `request_timeout_s=600`

### Benchmark configuration
```json
{
  "runner": "llama_server",
  "port": 18102,
  "gpu_layers": 99,
  "ctx_size": 4096,
  "batch_size": 256,
  "ubatch_size": 128,
  "flash_attn": false,
  "api": "completion"
}
```

## Known llama.cpp issues

**llama.cpp PR #19468** (merged Feb 2026) added `qwen35` architecture support. However, there are known issues:

1. **Throughput bug #20072**: Reported in llama.cpp issue tracker. Hybrid attention models may experience suboptimal decode throughput on some backends. The ET backend has not been explicitly tested with this architecture yet.

2. **Thinking loops**: Models with reasoning/thinking capabilities (like Qwen3.5) can enter extended thinking loops if not properly constrained. The benchmark uses `temperature=0`, `ignore_eos=true`, and `min_completion_tokens=32` to mitigate this.

3. **Vision benchmark**: This benchmark is text-only. Vision capability via the mmproj projector is documented but not exercised by the standard completion benchmark. Separate vision benchmarking would require image inputs and the `llama-vision` API.

## Infrastructure reuse

This port reuses existing Qwen3 infrastructure:
- Same vocabulary (248,320 tokens) as Qwen3-8B
- Similar ET backend settings as `qwen3_8b` and `qwen3_4b`
- Same benchmark harness and validation dataset (WikiText-2 raw test)
- Same `llama.cpp-et` framework (no architecture-specific patches needed beyond PR #19468)

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models qwen35_08b --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llama_server_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Similar Qwen3 benchmark: `benchmarks/qwen3_8b.json`
- Similar VLM benchmark: `benchmarks/smolvlm_256m.json`
- Qwen3.5 architecture: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- llama.cpp PR #19468: qwen35 architecture support

## Notes

- **No separate GGUF conversion needed**: ggml-org provides official Q8_0 GGUF
- **mmproj from unsloth**: The official ggml-org repo does not include mmproj; unsloth provides pre-converted vision projector
- **Early fusion advantage**: Unlike traditional VLMs (e.g., LLaVA, Idefics3), Qwen3.5 does not require a separate vision encoder — the ViT is natively integrated
- **Board memory**: Total ~1.0 GB deployment fits within ET-SoC1 memory constraints with room for KV cache
- **Architecture novelty**: Hybrid GatedDeltaNet (75% linear attention) may exhibit different performance characteristics than pure attention models on the ET backend
