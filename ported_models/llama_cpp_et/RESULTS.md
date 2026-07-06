# ET backend benchmark results

Decode throughput on the ET backend, measured with `llama-bench` on the `ET`
device (`-ngl 99`). Prefill is the `pp` figure; decode is the `tg` figure.

The **`UK=0` (baseline) column is what the board benchmark measures** — the
uberkernel path is off unless `GGML_ET_UBERKERNEL` is set, which the board
config does not set. The `UK=1` column is a separate, in-progress decode
optimization (not enabled on the board); it is included here for reference.
"Output OK" records whether the generated text was coherent in each mode.

| Model | Size | Quant | Prefill t/s | Decode t/s (UK=0) | Decode t/s (UK=1) | Decode speedup | Output OK |
|-------|------|-------|------------:|------------------:|------------------:|---------------:|-----------|
| Qwen3-0.6B | 0.80 GB | Q8_0 | 68.4 | 11.7 | 20.9 | 1.79× (+78.6%) | ✅ both |
| Qwen3.5-0.8B | 0.81 GB | Q8_0 | 43.7 | 10.0 | 13.0 | 1.30× (+30.0%) | ✅ both |
| gemma-3-1b-it | 0.72 GB | Q4_0 | 43.1 | 10.6 | — | — | ✅ UK=0 · ❌ UK=1 crash on load |
| gemma-3-1b-it | 1.07 GB | Q8_0 | 41.1 | 10.4 | — | — | ✅ UK=0 · ❌ UK=1 garbled |
| LFM2.5-1.2B-Instruct | 0.70 GB | Q4_0 | 73.8 | 19.4 | — | — | ✅ UK=0 · ❌ UK=1 broken |
| LFM2.5-1.2B-Instruct | 1.25 GB | Q8_0 | 61.1 | 18.0 | — | — | ✅ UK=0 · ❌ UK=1 garbled |
| Llama-3.2-1B-Instruct | 1.32 GB | Q8_0 | 74.1 | 17.7 | 27.0 | 1.53× (+52.5%) | ✅ both |
| tinyllama-1.1b-chat-v1.0 | 1.17 GB | Q8_0 | 75.3 | 15.8 | 27.7 | 1.75× (+75.3%) | ✅ both |
| SmolLM2-1.7B-Instruct | 0.99 GB | Q4_0 | 70.5 | 10.1 | 13.7 | 1.36× (+35.6%) | ✅ both |
| SmolLM2-1.7B-Instruct | 1.82 GB | Q8_0 | 56.0 | 9.9 | — | — | ✅ UK=0 · ❌ UK=1 broken |
| rwkv7-1.5B-g1 | 1.69 GB | Q8_0 | 16.8 | 3.4 | — | — | ✅ UK=0 · ❌ UK=1 broken |
| Qwen3.5-2B | 2.01 GB | Q8_0 | 32.1 | 8.5 | 10.3 | 1.21× (+21.2%) | ✅ both |
| SmolVLM-500M-Instruct | 0.44 GB | Q8_0 | 78.5 | 13.9 | 30.2 | 2.17× (+117.3%) | ✅ both |
| SmolVLM-Instruct | 1.93 GB | Q8_0 | 44.9 | 9.9 | 13.7 | 1.38× (+38.4%) | ✅ both |
| SmolVLM2-2.2B-Instruct | 1.93 GB | Q8_0 | 44.8 | 9.9 | 13.6 | 1.37× (+37.4%) | ✅ both |
| llama-3.2-3b-instruct | 3.42 GB | Q8_0 | 27.9 | 8.1 | 11.2 | 1.38× (+38.3%) | ✅ both |
| Ministral-3b-instruct | 3.52 GB | Q8_0 | 18.7 | 10.7 | — | — | ⚠️ UK=0 garbled · ⚠️ UK=1 coherent but factually wrong |
| gemma-4-E2B-it | 5.05 GB | Q8_0 | 27.7 | 6.3 | 9.5 | 1.51× (+50.8%) | ✅ both |
| llama-7b | 3.83 GB | Q4_0 | 21.2 | 4.8 | 6.0 | 1.25× (+25.0%) | ⚠️ base model — echoes prompt |
| llama-7b | 7.16 GB | Q8_0 | 9.4 | 3.9 | 4.8 | 1.23× (+23.1%) | ⚠️ base model — echoes prompt |
| rwkv7-7.2B-g0 | 7.93 GB | Q8_0 | 4.9 | 1.6 | 1.8 | 1.13× (+12.5%) | ✅ both |
| Llama-3.2-8B-Instruct | 8.54 GB | Q8_0 | 8.0 | 4.8 | 6.2 | 1.29× (+29.2%) | ✅ both |
| Llama-3-8B-Instruct | 8.54 GB | Q8_0 | 8.0 | 4.7 | 6.2 | 1.32× (+31.9%) | ✅ both |
| Qwen3VL-8B-Instruct | 8.71 GB | Q8_0 | 7.5 | 4.3 | 5.8 | 1.35× (+34.9%) | ✅ both |

Notes:

- Models marked `❌ UK=1` produce correct output with the uberkernel path
  disabled; enabling it for those is still in progress.
- Not every model × quant combination has been exercised; the table lists the
  configurations that were measured.
- The three canonical, fully-coherent Q8_0 wins (Llama-3.2-1B, Qwen3-0.6B,
  tinyllama-1.1b) already have benchmark configs under `benchmarks/`, so the
  board can measure them directly against this framework pin.
