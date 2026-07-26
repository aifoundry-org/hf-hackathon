# ERNIE-4.5-0.3B-PT Porting Recipe

## Overview

Adds `baidu/ERNIE-4.5-0.3B-PT` (360M-parameter causal LM) to the
`llama_cpp_et` benchmark suite as a fully automated, board-scored model
port, using `bartowski/baidu_ERNIE-4.5-0.3B-PT-GGUF`'s Q8_0 GGUF. Same
rationale as the other ports in this session: decoder-only causal LM,
uses the existing, unmodified `"api": "completion"` benchmark path -- no
protected file touched.

This introduces the **ERNIE 4.5** execution family to the board.

## Why this port needed no new ET-SoC1 kernel work

ERNIE 4.5 (dense variant) uses grouped-query attention (16 query heads / 2
KV heads), RoPE, RMSNorm, and a standard FFN. Every op involved
(`GGML_OP_RMS_NORM`, `GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`,
`GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.
`convert_hf_to_gguf.py` registers `Ernie4_5_ForCausalLM` (distinct from
`Ernie4_5_MoeForCausalLM`, which this is not), and the GGUF used here is
a pre-made community conversion, so this port is pure configuration
wiring.

## Model Reference

- **GGUF source**: `bartowski/baidu_ERNIE-4.5-0.3B-PT-GGUF` (Hugging Face)
- **File**: `baidu_ERNIE-4.5-0.3B-PT-Q8_0.gguf`
- **License**: Apache-2.0 (matches upstream `baidu/ERNIE-4.5-0.3B-PT`)
- **Architecture**: `Ernie4_5_ForCausalLM`, 18 transformer layers, GQA (16
  query heads / 2 KV heads), 131072-token native context.
- **Quantization**: Q8_0 (~366 MiB per the model's own reported file size;
  ~386 MB on disk).

## Steps Taken

1. **Downloaded** `baidu_ERNIE-4.5-0.3B-PT-Q8_0.gguf` from the Hugging
   Face repo above; verified `sha256=022ce1cfc46a2818a96b8a683aec0944047cac6745f35aa5a2a48729a58caa74`,
   `size=385789248` bytes (exact match) against the download.
2. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18111`. Confirmed the model correctly
   identifies as `arch = ernie4_5`, `360.75 M` params, full `19/19` layer
   ET offload (17 repeating layers + output layer), the model's native
   131072-token context correctly detected at load time, and a clean
   654-node / 2-split compute graph. As with the other ports in this
   session, full multi-token `/completion` decode output was not
   captured locally within this session -- the real per-PR board score
   for `"board": true` models comes from the self-hosted **real ET-SoC1
   hardware** runner, not local sysemu.
3. **Registered**:
   - `ported_models/llama_cpp_et/artifacts.json` -- new
     `ernie45_03b_q8_gguf` artifact entry, following the
     `qwen25_05b_q8_gguf` pattern.
   - `ported_models/llama_cpp_et/benchmarks/ernie45_03b.json` -- new
     benchmark config, `ctx_size: 2048` (kept comparable to other small
     models rather than the full native 131072), port `18111` (next free
     port after this session's earlier additions).
   - `.github/ci/benchmark_config.json` -- new `"ernie45_03b"` entry
     pointing at the benchmark config above.

## Instructions for Reproduction

```bash
huggingface-cli download bartowski/baidu_ERNIE-4.5-0.3B-PT-GGUF baidu_ERNIE-4.5-0.3B-PT-Q8_0.gguf --local-dir .
./bin/llama-server -m baidu_ERNIE-4.5-0.3B-PT-Q8_0.gguf --device ET -ngl 99 --port 18111
curl -s http://127.0.0.1:18111/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK","n_predict":96,"temperature":0}'
```

## Open items for maintainer review

- No changes were made to any protected file.
- As with the other ports added this session, this has not been claimed
  against the "most models ported" track -- that requires a maintainer to
  first register an `identity_id` on `main` (asked separately). This PR is
  a general board addition only.

## Negative result from this same investigation (not included in this PR)

`pfnet/plamo-2-1b` (via `mmnga/plamo-2-1b-gguf`, Q8_0) was also loaded
against this same ET backend and **crashed at graph-reserve time** with
the identical error already seen with RWKV-6 in the previous PR:
`pre-allocated tensor (cache_r_l0 (view) (copy of )) in a buffer (ET)
that cannot run the operation (CPY)`. PLaMo-2 uses a hybrid Mamba-style
recurrent layer, and its recurrent state cache needs the same `GGML_OP_CPY`
this ET backend doesn't implement. Combined with the RWKV-6 finding, this
is now two independent confirmations that **any architecture using
llama.cpp's recurrent-cache mechanism (Mamba/Mamba2/RWKV6/PLaMo2-hybrid/
Jamba-style) is not portable to this backend as-is**. Not included in this
or any PR; flagging as useful signal for future ET backend op coverage.
