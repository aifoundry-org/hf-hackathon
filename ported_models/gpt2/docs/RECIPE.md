# `gpt2` (`gpt2`) — model-ports reproduce recipe

Ports **GPT-2 124M** (OpenAI, MIT) to ET-SoC1 via the shared `llama.cpp-et` runtime. GPT-2 is
a new execution family (`LLM_ARCH_GPT2`) already implemented in the committed framework — no
framework change is made here. Two-stage `docs/SUBMISSION_GUIDE.md` flow.

## 0. Provenance

- Base weights: `openai-community/gpt2`, MIT.
- Benchmark GGUF: `mradermacher/gpt2-GGUF` @ `0cda0c2b1459ccd32256c6ddde9d230934112c1c`,
  `gpt2.Q8_0.gguf`, sha256 `9ab5d3c0b9ac838651c2bfd2db2d5b75d40077562557ccd23fca9569bdc2eee0`,
  177,669,376 bytes. See `docs/HF_REFERENCES.md` and `artifacts.json`.
- No custom quantization/packing.

## 1. Stage 1 — identity + contract approval (maintainer)

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/gpt2.json`.

Precomputed hashes in the identity entry:

- `benchmark_config_sha256` = `canonical_sha256(effective gpt2 model config)`:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/gpt2/benchmarks/gpt2.json"))
  cfg["config"] = "ported_models/gpt2/benchmarks/gpt2.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/gpt2.json` after the
  maintainer commits the contract (byte-identical to `docs/proposed_reference_contract.json`).

## 2. Stage 2 — implementation PR (this port)

- New standalone root `ported_models/gpt2/`.
- Claim `ported_models/submissions/model_ports/gpt2.json`.
- One `gpt2` entry in `.github/ci/benchmark_config.json`:
  ```json
  "gpt2": { "config": "ported_models/gpt2/benchmarks/gpt2.json" }
  ```

## 3. Build (CI, reproducible by hand)

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

build-et/bin/llama-server --model gpt2.Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 1024 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

`--ctx-size 1024` matches GPT-2's trained context length (no RoPE, so it cannot be extended).

## 4. Board correctness + metric

Main-owned `.github/ci/reference/gpt2.json` contract: decode validation (≥32 tokens, temp 0,
device ET, full offload, host-CPU agreement 1.0), zero op fallbacks, WikiText-2 perplexity
within bound (ET-vs-CPU relative diff ≤ 0.01), metric decode `tokens_per_second`.

## 5. Op-coverage rationale (confidence MEDIUM-HIGH)

GPT-2 ops — `GET_ROWS` (token + learned position embeddings), `MUL_MAT` (Q8_0×F32; fused
QKV/attn/FFN), `NORM` (LayerNorm+bias), `UNARY` (GELU), `SOFT_MAX`, `ADD` (biases + residual),
`MUL`, `CONT` — all have real ET kernels. There is **no rotary embedding**, eliminating the
RoPE edge cases; positions are a plain `GET_ROWS`+`ADD`. The two non-seed-exercised ops,
`NORM` and `UNARY`-GELU, both have clean F32 kernels (`norm_f32.c`, `unary_f32.c`). See
`MODEL.md` for the per-op mapping.
