# `exaone` (`exaone_2_4b`) — model-ports reproduce recipe

Ports **EXAONE-3.5-2.4B-Instruct** (LG AI Research) to ET-SoC1 via the shared `llama.cpp-et`
runtime. EXAONE is a new execution family (`LLM_ARCH_EXAONE`) already implemented in the
committed framework — no framework change is made here. Two-stage `docs/SUBMISSION_GUIDE.md`
flow.

**License caveat:** EXAONE AI Model License (non-commercial). Used only as a CI-downloaded
board benchmark artifact; weights not committed. See `docs/HF_REFERENCES.md`. The Apache-2.0
ports (olmo2, internlm2, granite) are the license-clean alternatives.

## 0. Provenance

- Base weights: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` (EXAONE AI Model License).
- Benchmark GGUF: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF` @
  `142acae803a41c206e8d0fa978c6102c748911bb`, `EXAONE-3.5-2.4B-Instruct-Q8_0.gguf`,
  sha256 `464d3b40dabdc0fb0d1c05c84d51372bc7da44e038708e6924dd2bd4c9128a35`,
  2,838,845,952 bytes. See `docs/HF_REFERENCES.md` and `artifacts.json`.
- No custom quantization/packing.

## 1. Stage 1 — identity + contract approval (maintainer)

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/exaone_2_4b.json`.

Precomputed hashes in the identity entry:

- `benchmark_config_sha256` = `canonical_sha256(effective exaone model config)`:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/exaone/benchmarks/exaone.json"))
  cfg["config"] = "ported_models/exaone/benchmarks/exaone.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/exaone_2_4b.json` after the
  maintainer commits the contract (byte-identical to `docs/proposed_reference_contract.json`).

## 2. Stage 2 — implementation PR (this port)

- New standalone root `ported_models/exaone/`.
- Claim `ported_models/submissions/model_ports/exaone.json`.
- One `exaone_2_4b` entry in `.github/ci/benchmark_config.json`:
  ```json
  "exaone_2_4b": { "config": "ported_models/exaone/benchmarks/exaone.json" }
  ```

## 3. Build (CI, reproducible by hand)

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

build-et/bin/llama-server --model EXAONE-3.5-2.4B-Instruct-Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 2048 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

## 4. Board correctness + metric

Main-owned `.github/ci/reference/exaone_2_4b.json` contract: decode validation (≥32 tokens,
temp 0, device ET, full offload, host-CPU agreement 1.0), zero op fallbacks, WikiText-2
perplexity within bound (ET-vs-CPU relative diff ≤ 0.01), metric decode `tokens_per_second`.

## 5. Op-coverage rationale (confidence HIGH)

EXAONE 3.5 ops — `RMS_NORM`, `ROPE`-NEOX (n_dims=80), `MUL_MAT` (Q8_0×F32), `GLU` SwiGLU,
`SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` — are exactly the ops the seed `llama32_1b` /
`qwen25_05b` / `smollm2` graphs already prove on ET-SoC1. Strict subset of the seed-proven
set; zero implemented-but-unproven ops. GQA (8 KV) reuses the seed GQA path. See `MODEL.md`
for the per-op mapping.
