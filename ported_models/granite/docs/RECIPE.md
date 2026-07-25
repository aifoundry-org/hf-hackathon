# `granite` (`granite_2b`) — model-ports reproduce recipe

Ports **Granite-3.0-2B-Instruct** (IBM, Apache-2.0) to ET-SoC1 via the shared
`llama.cpp-et` runtime. Granite is a new execution family (`LLM_ARCH_GRANITE`) already
implemented in the committed framework — no framework change is made here. Two-stage
`docs/SUBMISSION_GUIDE.md` flow.

## 0. Provenance

- Base weights: `ibm-granite/granite-3.0-2b-instruct`, Apache-2.0.
- Benchmark GGUF: `lmstudio-community/granite-3.0-2b-instruct-GGUF` @
  `0f35cb534c61d4f1ea9a8e266efc522db70dc2fa`, `granite-3.0-2b-instruct-Q8_0.gguf`,
  sha256 `41f268169c7f0ab6758d0a51f497d9e55af0226bc71723e0a99a291b08e2ebda`,
  2,801,068,896 bytes. See `docs/HF_REFERENCES.md` and `artifacts.json`.
- No custom quantization/packing.

## 1. Stage 1 — identity + contract approval (maintainer)

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/granite_2b.json`.

Precomputed hashes in the identity entry:

- `benchmark_config_sha256` = `canonical_sha256(effective granite model config)`:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/granite/benchmarks/granite.json"))
  cfg["config"] = "ported_models/granite/benchmarks/granite.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/granite_2b.json` after the
  maintainer commits the contract (byte-identical to `docs/proposed_reference_contract.json`).

## 2. Stage 2 — implementation PR (this port)

- New standalone root `ported_models/granite/`.
- Claim `ported_models/submissions/model_ports/granite.json`.
- One `granite_2b` entry in `.github/ci/benchmark_config.json`:
  ```json
  "granite_2b": { "config": "ported_models/granite/benchmarks/granite.json" }
  ```

## 3. Build (CI, reproducible by hand)

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

build-et/bin/llama-server --model granite-3.0-2b-instruct-Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 2048 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

## 4. Board correctness + metric

Main-owned `.github/ci/reference/granite_2b.json` contract: decode validation (≥32 tokens,
temp 0, device ET, full offload, host-CPU agreement 1.0), zero op fallbacks, WikiText-2
perplexity within bound (ET-vs-CPU relative diff ≤ 0.01), metric decode `tokens_per_second`.

## 5. Op-coverage rationale (confidence MEDIUM-HIGH)

Granite ops — `RMS_NORM`, `ROPE`-NEOX (n_dims=64), `MUL_MAT` (Q8_0×F32), `GLU` **SwiGLU**,
`SCALE` (embedding ×12, residual ×0.22, attention ×0.015625, logits ÷8), `SOFT_MAX`,
`GET_ROWS`, `ADD`, `MUL`, `CONT` — all have real ET kernels. The FFN is SwiGLU
(seed-proven); the only non-seed-exercised op is the trivial elementwise `SCALE`. GQA (8 KV)
and head_dim 64 reuse seed-proven paths. See `MODEL.md` for the per-op mapping.
