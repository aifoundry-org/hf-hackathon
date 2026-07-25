# `phi2` (`phi1_5`) — model-ports reproduce recipe

Ports **Phi-1.5** (Microsoft, MIT) to ET-SoC1 via the shared `llama.cpp-et` runtime. Phi-1.5
uses the Phi-2 execution family (`LLM_ARCH_PHI2`) already implemented in the committed
framework — no framework change is made here. Two-stage `docs/SUBMISSION_GUIDE.md` flow.

## 0. Provenance

- Base weights: `microsoft/phi-1_5`, MIT.
- Benchmark GGUF: `mradermacher/phi-1_5-GGUF` @ `fb3f2464a557228caf113b4d6ca7bebb2dc6c08c`,
  `phi-1_5.Q8_0.gguf`, sha256 `d44836d19a3203c1f5137965cd7244ceddd69bebe42075e2d5979795f4f36ba7`,
  1,510,471,040 bytes. See `docs/HF_REFERENCES.md` and `artifacts.json`.
- No custom quantization/packing.

## 1. Stage 1 — identity + contract approval (maintainer)

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/phi1_5.json`.

Precomputed hashes in the identity entry:

- `benchmark_config_sha256` = `canonical_sha256(effective phi1_5 model config)`:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/phi2/benchmarks/phi2.json"))
  cfg["config"] = "ported_models/phi2/benchmarks/phi2.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/phi1_5.json` after the
  maintainer commits the contract (byte-identical to `docs/proposed_reference_contract.json`).

## 2. Stage 2 — implementation PR (this port)

- New standalone root `ported_models/phi2/`.
- Claim `ported_models/submissions/model_ports/phi2.json`.
- One `phi1_5` entry in `.github/ci/benchmark_config.json`:
  ```json
  "phi1_5": { "config": "ported_models/phi2/benchmarks/phi2.json" }
  ```

## 3. Build (CI, reproducible by hand)

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

build-et/bin/llama-server --model phi-1_5.Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 2048 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

## 4. Board correctness + metric

Main-owned `.github/ci/reference/phi1_5.json` contract: decode validation (≥32 tokens, temp 0,
device ET, full offload, host-CPU agreement 1.0), zero op fallbacks, WikiText-2 perplexity
within bound (ET-vs-CPU relative diff ≤ 0.01), metric decode `tokens_per_second`.

## 5. Op-coverage rationale (confidence MEDIUM)

Phi-1.5 ops — `ROPE` (NEOX partial, n_rot=32), `MUL_MAT` (Q8_0×F32), `NORM` (LayerNorm+bias),
`UNARY` (GELU), `SOFT_MAX`, `GET_ROWS`, `ADD` (biases + parallel residual), `MUL`, `CONT` —
all have real ET kernels. Three are not exercised by any seed model (partial rotary, `NORM`,
`UNARY`-GELU), which is why this is the lowest-confidence port in the set; each has a clean F32
kernel and passes `supports_op` (`rope_f32.c` explicitly handles the partial-rotary tail). See
`MODEL.md` for the per-op mapping. Board-smoke recommended before relying on this port.
