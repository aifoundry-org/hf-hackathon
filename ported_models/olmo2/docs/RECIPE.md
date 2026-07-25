# `olmo2` (`olmo2_1b`) — model-ports reproduce recipe

Ports **OLMo-2-0425-1B-Instruct** (AllenAI, Apache-2.0) to ET-SoC1 via the shared
`llama.cpp-et` runtime. OLMo 2 is a new execution family (`LLM_ARCH_OLMO2`) already
implemented in the committed framework — no framework change is made here. This recipe
follows the two-stage `docs/SUBMISSION_GUIDE.md` flow.

## 0. Provenance

- Base weights: `allenai/OLMo-2-0425-1B-Instruct`, Apache-2.0.
- Benchmark GGUF: `allenai/OLMo-2-0425-1B-Instruct-GGUF` @
  `62f8c199538474c3e33ed5d7e0580abd66686a27`, `OLMo-2-0425-1B-Instruct-Q8_0.gguf`,
  sha256 `4a180742923d27d2434e53062b21f508257a5ee15abda398a3db13385e33d6e6`,
  1,581,667,488 bytes. See `docs/HF_REFERENCES.md` and `artifacts.json`.
- No custom quantization/packing — the Q8_0 GGUF is a direct AllenAI release.

## 1. Stage 1 — identity + contract approval (maintainer)

Per the track plan, a maintainer first registers the identity on `main`. Ready-made
inputs are provided in this root:

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/olmo2_1b.json`.

The two hashes in the identity entry are precomputed to match this port exactly:

- `benchmark_config_sha256` = `canonical_sha256(effective olmo2 model config)`, where the
  effective config is `benchmarks/olmo2.json` with `"config"` set to its include path
  (this is how `.github/ci/scripts/model_port_claim.py::effective_model_config` derives
  it). Recompute:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/olmo2/benchmarks/olmo2.json"))
  cfg["config"] = "ported_models/olmo2/benchmarks/olmo2.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/olmo2_1b.json` after the
  maintainer commits the contract (byte-identical to
  `docs/proposed_reference_contract.json`).

This stage earns no credit; it only makes the identity `eligible`.

## 2. Stage 2 — implementation PR (this port)

Already prepared:

- New standalone root `ported_models/olmo2/` (this directory).
- Claim `ported_models/submissions/model_ports/olmo2.json`.
- One `olmo2` entry in `.github/ci/benchmark_config.json`:
  ```json
  "olmo2": { "config": "ported_models/olmo2/benchmarks/olmo2.json" }
  ```

## 3. Build (CI, reproducible by hand)

The board runner builds `llama-server` from the committed `llama.cpp-et` submodule and
runs OLMo-2 with full ET offload:

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

# download the pinned GGUF (url + sha256 in artifacts.json), then:
build-et/bin/llama-server --model OLMo-2-0425-1B-Instruct-Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 2048 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

## 4. Board correctness + metric

CI applies the main-owned `.github/ci/reference/olmo2_1b.json` contract:

- **Decode validation:** the fixed prompt (`benchmarks/olmo2.json`) must emit
  `>= 32` completion tokens at temperature 0, device `ET`, full offload (`gpu_layers=99`),
  and match host (CPU) execution of the same GGUF (`minimum_host_agreement = 1.0`).
- **No host fallback:** every graph op must run on ET (`require_zero_op_fallbacks`). This
  is the smoke's real gate — see `MODEL.md` for the op-coverage proof that OLMo-2's graph
  is a subset of the seed-proven ET kernel set.
- **Quality:** WikiText-2 raw perplexity within the contract bound, and ET-vs-CPU
  perplexity relative difference `<= 0.01`.
- **Metric:** decode `tokens_per_second` (higher better) recorded to the leaderboard.

## 5. Op-coverage rationale (why this passes first try)

OLMo-2's ops — `RMS_NORM` (incl. QK-RMSNorm), full `ROPE`-NEOX (n_dims=128), `MUL_MAT`
(Q8_0×F32), `GLU` (SwiGLU), `SOFT_MAX`, `GET_ROWS`, `ADD`, `MUL`, `CONT` — are exactly the
ops the seed `llama32_1b` / `qwen25_05b` / `smollm2` graphs already prove on ET-SoC1. No
LayerNorm, no unary activation, no partial rotary, no learned positional embeddings. See
`MODEL.md` for the per-op kernel mapping (`ggml-et` gate at `ggml-et.cpp` +
`et-kernels/src/`).
