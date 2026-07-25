# `gemma` (`gemma_2b`) — model-ports reproduce recipe

Ports **Gemma-2B-it (v1)** (Google, Gemma license) to ET-SoC1 via the shared
`llama.cpp-et` runtime. Gemma v1 is a new execution family (`LLM_ARCH_GEMMA`) already
implemented in the committed framework — no framework change is made here. This recipe
follows the two-stage `docs/SUBMISSION_GUIDE.md` flow.

## 0. Provenance

- Base weights: `google/gemma-2b-it`, Gemma license (gated on HF).
- Benchmark GGUF: `MaziyarPanahi/gemma-2b-it-GGUF` @
  `72164ae6fc4003cecf37bc07f3a825b8a20b8cbb`, `gemma-2b-it.Q8_0.gguf`,
  sha256 `dae8a0a75dda3553c06af737adfff0c003ed1b6393932964cce72bb4f0fd41f6`,
  2,669,070,080 bytes. Ungated third-party Q8_0 conversion. See `docs/HF_REFERENCES.md`
  and `artifacts.json`.
- No custom quantization/packing.

## 1. Stage 1 — identity + contract approval (maintainer)

A maintainer first registers the identity on `main`. Ready-made inputs are provided:

- `docs/proposed_identity_entry.json` → add to `data/model-port-identities.json`.
- `docs/proposed_reference_contract.json` → add as `.github/ci/reference/gemma_2b.json`.

The two hashes in the identity entry are precomputed to match this port exactly:

- `benchmark_config_sha256` = `canonical_sha256(effective gemma model config)`, where the
  effective config is `benchmarks/gemma.json` with `"config"` set to its include path.
  Recompute:
  ```
  python3 - <<'PY'
  import hashlib, json
  cfg = json.load(open("ported_models/gemma/benchmarks/gemma.json"))
  cfg["config"] = "ported_models/gemma/benchmarks/gemma.json"
  print(hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest())
  PY
  ```
- `validation_contract_sha256` = `sha256sum .github/ci/reference/gemma_2b.json` after the
  maintainer commits the contract (byte-identical to `docs/proposed_reference_contract.json`).

This stage earns no credit; it only makes the identity `eligible`.

## 2. Stage 2 — implementation PR (this port)

Already prepared:

- New standalone root `ported_models/gemma/` (this directory).
- Claim `ported_models/submissions/model_ports/gemma.json`.
- One `gemma_2b` entry in `.github/ci/benchmark_config.json`:
  ```json
  "gemma_2b": { "config": "ported_models/gemma/benchmarks/gemma.json" }
  ```

## 3. Build (CI, reproducible by hand)

```
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et -B build-et \
  -DGGML_ET=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release --target llama-server llama-perplexity

# download the pinned GGUF (url + sha256 in artifacts.json), then:
build-et/bin/llama-server --model gemma-2b-it.Q8_0.gguf \
  --device ET --gpu-layers 99 --ctx-size 2048 --batch-size 256 --ubatch-size 128 \
  --host 127.0.0.1 --port 18081
```

## 4. Board correctness + metric

CI applies the main-owned `.github/ci/reference/gemma_2b.json` contract: decode validation
(≥32 tokens, temp 0, device ET, full offload, host-CPU agreement 1.0), zero op fallbacks,
WikiText-2 perplexity within bound (ET-vs-CPU relative diff ≤ 0.01), metric decode
`tokens_per_second`.

## 5. Op-coverage rationale (confidence MEDIUM-HIGH)

Gemma v1 ops — `RMS_NORM`, `ROPE`-NEOX (n_dims=256, at the kernel limit), `MUL_MAT`
(Q8_0×F32), **`GLU` GeGLU**, **`SCALE`** (embed ×√hidden, query ×1/√head_dim), `SOFT_MAX`,
`GET_ROWS`, `ADD`, `MUL`, `CONT` — all have real ET kernels. GeGLU (`glu_f32.c`
`GGML_GLU_OP_GEGLU`/`block_geglu`) and SCALE are kernel-backed but not exercised by any seed
model, so unlike OLMo-2/InternLM2 there is a small non-zero fallback risk. Board-smoke this
port before relying on it. See `MODEL.md` for the per-op mapping.
