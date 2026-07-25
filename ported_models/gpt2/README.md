# GPT-2 (124M) — ET-SoC1 model port

Port of **GPT-2 124M** (OpenAI, MIT) to ET-SoC1 through the shared `llama.cpp-et` runtime,
for the "Most Models Ported by One Individual" track.

GPT-2 is a new `llama.cpp` execution family (`LLM_ARCH_GPT2`), already implemented in the
committed framework submodule (`src/models/gpt2.cpp`). This port adds no framework code — it
pins the Q8_0 GGUF, wires the `llama_server` benchmark, and files the track claim.

Confidence is **MEDIUM-HIGH**: no rotary at all (positions are a learned `GET_ROWS`+`ADD`),
and the only non-seed-exercised ops are LayerNorm (`NORM`) and GELU (`UNARY`), both with real
ET kernels. At ~178 MB it is by far the smallest port here, so the board run is near-instant.
See `MODEL.md` for the per-op mapping.

## Layout

```
ported_models/gpt2/
├── MODEL.md                              model card, architecture, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/gpt2.json                 llama_server board benchmark config (ctx_size 1024)
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/gpt2.json` — the track claim.
- one `gpt2` entry in `.github/ci/benchmark_config.json`.
