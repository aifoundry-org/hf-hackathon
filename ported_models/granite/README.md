# Granite-3.0-2B-Instruct — ET-SoC1 model port

Port of **Granite-3.0-2B-Instruct** (IBM, Apache-2.0) to ET-SoC1 through the shared
`llama.cpp-et` runtime, for the "Most Models Ported by One Individual" track.

Granite 3.0 is a new `llama.cpp` execution family (`LLM_ARCH_GRANITE`), already implemented
in the committed framework submodule (`src/models/granite.cpp`). This port adds no framework
code — it pins the Q8_0 GGUF, wires the `llama_server` benchmark, and files the track claim.

Confidence is **MEDIUM-HIGH**: the FFN is SwiGLU (seed-proven), and the only op not already
exercised by a seed model is the trivial elementwise `SCALE` (Granite's four scalar
multipliers), which has a real ET kernel. See `MODEL.md` for the per-op mapping.

## Layout

```
ported_models/granite/
├── MODEL.md                              model card, architecture, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/granite.json              llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/granite.json` — the track claim.
- one `granite_2b` entry in `.github/ci/benchmark_config.json`.
