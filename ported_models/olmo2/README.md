# OLMo-2-0425-1B — ET-SoC1 model port

Port of **OLMo-2-0425-1B-Instruct** (AllenAI, Apache-2.0) to ET-SoC1 through the shared
`llama.cpp-et` runtime, for the "Most Models Ported by One Individual" track.

OLMo 2 is a new `llama.cpp` execution family (`LLM_ARCH_OLMO2`), already implemented in
the committed framework submodule (`src/models/olmo2.cpp`). This port adds no framework
code — it pins the Q8_0 GGUF, wires the `llama_server` benchmark, and files the track
claim. Its op graph is a strict subset of the ops the seed Llama/Qwen models already run
on the board, so the correctness smoke has no host-fallback risk.

## Layout

```
ported_models/olmo2/
├── MODEL.md                              model card, architecture, op-coverage proof
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/olmo2.json                llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/olmo2.json` — the track claim.
- one `olmo2` entry in `.github/ci/benchmark_config.json`.

## Reproduce

See `docs/RECIPE.md`. In short: the board CI downloads the pinned GGUF by url + sha256,
builds `llama-server` from the committed `llama.cpp-et` submodule with `-DGGML_ET=ON`,
runs full-offload decode on ET-SoC1 (`gpu_layers=99`, device `ET`), and validates decode
output + WikiText-2 perplexity against the main-owned contract.
