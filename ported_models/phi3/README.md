# Phi-3-mini-4k-Instruct — ET-SoC1 model port

Port of **Phi-3-mini-4k-Instruct** (Microsoft, MIT) to ET-SoC1 through the shared
`llama.cpp-et` runtime, for the "Most Models Ported by One Individual" track.

Phi-3 is a new `llama.cpp` execution family (`LLM_ARCH_PHI3`), already implemented in the
committed framework submodule (`src/models/phi3.cpp`) — distinct from the `phi2` family. This
port adds no framework code — it pins the Q8_0 GGUF, wires the `llama_server` benchmark, and
files the track claim.

Confidence is **MED-HIGH**: the graph (RMSNorm + RoPE + SwiGLU, with fused QKV/gate-up split
by `VIEW`/`CONT`) is a subset of the seed-proven ET op set — no unproven ops. At 4 GB it is
the largest port here; the board already runs 8B seed models, so size is not expected to
matter. See `MODEL.md` for the per-op mapping.

## Layout

```
ported_models/phi3/
├── MODEL.md                              model card, architecture, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/phi3.json                 llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/phi3.json` — the track claim.
- one `phi3_mini` entry in `.github/ci/benchmark_config.json`.
