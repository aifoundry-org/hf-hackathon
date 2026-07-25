# Phi-1.5 — ET-SoC1 model port

Port of **Phi-1.5** (Microsoft, MIT) to ET-SoC1 through the shared `llama.cpp-et` runtime,
for the "Most Models Ported by One Individual" track.

Phi-1.5 is a new `llama.cpp` execution family (`LLM_ARCH_PHI2`), already implemented in the
committed framework submodule (`src/models/phi2.cpp`). This port adds no framework code — it
pins the Q8_0 GGUF, wires the `llama_server` benchmark, and files the track claim.

Confidence is **MEDIUM** — the most of any port in this set. All ops have real ET kernels, but
three are not exercised by any seed model: partial rotary (`ROPE` with n_rot=32), LayerNorm
(`NORM`), and GELU (`UNARY`). Each has a clean F32 kernel and passes `supports_op`, so it
should be board-smoked before it is relied on. See `MODEL.md` for the per-op mapping. The
Apache/MIT RMSNorm ports (olmo2, internlm2, granite, exaone) are the higher-confidence picks.

## Layout

```
ported_models/phi2/
├── MODEL.md                              model card, architecture, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/phi2.json                 llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/phi2.json` — the track claim.
- one `phi1_5` entry in `.github/ci/benchmark_config.json`.
