# EXAONE-3.5-2.4B-Instruct — ET-SoC1 model port

Port of **EXAONE-3.5-2.4B-Instruct** (LG AI Research) to ET-SoC1 through the shared
`llama.cpp-et` runtime, for the "Most Models Ported by One Individual" track.

EXAONE is a new `llama.cpp` execution family (`LLM_ARCH_EXAONE`), already implemented in the
committed framework submodule (`src/models/exaone.cpp`). This port adds no framework code —
it pins the official Q8_0 GGUF, wires the `llama_server` benchmark, and files the track
claim.

Board-pass confidence is **HIGH** — its op graph is a strict subset of the seed-proven ET
kernel set (RMSNorm + RoPE-NEOX + SwiGLU + Q8_0 MatMul), with zero unproven ops, on par with
the OLMo-2 and InternLM2 ports.

**License caveat:** EXAONE uses the EXAONE AI Model License (non-commercial), not a
permissive OSI license — see `MODEL.md`. Flagged for maintainer review; the Apache-2.0 ports
(olmo2, internlm2, granite) are the license-clean options.

## Layout

```
ported_models/exaone/
├── MODEL.md                              model card, license note, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/exaone.json               llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance + license detail
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/exaone.json` — the track claim.
- one `exaone_2_4b` entry in `.github/ci/benchmark_config.json`.
