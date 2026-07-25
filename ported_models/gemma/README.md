# Gemma-2B-it (v1) — ET-SoC1 model port

Port of **Gemma-2B-it** (Google, Gemma license) to ET-SoC1 through the shared
`llama.cpp-et` runtime, for the "Most Models Ported by One Individual" track.

Gemma v1 is a new `llama.cpp` execution family (`LLM_ARCH_GEMMA`), already implemented in
the committed framework submodule (`src/models/gemma.cpp`) — distinct from the seed
`gemma3n` family and from the `gemma2`/`gemma3` families other participants target. This
port adds no framework code — it pins the Q8_0 GGUF, wires the `llama_server` benchmark,
and files the track claim.

Confidence is **MEDIUM-HIGH**: every graph op has a real ET kernel, but Gemma's GeGLU FFN
and embedding/query `SCALE` are kernel-backed yet not exercised by any seed model, so it
should be board-smoked before being relied on. See `MODEL.md` for the per-op mapping. The
two RMSNorm-SwiGLU ports (OLMo-2, InternLM2) are strictly lower risk.

## Layout

```
ported_models/gemma/
├── MODEL.md                              model card, architecture, op-coverage analysis
├── README.md                            this file
├── artifacts.json                       GGUF + shared framework artifact wiring
├── benchmarks/gemma.json                llama_server board benchmark config
└── docs/
    ├── RECIPE.md                        end-to-end reproduce recipe
    ├── HF_REFERENCES.md                 pinned provenance
    ├── proposed_identity_entry.json     stage-1 registry entry (maintainer adds)
    └── proposed_reference_contract.json stage-1 correctness contract (maintainer adds)
```

Plus, outside this root:
- `ported_models/submissions/model_ports/gemma.json` — the track claim.
- one `gemma_2b` entry in `.github/ci/benchmark_config.json`.
