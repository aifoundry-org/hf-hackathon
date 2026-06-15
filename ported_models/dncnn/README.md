# DnCNN Reference Model

Hugging Face refs are downloaded on demand with
`scripts/download_hf_refs.sh`; do not commit downloaded model packages or
generated metadata JSON. The current base is `deepinv/dncnn` pinned in
[`docs/HF_REFERENCES.md`](../../docs/HF_REFERENCES.md).

Current checked-in source focuses on the ET-SoC1 DnCNN kernels and PMC sweep
variants. Runtime inputs, packed weights, and generated ELFs are not committed.
Download external references with `scripts/download_hf_refs.sh` and build
ET-SoC1 ELFs locally from source.

Start with `docs/optimizations.md`, then use the manifests in `manifests/` for
smoke and 100-variant sweeps.
