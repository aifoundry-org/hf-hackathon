# ET llama.cpp Model Ports

This folder owns GGUF model ports that run through the public ET-backed
`llama.cpp` fork instead of through handwritten ELF kernels.

The framework source is committed under `src/llama.cpp-et` as a pinned git
submodule, mirrored from:

- `https://github.com/AFOliveira/llama.cpp.git`
- branch `agent/fail-fast-et-runtime`
- revision `27e3f108b759a6e4513ff73cb0f122ac50457b80`
- upstream review: `https://github.com/aifoundry-org/llama.cpp/pull/18`

**CI builds llama.cpp from the committed submodule.** A reviewer reading a PR
can see exactly which framework version will run on the board. To change it,
bump the submodule pin in the same PR — the SHA shift shows up in the diff.

Initialize the submodule locally with:

```sh
git submodule update --init --recursive ported_models/llama_cpp_et/src/llama.cpp-et
```

CI runs `actions/checkout@v4` with `submodules: recursive`, so the submodule is
present on the runner without any extra step.

Each benchmark under `benchmarks/` selects one GGUF artifact, runs
`llama-server` on the `ET` device, records decode tokens/second, and runs
`llama-perplexity` on WikiText-2 raw.

The pinned backend propagates every failed operation, records asynchronous
runtime callbacks, and drains the ET stream after at most 4,096 queued kernels.
That bound prevents the 16-bit runtime event namespace from wrapping while an
older event is still live. The corresponding allocator fix is under upstream
review in `aifoundry-org/et-platform#134`.

The default main-branch board run only executes the canonical smoke set. Extra
ET-supported candidates can be run explicitly, for example:

```sh
MODELS="qwen3_06b smollm2_135m llama32_3b" \
  .github/ci/platform/deploy/soc3-benchmark.sh
```

Candidate artifacts currently include non-MoE Qwen3 q8_0, SmolLM2 q8_0,
Llama 3.2 3B q8_0, and Llama 3.1 8B q8_0. They are excluded from the default
CI sweep until they are promoted to the public leaderboard set.

Large model files, framework checkouts, builds, and datasets are cached under
`local-artifacts/` on the board host and are not committed.

See `THIRD_PARTY.md` for source licensing notes. The ET `llama.cpp` fork is MIT
licensed, so it can be redistributed if the upstream copyright and permission
notice are preserved. GGONNX is vendored as a separate port under
`ported_models/ggonnx/`; see `ported_models/ggonnx/THIRD_PARTY.md` for its
licensing record.
