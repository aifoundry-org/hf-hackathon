# GGONNX Port

ONNX Runtime Execution Provider that lowers ONNX graph partitions onto GGML —
the bridge that lets Martin's ONNX model collection run on the ET-SoC1 board.

Upstream: <https://github.com/marty1885/ggonnx>.

## Source

Vendored under `src/ggonnx/` at upstream SHA `c0453fa516af3e2ae4140fafa798fccb6daff46c`
(committed 2026-05-02). See `THIRD_PARTY.md` for the licensing record at the time
of vendoring.

## Status

Vendored, **not yet wired to a benchmark runner.** ggonnx is the intended bridge for
running ONNX models on the ET-SoC1 board via the same GGML-ET backend the llama.cpp
baseline uses. It is not registered in `.github/ci/benchmark_config.json`, so PRs that
touch this port get a "new port detected without a benchmark entry" note from the
board CI (see `changed_benchmark_models.py`) until a runner is added.

To benchmark it on the board, a future change needs to:

1. build the ggonnx EP against the board's GGML-ET (the llama.cpp-et build now works
   on the board host — see that port's README for the toolchain story);
2. add a `ggonnx` runner + dispatch (analogous to the `llama_server` runner in
   `.github/ci/scripts/run_model_benchmark.sh`);
3. register the model(s) in `benchmark_config.json`.

Note that the board's ET backend currently has op-support gaps (e.g. recurrent
`CPY`), so per-ONNX-op coverage will need checking.

The inventory entries in `artifacts.json` list candidate ONNX models with their
intended validation strategy (`compare_outputs_to_onnxruntime_cpu`, `*_smoke_pending`,
etc.); artifact URLs and the intended validation category live there.

## Related public artifacts

Martin's public Hugging Face artifacts that line up with the GGONNX path:

- `marty1885/streaming-piper`: LJSpeech encoder, decoder, and config.
- `marty1885/zeroeggs-onnx`: small ONNX speech/motion models.
