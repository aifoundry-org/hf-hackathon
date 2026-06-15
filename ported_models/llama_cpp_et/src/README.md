# Source

`llama.cpp-et` is a pinned submodule for the public ET-enabled `llama.cpp`
fork used by these GGUF model ports. **This is the source-of-truth for what CI
builds and runs.** No URL-based clone fallback exists; the benchmark runner
errors out if the submodule is not initialized.

Initialize it with:

```sh
git submodule update --init --recursive ported_models/llama_cpp_et/src/llama.cpp-et
```

CI runs `actions/checkout@v4` with `submodules: recursive`, so the submodule is
present on the board runner without any operator action.

## Changing the framework version

Bump the submodule pin in a PR. The SHA delta appears in the diff and the
reviewer sees exactly what changed before any benchmark runs:

```sh
cd ported_models/llama_cpp_et/src/llama.cpp-et
git fetch origin <branch>
git checkout <new-sha>
cd ../../../..
git add ported_models/llama_cpp_et/src/llama.cpp-et
git commit -m "bump llama.cpp-et to <short-sha>"
```

If you also want to update `ported_models/llama_cpp_et/artifacts.json` to
record the new upstream branch/revision metadata, do so in the same PR — but
the submodule SHA itself is authoritative for the build.

## License

The pinned source is MIT licensed. Keep the upstream `LICENSE` file and
copyright notice intact if this source is copied or redistributed outside the
submodule. See `../THIRD_PARTY.md`.
