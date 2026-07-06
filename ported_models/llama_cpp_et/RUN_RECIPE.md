# Running llama.cpp models on the ET backend

This fork runs GGUF models through the ET backend of `llama.cpp` and supports
`Q8_0`, `Q4_0`, and `Q4_K` quantized models.

## Build

Configure and build with the ET backend enabled:

```sh
cmake -S ported_models/llama_cpp_et/src/llama.cpp-et \
      -B build-et \
      -DGGML_ET=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD_LIBRARIES=-lglog
cmake --build build-et --config Release -j
```

Binaries land in `build-et/bin/` (`llama-cli`, `llama-server`, `llama-bench`,
`llama-perplexity`).

## Run a model

Select the ET device with `--device ET` and offload all layers with `-ngl 99`.
The same commands work for `Q8_0`, `Q4_0`, and `Q4_K` GGUF files — only the
model path changes.

### Single prompt (`llama-cli`)

```sh
build-et/bin/llama-cli \
  -m <model>.gguf \
  --device ET -ngl 99 \
  -p "Explain what a matrix engine is in two sentences." \
  -n 128
```

### Server (`llama-server`)

```sh
build-et/bin/llama-server \
  -m <model>.gguf \
  --device ET -ngl 99 \
  --ctx-size 2048 \
  --host 127.0.0.1 --port 18081
```

Send a completion request:

```sh
curl -s http://127.0.0.1:18081/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "n_predict": 64, "temperature": 0}'
```

### Throughput (`llama-bench`)

```sh
build-et/bin/llama-bench \
  -m <model>.gguf \
  --device ET -ngl 99 \
  -p 512 -n 128
```

`-p 512` reports prefill tokens/s (pp512); `-n 128` reports decode tokens/s
(tg128).

### Perplexity (`llama-perplexity`)

```sh
build-et/bin/llama-perplexity \
  -m <model>.gguf \
  --device ET -ngl 99 \
  -f wiki.test.raw \
  --ctx-size 128
```

## Quantization notes

- `Q8_0` and `Q4_0` run on the ET backend directly.
- `Q4_K` is supported, with a matrix-engine (TensorFMA32) `mul_mat` path that
  accelerates prompt processing (prefill) for larger prompts; token generation
  uses the scalar path.

## Questions

Discussion thread:
<https://discord.com/channels/1246258403441905817/1428532748703760394/1512198362336592037>
