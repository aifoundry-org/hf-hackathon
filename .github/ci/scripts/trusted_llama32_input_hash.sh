#!/usr/bin/env bash
set -euo pipefail

ref="${1:-HEAD}"
paths=(
  .github/ci/reference/llama32_1b.json
  .github/ci/reference/llama32_1b_track.json
  .github/ci/reference/rwkv7_15b.json
  .github/workflows/trusted-llama32-pr.yml
  .github/ci/scripts/prepare_trusted_llama32_candidate.py
  .github/ci/scripts/run_llama_server_benchmark.py
  .github/ci/scripts/run_trusted_llama_benchmark.py
  .github/ci/scripts/trusted_llama32_gate.py
  .github/ci/scripts/run_trusted_llama32_candidate.sh
  .github/ci/benchmark_config.json
  ported_models/llama_cpp_et/benchmarks/llama32_1b.json
  ported_models/llama_cpp_et/benchmarks/lfm25.json
  ported_models/llama_cpp_et/benchmarks/gemma3n_e2b.json
  ported_models/llama_cpp_et/benchmarks/tinyllama11b.json
  ported_models/llama_cpp_et/benchmarks/rwkv7_15b.json
  ported_models/llama_cpp_et/benchmarks/qwen25_05b.json
  ported_models/llama_cpp_et/benchmarks/qwen3_8b.json
  ported_models/llama_cpp_et/benchmarks/deepseek_r1_15b.json
  ported_models/llama_cpp_et/submissions/llama32_1b.json
  ported_models/llama_cpp_et/submissions/llama32_1b.track.json
  ported_models/llama_cpp_et/artifacts.json
  ported_models/llama_cpp_et/src/llama.cpp-et
  data/llama32_1b.json
  data/lfm25.json
  data/gemma3n_e2b.json
  data/tinyllama11b.json
  data/rwkv7_15b.json
  data/qwen25_05b.json
  data/qwen3_8b.json
  data/deepseek_r1_15b.json
)

for path in "${paths[@]}"; do
  printf '%s\0' "$path"
  git show "$ref:$path" 2>/dev/null || true
  printf '\0'
done | sha256sum | awk '{print $1}'
