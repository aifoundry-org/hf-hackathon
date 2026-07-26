---
license: apache-2.0
tags:
- et-soc1
- core-et
- openhw
- hackathon
- benchmarking
- model-porting
---

# CORE-ET Model Porting Hackathon

This is an AIFoundry + OpenHW hackathon for porting open AI models onto the
CORE-ET open hardware platform. The current board workflow runs submissions on
ET-SoC1 boards and reports reproducible benchmark results.

<!-- leaderboard:start -->
## ET-SoC1 Board Leaderboard

Results are from real ET-SoC1 silicon via the main-branch board workflow, hardware epoch `et-soc1-aifoundry3-600-400-tdp0-v1`. Each model uses its own primary metric.

| Model | Best participant | Variant | Metric | Score | PPL | Run |
|-------|------------------|---------|--------|-------|-----|-----|
| yolo | AFOliveira | `yolo_m30` | Mean end-to-end latency | 0.917323s | - | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| dncnn | karabambus | `dncnn20l64` | Denoise kernel wait (64x64 tile) | 0.221179s | - | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| lfm25 | Afonso Oliveira | `LFM2.5-1.2B-Instruct-Q8_0` | Decode tokens/s | 3.25 | 21.90 (+/- 4.44) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| llama32_1b | RehanQasim-dev | `Llama-3.2-1B-Instruct-Q8_0` | Decode tokens/s | 13.53 | 15.21 (+/- 2.87) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| gemma3n_e2b | RehanQasim-dev | `gemma-3n-E2B-it-Q8_0` | Decode tokens/s | 1.84 | 35.05 (+/- 11.06) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| tinyllama11b | RehanQasim-dev | `TinyLlama-1.1B-Chat-v1.0-Q8_0` | Decode tokens/s | 11.40 | 29.24 (+/- 7.71) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| rwkv7_15b | RehanQasim-dev | `rwkv7-1.5B-world-q8_0` | Decode tokens/s | 1.83 | 16.47 (+/- 3.01) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| qwen25_05b | RehanQasim-dev | `Qwen2.5-0.5B-Instruct-Q8_0` | Decode tokens/s | 11.46 | 18.23 (+/- 3.85) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| qwen3_8b | Ashish Soni | `Qwen3-8B-Q8_0` | Decode tokens/s | 3.36 | 10.49 (+/- 2.28) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smollm2_135m | RehanQasim-dev | `SmolLM2-135M-Instruct-Q8_0` | Decode tokens/s | 11.70 | 25.43 (+/- 5.40) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smollm2_360m | RehanQasim-dev | `SmolLM2-360M-Instruct-Q8_0` | Decode tokens/s | 10.17 | 18.31 (+/- 3.83) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smollm2_17b | RehanQasim-dev | `SmolLM2-1.7B-Instruct-Q8_0` | Decode tokens/s | 9.36 | 13.82 (+/- 3.01) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| deepseek_r1_15b | RehanQasim-dev | `DeepSeek-R1-Distill-Qwen-1.5B-Q8_0` | Decode tokens/s | 8.05 | 47.37 (+/- 12.09) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smolvlm_256m | noor-malaika | `SmolVLM-256M-Instruct-Q8_0` | Decode tokens/s | 11.70 | 31.21 (+/- 6.43) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smolvlm2_500m_video | RehanQasim-dev | `SmolVLM2-500M-Video-Instruct` | ET firmware cycles | 1805649301 | 22.28 (+/- 4.80) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |
| smolvlm_500m | Ashish-Soni08 | `SmolVLM-500M-Instruct` | ET firmware cycles | 608248522 | 25.37 (+/- 5.55) | [9910b2a](https://github.com/aifoundry-org/hf-hackathon/actions/runs/30111392825) |

Full JSON data lives in [`data/`](data/).
<!-- leaderboard:end -->

<!-- model-port-standings:start -->
## Most Models Ported by One Individual

| Rank | GitHub participant | Distinct ports | Credited models |
|------|--------------------|----------------|-----------------|
| 1 | [@Ashish-Soni08](https://github.com/Ashish-Soni08) | 1 | [smolvlm_500m](https://github.com/aifoundry-org/hf-hackathon/actions/runs/29910365911) |

<!-- model-port-standings:end -->

In this repo, you will find ready-to-deploy ported models, guides to deploy
them, and opinionated files on how to optimize them for CORE-ET and ET-SoC1
board runs.

## Submitting Results

Use the GitHub repository for pull requests and board-CI results:
<https://github.com/aifoundry-org/hf-hackathon>.

The Hugging Face repo under `AIFoundry-hackathon/hf-hackathon` is a read-only
mirror synced from GitHub `main`. It is for distribution and visibility only:
do not submit hackathon PRs on Hugging Face. If you want to submit results or a
new port, open a GitHub PR against the hackathon repository so the ET-SoC1 board
CI can run and comment on your submission.

Start here:

- [`docs/ET_SOC1_QUICKSTART.md`](docs/ET_SOC1_QUICKSTART.md): CORE-ET quickstart on ET-SoC1 boards; install the toolchain, download Hugging Face refs, build, and run.
- [`docs/HF_REFERENCES.md`](docs/HF_REFERENCES.md): pinned Hugging Face base models for the showcased workloads and new submissions.
- [`docs/BOARD_ACCESS.md`](docs/BOARD_ACCESS.md): join Discord and request Tailscale access to the board pool.
- [`docs/opinionated_porting_options/afonso.md`](docs/opinionated_porting_options/afonso.md): layer-by-layer porting, PMC measurement, and
  optimization workflow.
- [`docs/opinionated_porting_options/martin.md`](docs/opinionated_porting_options/martin.md): ET-SoC1 board mental model, correctness footguns, and
  performance playbook.
- `ported_models/yolo/`
- `ported_models/yolov10n_hf_reference/` (separate pinned-ONNX scalar FP32 correctness path)
- `ported_models/llama_cpp_et/`
- `ported_models/ggonnx/`

The `llama.cpp-et` leaderboard rows are board-only: CI resolves the
model/runtime artifacts declared in `ported_models/llama_cpp_et/artifacts.json`,
runs GGUFs through the ET-backed framework runners, and scores decode
tokens/second plus WikiText-2 raw PPL for transformer models. Extra supported
GGUF and TTS candidates can be run explicitly from the same manifest without
joining the default main-branch sweep.

## What To Point Your Agents At

For agent-assisted submissions, point your agent at the repo docs first, then at
the platform sources:

- [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md): required PR shape and checklist.
- [`docs/ET_SOC1_QUICKSTART.md`](docs/ET_SOC1_QUICKSTART.md): CORE-ET quickstart on ET-SoC1 boards.
- [`docs/HF_REFERENCES.md`](docs/HF_REFERENCES.md): pinned base model references.
- [`docs/opinionated_porting_options/afonso.md`](docs/opinionated_porting_options/afonso.md): model-porting workflow.
- [`docs/opinionated_porting_options/martin.md`](docs/opinionated_porting_options/martin.md): board and performance guidance.
- [`ported_models/`](ported_models/): existing ports and artifacts to copy from.
- [OpenHW CORE-ET RTL](https://github.com/openhwgroup/core-et): RTL and hardware source context.
- [AIFoundry ET platform](https://github.com/aifoundry-org/et-platform): platform support, runtime pieces, and toolchain setup helpers.
- [AIFoundry RISC-V GNU toolchain](https://github.com/aifoundry-org/riscv-gnu-toolchain): toolchain source used by the setup flow.

Submissions should include a reusable `.md` recipe or equivalent
agent-readable notes. Capture the task breakdown, markdown instructions or
prompt files used, repos/docs/RTL/model files you pointed tools at, commands
that worked, checks that failed, and the final verification path so another
participant or agent can reproduce the result.

## Submitting a port

Submit through GitHub PRs. The Hugging Face repo is a read-only mirror and does
not run board CI. Follow [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md);
ask questions in Discord `#Lab` if you need help.

New model submissions should start from pinned Hugging Face model repos whenever
the model family exists there. Record the repo, revision, filename, license, and
any export or packing step needed to reproduce the board artifact. Include a
reusable recipe or agent-readable notes with the PR.

## License

First-party code in this repository (CI scripts, configs, docs, porting harness)
is licensed under **Apache-2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Bundled or referenced third-party components (the ET `llama.cpp` fork, GGONNX,
downloaded model weights) keep their own upstream licenses and are **not**
covered by Apache-2.0. See [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) for the
full inventory and the per-port `THIRD_PARTY.md` files.
