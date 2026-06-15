---
license: apache-2.0
tags:
- et-soc1
- hackathon
- benchmarking
- model-porting
---

# ET-SoC1 Model Porting

<!-- leaderboard:start -->
## ET-SoC1 Board Leaderboard

Results are from real ET-SoC1 silicon via the main-branch board workflow. Each model uses its own primary metric.

| Model | Best participant | Variant | Metric | Score | PPL | Run |
|-------|------------------|---------|--------|-------|-----|-----|
| dncnn | AFOliveira | `v3x_01_oc2_base` | Kernel wait | 0.040075s | - | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| yolo | AFOliveira | `y10_00_base` | Kernel wait | 0.134534s | - | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| whisper | AFOliveira | `w10_00_base` | Kernel wait | 0.041361s | - | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| lfm25 | AFOliveira | `LFM2.5-1.2B-Instruct-Q8_0` | Decode tokens/s | 3.10 | 21.70 (+/- 4.45) | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| llama32_1b | AFOliveira | `Llama-3.2-1B-Instruct-Q8_0` | Decode tokens/s | 13.54 | 15.31 (+/- 2.89) | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| gemma3n_e2b | AFOliveira | `gemma-3n-E2B-it-Q8_0` | Decode tokens/s | 1.49 | 32.83 (+/- 10.20) | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| tinyllama11b | AFOliveira | `TinyLlama-1.1B-Chat-v1.0-Q8_0` | Decode tokens/s | 11.33 | 29.23 (+/- 7.70) | [5cdcfc8](https://github.com/aifoundry-org/hf-hackathon/actions/runs/27550241277) |
| rwkv7_15b | AFOliveira | `rwkv7-1.5B-world-q8_0` | Decode tokens/s | 3.65 | 16.46 (+/- 3.01) | [1bb9a0c](https://github.com/nekkoai/hf-hackathon/actions/runs/26504752353) |

Full JSON data lives in [`data/`](data/).
<!-- leaderboard:end -->

This is an initiation to model porting on the ET-SoC1 board.

In this repo, you will find ready-to-deploy ported models, guides to deploy
them, and opinionated files on how to optimize them by the nekkoai team.

## Submitting Results

Use the GitHub repository for pull requests and board-CI results:
<https://github.com/aifoundry-org/hf-hackathon>.

The Hugging Face repo under `AIFoundry-hackathon/hf-hackathon` is a read-only
mirror synced from GitHub `main`. It is for distribution and visibility only:
do not submit hackathon PRs on Hugging Face. If you want to submit results or a
new port, open a GitHub PR against the hackathon repository so the ET-SoC1 board
CI can run and comment on your submission.

Start here:

- [`docs/ET_SOC1_QUICKSTART.md`](docs/ET_SOC1_QUICKSTART.md): install the toolchain, download Hugging Face refs, build, and run.
- [`docs/HF_REFERENCES.md`](docs/HF_REFERENCES.md): pinned Hugging Face base models for the showcased workloads and new submissions.
- [`docs/BOARD_ACCESS.md`](docs/BOARD_ACCESS.md): join Discord and request Tailscale access to the board pool.
- [`docs/opinionated_porting_options/afonso.md`](docs/opinionated_porting_options/afonso.md): layer-by-layer porting, PMC measurement, and
  optimization workflow.
- [`docs/opinionated_porting_options/martin.md`](docs/opinionated_porting_options/martin.md): ET-SoC1 mental model, correctness footguns, and
  performance playbook.
- `ported_models/dncnn/`
- `ported_models/yolo/`
- `ported_models/whisper/`
- `ported_models/llama_cpp_et/`
- `ported_models/ggonnx/`

The `llama.cpp-et` leaderboard rows are board-only: CI resolves the
model/runtime artifacts declared in `ported_models/llama_cpp_et/artifacts.json`,
runs GGUFs through the ET-backed framework runners, and scores decode
tokens/second plus WikiText-2 raw PPL for transformer models. Extra supported
GGUF and TTS candidates can be run explicitly from the same manifest without
joining the default main-branch sweep.

## Submitting a port

Submit through GitHub PRs. The Hugging Face repo is a read-only mirror and does
not run board CI. Follow [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md);
ask questions in Discord `#Lab` if you need help.

New model submissions should start from pinned Hugging Face model repos whenever
the model family exists there. Record the repo, revision, filename, license, and
any export or packing step needed to reproduce the board artifact.

## License

First-party code in this repository (CI scripts, configs, docs, porting harness)
is licensed under **Apache-2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Bundled or referenced third-party components (the ET `llama.cpp` fork, GGONNX,
downloaded model weights) keep their own upstream licenses and are **not**
covered by Apache-2.0. See [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) for the
full inventory and the per-port `THIRD_PARTY.md` files.
