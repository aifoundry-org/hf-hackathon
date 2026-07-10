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

Results are from real ET-SoC1 silicon via the main-branch board workflow. Each model uses its own primary metric.

| Model | Best participant | Variant | Metric | Score | PPL | Run |
|-------|------------------|---------|--------|-------|-----|-----|
| yolo | DarthCeltic | `yolo_m30` | Mean end-to-end latency | 3.847858s | - | [2fd2537](https://github.com/aifoundry-org/hf-hackathon/actions/runs/29094066659) |
| lfm25 | AFOliveira | `LFM2.5-1.2B-Instruct-Q8_0` | Decode tokens/s | 3.12 | 21.70 (+/- 4.45) | [7c5d830](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28858491662) |
| llama32_1b | AFOliveira | `Llama-3.2-1B-Instruct-Q8_0` | Decode tokens/s | 13.58 | 15.31 (+/- 2.89) | [7c5d830](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28858491662) |
| gemma3n_e2b | AFOliveira | `gemma-3n-E2B-it-Q8_0` | Decode tokens/s | 1.50 | 32.83 (+/- 10.20) | [7c5d830](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28858491662) |
| tinyllama11b | AFOliveira | `TinyLlama-1.1B-Chat-v1.0-Q8_0` | Decode tokens/s | 11.29 | 29.23 (+/- 7.70) | [7c5d830](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28858491662) |
| rwkv7_15b | AFOliveira | `rwkv7-1.5B-world-q8_0` | Decode tokens/s | 3.65 | 16.46 (+/- 3.01) | [1bb9a0c](https://github.com/nekkoai/hf-hackathon/actions/runs/26504752353) |
| qwen25_05b | Razvan-Mihai Hanghicel | `Qwen2.5-0.5B-Instruct-Q8_0` | Decode tokens/s | 11.37 | 16.97 (+/- 3.53) | [d3f9502](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28964644188) |
| qwen3_8b | Ashish Soni | `Qwen3-8B-Q8_0` | Decode tokens/s | 3.30 | 10.50 (+/- 2.29) | [af95906](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28949135516) |
| deepseek_r1_15b | alien1403 | `DeepSeek-R1-Distill-Qwen-1.5B-Q8_0` | Decode tokens/s | 7.91 | 47.36 (+/- 12.09) | [7c5d830](https://github.com/aifoundry-org/hf-hackathon/actions/runs/28858491662) |

Full JSON data lives in [`data/`](data/).
<!-- leaderboard:end -->

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
