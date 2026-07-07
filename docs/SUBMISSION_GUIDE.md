# Submission Guide

This guide is for the AIFoundry + OpenHW CORE-ET hackathon. Board CI currently
runs submissions on ET-SoC1 boards and comments back with reproducible results.

Submit through GitHub PRs against:

<https://github.com/aifoundry-org/hf-hackathon>

The Hugging Face repo is a read-only mirror. Do not submit hackathon PRs there.

Need help? Ask in Discord `#Lab`: <https://discord.gg/CbSA2umxf6>

## PR Checklist

1. Add or update a folder under `ported_models/`.
2. For an existing framework, update its `artifacts.json`, benchmark JSON, and
   `.github/ci/benchmark_config.json` entry as needed.
3. For a new framework, commit the framework source under
   `ported_models/<name>/src/`, add `artifacts.json`, add a runner under
   `.github/ci/scripts/`, and register it in the benchmark config/helper.
4. Use a Hugging Face model repo as the base reference whenever the model family
   exists there. Pin the repo, revision, filename(s), and license; document any
   export, quantization, preprocessing, shape, or packing step needed to
   reproduce the ET-SoC1 artifact.
5. Add a reusable `.md` recipe or equivalent agent-readable notes. Include the
   task breakdown, instructions or prompt files used, repos/docs/RTL/model files
   you pointed tools at, relevant commands, environment assumptions,
   verification steps, and failures or dead ends that would help another
   participant or agent reproduce the path. A `SKILLS.md`-style file is fine,
   but not required.
6. Do not edit `data/*.json` or the generated leaderboard block in `README.md`;
   board CI updates those after merge.
7. Do not commit model blobs, local outputs, secrets, or machine-specific paths.
8. Run:

   ```bash
   bash .github/ci/scripts/ci_preflight.sh
   ```

9. Open the GitHub PR.

PR board CI selects the configured models touched by the diff and comments with
the ET-SoC1 board result. External fork PRs reach the board runner only after
GitHub's external-contributor workflow approval gate releases the run.

The `Leaderboard gate` check is the merge signal for benchmarked submissions.
Every selected model must produce a passing board score and strictly improve the
current base-branch leaderboard value for that model's primary metric. Higher is
better for token/second metrics; lower is better for kernel wait time. A new
configured model with no prior leaderboard entry can pass by producing its first
valid board score. Set the optional repository variable
`LEADERBOARD_MIN_RELATIVE_IMPROVEMENT` to require a larger fractional gain, such
as `0.01` for 1%.
