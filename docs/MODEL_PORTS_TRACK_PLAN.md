# Most Models Ported by One Individual: Trusted CI Plan

Status: implemented in trusted shadow mode. The PR gate, ET-SoC1 execution,
credit ledger, and deterministic standings are implemented. Award writes remain
disabled until the organizer freezes the contest end time and approves the
historical identity/backfill review.

## Award definition

The score is the number of active, unique model-port identities credited to a
canonical GitHub login. The winner has the largest count. Ties are ordered by
the earliest merge time of the participant's final count-increasing credit.

Latency is still recorded in each trusted run and can appear on the normal
per-model leaderboard, but models are not compared with unrelated models for
this award.

One identity counts once globally. A renamed checkpoint, quantization, weight
update, tokenizer variant, or parameter-size variant that reuses the same
execution family does not create another credit. Deciding whether a new model
is a distinct port requires organizer review; CI enforces the reviewed decision
from the main-owned identity registry.

Credit belongs to the qualifying pull request author. Commit author strings,
the person who merges the PR, labels, and participant-controlled `team` fields
never decide ownership. Organizer and bot accounts listed in policy are
excluded.

## Frozen policy and baseline

`.github/ci/reference/model_ports_track.json` is the authoritative policy. It
contains:

- the contest interval and immutable seed commit;
- excluded GitHub logins and the tie-break rule;
- the required `soc1sim` ET-SoC1 device;
- the permitted main-owned runner types;
- the claim root and submission file/byte limits; and
- `activation_mode`, which is either `shadow` or `enforce`.

The seed commit predates the central benchmark config, so
`data/model-port-identities.json` also freezes the exact `ported_models/` roots
present at that commit. Preflight compares this inventory with Git rather than
assuming the current benchmark model list existed at the baseline.

Current release blockers are explicit:

- `contest_end` is not published in the repository;
- the historical contest submissions have not been mapped to reviewed trusted
  artifacts and canonical identities; and
- branch protection cannot require the new status until this workflow is on
  `main`.

For those reasons, `activation_mode` is `shadow`: CI produces real verdicts but
does not create award credits. Policy validation refuses `enforce` unless
`contest_end` is set and `historical_review_complete` is true.

## Two-stage identity approval

A participant first asks a maintainer to add an eligible identity to
`data/model-port-identities.json`. That main-owned entry pins:

- a stable identity and execution-family ID;
- all known model names and aliases;
- the upstream Hugging Face repository, 40-character revision, and license;
- the approved runner;
- the exact effective benchmark-config SHA-256; and
- an existing validation contract under `.github/ci/reference/`.

This approval PR cannot earn a credit. A new runner, scorer, oracle, fixture, or
workflow must also be reviewed and merged in this stage. A model implementation
cannot introduce the code that certifies its own correctness.

## Participant claim

The implementation PR adds a new standalone `ported_models/<model>` root and
one claim at `ported_models/submissions/model_ports/<model>.json`:

```json
{
  "schema_version": 1,
  "track": "most_models_ported",
  "benchmark_model": "new_model",
  "identity_id": "approved-identity-id",
  "source": {
    "repo": "upstream-owner/upstream-model",
    "revision": "40-character-hugging-face-commit",
    "license": "upstream-license-id"
  },
  "implementation_paths": ["ported_models/new_model"],
  "benchmark_config": ".github/ci/benchmark_config.json",
  "recipe": "ported_models/new_model/docs/RECIPE.md"
}
```

The implementation may only add regular files beneath its new root. The claim
has no owner or score fields. It cannot change workflows, scorers, contracts,
policy, registries, ledger data, existing benchmark entries, or global
benchmark settings. The benchmark entry must hash exactly to the preapproved
configuration. Claims are capped at 500 files and 50 MB of committed content.

## Pull-request gate

`.github/workflows/trusted-model-port-pr.yml` is loaded from `main` with
`pull_request_target` and always publishes
`trusted-track/model-port-credit` on the participant head SHA. Irrelevant PRs
receive no-op success, so the status can safely be required on every PR.

For a claim, the workflow:

1. Resolves the current PR head and canonical author through the GitHub API.
2. Validates identity, novelty, diff scope, source pin, config hash, contract,
   ledger state, and policy using only main-owned validation code.
3. Starts from the selected trusted `main` commit and copies only the claim and
   added regular files under the approved implementation root. It overlays only
   the claimed model entry into the benchmark config; unrelated PR files never
   enter the board workspace.
4. Runs the approved smoke configuration on real ET-SoC1 through `soc1sim`.
   The trusted job skips redundant board smoke, caps device execution at two
   minutes, and has a ten-minute hard job timeout for checkout/build overhead.
5. Requires a passing main-owned correctness contract and a score bound to the
   PR SHA, ref, GitHub Actions run URL, contract hash, device, and canonical
   author.
6. Uploads the verdict for 90 days and removes the participant workspace.

Participant code executes on a persistent self-hosted machine. Every external
participant head therefore remains pending until a maintainer explicitly
dispatches the workflow with `approve_board_execution=true`. Owners, members,
and collaborators may run automatically. A new push creates a new head and
requires fresh approval. No repository write credential or benchmark secret is
provided to participant code.

The smoke is an eligibility test, not a quality benchmark. Its validation
contract must prove the intended graph/output and reject stubs, host fallback,
partial execution, skipped cases, non-finite metrics, or the wrong device.

## Merge-time credit issuance

A passing PR status is not itself an award: the PR may never merge or may lose
a first-credit race. On a successful board run for a push to `main`, the
main-owned leaderboard job:

1. Resolves the associated merged PR, canonical author, participant head, and
   merge time through the GitHub API.
2. Requires the claim paths in the pushed diff to exactly match that PR's files.
3. Synchronizes to the latest `origin/main` before reading mutable ledger and
   leaderboard state while retaining the triggering commit as immutable input.
4. Revalidates the claim and trusted score provenance.
5. Appends at most one credit for each identity, regenerates standings and the
   README, and performs a normal non-force push.

The credit ID is SHA-256 of `track + NUL + identity_id`; repeated processing of
the same merge is byte-identical and reported as idempotent. If another update
advances `main`, the push fails closed instead of rebasing a stale ledger. A
rerun reads the winning credit and rejects a conflicting claimant.

## Append-only ledger and standings

`data/model-port-credits.json` is an event ledger. Every record is
content-addressed: `record_id` is the SHA-256 of the canonical JSON record body.
A credit preserves:

- identity and benchmark model;
- canonical participant login, PR, participant SHA, merge SHA, and merge time;
- pinned source, recipe, and benchmark-config hash; and
- trusted run URL, score SHA, contract hash, `soc1sim` device, metric name, and
  finite metric value.

Records are never edited or deleted. An organizer correction appends a
content-addressed `revocation` or `supersession` with a reason and an existing
target credit ID. Unknown targets, duplicate identities, forged hashes, invalid
provenance, and non-finite metrics fail validation.

`data/model-port-standings.json` and the README table are generated from active
ledger credits. They must not be hand-edited.

## Test and security standard

### ET-SoC1 verification

On 2026-07-14 the approved `elf` runner path was exercised on
`esperanto-soc3` with the same trusted timeout and provenance settings used by
the model-port workflow. The representative main-owned YOLO contract built and
ran five real `soc1sim` cases in 32 seconds end to end on a warm host. All five
cases passed with precision and recall of 1.0, mean kernel wait of 1.508664
seconds, and the expected validation-contract SHA-256. This seed identity does
not receive a model-port credit; it is only a hardware-path verification.

The run also established three host compatibility requirements now enforced by
the deployment path: Docker compiler wrappers must replace both `ET_PLATFORM`
and `ET_INSTALL`, the complete `ET_PLATFORM_SRC` must be selected before wrapper
mounts are created, and a provisioned launcher's matching library directory
must precede unrelated SDK libraries. Board scoring remains compatible with the
host's Python 3.8 runtime.

The same candidate was then exercised on the production `aifoundry2-et-soc1`
runner after recovering its wedged card through the lab iBoot controller. A
cold end-to-end run, including trusted launcher build and host-reference setup,
completed in 59 seconds. All five cases passed with precision and recall of 1.0
and mean kernel wait of 1.473358 seconds; repository preflight completed in 4.8
seconds. The production test also proved that ET stream-error log markers are
rejected even when a stale host launcher exits zero, and that the workflow uses
a cached, input-hashed launcher built from the main-owned source.

The focused unit suite covers:

- valid standalone claims and allowlisted synthetic-tree construction;
- seed/baseline, excluded-user, deadline, duplicate-credit, and size rejection;
- protected measurement files, unclaimed config entries, and mixed invalid
  claim-root payloads;
- author, SHA, ref, run URL, contract hash, device, and correctness spoofing;
- exact merged-PR claim attribution;
- idempotent issuance and first-writer-wins race handling;
- content-addressed ledger tamper detection and append-only revocation; and
- deterministic count/tie ordering and shadow-mode no-write behavior.

Repository preflight additionally parses both workflows, validates JSON and
shell syntax, checks baseline-root inventory, regenerates standings in memory,
and runs the full existing CI test suite.

## Activation and operations plan

Before changing `activation_mode` to `enforce`:

1. Publish `contest_end` and confirm the start/baseline boundary.
2. Review historical qualifying PRs. For each, record the canonical PR author,
   distinct identity, source pin, and a passing trusted ET-SoC1 run. Publish the
   proposed backfill for disputes.
3. Decide whether collaboration is excluded or handled in a separate team
   prize; this implementation credits exactly one PR author.
4. Merge the workflow and require `trusted-track/model-port-credit` in branch
   protection, alongside the normal leaderboard gate.
5. Run shadow mode over representative real PRs and confirm organizer decisions
   match CI.
6. Append reviewed historical credits, regenerate standings, then switch to
   `enforce` in a main-owned policy PR.
7. At the deadline, reject later merge timestamps, resolve recorded disputes,
   and archive the final ledger plus trusted run links.

Until steps 1–5 are complete, the label `track: model-ports` is classification
only and the README must say that no award credits have been issued.
