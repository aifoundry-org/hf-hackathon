#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_config_helpers import sysemu_timeouts
from model_port_claim import ClaimError, active_credits, canonical_sha256, inspect_claims
from model_port_credit import issue_credits, record_hash
from prepare_trusted_model_port_tree import prepare
from render_model_port_standings import standings


ACTOR = "participant"
SOURCE = {
    "repo": "owner/novel-model",
    "revision": "a" * 40,
    "license": "apache-2.0",
}
CONTRACT_PATH = ".github/ci/reference/novel.json"
CLAIM_PATH = "ported_models/submissions/model_ports/novel.json"


def git_env() -> dict[str, str]:
    """Remove hook-provided repository bindings before using fixture repos."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, env=git_env(), text=True
    ).strip()


def write_json(repo: Path, path: str, value: object) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")


def commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, env=git_env(), check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        env=git_env(),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return git(repo, "rev-parse", "HEAD")


def benchmark_entry() -> dict:
    return {
        "runner": "elf",
        "board": True,
        "reference_contract": CONTRACT_PATH,
        "canonical_variant": "novel-int8",
        "source": "ported_models/novel/src/novel.c",
        "bench_dir": "novel-bench",
        "manifest": "novel_variants.txt",
        "score": {
            "metric": "kernel_wait_s",
            "label": "End-to-end latency",
            "higher_is_better": False,
        },
        "build": {"opt": "-O3", "defines": []},
        "mem_size": "0x10000",
        "region_size": "0x10000",
        "file_loads": [],
        "dump_size": "0x2000",
        "dump_magic": "0x12345678",
    }


class TrackRepo:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, env=git_env(), check=True)
        subprocess.run(
            ["git", "config", "user.email", "ci@example.com"],
            cwd=self.repo,
            env=git_env(),
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "CI"],
            cwd=self.repo,
            env=git_env(),
            check=True,
        )
        write_json(
            self.repo,
            ".github/ci/benchmark_config.json",
            {"primary_metric": "kernel_wait_s", "lower_is_better": True, "models": {}},
        )
        (self.repo / "README.md").write_text(
            "<!-- model-port-standings:start -->\nold\n<!-- model-port-standings:end -->\n"
        )
        self.baseline = commit(self.repo, "baseline")

        self.entry = benchmark_entry()
        self.policy = {
            "schema_version": 1,
            "track": "most_models_ported",
            "activation_mode": "enforce",
            "historical_review_complete": True,
            "contest_start": "2026-06-01T00:00:00Z",
            "contest_end": "2026-12-31T23:59:59Z",
            "baseline_sha": self.baseline,
            "excluded_logins": ["organizer"],
            "tie_break": "earliest_final_qualifying_merge",
            "credit_owner": "pull_request_author",
            "required_device": "soc1sim",
            "claim_root": "ported_models/submissions/model_ports",
            "allowed_runners": ["elf"],
            "max_added_files": 500,
            "max_added_bytes": 50_000_000,
            "require_new_standalone_root": True,
            "require_main_owned_validation_contract": True,
        }
        self.registry = {
            "schema_version": 1,
            "track": "most_models_ported",
            "baseline_port_roots": ["ported_models/seed"],
            "identities": [
                {
                    "identity_id": "novel-family",
                    "execution_family": "novel",
                    "benchmark_models": ["novel"],
                    "aliases": ["novel-q8"],
                    "eligible": True,
                    "canonical_source": SOURCE,
                    "approved_runner": "elf",
                    "benchmark_config_sha256": canonical_sha256(self.entry),
                    "validation_contract": CONTRACT_PATH,
                }
            ],
        }
        self.ledger = {
            "schema_version": 1,
            "track": "most_models_ported",
            "records": [],
        }
        write_json(self.repo, ".github/ci/reference/model_ports_track.json", self.policy)
        write_json(self.repo, "data/model-port-identities.json", self.registry)
        write_json(self.repo, "data/model-port-credits.json", self.ledger)
        write_json(
            self.repo,
            CONTRACT_PATH,
            {"schema_version": 1, "model": "novel", "expected_magic": "0x12345678"},
        )
        self.base = commit(self.repo, "preapprove identity and contract")

        config = json.loads((self.repo / ".github/ci/benchmark_config.json").read_text())
        config["models"]["novel"] = self.entry
        write_json(self.repo, ".github/ci/benchmark_config.json", config)
        (self.repo / "ported_models/novel/src").mkdir(parents=True)
        (self.repo / "ported_models/novel/src/novel.c").write_text("int main(void) { return 0; }\n")
        (self.repo / "ported_models/novel/docs").mkdir(parents=True)
        (self.repo / "ported_models/novel/docs/RECIPE.md").write_text("# Reproducible recipe\n")
        write_json(
            self.repo,
            CLAIM_PATH,
            {
                "schema_version": 1,
                "track": "most_models_ported",
                "benchmark_model": "novel",
                "identity_id": "novel-family",
                "source": SOURCE,
                "implementation_paths": ["ported_models/novel"],
                "benchmark_config": ".github/ci/benchmark_config.json",
                "recipe": "ported_models/novel/docs/RECIPE.md",
            },
        )
        (self.repo / "unrelated.txt").write_text("must not enter trusted candidate\n")
        self.head = commit(self.repo, "candidate")

    def close(self):
        self.temp.cleanup()

    def inspect(self, **overrides):
        values = {
            "repo": self.repo,
            "base": self.base,
            "head": self.head,
            "actor": ACTOR,
            "pr_number": 123,
            "policy": self.policy,
            "registry": self.registry,
            "ledger": self.ledger,
        }
        values.update(overrides)
        return inspect_claims(**values)


class ModelPortTrackTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TrackRepo()

    def tearDown(self):
        self.fixture.close()

    def test_valid_new_standalone_port_is_eligible(self):
        result = self.fixture.inspect()
        self.assertTrue(result["targeted"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["models"], ["novel"])

    def test_enforcement_cannot_activate_before_historical_review(self):
        policy = copy.deepcopy(self.fixture.policy)
        policy["historical_review_complete"] = False
        with self.assertRaises(ClaimError):
            self.fixture.inspect(policy=policy)

    def test_trusted_board_timeout_caps_override_long_global_defaults(self):
        cfg = {"board": {"launcher_timeout_s": 600, "outer_timeout_s": 660}}
        with patch.dict(
            os.environ,
            {
                "BOARD_BENCHMARK": "1",
                "TRUSTED_BOARD_LAUNCHER_TIMEOUT_CAP": "100",
                "TRUSTED_BOARD_OUTER_TIMEOUT_CAP": "120",
            },
            clear=False,
        ):
            self.assertEqual(sysemu_timeouts(cfg), (120, 100))

    def test_exact_config_hash_is_main_owned(self):
        registry = copy.deepcopy(self.fixture.registry)
        registry["identities"][0]["benchmark_config_sha256"] = "0" * 64
        result = self.fixture.inspect(registry=registry)
        self.assertFalse(result["passed"])
        self.assertIn("organizer-approved configuration hash", " ".join(result["errors"]))

    def test_seed_model_at_frozen_baseline_is_ineligible(self):
        policy = copy.deepcopy(self.fixture.policy)
        policy["baseline_sha"] = self.fixture.head
        result = self.fixture.inspect(policy=policy)
        self.assertFalse(result["passed"])
        self.assertIn("contest baseline", " ".join(result["errors"]))

    def test_excluded_login_and_out_of_window_merge_are_rejected(self):
        excluded = self.fixture.inspect(actor="organizer")
        self.assertFalse(excluded["passed"])
        self.assertIn("excluded", " ".join(excluded["errors"]))

        policy = copy.deepcopy(self.fixture.policy)
        policy["contest_end"] = "2026-06-30T23:59:59Z"
        late = self.fixture.inspect(
            policy=policy,
            merged_at="2026-07-01T00:00:00Z",
        )
        self.assertFalse(late["passed"])
        self.assertIn("deadline", " ".join(late["errors"]))

    def test_existing_execution_identity_cannot_receive_another_credit(self):
        score = self.score()
        ledger, report = self.issue(score)
        self.assertTrue(report["passed"])
        result = self.fixture.inspect(ledger=ledger)
        self.assertFalse(result["passed"])
        self.assertIn("already has an active port credit", " ".join(result["errors"]))

    def test_claim_cannot_change_trusted_measurement_code(self):
        (self.fixture.repo / ".github/ci/scripts").mkdir(parents=True)
        (self.fixture.repo / ".github/ci/scripts/evil.py").write_text("print('weaken gate')\n")
        bad_head = commit(self.fixture.repo, "try to replace gate")
        result = self.fixture.inspect(head=bad_head)
        self.assertFalse(result["passed"])
        self.assertIn("trusted track/measurement files", " ".join(result["errors"]))

    def test_mixed_valid_and_invalid_claim_paths_fail_closed(self):
        bad = self.fixture.repo / "ported_models/submissions/model_ports/README.md"
        bad.write_text("not a claim\n")
        bad_head = commit(self.fixture.repo, "add invalid claim-root payload")
        result = self.fixture.inspect(head=bad_head)
        self.assertFalse(result["passed"])
        self.assertIn("invalid files under", " ".join(result["errors"]))

    def test_claim_cannot_add_an_unclaimed_benchmark_entry(self):
        config_path = self.fixture.repo / ".github/ci/benchmark_config.json"
        config = json.loads(config_path.read_text())
        config["models"]["unclaimed"] = {"runner": "participant-controlled"}
        write_json(self.fixture.repo, ".github/ci/benchmark_config.json", config)
        bad_head = commit(self.fixture.repo, "candidate also adds unclaimed model")
        result = self.fixture.inspect(head=bad_head)
        self.assertFalse(result["passed"])
        self.assertIn("only add their claimed benchmark entries", " ".join(result["errors"]))

    def test_claim_size_limit_is_fail_closed(self):
        policy = copy.deepcopy(self.fixture.policy)
        policy["max_added_bytes"] = 1
        result = self.fixture.inspect(policy=policy)
        self.assertFalse(result["passed"])
        self.assertIn("committed-byte limit", " ".join(result["errors"]))

    def test_symlink_inside_participant_root_is_rejected(self):
        os.symlink(
            "src/novel.c",
            self.fixture.repo / "ported_models/novel/link-to-source",
        )
        bad_head = commit(self.fixture.repo, "add symlink")
        result = self.fixture.inspect(head=bad_head)
        self.assertFalse(result["passed"])
        self.assertIn("regular file", " ".join(result["errors"]))

    def test_trusted_tree_applies_only_new_root_and_model_entry(self):
        git(self.fixture.repo, "checkout", "--detach", self.fixture.base)
        metadata = prepare(
            repo=self.fixture.repo,
            base=self.fixture.base,
            head=self.fixture.head,
            main_ref=self.fixture.base,
            actor=ACTOR,
            pr_number=123,
            policy_path=self.fixture.repo / ".github/ci/reference/model_ports_track.json",
            identities_path=self.fixture.repo / "data/model-port-identities.json",
            ledger_path=self.fixture.repo / "data/model-port-credits.json",
        )
        self.assertTrue((self.fixture.repo / "ported_models/novel/src/novel.c").is_file())
        self.assertFalse((self.fixture.repo / "unrelated.txt").exists())
        config = json.loads((self.fixture.repo / ".github/ci/benchmark_config.json").read_text())
        self.assertEqual(config["models"]["novel"], self.fixture.entry)
        self.assertIn("unrelated.txt", metadata["ignored_paths"])

    def score(self, **updates):
        contract = (self.fixture.repo / CONTRACT_PATH).read_bytes()
        payload = {
            "model": "novel",
            "variant": "novel-int8",
            "passed": True,
            "kernel_wait_s": 0.25,
            "sha": self.fixture.head,
            "ref": "refs/heads/main",
            "team": ACTOR,
            "benchmark_device": "soc1sim",
            "run_url": "https://github.com/org/repo/actions/runs/1",
            "validation_contract_sha256": __import__("hashlib").sha256(contract).hexdigest(),
        }
        payload.update(updates)
        return payload

    def issue(self, score: dict, ledger=None):
        scores = self.fixture.repo / "scores"
        scores.mkdir(exist_ok=True)
        write_json(self.fixture.repo, "scores/score-novel.json", score)
        return issue_credits(
            repo=self.fixture.repo,
            before=self.fixture.base,
            head=self.fixture.head,
            actor=ACTOR,
            pr_number=123,
            participant_head_sha="b" * 40,
            merged_at="2026-07-01T00:00:00Z",
            expected_run_url="https://github.com/org/repo/actions/runs/1",
            scores_dir=scores,
            policy=self.fixture.policy,
            registry=self.fixture.registry,
            ledger=ledger or self.fixture.ledger,
        )

    def test_credit_requires_exact_score_provenance(self):
        mutations = {
            "team": "spoofed",
            "sha": "f" * 40,
            "ref": "refs/pull/123/head",
            "run_url": "https://github.com/org/repo/actions/runs/2",
            "benchmark_device": "sys_emu",
            "validation_contract_sha256": "0" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                _, report = self.issue(self.score(**{field: value}))
                self.assertFalse(report["passed"])
                self.assertIn("provenance mismatch", " ".join(report["errors"]))

    def test_non_claim_direct_push_is_a_noop(self):
        ledger, report = issue_credits(
            repo=self.fixture.repo,
            before=self.fixture.base,
            head=self.fixture.base,
            actor="Direct Push Name",
            pr_number=0,
            participant_head_sha=self.fixture.base,
            merged_at="2026-07-01T00:00:00Z",
            expected_run_url="https://github.com/org/repo/actions/runs/1",
            scores_dir=self.fixture.repo / "missing-scores",
            policy=self.fixture.policy,
            registry=self.fixture.registry,
            ledger=self.fixture.ledger,
            expected_claim_paths=set(),
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["targeted"])
        self.assertEqual(ledger, self.fixture.ledger)

    def test_failed_correctness_cannot_issue_credit(self):
        _, report = self.issue(self.score(passed=False))
        self.assertFalse(report["passed"])
        self.assertIn("trusted board score failed", " ".join(report["errors"]))

    def test_merged_claims_must_match_resolved_pull_request(self):
        scores = self.fixture.repo / "scores"
        scores.mkdir(exist_ok=True)
        write_json(self.fixture.repo, "scores/score-novel.json", self.score())
        _, report = issue_credits(
            repo=self.fixture.repo,
            before=self.fixture.base,
            head=self.fixture.head,
            actor=ACTOR,
            pr_number=123,
            participant_head_sha="b" * 40,
            merged_at="2026-07-01T00:00:00Z",
            expected_run_url="https://github.com/org/repo/actions/runs/1",
            scores_dir=scores,
            policy=self.fixture.policy,
            registry=self.fixture.registry,
            ledger=self.fixture.ledger,
            expected_claim_paths={"ported_models/submissions/model_ports/other.json"},
        )
        self.assertFalse(report["passed"])
        self.assertIn("resolved pull request files", " ".join(report["errors"]))

    def test_credit_issuance_is_idempotent(self):
        ledger, report = self.issue(self.score())
        self.assertEqual(report["issued"], ["novel-family"])
        self.assertEqual(len(ledger["records"]), 1)
        second, report = self.issue(self.score(), ledger=ledger)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["idempotent"], ["novel-family"])
        self.assertEqual(second, ledger)

    def test_first_credit_wins_a_concurrent_identity_race(self):
        ledger, first = self.issue(self.score())
        self.assertTrue(first["passed"])
        competitor_score = self.score(team="competitor")
        scores = self.fixture.repo / "scores"
        write_json(self.fixture.repo, "scores/score-novel.json", competitor_score)
        second, report = issue_credits(
            repo=self.fixture.repo,
            before=self.fixture.base,
            head=self.fixture.head,
            actor="competitor",
            pr_number=124,
            participant_head_sha="d" * 40,
            merged_at="2026-07-02T00:00:00Z",
            expected_run_url="https://github.com/org/repo/actions/runs/1",
            scores_dir=scores,
            policy=self.fixture.policy,
            registry=self.fixture.registry,
            ledger=ledger,
        )
        self.assertFalse(report["passed"])
        self.assertIn("already credited", " ".join(report["errors"]))
        self.assertEqual(second, ledger)

    def test_ledger_is_content_addressed_and_revocations_are_append_only(self):
        ledger, _ = self.issue(self.score())
        tampered = copy.deepcopy(ledger)
        tampered["records"][0]["participant_login"] = "mallory"
        with self.assertRaises(ClaimError):
            active_credits(tampered, self.fixture.policy)

        credit = ledger["records"][0]
        revocation_body = {
            "record_type": "revocation",
            "target_credit_id": credit["credit_id"],
            "reason": "organizer-reviewed correctness defect",
        }
        revoked = copy.deepcopy(ledger)
        revoked["records"].append(
            {"record_id": record_hash(revocation_body), **revocation_body}
        )
        self.assertEqual(active_credits(revoked, self.fixture.policy), {})
        self.assertEqual(revoked["records"][0], credit)

    def test_shadow_mode_never_writes_credit(self):
        policy = copy.deepcopy(self.fixture.policy)
        policy["activation_mode"] = "shadow"
        scores = self.fixture.repo / "scores"
        scores.mkdir(exist_ok=True)
        write_json(self.fixture.repo, "scores/score-novel.json", self.score())
        ledger, report = issue_credits(
            repo=self.fixture.repo,
            before=self.fixture.base,
            head=self.fixture.head,
            actor=ACTOR,
            pr_number=123,
            participant_head_sha="b" * 40,
            merged_at="2026-07-01T00:00:00Z",
            expected_run_url="https://github.com/org/repo/actions/runs/1",
            scores_dir=scores,
            policy=policy,
            registry=self.fixture.registry,
            ledger=self.fixture.ledger,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["would_issue"], ["novel-family"])
        self.assertEqual(ledger, self.fixture.ledger)

    def test_standings_use_count_then_earliest_final_merge(self):
        def credit(identity, login, merged_at):
            body = {
                "record_type": "credit",
                "credit_id": hashlib.sha256(
                    f"most_models_ported\0{identity}".encode()
                ).hexdigest(),
                "identity_id": identity,
                "benchmark_model": identity,
                "participant_login": login,
                "pr_number": 1,
                "participant_head_sha": "b" * 40,
                "merge_sha": "c" * 40,
                "merged_at": merged_at,
                "source": SOURCE,
                "recipe": "recipe.md",
                "benchmark_config_sha256": "d" * 64,
                "trusted_run": {
                    "url": "https://github.com/org/repo/actions/runs/1",
                    "score_sha": "c" * 40,
                    "validation_contract_sha256": "e" * 64,
                    "benchmark_device": "soc1sim",
                    "metric": "kernel_wait_s",
                    "metric_value": 0.25,
                },
            }
            return {"record_id": record_hash(body), **body}

        ledger = {
            "schema_version": 1,
            "track": "most_models_ported",
            "records": [
                credit("one", "alice", "2026-07-01T01:00:00+01:00"),
                credit("two", "bob", "2026-07-01T00:30:00Z"),
            ],
        }
        payload = standings(self.fixture.policy, ledger)
        self.assertEqual([row["participant_login"] for row in payload["standings"]], ["alice", "bob"])


if __name__ == "__main__":
    unittest.main()
