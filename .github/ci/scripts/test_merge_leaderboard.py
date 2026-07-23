#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from merge_leaderboard import (
    bootstrap_participant,
    hardware_epoch,
    merge_entry,
    metric_config,
    validate_score_set,
)


class MergeLeaderboardTests(unittest.TestCase):
    EXPECTED_SHA = "candidate"
    EXPECTED_REF = "refs/heads/main"
    EXPECTED_RUN_URL = "https://github.example/actions/runs/1"

    @staticmethod
    def valid_score(model: str) -> dict:
        metric, _ = metric_config(model)
        return {
            "model": model,
            "passed": True,
            metric: 1.0,
            "hardware_epoch": hardware_epoch(),
            "hardware": {
                "board_id": "aifoundry3",
                "minion_frequency_mhz": 600,
                "noc_frequency_mhz": 400,
                "tdp_w": 0,
                "boot_id": "test-boot",
            },
            "sha": MergeLeaderboardTests.EXPECTED_SHA,
            "ref": MergeLeaderboardTests.EXPECTED_REF,
            "run_url": MergeLeaderboardTests.EXPECTED_RUN_URL,
        }

    def validate_scores(self, scores_dir: Path, models: list[str]) -> dict[str, dict]:
        return validate_score_set(
            scores_dir,
            models,
            expected_sha=self.EXPECTED_SHA,
            expected_ref=self.EXPECTED_REF,
            expected_run_url=self.EXPECTED_RUN_URL,
        )

    def test_complete_passing_score_set_is_accepted(self):
        models = ["lfm25", "tinyllama11b"]
        with tempfile.TemporaryDirectory() as raw:
            scores_dir = Path(raw)
            for model in models:
                (scores_dir / f"score-{model}.json").write_text(
                    json.dumps(self.valid_score(model))
                )
            scores = self.validate_scores(scores_dir, models)
        self.assertEqual(set(scores), set(models))

    def test_missing_score_rejects_entire_set(self):
        with tempfile.TemporaryDirectory() as raw:
            scores_dir = Path(raw)
            (scores_dir / "score-lfm25.json").write_text(
                json.dumps(self.valid_score("lfm25"))
            )
            with self.assertRaisesRegex(SystemExit, "tinyllama11b: missing"):
                self.validate_scores(scores_dir, ["lfm25", "tinyllama11b"])

    def test_failed_or_wrong_epoch_score_rejects_entire_set(self):
        score = self.valid_score("lfm25")
        score["passed"] = False
        score["hardware_epoch"] = "some-other-board"
        with tempfile.TemporaryDirectory() as raw:
            scores_dir = Path(raw)
            (scores_dir / "score-lfm25.json").write_text(json.dumps(score))
            with self.assertRaisesRegex(
                SystemExit, "passed is not true.*hardware mismatch"
            ):
                self.validate_scores(scores_dir, ["lfm25"])

    def test_score_provenance_must_match_exact_commit_ref_and_run(self):
        for field, value in (
            ("sha", "newer-main"),
            ("ref", "refs/heads/other"),
            ("run_url", "https://github.example/actions/runs/2"),
        ):
            with self.subTest(field=field):
                score = self.valid_score("lfm25")
                score[field] = value
                with tempfile.TemporaryDirectory() as raw:
                    scores_dir = Path(raw)
                    (scores_dir / "score-lfm25.json").write_text(json.dumps(score))
                    with self.assertRaisesRegex(SystemExit, field):
                        self.validate_scores(scores_dir, ["lfm25"])

    def test_lower_is_better_score_must_strictly_improve(self):
        existing = [
            {
                "team": "octocat",
                "participant_login": "octocat",
                "pmc_cycles": 100,
                "sha": "best",
            }
        ]

        for pmc_cycles in (100, 101):
            with self.subTest(pmc_cycles=pmc_cycles):
                score = {
                    "model": "smolvlm2_500m_video",
                    "passed": True,
                    "pmc_cycles": pmc_cycles,
                    "sha": "candidate",
                }
                self.assertEqual(
                    merge_entry(existing, score, participant_login="octocat"),
                    existing,
                )

        improved = merge_entry(
            existing,
            {
                "model": "smolvlm2_500m_video",
                "passed": True,
                "pmc_cycles": 99,
                "sha": "candidate",
            },
            participant_login="octocat",
        )
        self.assertEqual(len(improved), 1)
        self.assertEqual(improved[0]["pmc_cycles"], 99)
        self.assertEqual(improved[0]["sha"], "candidate")

    def test_higher_is_better_score_must_strictly_improve(self):
        existing = [
            {
                "team": "octocat",
                "participant_login": "octocat",
                "tokens_per_second": 10.0,
                "sha": "best",
            }
        ]

        for tokens_per_second in (10.0, 9.0):
            with self.subTest(tokens_per_second=tokens_per_second):
                score = {
                    "model": "llama32_1b",
                    "passed": True,
                    "tokens_per_second": tokens_per_second,
                    "sha": "candidate",
                }
                self.assertEqual(
                    merge_entry(existing, score, participant_login="octocat"),
                    existing,
                )

        improved = merge_entry(
            existing,
            {
                "model": "llama32_1b",
                "passed": True,
                "tokens_per_second": 11.0,
                "sha": "candidate",
            },
            participant_login="octocat",
        )
        self.assertEqual(len(improved), 1)
        self.assertEqual(improved[0]["tokens_per_second"], 11.0)
        self.assertEqual(improved[0]["sha"], "candidate")

    def test_main_epoch_bootstrap_preserves_the_incumbent_owner(self):
        legacy = [
            {
                "team": "Display Name",
                "participant_login": "incumbent",
                "variant": "dncnn20l64",
                "kernel_wait_s": 1.0,
            }
        ]
        score = {
            "model": "dncnn",
            "passed": True,
            "hardware_epoch": hardware_epoch(),
            "ref": "refs/heads/main",
            "kernel_wait_s": 0.9,
        }
        self.assertEqual(
            bootstrap_participant("dncnn", score, legacy),
            "incumbent",
        )

    def test_pull_request_cannot_claim_the_epoch_bootstrap(self):
        legacy = [
            {
                "participant_login": "incumbent",
                "variant": "dncnn20l64",
                "kernel_wait_s": 1.0,
            }
        ]
        score = {
            "model": "dncnn",
            "hardware_epoch": hardware_epoch(),
            "ref": "refs/pull/123/head",
        }
        self.assertIsNone(bootstrap_participant("dncnn", score, legacy))


if __name__ == "__main__":
    unittest.main()
