#!/usr/bin/env python3

import unittest

from merge_leaderboard import merge_entry


class MergeLeaderboardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
