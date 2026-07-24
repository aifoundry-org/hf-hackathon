#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from model_port_claim import ClaimError
from model_port_history import validate_historical_review


REPO_ROOT = Path(__file__).resolve().parents[3]


def load(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text())


class HistoricalModelPortReviewTests(unittest.TestCase):
    def setUp(self):
        self.policy = load(".github/ci/reference/model_ports_track.json")
        self.registry = load("data/model-port-identities.json")
        self.ledger = load("data/model-port-credits.json")
        self.review = load("data/model-port-historical-review.json")

    def validate(self, review=None, ledger=None):
        return validate_historical_review(
            repo=REPO_ROOT,
            policy=self.policy,
            registry=self.registry,
            ledger=ledger or self.ledger,
            review=review or self.review,
        )

    def test_committed_historical_review_matches_backfill(self):
        result = self.validate()
        self.assertEqual(result["decisions"], 11)
        self.assertEqual(result["credited"], 1)
        self.assertEqual(result["ineligible"], 10)

    def test_backfill_requires_exact_oracle_run(self):
        review = copy.deepcopy(self.review)
        credited = next(
            decision for decision in review["decisions"] if decision["outcome"] == "credited"
        )
        credited["oracle"]["trusted_run_url"] = (
            "https://github.com/aifoundry-org/hf-hackathon/actions/runs/1"
        )
        with self.assertRaisesRegex(ClaimError, "wrong run URL"):
            self.validate(review=review)

    def test_every_historical_credit_requires_a_review_decision(self):
        review = copy.deepcopy(self.review)
        review["decisions"] = [
            decision for decision in review["decisions"] if decision["outcome"] != "credited"
        ]
        with self.assertRaisesRegex(ClaimError, "review and backfilled ledger credits"):
            self.validate(review=review)


if __name__ == "__main__":
    unittest.main()
