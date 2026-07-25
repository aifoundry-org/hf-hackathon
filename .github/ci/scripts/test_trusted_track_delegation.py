#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

from trusted_track_delegation import (
    LLAMA_MODEL,
    SMOLVLM2_MODEL,
    delegated_runtime_models,
    remaining_models,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads(
    (ROOT / ".github/ci/reference/llama32_1b.json").read_text()
)


class TrustedTrackDelegationTests(unittest.TestCase):
    def test_delegates_smol_llama_and_shared_runtime_regressions(self) -> None:
        delegated = delegated_runtime_models(CONTRACT)

        self.assertIn(SMOLVLM2_MODEL, delegated)
        self.assertIn(LLAMA_MODEL, delegated)
        self.assertEqual(
            delegated - {SMOLVLM2_MODEL, LLAMA_MODEL},
            set(CONTRACT["runtime"]["regression_models"]),
        )

    def test_preserves_unrelated_models_in_original_order(self) -> None:
        selected = [
            "yolo",
            SMOLVLM2_MODEL,
            "dncnn",
            LLAMA_MODEL,
            *CONTRACT["runtime"]["regression_models"],
        ]

        self.assertEqual(
            remaining_models(selected, CONTRACT),
            ["yolo", "dncnn"],
        )


if __name__ == "__main__":
    unittest.main()
