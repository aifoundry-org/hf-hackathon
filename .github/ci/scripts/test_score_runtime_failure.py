#!/usr/bin/env python3
"""Regression tests for ET runtime-failure classification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from score_results import evaluate_row


class RuntimeFailureTests(unittest.TestCase):
    def evaluate_log(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.log"
            log.write_text(text)
            return evaluate_row(
                "dncnn",
                {},
                {
                    "status": "fail",
                    "log": "run.log",
                    "dump": "",
                    "kernel_wait_s": "",
                    "elapsed_s": "60.0",
                },
                root,
                None,
            )

    def test_launcher_timeout_is_a_runtime_failure(self):
        result = self.evaluate_log("Error: kernel execution timed out\n")
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_kind"], "runtime")
        self.assertIn("kernel execution timed out", result["valid_note"])

    def test_stale_abort_response_is_a_runtime_failure(self):
        result = self.evaluate_log(
            "DeviceLayer could be in a BAD STATE\n"
            "FATAL Unbalanced number of abort unblockers\n"
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_kind"], "runtime")
        self.assertIn("DeviceLayer could be in a BAD STATE", result["valid_note"])


if __name__ == "__main__":
    unittest.main()
