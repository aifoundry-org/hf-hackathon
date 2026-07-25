#!/usr/bin/env python3
"""Tests for the nested trusted SmolVLM2 runtime boundary."""

import unittest

from trusted_smolvlm2_runtime_scope import build_report, classify_change


class RuntimeScopeTests(unittest.TestCase):
    def test_kernel_sources_are_allowed_and_tracked(self) -> None:
        for suffix in ("c", "h", "S", "inc"):
            change = classify_change(
                "ggml/src/ggml-et/et-kernels/src/candidate." + suffix,
                "M",
                "100644",
            )
            self.assertTrue(change["allowed"])
            self.assertEqual(change["scope"], "kernel")

    def test_exact_registration_and_dispatch_files_are_allowed(self) -> None:
        for path in (
            "ggml/src/ggml-et/CMakeLists.txt",
            "ggml/src/ggml-et/ggml-et-ops.cpp",
            "ggml/src/ggml-et/ggml-et-ops.h",
        ):
            change = classify_change(path, "M", "100644")
            self.assertTrue(change["allowed"])
            self.assertEqual(change["scope"], "integration")

    def test_unrelated_runtime_files_are_rejected(self) -> None:
        change = classify_change(
            "ggml/src/ggml-et/ggml-et-backend.cpp", "M", "100644"
        )
        self.assertFalse(change["allowed"])
        self.assertEqual(change["scope"], "rejected")

    def test_deletions_and_non_regular_modes_are_rejected(self) -> None:
        path = "ggml/src/ggml-et/et-kernels/src/mul_mat_Q8_0.c"
        self.assertFalse(classify_change(path, "D", "")["allowed"])
        self.assertFalse(classify_change(path, "M", "100755")["allowed"])
        self.assertFalse(classify_change(path, "T", "120000")["allowed"])

    def test_report_keeps_every_classification(self) -> None:
        report = build_report(
            "base",
            "candidate",
            [
                {
                    "path": "ggml/src/ggml-et/et-kernels/src/kernel.c",
                    "status": "A",
                    "mode": "100644",
                },
                {
                    "path": "ggml/src/ggml-et/ggml-et-ops.cpp",
                    "status": "M",
                    "mode": "100644",
                },
                {
                    "path": "examples/server/server.cpp",
                    "status": "M",
                    "mode": "100644",
                },
            ],
        )
        self.assertFalse(report["allowed"])
        self.assertEqual(
            report["summary"],
            {"total": 3, "kernel": 1, "integration": 1, "rejected": 1},
        )
        self.assertEqual(report["rejected_paths"], ["examples/server/server.cpp"])


if __name__ == "__main__":
    unittest.main()
