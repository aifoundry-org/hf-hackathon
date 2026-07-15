#!/usr/bin/env python3
"""Regression tests for trusted SmolVLM2 candidate scoping."""

import unittest

from prepare_trusted_smolvlm2_candidate import validate_candidate_paths


class CandidatePathTests(unittest.TestCase):
    def test_unrelated_protected_paths_are_a_noop(self) -> None:
        validate_candidate_paths(
            [
                ".github/ci/benchmark_config.json",
                "ported_models/llama_cpp_et/benchmarks/another_model.json",
            ],
            runtime_changed=False,
        )

    def test_runtime_candidate_cannot_change_protected_paths(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "protected SmolVLM2 gate paths"):
            validate_candidate_paths(
                [
                    "ported_models/llama_cpp_et/src/llama.cpp-et",
                    "ported_models/llama_cpp_et/artifacts.json",
                ],
                runtime_changed=True,
            )

    def test_runtime_only_candidate_is_allowed(self) -> None:
        validate_candidate_paths(
            ["ported_models/llama_cpp_et/src/llama.cpp-et"],
            runtime_changed=True,
        )


if __name__ == "__main__":
    unittest.main()
