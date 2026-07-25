#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrustedSmolVLM2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "quality": {"perplexity": {"maximum_perplexity": 26.739}},
            "performance": {
                "minimum_relative_improvement": 0.01,
                "maximum_paired_baseline_drift": 0.005,
                "minimum_end_to_end_improvement": 0.0025,
                "maximum_paired_end_to_end_drift": 0.01,
            },
        }

    def run_gate(
        self,
        candidate_cycles: int,
        *,
        candidate_ppl: float = 22.0,
        candidate_wall: float = 99.0,
        candidate_boot: str = "test-boot",
        after: int = 1002,
    ) -> int:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(self.contract))
            contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()

            def score(cycles: int, ppl: float = 22.0, wall: float = 100.0, boot: str = "test-boot") -> dict:
                return {
                    "hardware_epoch": "et-soc1-aifoundry3-600-400-tdp0-v1",
                    "hardware": {
                        "board_id": "aifoundry3",
                        "boot_id": boot,
                        "minion_frequency_mhz": 600,
                        "noc_frequency_mhz": 400,
                        "tdp_w": 0,
                    },
                    "passed": True,
                    "pmc_cycles": cycles,
                    "median_end_to_end_s": wall,
                    "perplexity": ppl,
                    "cpu_perplexity": ppl,
                    "task_accuracy": 1.0,
                    "host_agreement": 1.0,
                    "vision_fallback_ops": [],
                    "trusted_cpu_reference": True,
                    "cpu_reference_executed": True,
                    "cpu_perplexity_reference_executed": True,
                    "validation_contract_sha256": contract_sha,
                }

            paths = []
            for name, value in (
                ("before", score(1000)),
                ("candidate", score(candidate_cycles, candidate_ppl, candidate_wall, candidate_boot)),
                ("after", score(after, wall=100.2)),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value))
                paths.append(path)
            gate = Path(__file__).with_name("trusted_smolvlm2_gate.py")
            return subprocess.run(
                [
                    sys.executable,
                    str(gate),
                    "--contract",
                    str(contract_path),
                    "--baseline-before",
                    str(paths[0]),
                    "--candidate",
                    str(paths[1]),
                    "--baseline-after",
                    str(paths[2]),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).returncode

    def test_gate_passes_real_improvement(self) -> None:
        self.assertEqual(self.run_gate(990), 0)

    def test_gate_rejects_ppl_regression(self) -> None:
        self.assertEqual(self.run_gate(990, candidate_ppl=27.0), 1)

    def test_gate_rejects_pmc_only_improvement(self) -> None:
        self.assertEqual(self.run_gate(990, candidate_wall=100.0), 1)

    def test_gate_marks_main_drift_as_infrastructure(self) -> None:
        self.assertEqual(self.run_gate(900, after=1100), 2)

    def test_gate_rejects_a_candidate_from_another_board_boot(self) -> None:
        self.assertEqual(self.run_gate(990, candidate_boot="different-boot"), 2)


if __name__ == "__main__":
    unittest.main()
