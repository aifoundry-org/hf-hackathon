#!/usr/bin/env python3

import json
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_trusted_llama32_candidate import evaluation_mode, validate_track_claim
from merge_leaderboard import merge_entry
from run_trusted_llama_benchmark import gguf_parameter_count
from trusted_llama32_gate import regression_floor


ROOT = Path(__file__).resolve().parents[3]
POLICY = json.loads(
    (ROOT / ".github/ci/reference/llama32_1b_track.json").read_text()
)
CONTRACT = json.loads(
    (ROOT / ".github/ci/reference/llama32_1b.json").read_text()
)


class TrustedLlamaPolicyTests(unittest.TestCase):
    def test_policy_is_bound_to_the_canonical_claim_and_sane_tolerance(self):
        self.assertEqual(POLICY["model"], "llama32_1b")
        self.assertEqual(
            POLICY["candidate_claim"],
            "ported_models/llama_cpp_et/submissions/llama32_1b.track.json",
        )
        tolerance = POLICY["shared_runtime"]["max_relative_throughput_regression"]
        self.assertGreaterEqual(tolerance, 0)
        self.assertLess(tolerance, 1)

    def test_runtime_change_without_claim_is_regression_only(self):
        self.assertEqual(
            evaluation_mode(
                runtime_changed=True,
                manifest_changed=False,
                claim_changed=False,
            ),
            "regression",
        )

    def test_claim_makes_candidate_competitive(self):
        self.assertEqual(
            evaluation_mode(
                runtime_changed=True,
                manifest_changed=False,
                claim_changed=True,
            ),
            "competition",
        )

    def test_manifest_cannot_enter_track_implicitly(self):
        with self.assertRaisesRegex(RuntimeError, "requires an explicit track claim"):
            evaluation_mode(
                runtime_changed=False,
                manifest_changed=True,
                claim_changed=False,
            )

    def test_claim_must_bind_exact_runtime_and_submission(self):
        runtime = "a" * 40
        claim = {
            "schema_version": 1,
            "track": POLICY["track"],
            "model": "llama32_1b",
            "submission_id": "alice-v2",
            "runtime_revision": runtime,
        }
        validate_track_claim(claim, POLICY, runtime)
        claim["runtime_revision"] = "b" * 40
        with self.assertRaisesRegex(RuntimeError, "candidate llama.cpp-et gitlink"):
            validate_track_claim(claim, POLICY, runtime)

    def test_regression_floor_allows_configured_noise_only(self):
        self.assertAlmostEqual(regression_floor(100.0, 0.01), 99.0)

    def test_contract_uses_one_bounded_et_server_process(self):
        self.assertEqual(CONTRACT["performance"]["tool"], "llama-server")
        self.assertEqual(CONTRACT["performance"]["repetitions"], 1)
        self.assertEqual(CONTRACT["performance"]["generation_tokens"], 24)
        self.assertEqual(CONTRACT["quality"]["reference_device"], "CPU")

    def test_gguf_parameter_count_uses_exact_tensor_shapes(self):
        def string(value: bytes) -> bytes:
            return struct.pack("<Q", len(value)) + value

        payload = bytearray(b"GGUF")
        payload += struct.pack("<IQQ", 3, 2, 1)
        payload += string(b"general.name")
        payload += struct.pack("<I", 8)
        payload += string(b"fixture")
        payload += string(b"tensor_a")
        payload += struct.pack("<IQQIQ", 2, 2, 3, 0, 0)
        payload += string(b"tensor_b")
        payload += struct.pack("<IQIQ", 1, 5, 0, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(payload)
            self.assertEqual(gguf_parameter_count(path), 11)

    def test_trusted_login_overrides_score_identity_and_deduplicates(self):
        existing = [
            {
                "team": "Old Display Name",
                "participant_login": "octocat",
                "tokens_per_second": 1.0,
            }
        ]
        score = {
            "model": "llama32_1b",
            "passed": True,
            "team": "spoofed-name",
            "tokens_per_second": 2.0,
        }
        entries = merge_entry(existing, score, participant_login="octocat")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["team"], "octocat")
        self.assertEqual(entries[0]["participant_login"], "octocat")


if __name__ == "__main__":
    unittest.main()
