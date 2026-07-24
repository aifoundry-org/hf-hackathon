#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_labels import classify, desired_labels


class TrackLabelTests(unittest.TestCase):
    def test_file_rules(self):
        cases = (
            ([{"filename": "ported_models/yolo/src/kernel.c"}], ["track: yolo-performance"]),
            (
                [{"filename": "ported_models/llama_cpp_et/src/llama.cpp-et"}],
                ["track: week-2-challenge"],
            ),
            (
                [{"filename": "ported_models/llama_cpp_et/benchmarks/smolvlm2_500m_video.json"}],
                ["track: week-2-challenge"],
            ),
            ([{"filename": "ported_models/llama_cpp_et/benchmarks/llama32_1b.json"}], ["track: llama-3.2-1b-performance"]),
            ([{"filename": "ported_models/llama_cpp_et/benchmarks/new.json", "status": "added"}], ["track: model-ports"]),
            ([{"filename": "ported_models/submissions/model_ports/new_model.json", "status": "added"}], ["track: model-ports"]),
            ([{"filename": "docs/BOARD_ACCESS.md"}], ["track: community"]),
            ([{"filename": "LICENSE"}], ["misc"]),
            (
                [
                    {"filename": "ported_models/yolo/src/kernel.c"},
                    {"filename": "ported_models/yolo/docs/recipe.md"},
                ],
                ["track: yolo-performance", "track: community"],
            ),
        )
        for files, expected in cases:
            with self.subTest(files=files):
                self.assertEqual(desired_labels(files), expected)

    def test_week2_prefix(self):
        self.assertEqual(
            desired_labels(
                [{"filename": "ported_models/robotics/src/model.c"}],
                week2_prefixes=("ported_models/robotics",),
            ),
            ["track: week-2-challenge"],
        )

    def test_new_port_uses_trusted_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ported_models" / "yolo").mkdir(parents=True)
            result = classify(
                [{"filename": "ported_models/new_model/MODEL.md", "status": "added"}],
                root,
            )
        self.assertEqual(result["new_port_roots"], ["new_model"])
        self.assertEqual(result["desired_labels"], ["track: model-ports"])


if __name__ == "__main__":
    unittest.main()
