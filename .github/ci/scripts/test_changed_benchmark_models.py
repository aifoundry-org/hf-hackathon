from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_config_helpers import load_config
from changed_benchmark_models import (
    HARDWARE_MIGRATION_SMOKE_MODELS,
    existing_leaderboard_models,
)


class HardwareMigrationSelectionTests(unittest.TestCase):
    def test_smoke_covers_every_board_runner(self) -> None:
        cfg = load_config()
        runners = {
            cfg["models"][model].get("runner", "elf")
            for model in HARDWARE_MIGRATION_SMOKE_MODELS
        }
        self.assertEqual(runners, {"elf", "llama_server", "smolvlm2_video"})

    def test_bootstrap_selects_only_models_with_existing_entries(self) -> None:
        cfg = load_config()
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            (data_root / "yolo.json").write_text(
                json.dumps({"entries": [{"kernel_wait_s": 1.0}]})
            )
            (data_root / "dncnn.json").write_text(json.dumps({"entries": []}))
            (data_root / "lfm25.json").write_text("{invalid")

            selected = existing_leaderboard_models(cfg, "board", data_root)

        self.assertEqual(selected, ["yolo"])


if __name__ == "__main__":
    unittest.main()
