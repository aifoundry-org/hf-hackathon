from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import board_lock


class BoardLockTests(unittest.TestCase):
    def test_open_never_requests_create(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "board.lock"
            path.touch()
            real_open = os.open
            observed_flags: list[int] = []

            def recording_open(name: str, flags: int, *args: object) -> int:
                observed_flags.append(flags)
                return real_open(name, flags, *args)

            with mock.patch.object(board_lock.os, "open", side_effect=recording_open):
                with board_lock.exclusive_board_lock(path):
                    pass

        self.assertEqual(len(observed_flags), 1)
        self.assertEqual(observed_flags[0] & os.O_CREAT, 0)

    def test_missing_lock_fails_with_provisioning_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "missing.lock"
            with self.assertRaisesRegex(RuntimeError, "prepare_board_lock.sh"):
                board_lock.open_board_lock(path)

    def test_cli_times_out_while_another_process_holds_lock(self) -> None:
        helper = Path(board_lock.__file__).resolve()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "board.lock"
            path.touch()
            with board_lock.exclusive_board_lock(path):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(helper),
                        "--lock",
                        str(path),
                        "--timeout",
                        "0.05",
                        "--",
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

        self.assertEqual(proc.returncode, 73)
        self.assertIn("timed out", proc.stderr)

    def test_cli_returns_child_status_after_lock_release(self) -> None:
        helper = Path(board_lock.__file__).resolve()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "board.lock"
            path.touch()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(helper),
                    "--lock",
                    str(path),
                    "--timeout",
                    "1",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(proc.returncode, 7)

    def test_every_real_board_entrypoint_uses_shared_lock_helpers(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected_markers = {
            ".github/ci/scripts/run_llama_server_benchmark.py": "open_board_lock(board_lock)",
            ".github/ci/scripts/run_smolvlm2_video_benchmark.py": "open_board_lock(board_lock)",
            "scripts/run_sysemu_model_ports.sh": ".github/ci/scripts/board_lock.py",
            ".github/ci/platform/et_jobs/runner.py": ".github/ci/scripts/board_lock.py",
            ".github/ci/platform/deploy/soc3-benchmark.sh": ".github/ci/scripts/board_lock.py",
        }
        for relative, marker in expected_markers.items():
            with self.subTest(path=relative):
                text = (root / relative).read_text()
                self.assertIn(marker, text)
                self.assertNotIn('open("a")', text)
                self.assertNotIn("flock -x", text)

        provisioning_entrypoints = [
            ".github/ci/platform/deploy/install-board-host.sh",
            ".github/ci/platform/deploy/soc3-benchmark-jobs.sh",
            ".github/ci/platform/deploy/soc3-benchmark.sh",
            ".github/ci/platform/deploy/soc3-e2e.sh",
            "scripts/run_sysemu_model_ports.sh",
        ]
        for relative in provisioning_entrypoints:
            with self.subTest(provisioning=relative):
                self.assertIn(
                    "prepare_board_lock.sh",
                    (root / relative).read_text(),
                )

        selector = (root / ".github/ci/scripts/changed_benchmark_models.py").read_text()
        workflow = (root / ".github/workflows/benchmark-board.yml").read_text()
        for relative in (
            ".github/ci/scripts/board_lock.py",
            ".github/ci/scripts/prepare_board_lock.sh",
        ):
            with self.subTest(protected_infra=relative):
                self.assertIn(f'"{relative}"', selector)
                self.assertIn(relative, workflow)
                selection = subprocess.run(
                    [
                        sys.executable,
                        str(root / ".github/ci/scripts/changed_benchmark_models.py"),
                        "--target",
                        "board",
                        "--changed-file",
                        relative,
                        "--scope",
                        "changed",
                        "--format",
                        "space",
                    ],
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                self.assertEqual(selection.stdout.split(), ["dncnn", "smollm2_135m"])


if __name__ == "__main__":
    unittest.main()
