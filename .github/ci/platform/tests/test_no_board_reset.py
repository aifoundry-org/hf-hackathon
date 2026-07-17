from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
BOARD_RUNTIME_FILES = [
    REPO_ROOT / ".github/ci/platform/deploy/soc3-benchmark.sh",
    REPO_ROOT / ".github/ci/platform/deploy/soc3-benchmark-jobs.sh",
    REPO_ROOT / ".github/ci/platform/deploy/soc3-e2e.sh",
    REPO_ROOT / ".github/ci/platform/deploy/config.env.example",
    REPO_ROOT / ".github/ci/platform/et_jobs/config.py",
    REPO_ROOT / ".github/ci/platform/et_jobs/runner.py",
]
FORBIDDEN_RESET_MARKERS = (
    "SOC_RESET_SYSFS",
    "/soc_reset/reinitiate",
    "def soc_reset(",
    "reset_board()",
)


class BoardRecoveryPolicyTests(unittest.TestCase):
    def test_board_runtime_never_resets_card(self) -> None:
        for path in BOARD_RUNTIME_FILES:
            with self.subTest(path=path.name):
                text = path.read_text()
                found = [marker for marker in FORBIDDEN_RESET_MARKERS if marker in text]
                self.assertFalse(
                    found,
                    f"{path.relative_to(REPO_ROOT)} contains forbidden reset controls: {found}",
                )


if __name__ == "__main__":
    unittest.main()
