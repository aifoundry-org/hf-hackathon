import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_RESET_MARKERS = (
    "SOC_RESET_SYSFS",
    "/soc_reset/reinitiate",
    "def soc_reset(",
    "reset_board()",
)
FORBIDDEN_BYPASS_MARKERS = (
    "SOC3_SKIP_BOARD_SMOKE",
    "SOC3_FAIL_ON_MODEL_FAILURE",
)
POLICY_TEST = Path(__file__).resolve()


def board_execution_files() -> list[Path]:
    roots = [
        REPO_ROOT / ".github/workflows",
        REPO_ROOT / ".github/ci/scripts",
        REPO_ROOT / ".github/ci/platform",
    ]
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path == POLICY_TEST:
                continue
            if path.name == "README.md":
                continue
            if path.suffix in {".py", ".sh", ".yml", ".yaml"} or ".env.example" in path.name:
                files.append(path)
    return files


class BoardRecoveryPolicyTests(unittest.TestCase):
    def test_all_board_execution_paths_never_reset_card(self) -> None:
        for path in board_execution_files():
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                text = path.read_text()
                found = [marker for marker in FORBIDDEN_RESET_MARKERS if marker in text]
                self.assertFalse(
                    found,
                    f"{path.relative_to(REPO_ROOT)} contains forbidden reset controls: {found}",
                )

    def test_board_preflight_and_failure_cannot_be_bypassed(self) -> None:
        for path in board_execution_files():
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                text = path.read_text()
                found = [marker for marker in FORBIDDEN_BYPASS_MARKERS if marker in text]
                self.assertFalse(
                    found,
                    f"{path.relative_to(REPO_ROOT)} contains a board safety bypass: {found}",
                )

    def test_direct_runner_is_fail_closed_and_quarantines_hardware_errors(self) -> None:
        script = (
            REPO_ROOT / ".github/ci/platform/deploy/soc3-benchmark.sh"
        ).read_text()
        for marker in (
            "require_clean_board_state",
            "mark_board_quarantined",
            "reject_new_board_errors",
            "runtime_failure_marker",
            "reject_candidate_card_controls",
            "cmp -s",
            "exit \"$FAIL\"",
        ):
            self.assertIn(marker, script)
        self.assertLess(
            script.index("is quarantined; refusing the board job before any build"),
            script.index("Pre-building ELFs locally"),
        )
        self.assertNotIn("leaving board infrastructure job green", script)

    def test_external_pr_code_never_runs_directly_on_root_board_runner(self) -> None:
        workflow = (
            REPO_ROOT / ".github/workflows/benchmark-board.yml"
        ).read_text()
        self.assertIn(
            "External PR code is never executed directly as root on the board host",
            workflow,
        )
        external_case = workflow.split('elif [[ "$external_pr" == "1" ]]', 1)[1]
        external_case = external_case.split("else", 1)[0]
        self.assertIn("allowed=0", external_case)

    def test_root_runner_has_kernel_control_sandbox_policy(self) -> None:
        installer = (
            REPO_ROOT
            / ".github/ci/platform/deploy/install-actions-runner-safety.sh"
        ).read_text()
        for setting in (
            "ProtectKernelTunables=yes",
            "ProtectKernelModules=yes",
            "ProtectControlGroups=yes",
            "ReadOnlyPaths=/sys/bus/pci/devices /sys/devices /opt/et /opt/et-platform",
            "IPAddressDeny=10.20.10.117",
            "NoNewPrivileges=yes",
            "CapabilityBoundingSet=~CAP_SYS_ADMIN CAP_SYS_BOOT CAP_SYS_MODULE CAP_SYS_RAWIO",
            "SystemCallFilter=~@mount @reboot @module",
        ):
            self.assertIn(setting, installer)
        self.assertIn('systemctl show "$unit" -p User --value', installer)
        self.assertIn('state_dir="${ET_BOARD_STATE_DIR:-/var/lib/et-soc1-ci}"', installer)
        self.assertIn('board_lock="${BOARD_LOCK:-$state_dir/board.lock}"', installer)
        self.assertNotIn("root:etsoc", installer)
        self.assertNotIn("systemctl restart", installer)
        self.assertNotIn("systemctl start", installer)

    def test_board_runner_requires_audited_host_runtime_before_build(self) -> None:
        runner = (
            REPO_ROOT / ".github/ci/platform/deploy/soc3-benchmark.sh"
        ).read_text()
        verifier = (
            REPO_ROOT
            / ".github/ci/platform/deploy/verify-et-runtime-contract.sh"
        ).read_text()
        installer = (
            REPO_ROOT
            / ".github/ci/platform/deploy/install-et-runtime-contract.sh"
        ).read_text()
        self.assertIn("verify-et-runtime-contract.sh", runner)
        self.assertLess(
            runner.index("verify-et-runtime-contract.sh"),
            runner.index("Pre-building ELFs locally"),
        )
        for marker in (
            "source_revision",
            "sha256sum",
            "required_marker",
            "does not match the audited contract",
        ):
            self.assertIn(marker, verifier)
        self.assertIn("refusing runtime replacement", installer)
        self.assertIn("The runner was not started and the board was not accessed.", installer)

    def test_board_workflows_cannot_escape_sandbox_over_ssh(self) -> None:
        runner = (
            REPO_ROOT / ".github/ci/platform/deploy/soc3-benchmark.sh"
        ).read_text()
        self.assertIn('SOC3_REQUIRE_LOCAL:-0}" == "1"', runner)
        for name in (
            "benchmark-board.yml",
            "trusted-llama32-pr.yml",
            "trusted-smolvlm2-pr.yml",
            "trusted-model-port-pr.yml",
        ):
            workflow = (REPO_ROOT / ".github/workflows" / name).read_text()
            with self.subTest(workflow=name):
                self.assertIn('SOC3_REQUIRE_LOCAL: "1"', workflow)

    def test_llama_generation_stays_inside_empirical_event_budget(self) -> None:
        contract = json.loads(
            (REPO_ROOT / ".github/ci/reference/llama32_1b.json").read_text()
        )
        performance = contract["performance"]
        self.assertLessEqual(
            int(performance["generation_tokens"]) * int(performance["repetitions"]),
            72,
        )
        self.assertLessEqual(
            int(contract["generation_validation"]["max_tokens"]),
            24,
        )
        for path in sorted(
            (REPO_ROOT / "ported_models/llama_cpp_et/benchmarks").glob("*.json")
        ):
            config = json.loads(path.read_text())
            if config.get("runner") != "llama_server":
                continue
            llama = config["llama_server"]
            with self.subTest(path=path.name):
                self.assertLessEqual(int(llama["max_tokens"]), 24)
                self.assertLessEqual(
                    int(llama["min_completion_tokens"]),
                    int(llama["max_tokens"]),
                )

    def test_llama_backend_drains_before_runtime_event_id_wrap(self) -> None:
        source_root = (
            REPO_ROOT
            / "ported_models/llama_cpp_et/src/llama.cpp-et/ggml/src/ggml-et"
        )
        common = (source_root / "ggml-et-common.h").read_text()
        kernels = (source_root / "ggml-et-kernels.cpp").read_text()
        backend = (source_root / "ggml-et.cpp").read_text()
        self.assertIn("pending_kernel_events = 0", common)
        self.assertIn("GGML_ET_MAX_PENDING_KERNEL_EVENTS = 4096", kernels)
        self.assertIn(
            "pending_kernel_events >= GGML_ET_MAX_PENDING_KERNEL_EVENTS",
            kernels,
        )
        self.assertIn("waitForStream(dev_ctx->default_stream)", kernels)
        self.assertIn("dev_ctx->pending_kernel_events = 0", backend)


if __name__ == "__main__":
    unittest.main()
