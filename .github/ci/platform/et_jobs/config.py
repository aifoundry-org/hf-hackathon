from __future__ import annotations

import json
import os
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


DATA_DIR = _path("JOBS_DATA_DIR", "/var/lib/et-jobs")
DB_PATH = _path("JOBS_DB_PATH", str(DATA_DIR / "jobs.db"))
ARTIFACTS_DIR = _path("JOBS_ARTIFACTS_DIR", str(DATA_DIR / "artifacts"))
LOGS_DIR = _path("JOBS_LOGS_DIR", str(DATA_DIR / "logs"))

API_BIND = os.environ.get("JOBS_BIND", "127.0.0.1:8080")
API_URL = os.environ.get("JOBS_API_URL", f"http://{API_BIND}")
HOST_ID = os.environ.get("HOST_ID", "localhost")
REPO_ROOT = _path("JOBS_REPO_ROOT", str(Path(__file__).resolve().parents[4]))
BENCHMARK_CONFIG = _path(
    "BENCHMARK_CONFIG",
    str(REPO_ROOT / ".github" / "ci" / "benchmark_config.json"),
)


def benchmark_models() -> list[str]:
    try:
        cfg = json.loads(BENCHMARK_CONFIG.read_text())
    except FileNotFoundError:
        return []
    return list(cfg.get("models", {}).keys())


# Security
PRODUCTION = os.environ.get("ET_JOBS_PRODUCTION", "0") == "1"
SUBMITTER_TOKEN = os.environ.get("SUBMITTER_TOKEN", "")
OPERATOR_TOKEN = os.environ.get("OPERATOR_TOKEN", "")
RATE_LIMIT_SUBMIT = _int("RATE_LIMIT_SUBMIT_PER_HOUR", 30)
RATE_LIMIT_READ = _int("RATE_LIMIT_READ_PER_MIN", 120)
MAX_QUEUE_DEPTH = _int("MAX_QUEUE_DEPTH", 500)
MAX_RUNNING_JOBS = _int("MAX_RUNNING_JOBS", 8)
MAX_SUBMITTER_JOBS_PER_HOUR = _int("MAX_SUBMITTER_JOBS_PER_HOUR", 10)

# Board / launcher
LAUNCHER = os.environ.get("LAUNCHER", "")
ET_INSTALL = os.environ.get("ET_INSTALL", "/opt/et")
ET_PLATFORM = os.environ.get("ET_PLATFORM", ET_INSTALL)
ET_PLATFORM_SRC = os.environ.get("ET_PLATFORM_SRC", "")
ET_LIB_PATH = os.environ.get("ET_LIB_PATH", "")
BOARD_LOCK = os.environ.get("BOARD_LOCK", "/var/lock/etsoc-shire0.lock")
SOC_RESET = os.environ.get(
    "SOC_RESET_SYSFS",
    "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/soc_reset/reinitiate",
)
DEFAULT_SHIRE = _int("DEFAULT_SHIRE", 0)
KERNEL_TIMEOUT = _int("KERNEL_TIMEOUT", 120)
MEM_SIZE = _int("MEM_SIZE", 16777216)
ZERO_BIN = os.environ.get("ZERO_BIN", "")

SMOKE_ELF = os.environ.get(
    "SMOKE_ELF", "/opt/et/kernels/histogram.erbium-soc1sim.elf"
)

SIM_ONLY_PUBLIC = os.environ.get("SIM_ONLY_PUBLIC", "1") == "1"
DRY_RUN = os.environ.get("ET_JOBS_DRY_RUN", "0") == "1"
BOARD_ENABLED = os.environ.get("ET_JOBS_BOARD_ENABLED", "0") == "1"

# Legacy single worker token (dev); prefer WORKER_TOKENS
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "change-me-in-production")
