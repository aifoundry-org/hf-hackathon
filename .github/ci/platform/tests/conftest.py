from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Test env before importing et_jobs
_test_root = tempfile.mkdtemp(prefix="et-jobs-test-")
os.environ["JOBS_DATA_DIR"] = _test_root
os.environ["ET_JOBS_PRODUCTION"] = "0"
os.environ["ET_JOBS_DRY_RUN"] = "1"
os.environ["SUBMITTER_TOKEN"] = "test-submitter-token-32chars-minimumxx"
os.environ["OPERATOR_TOKEN"] = "test-operator-token-32chars-minimumxxx"
os.environ["WORKER_TOKENS"] = "sim-host:test-worker-sim-token-32chars-minimum,board-host:test-worker-board-token-32chars"
os.environ["WORKER_TOKEN"] = ""
os.environ["RATE_LIMIT_SUBMIT_PER_HOUR"] = "1000"
os.environ["RATE_LIMIT_READ_PER_MIN"] = "10000"
os.environ["MAX_QUEUE_DEPTH"] = "100"
os.environ["MAX_RUNNING_JOBS"] = "10"
os.environ["MAX_SUBMITTER_JOBS_PER_HOUR"] = "100"
os.environ["SIM_ONLY_PUBLIC"] = "0"

# Reload config-dependent modules
import importlib

import et_jobs.config as config_mod
import et_jobs.security as security_mod

import et_jobs.api as api_mod
import et_jobs.db as db_mod

importlib.reload(config_mod)
importlib.reload(security_mod)
importlib.reload(db_mod)
importlib.reload(api_mod)

app = api_mod.app
db = db_mod
db.init_db()


@pytest.fixture(autouse=True)
def clean_jobs_table():
    with db.connect() as conn:
        conn.execute("DELETE FROM jobs")
        conn.commit()
    from et_jobs import security

    security._limiter._events.clear()
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def submitter_headers():
    return {"Authorization": "Bearer test-submitter-token-32chars-minimumxx"}


@pytest.fixture
def sim_worker_headers():
    return {"Authorization": "Bearer test-worker-sim-token-32chars-minimum"}


@pytest.fixture
def board_worker_headers():
    return {"Authorization": "Bearer test-worker-board-token-32chars"}


@pytest.fixture
def operator_headers():
    return {"Authorization": "Bearer test-operator-token-32chars-minimumxxx"}
