from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


def test_health_public(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200


def test_read_jobs_requires_auth(client: TestClient):
    assert client.get("/v1/jobs").status_code == 401
    assert client.get("/v1/jobs", headers={"Authorization": "Bearer bad"}).status_code == 403


def test_submit_requires_auth(client: TestClient):
    body = {"job_type": "smoke_histogram", "submitter": "t", "want_board": False}
    assert client.post("/v1/jobs", json=body).status_code == 401


def test_submit_and_read(client: TestClient, submitter_headers: dict):
    r = client.post(
        "/v1/jobs",
        json={"job_type": "smoke_histogram", "submitter": "pytest", "want_board": False},
        headers=submitter_headers,
    )
    assert r.status_code == 200
    jid = r.json()["id"]
    g = client.get(f"/v1/jobs/{jid}", headers=submitter_headers)
    assert g.status_code == 200
    assert g.json()["submitter"] == "pytest"


def test_worker_claim_requires_token(client: TestClient):
    r = client.post(
        "/v1/internal/claim",
        json={"pool": "sim", "host_id": "sim-host"},
    )
    assert r.status_code == 401


def test_worker_wrong_host_token(client: TestClient, board_worker_headers: dict):
    r = client.post(
        "/v1/internal/claim",
        json={"pool": "sim", "host_id": "sim-host"},
        headers=board_worker_headers,
    )
    assert r.status_code == 403


def test_worker_claim_sim(client: TestClient, submitter_headers: dict, sim_worker_headers: dict):
    cr = client.post(
        "/v1/jobs",
        json={"job_type": "smoke_histogram", "submitter": "w", "want_board": False},
        headers=submitter_headers,
    )
    jid = cr.json()["id"]
    r = client.post(
        "/v1/internal/claim",
        json={"pool": "sim", "host_id": "sim-host"},
        headers=sim_worker_headers,
    )
    assert r.status_code == 200
    claimed = r.json()
    assert claimed["id"] == jid
    assert claimed["status"] == "running"


def test_validate_score_rejects_missing_wait(monkeypatch):
    from fastapi import HTTPException

    from et_jobs.security import validate_score

    monkeypatch.setenv("ET_JOBS_DRY_RUN", "0")
    with pytest.raises(HTTPException) as exc:
        validate_score("sim", True, {}, "sim-host")
    assert exc.value.status_code == 400


def test_complete_with_valid_score(
    client: TestClient, submitter_headers: dict, sim_worker_headers: dict, monkeypatch
):
    monkeypatch.setenv("ET_JOBS_DRY_RUN", "0")
    cr = client.post(
        "/v1/jobs",
        json={"job_type": "smoke_histogram", "submitter": "s", "want_board": False},
        headers=submitter_headers,
    )
    jid = cr.json()["id"]
    client.post(
        "/v1/internal/claim",
        json={"pool": "sim", "host_id": "sim-host"},
        headers=sim_worker_headers,
    )
    good = client.post(
        "/v1/internal/complete",
        json={
            "job_id": jid,
            "pool": "sim",
            "ok": True,
            "score": {"kernel_wait_s": 1.0, "host_id": "sim-host"},
        },
        headers=sim_worker_headers,
    )
    assert good.status_code == 200


def test_operator_advance(client: TestClient, submitter_headers: dict, operator_headers: dict):
    cr = client.post(
        "/v1/jobs",
        json={"job_type": "smoke_histogram", "submitter": "o", "want_board": True},
        headers=submitter_headers,
    )
    jid = cr.json()["id"]
    adv = client.post(
        "/v1/internal/operator/advance-sim",
        json={"job_id": jid, "want_board": True},
        headers=operator_headers,
    )
    assert adv.status_code == 200
    assert adv.json()["status"] == "sim_passed"
    assert adv.json()["stage"] == "board"


def test_operator_advance_enables_board_after_sim_only_public(
    client: TestClient,
    submitter_headers: dict,
    operator_headers: dict,
    board_worker_headers: dict,
    monkeypatch,
):
    import et_jobs.config as cfg

    monkeypatch.setattr(cfg, "SIM_ONLY_PUBLIC", True)
    monkeypatch.setattr("et_jobs.db.config", cfg)
    cr = client.post(
        "/v1/jobs",
        json={"job_type": "smoke_histogram", "submitter": "o", "want_board": True},
        headers=submitter_headers,
    )
    assert cr.status_code == 200
    assert cr.json()["want_board"] is False
    jid = cr.json()["id"]

    adv = client.post(
        "/v1/internal/operator/advance-sim",
        json={"job_id": jid, "want_board": True},
        headers=operator_headers,
    )
    assert adv.status_code == 200
    assert adv.json()["want_board"] is True
    assert adv.json()["status"] == "sim_passed"
    assert adv.json()["stage"] == "board"

    claim = client.post(
        "/v1/internal/claim",
        json={"pool": "board", "host_id": "board-host"},
        headers=board_worker_headers,
    )
    assert claim.status_code == 200
    assert claim.json()["id"] == jid


def test_invalid_job_id(client: TestClient, submitter_headers: dict):
    r = client.get("/v1/jobs/not-a-uuid", headers=submitter_headers)
    assert r.status_code == 400


def test_no_upload_endpoint(client: TestClient):
    assert client.post("/v1/upload").status_code in (404, 405)


def test_production_config_rejects_defaults():
    import importlib

    import et_jobs.config as cfg
    import et_jobs.security as sec

    saved = {
        "ET_JOBS_PRODUCTION": os.environ.get("ET_JOBS_PRODUCTION"),
        "SUBMITTER_TOKEN": os.environ.get("SUBMITTER_TOKEN"),
        "WORKER_TOKENS": os.environ.get("WORKER_TOKENS"),
    }
    try:
        os.environ["ET_JOBS_PRODUCTION"] = "1"
        os.environ["SUBMITTER_TOKEN"] = "local-dev-submitter-token"
        os.environ["WORKER_TOKENS"] = "h:short"
        importlib.reload(cfg)
        importlib.reload(sec)
        with pytest.raises(RuntimeError, match="production config invalid"):
            sec.validate_production_config()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(cfg)
        importlib.reload(sec)


def test_rate_limit_submit(client: TestClient, submitter_headers: dict, monkeypatch):
    import et_jobs.config as cfg

    monkeypatch.setattr(cfg, "RATE_LIMIT_SUBMIT", 2)
    body = {"job_type": "smoke_histogram", "submitter": "rl", "want_board": False}
    assert client.post("/v1/jobs", json=body, headers=submitter_headers).status_code == 200
    assert client.post("/v1/jobs", json=body, headers=submitter_headers).status_code == 200
    assert client.post("/v1/jobs", json=body, headers=submitter_headers).status_code == 429


def test_queue_full(client: TestClient, submitter_headers: dict, monkeypatch):
    import et_jobs.config as cfg

    monkeypatch.setattr(cfg, "MAX_QUEUE_DEPTH", 1)
    monkeypatch.setattr("et_jobs.db.config", cfg)
    body = {"job_type": "smoke_histogram", "submitter": "q", "want_board": False}
    assert client.post("/v1/jobs", json=body, headers=submitter_headers).status_code == 200
    assert client.post("/v1/jobs", json=body, headers=submitter_headers).status_code == 503


def test_docs_hidden_in_production(monkeypatch):
    import importlib

    monkeypatch.setenv("ET_JOBS_PRODUCTION", "1")
    monkeypatch.setenv("SUBMITTER_TOKEN", "x" * 32)
    monkeypatch.setenv("OPERATOR_TOKEN", "y" * 32)
    monkeypatch.setenv("WORKER_TOKENS", f"sim:{'a' * 32}")
    monkeypatch.setenv("ET_JOBS_DRY_RUN", "0")
    import et_jobs.config as cfg
    import et_jobs.security as sec

    importlib.reload(cfg)
    importlib.reload(sec)
    sec.validate_production_config()
    import et_jobs.api as api_mod

    importlib.reload(api_mod)
    assert api_mod.app.docs_url is None
