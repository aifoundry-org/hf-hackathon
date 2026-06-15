from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import db
from .config import API_BIND, PRODUCTION, SUBMITTER_TOKEN, benchmark_models
from .security import (
    check_rate_limit,
    client_key_from_auth,
    validate_job_id,
    validate_production_config,
    validate_score,
    verify_api_read,
    verify_operator_token,
    verify_submitter_token,
    verify_worker_token,
)

_docs = None if PRODUCTION else "/docs"
_redoc = None if PRODUCTION else "/redoc"

app = FastAPI(
    title="ET-SoC1 Job API",
    version="0.2.0",
    description="Secure job queue for ET-SoC1 hackathon benchmarks.",
    docs_url=_docs,
    redoc_url=_redoc,
    openapi_url=None if PRODUCTION else "/openapi.json",
)


class CreateJob(BaseModel):
    job_type: str = Field(..., description="smoke_histogram | benchmark")
    model: str | None = None
    want_board: bool = False
    submitter: str = Field(..., min_length=1, max_length=128)


class ClaimRequest(BaseModel):
    pool: str
    host_id: str = Field(..., min_length=1, max_length=128)


class CompleteRequest(BaseModel):
    job_id: str
    pool: str
    ok: bool
    score: dict[str, Any] | None = None
    error: str | None = None


class AdvanceSimRequest(BaseModel):
    job_id: str
    want_board: bool = True


@app.on_event("startup")
def startup() -> None:
    validate_production_config()
    db.init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "et-jobs", "production": str(PRODUCTION).lower()}


@app.get("/v1/info")
def info() -> dict[str, Any]:
    return {
        "version": "0.2.0",
        "job_types": ["smoke_histogram", "benchmark"],
        "models": benchmark_models(),
        "pools": ["sim", "board"],
        "auth": {
            "submit": "Bearer SUBMITTER_TOKEN",
            "read": "Bearer SUBMITTER_TOKEN | OPERATOR_TOKEN | worker token",
            "worker": "Bearer per-host WORKER_TOKENS",
        },
    }


@app.post("/v1/jobs")
def create_job(
    body: CreateJob,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_submitter_token(authorization)
    check_rate_limit(client_key_from_auth(authorization), kind="submit")
    if body.job_type == "benchmark" and not body.model:
        raise HTTPException(400, "model required for benchmark jobs")
    if body.job_type not in ("smoke_histogram", "benchmark"):
        raise HTTPException(400, "unknown job_type")
    models = benchmark_models()
    if body.model and body.model not in models:
        raise HTTPException(400, "model must be one of: " + ", ".join(models))
    job = db.create_job(
        submitter=body.submitter,
        job_type=body.job_type,
        model=body.model,
        want_board=body.want_board,
    )
    return job


@app.get("/v1/jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_api_read(authorization)
    check_rate_limit(client_key_from_auth(authorization), kind="read")
    jobs = db.list_jobs(limit=limit, status=status)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/v1/jobs/{job_id}")
def get_job(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_api_read(authorization)
    check_rate_limit(client_key_from_auth(authorization), kind="read")
    job_id = validate_job_id(job_id)
    try:
        return db.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")


@app.get("/v1/jobs/{job_id}/logs/{stage}")
def get_log(
    job_id: str,
    stage: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_api_read(authorization)
    check_rate_limit(client_key_from_auth(authorization), kind="read")
    job_id = validate_job_id(job_id)
    if stage not in ("sim", "board"):
        raise HTTPException(400, "stage must be sim or board")
    try:
        job = db.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    path = job["logs"].get(stage)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "log not ready")
    tail = Path(path).read_text(errors="replace")[-200_000:]
    return {"job_id": job_id, "stage": stage, "tail": tail}


@app.post("/v1/internal/claim")
def claim(
    req: ClaimRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_worker_token(authorization, req.host_id)
    if req.pool not in ("sim", "board"):
        raise HTTPException(400, "pool must be sim or board")
    from . import config as cfg

    if db.count_running() >= cfg.MAX_RUNNING_JOBS:
        return Response(status_code=204)
    job = db.claim_job(pool=req.pool, host_id=req.host_id)
    if job is None:
        return Response(status_code=204)
    return job


@app.post("/v1/internal/complete")
def complete(
    req: CompleteRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    job_id = validate_job_id(req.job_id)
    try:
        job = db.get_job(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")

    verify_worker_token(authorization, job.get("host_id"))
    validate_score(req.pool, req.ok, req.score, job.get("host_id"))

    if req.pool == "sim":
        if req.ok:
            fields: dict[str, Any] = {
                "status": "sim_passed",
                "sim_score_json": req.score or {},
            }
            if job["want_board"]:
                fields["stage"] = "board"
            else:
                fields["stage"] = "done"
                fields["status"] = "completed"
            job = db.update_job(job_id, **fields)
        else:
            job = db.update_job(
                job_id,
                status="failed",
                stage="done",
                error=req.error or "sim failed",
                sim_score_json=req.score or {},
            )
    elif req.pool == "board":
        if req.ok and job["status"] not in ("running", "sim_passed"):
            raise HTTPException(409, "job not in board-runnable state")
        if req.ok:
            job = db.update_job(
                job_id,
                status="completed",
                stage="done",
                board_score_json=req.score or {},
            )
        else:
            job = db.update_job(
                job_id,
                status="failed",
                stage="done",
                error=req.error or "board failed",
                board_score_json=req.score or {},
            )
    else:
        raise HTTPException(400, "invalid pool")
    return job


@app.post("/v1/internal/operator/advance-sim")
def operator_advance_sim(
    req: AdvanceSimRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_operator_token(authorization)
    job_id = validate_job_id(req.job_id)
    try:
        return db.advance_sim_passed(job_id, want_board=req.want_board)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def main() -> None:
    import uvicorn

    host, _, port = API_BIND.partition(":")
    uvicorn.run(app, host=host or "127.0.0.1", port=int(port or 8080))


if __name__ == "__main__":
    main()
