from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    submitter TEXT NOT NULL,
    job_type TEXT NOT NULL,
    model TEXT,
    want_board INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    host_id TEXT,
    sim_score_json TEXT,
    board_score_json TEXT,
    error TEXT,
    meta_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, stage);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _count_jobs_with_status(*, statuses: tuple[str, ...]) -> int:
    placeholders = ",".join("?" for _ in statuses)
    with connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM jobs WHERE status IN ({placeholders})",
            statuses,
        ).fetchone()
    return int(row["c"]) if row else 0


def count_queued() -> int:
    return _count_jobs_with_status(statuses=("queued",))


def count_running() -> int:
    return _count_jobs_with_status(statuses=("running",))


def count_submitter_recent(submitter: str, *, hours: int = 1) -> int:
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    with connect() as conn:
        rows = conn.execute(
            "SELECT created_at FROM jobs WHERE submitter = ? ORDER BY created_at DESC LIMIT 50",
            (submitter,),
        ).fetchall()
    n = 0
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if ts >= cutoff:
            n += 1
    return n


def enforce_queue_limits(*, submitter: str) -> None:
    from fastapi import HTTPException

    if count_queued() >= config.MAX_QUEUE_DEPTH:
        raise HTTPException(503, "queue full, try again later")
    if count_running() >= config.MAX_RUNNING_JOBS:
        raise HTTPException(503, "too many jobs running, try again later")
    if count_submitter_recent(submitter) >= config.MAX_SUBMITTER_JOBS_PER_HOUR:
        raise HTTPException(429, "submitter hourly job limit exceeded")


def advance_sim_passed(job_id: str, *, want_board: bool = True) -> dict[str, Any]:
    """Operator-only: move job to board queue without running sim (lab hosts without sys_emu)."""
    job = get_job(job_id)
    if job["status"] not in ("queued",):
        raise ValueError(f"cannot advance job in status {job['status']}")
    fields: dict[str, Any] = {
        "want_board": int(want_board),
        "status": "sim_passed",
        "stage": "board" if want_board else "done",
    }
    if not want_board:
        fields["status"] = "completed"
    return update_job(job_id, **fields)


def create_job(
    *,
    submitter: str,
    job_type: str,
    model: str | None,
    want_board: bool,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enforce_queue_limits(submitter=submitter)
    if config.SIM_ONLY_PUBLIC and want_board:
        want_board = False
    jid = str(uuid.uuid4())
    now = _now()
    status = "queued"
    stage = "sim"
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, created_at, updated_at, submitter, job_type, model,
                want_board, status, stage, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                jid,
                now,
                now,
                submitter,
                job_type,
                model,
                int(want_board),
                status,
                stage,
                json.dumps(meta or {}),
            ),
        )
        conn.commit()
    return get_job(jid)


def list_jobs(*, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM jobs"
    args: list[Any] = []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [_row_to_job(r) for r in rows]


def get_job(job_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    return _row_to_job(row)


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    logs = job_log_dir(row["id"])
    out = {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "submitter": row["submitter"],
        "job_type": row["job_type"],
        "model": row["model"],
        "want_board": bool(row["want_board"]),
        "status": row["status"],
        "stage": row["stage"],
        "host_id": row["host_id"],
        "sim_score": json.loads(row["sim_score_json"]) if row["sim_score_json"] else None,
        "board_score": json.loads(row["board_score_json"]) if row["board_score_json"] else None,
        "error": row["error"],
        "meta": json.loads(row["meta_json"]) if row["meta_json"] else {},
        "logs": {
            "sim": str(logs / "sim.log") if (logs / "sim.log").is_file() else None,
            "board": str(logs / "board.log") if (logs / "board.log").is_file() else None,
        },
    }
    return out


def job_log_dir(job_id: str) -> Path:
    d = config.LOGS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    allowed = {
        "status",
        "stage",
        "want_board",
        "host_id",
        "sim_score_json",
        "board_score_json",
        "error",
    }
    parts = ["updated_at = ?"]
    values: list[Any] = [_now()]
    for key, val in fields.items():
        if key not in allowed:
            continue
        parts.append(f"{key} = ?")
        if key.endswith("_json") and val is not None and not isinstance(val, str):
            val = json.dumps(val)
        values.append(val)
    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(parts)} WHERE id = ?", values)
        conn.commit()
    return get_job(job_id)


def claim_job(*, pool: str, host_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        if pool == "sim":
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued' AND stage = 'sim'
                ORDER BY created_at LIMIT 1
                """
            ).fetchone()
        elif pool == "board":
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'sim_passed' AND stage = 'board' AND want_board = 1
                ORDER BY created_at LIMIT 1
                """
            ).fetchone()
        else:
            return None
        if row is None:
            return None
        jid = row["id"]
        conn.execute(
            """
            UPDATE jobs SET status = 'running', host_id = ?, updated_at = ?
            WHERE id = ? AND status IN ('queued', 'sim_passed')
            """,
            (host_id, _now(), jid),
        )
        if conn.total_changes != 1:
            conn.rollback()
            return None
        conn.commit()
    return get_job(jid)
