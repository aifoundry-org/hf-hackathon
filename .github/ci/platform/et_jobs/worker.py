from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

from . import config
from .runner import parse_kernel_wait
from .db import job_log_dir
from .security import worker_token_for_host


def _api(method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
    base = config.API_URL
    url = f"{base.rstrip('/')}{path}"
    data = None
    token = worker_token_for_host(config.HOST_ID)
    if not token:
        raise RuntimeError(f"no WORKER_TOKENS entry for HOST_ID={config.HOST_ID}")
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        if e.code == 204:
            return 204, ""
        try:
            return e.code, json.loads(raw) if raw else {"detail": e.reason}
        except json.JSONDecodeError:
            return e.code, raw


def claim(pool: str) -> dict[str, Any] | None:
    code, payload = _api(
        "POST",
        "/v1/internal/claim",
        {"pool": pool, "host_id": config.HOST_ID},
    )
    if code == 204:
        return None
    if code != 200:
        raise RuntimeError(f"claim failed {code}: {payload}")
    return payload


def complete(pool: str, job_id: str, ok: bool, score: dict | None, error: str | None) -> None:
    code, payload = _api(
        "POST",
        "/v1/internal/complete",
        {
            "job_id": job_id,
            "pool": pool,
            "ok": ok,
            "score": score,
            "error": error,
        },
    )
    if code != 200:
        raise RuntimeError(f"complete failed {code}: {payload}")


def execute_job(job: dict[str, Any], pool: str) -> tuple[bool, dict[str, Any], str | None]:
    jid = job["id"]
    cmd = [
        sys.executable,
        "-m",
        "et_jobs.runner",
        "--job-id",
        jid,
        "--stage",
        pool,
        "--job-type",
        job["job_type"],
    ]
    if job.get("model"):
        cmd.extend(["--model", job["model"]])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = job_log_dir(jid) / f"{pool}.log"
    wait_s = parse_kernel_wait(log_path)
    score: dict[str, Any] = {"kernel_wait_s": wait_s, "host_id": config.HOST_ID}
    if proc.stdout.strip():
        try:
            score.update(json.loads(proc.stdout))
        except json.JSONDecodeError:
            pass
    nested = score.get("score") if isinstance(score.get("score"), dict) else {}
    passed = score.get("passed") or nested.get("passed")
    ok = (proc.returncode == 0 and wait_s is not None) or passed is True
    if nested:
        score.update(nested)
    err = None if ok else (proc.stderr or proc.stdout or "run failed")[:2000]
    return ok, score, err


def board_available() -> bool:
    return Path("/dev/et0_mgmt").exists() or config.DRY_RUN


def loop(pool: str, poll_s: float = 5.0) -> None:
    if pool == "board" and not board_available():
        print("board pool disabled: /dev/et0_mgmt missing (set ET_JOBS_DRY_RUN=1 to test)", flush=True)
        while True:
            time.sleep(60)
    print(f"worker pool={pool} host={config.HOST_ID} api={config.API_URL} dry_run={config.DRY_RUN}", flush=True)
    while True:
        try:
            job = claim(pool)
            if job is None:
                time.sleep(poll_s)
                continue
            print(f"claimed {job['id']} type={job['job_type']}", flush=True)
            ok, score, err = execute_job(job, pool)
            complete(pool, job["id"], ok, score, err)
            print(f"finished {job['id']} ok={ok} wait={score.get('kernel_wait_s')}", flush=True)
        except Exception as exc:
            print(f"worker error: {exc}", flush=True)
            time.sleep(poll_s)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ET job worker")
    p.add_argument("--pool", choices=["sim", "board"], required=True)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    if args.once:
        job = claim(args.pool)
        if not job:
            print("no job")
            return
        ok, score, err = execute_job(job, args.pool)
        complete(args.pool, job["id"], ok, score, err)
        print(json.dumps({"ok": ok, "score": score}, indent=2))
        return
    loop(args.pool)


if __name__ == "__main__":
    main()
