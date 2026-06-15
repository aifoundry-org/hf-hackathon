from __future__ import annotations

import hashlib
import os
import re
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from fastapi import Header, HTTPException

from . import config

_UNSAFE_DEFAULTS = frozenset(
    {
        "change-me-in-production",
        "dev-token-change-me",
        "local-dev-worker-token-change-me",
        "local-dev-submitter-token",
        "generate-a-long-random-token",
    }
)


def parse_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return ""
    return authorization[7:].strip()


def parse_worker_tokens() -> dict[str, str]:
    """host_id -> token. WORKER_TOKENS=host1:tok1,host2:tok2"""
    raw = os.environ.get("WORKER_TOKENS", "").strip()
    out: dict[str, str] = {}
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            host, tok = part.split(":", 1)
            out[host.strip()] = tok.strip()
    legacy = os.environ.get("WORKER_TOKEN", "").strip()
    if legacy and not out:
        out["*"] = legacy
    return out


def worker_tokens() -> dict[str, str]:
    return parse_worker_tokens()


def validate_production_config() -> None:
    if not config.PRODUCTION:
        return
    errors: list[str] = []
    if not config.SUBMITTER_TOKEN or len(config.SUBMITTER_TOKEN) < 32:
        errors.append("SUBMITTER_TOKEN must be set (>=32 chars) in production")
    if config.SUBMITTER_TOKEN in _UNSAFE_DEFAULTS:
        errors.append("SUBMITTER_TOKEN is a known unsafe default")
    tokens = worker_tokens()
    if not tokens:
        errors.append("WORKER_TOKENS must be set in production (host:token,...)")
    if "*" in tokens:
        errors.append("wildcard WORKER_TOKEN not allowed in production")
    for host, tok in tokens.items():
        if len(tok) < 32:
            errors.append(f"worker token for {host} too short")
        if tok in _UNSAFE_DEFAULTS:
            errors.append(f"worker token for {host} is unsafe default")
    if not config.OPERATOR_TOKEN or len(config.OPERATOR_TOKEN) < 32:
        errors.append("OPERATOR_TOKEN must be set (>=32 chars) in production")
    if config.DRY_RUN:
        errors.append("ET_JOBS_DRY_RUN must be 0 in production")
    bind_host = config.API_BIND.split(":")[0]
    if bind_host in ("0.0.0.0", "::"):
        errors.append("JOBS_BIND must not be 0.0.0.0 in production (use reverse proxy)")
    if errors:
        raise RuntimeError("production config invalid: " + "; ".join(errors))


def worker_token_for_host(host_id: str) -> str:
    tokens = worker_tokens()
    tok = tokens.get(host_id)
    if tok:
        return tok
    wildcard = tokens.get("*")
    if wildcard:
        return wildcard
    return ""


def verify_worker_token(authorization: str | None, host_id: str | None = None) -> str:
    token = parse_bearer(authorization)
    if not token:
        raise HTTPException(401, "Bearer token required")
    tokens = worker_tokens()
    if host_id:
        expected = tokens.get(host_id)
        if expected and token == expected:
            return host_id
    wildcard = tokens.get("*")
    if wildcard and token == wildcard:
        return host_id or "*"
    if token in tokens.values():
        # token valid but wrong host
        raise HTTPException(403, "token not valid for this host_id")
    raise HTTPException(403, "invalid worker token")


def verify_submitter_token(authorization: str | None) -> None:
    if not config.SUBMITTER_TOKEN:
        if config.PRODUCTION:
            raise HTTPException(500, "submitter auth not configured")
        return
    token = parse_bearer(authorization)
    if token != config.SUBMITTER_TOKEN:
        raise HTTPException(403 if token else 401, "invalid or missing submitter token")


def verify_operator_token(authorization: str | None) -> None:
    if not config.OPERATOR_TOKEN:
        raise HTTPException(501, "operator API disabled")
    token = parse_bearer(authorization)
    if token != config.OPERATOR_TOKEN:
        raise HTTPException(403 if token else 401, "invalid or missing operator token")


def verify_api_read(authorization: str | None) -> None:
    """Submitter, operator, or any registered worker may read jobs/logs."""
    token = parse_bearer(authorization)
    if not token:
        raise HTTPException(401, "Bearer token required for job access")
    if config.SUBMITTER_TOKEN and token == config.SUBMITTER_TOKEN:
        return
    if config.OPERATOR_TOKEN and token == config.OPERATOR_TOKEN:
        return
    if token in worker_tokens().values():
        return
    raise HTTPException(403, "invalid token for job access")


def validate_score(pool: str, ok: bool, score: dict[str, Any] | None, host_id: str | None) -> None:
    if not ok:
        return
    if score is None:
        raise HTTPException(400, "score required when ok=true")
    if os.environ.get("ET_JOBS_DRY_RUN", "0") == "1":
        return
    wait = score.get("kernel_wait_s")
    tokens_per_second = score.get("tokens_per_second")
    has_kernel_metric = isinstance(wait, (int, float)) and wait >= 0
    has_token_metric = isinstance(tokens_per_second, (int, float)) and tokens_per_second > 0
    if not has_kernel_metric and not has_token_metric:
        raise HTTPException(400, "score must include non-negative kernel_wait_s or positive tokens_per_second")
    score_host = score.get("host_id")
    if host_id and score_host and score_host != host_id:
        raise HTTPException(400, "score host_id must match claiming host")


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, *, limit: int, window_s: float) -> bool:
        now = time.time()
        cutoff = now - window_s
        with self._lock:
            bucket = self._events[key]
            self._events[key] = [t for t in bucket if t > cutoff]
            if len(self._events[key]) >= limit:
                return False
            self._events[key].append(now)
            return True


_limiter = SlidingWindowLimiter()


def check_rate_limit(client_key: str, *, kind: str) -> None:
    if kind == "submit":
        limit = config.RATE_LIMIT_SUBMIT
        window = 3600.0
    else:
        limit = config.RATE_LIMIT_READ
        window = 60.0
    if not _limiter.allow(f"{kind}:{client_key}", limit=limit, window_s=window):
        raise HTTPException(429, f"rate limit exceeded for {kind}")


def client_key_from_auth(authorization: str | None) -> str:
    tok = parse_bearer(authorization)
    if tok:
        return hashlib.sha256(tok.encode()).hexdigest()[:16]
    return "anonymous"


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def validate_job_id(job_id: str) -> str:
    if not _UUID_RE.match(job_id):
        raise HTTPException(400, "invalid job_id")
    return job_id
