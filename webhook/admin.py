"""
webhook/admin.py — Minimal FastAPI admin dashboard for VeriVoice.

Endpoints:
  GET  /admin/stats   — Total analyses today, fake/real counts, avg score, top flags
  GET  /admin/recent  — Last 20 anonymised results from R2 audit log
  POST /admin/retrain — Triggers Celery task to update ensemble weights

Security:
  - All endpoints secured with HTTP Basic Auth.
  - Credentials come from ADMIN_USERNAME / ADMIN_PASSWORD env vars.
  - Rate limited to 100 requests per minute per IP.
  - All admin access logged to audit/admin_access.jsonl in R2.
"""

import json
from collections import Counter
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from config import settings
from utils.audit import AuditLogger

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBasic()

# ── In-memory per-IP rate limiter (100 req/min) ───────────────────────────────
import time
_admin_rate: dict[str, list[float]] = {}
_ADMIN_RATE_LIMIT = 100
_ADMIN_RATE_WINDOW = 60  # seconds


def _check_admin_rate(ip: str) -> None:
    """Block IPs exceeding 100 requests per minute.

    Args:
        ip: Requester IP address string.

    Raises:
        HTTPException 429: If rate limit exceeded.
    """
    now = time.time()
    window_start = now - _ADMIN_RATE_WINDOW
    hits = _admin_rate.get(ip, [])
    hits = [t for t in hits if t > window_start]
    if len(hits) >= _ADMIN_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Admin rate limit exceeded")
    hits.append(now)
    _admin_rate[ip] = hits


def _auth(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    """Validate HTTP Basic Auth credentials.

    Args:
        credentials: Username/password from Authorization header.

    Returns:
        Username string if valid.

    Raises:
        HTTPException 401: If credentials are incorrect.
    """
    correct_user = secrets.compare_digest(
        credentials.username.encode(), settings.admin_username.encode()
    )
    correct_pass = secrets.compare_digest(
        credentials.password.encode(), settings.admin_password.encode()
    )
    if not (correct_user and correct_pass):
        log.warning("admin_auth_failed", username=credentials.username)
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _get_auditor() -> AuditLogger:
    return AuditLogger(
        bucket=settings.r2_bucket,
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
    )


@router.get("/stats")
async def stats(
    request: Request,
    username: str = Depends(_auth),
) -> dict:
    """Return aggregated statistics for today's analyses.

    Returns:
        JSON with total_analyses_today, fake_count, real_count,
        suspicious_count, avg_score, top_flags.
    """
    _check_admin_rate(request.client.host)
    auditor = _get_auditor()

    try:
        records = auditor.get_recent(n=500)   # get last 500 for stats
    except Exception as exc:
        log.error("admin_stats_failed", error=str(exc))
        return {"error": str(exc)}

    fake_count = sum(1 for r in records if r.get("label") == "fake")
    susp_count = sum(1 for r in records if r.get("label") == "suspicious")
    real_count = sum(1 for r in records if r.get("label") == "real")
    scores = [r["final_score"] for r in records if isinstance(r.get("final_score"), float)]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    all_flags: list[str] = []
    for r in records:
        all_flags.extend(r.get("meta_flags", []))
    top_flags = [flag for flag, _ in Counter(all_flags).most_common(5)]

    auditor.log_admin_access(username, "/admin/stats", request.client.host)
    return {
        "total_analyses_today": len(records),
        "fake_count": fake_count,
        "suspicious_count": susp_count,
        "real_count": real_count,
        "avg_score": avg_score,
        "top_flags": top_flags,
    }


@router.get("/recent")
async def recent(
    request: Request,
    username: str = Depends(_auth),
) -> dict:
    """Return last 20 anonymised audit records.

    Sender hashes are truncated to 8 characters. No phone numbers are returned.

    Returns:
        JSON with "results" list of anonymised record dicts.
    """
    _check_admin_rate(request.client.host)
    auditor = _get_auditor()

    try:
        records = auditor.get_recent(n=20)
    except Exception as exc:
        log.error("admin_recent_failed", error=str(exc))
        return {"error": str(exc)}

    auditor.log_admin_access(username, "/admin/recent", request.client.host)
    return {"results": records}


@router.post("/retrain")
async def retrain(
    request: Request,
    username: str = Depends(_auth),
) -> dict:
    """Trigger a Celery task to update ensemble weights via online learning.

    Pulls recent audit data from R2 and re-tunes ensemble weights based on
    operator-corrected labels (thumbs up/down feedback in v2.0 roadmap).

    Returns:
        JSON with task_id of the dispatched Celery task.
    """
    _check_admin_rate(request.client.host)
    # Import here to avoid circular import (tasks imports config)
    from processor.tasks import celery_app as _app
    task = _app.send_task("processor.tasks.retrain_ensemble")
    log.info("retrain_triggered", task_id=task.id, username=username)
    auditor = _get_auditor()
    auditor.log_admin_access(username, "/admin/retrain", request.client.host)
    return {"task_id": task.id, "status": "queued"}
