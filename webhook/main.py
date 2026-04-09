"""
webhook/main.py — Entry point that mounts both the webhook and admin routers.

Start the server:
    uvicorn webhook.main:app --host 0.0.0.0 --port $PORT
"""

import structlog
from fastapi import FastAPI

from webhook.server import app as webhook_app
from webhook.admin import router as admin_router
import detection.model_registry as registry

log = structlog.get_logger(__name__)

# Mount admin routes onto the webhook app
webhook_app.include_router(admin_router)
app = webhook_app


@app.on_event("startup")
async def startup() -> None:
    """Pre-load ML models when the API server starts (optional for CPU-only free tier).

    On the Celery worker, models are loaded by load_models() in the worker startup signal.
    On the webhook API server, we skip loading to keep boot time under 30 s.
    """
    log.info("verivoice_webhook_starting", version="1.0.0")
