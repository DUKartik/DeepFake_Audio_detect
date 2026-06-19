"""
webhook/server.py — FastAPI webhook server for WhatsApp Cloud API events.

Endpoints:
  GET  /webhook  — Meta verification handshake (hub.challenge echo)
  POST /webhook  — Incoming message events (audio messages → Celery task)
  GET  /health   — Health check (returns 200 OK)

Security:
  - GET  /webhook validates hub.verify_token against VERIFY_TOKEN env var.
  - POST /webhook validates X-Hub-Signature-256 HMAC-SHA256 against APP_SECRET.
  - Both return 403/401 immediately on mismatch.

Performance:
  - Webhook handler enqueues the Celery task and returns HTTP 200 within ~50 ms.
  - Meta drops webhook connections that take > 20 seconds.
"""

import os

import structlog
from fastapi import FastAPI, HTTPException, Request, Response

from processor.tasks import celery_app, handle_stop_command, process_audio
from utils.hashing import verify_hmac_sha256

log = structlog.get_logger(__name__)

app = FastAPI(
    title="VeriVoice Webhook",
    description="WhatsApp deepfake audio detection — webhook receiver",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint used by Render.com to keep the service alive.

    Returns:
        JSON {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/webhook")
async def verify(
    hub_mode: str,
    hub_challenge: str,
    hub_verify_token: str,
) -> Response:
    """Meta webhook verification handshake.

    Meta sends a GET request with hub.mode="subscribe",
    hub.challenge=<random_int>, and hub.verify_token=<your_token>.
    We must echo back hub.challenge as plain text if the token matches.

    Args:
        hub_mode:         Must be "subscribe".
        hub_challenge:    Random integer string from Meta to echo back.
        hub_verify_token: Must match the VERIFY_TOKEN env var.

    Returns:
        Plain text hub.challenge if token is valid.

    Raises:
        HTTPException 403: If the verify token does not match.
    """
    verify_token = os.environ.get("VERIFY_TOKEN", "")
    if hub_verify_token != verify_token:
        log.warning("webhook_verify_failed", token_head=hub_verify_token[:4])
        raise HTTPException(status_code=403, detail="Token mismatch")

    log.info("webhook_verified", mode=hub_mode)
    return Response(content=hub_challenge, media_type="text/plain")


@app.post("/webhook")
async def receive(request: Request) -> dict:
    """Receive incoming WhatsApp message events from Meta.

    Validates the HMAC-SHA256 signature, parses the payload, and
    enqueues a Celery task for audio messages. Non-audio messages and
    STOP commands are handled inline.

    Returns:
        JSON {"status": "ok"} immediately (before task completes).

    Raises:
        HTTPException 401: If the HMAC signature is invalid.
    """
    body = await request.body()

    # ── HMAC Signature Validation ──────────────────────────────────────────
    sig = request.headers.get("X-Hub-Signature-256", "")
    app_secret = os.environ.get("APP_SECRET", "")
    if not verify_hmac_sha256(app_secret, body, sig):
        log.warning("hmac_validation_failed", sig_head=sig[:12])
        raise HTTPException(status_code=401, detail="Invalid signature")

    # ── Parse Payload ─────────────────────────────────────────────────────
    data = await request.json()
    try:
        changes = data["entry"][0]["changes"][0]["value"]
        messages = changes.get("messages", [])
    except (KeyError, IndexError) as exc:
        log.warning("webhook_parse_error", error=str(exc))
        # Still return 200 — Meta will retry on non-200
        return {"status": "ok"}

    for msg in messages:
        sender: str = msg.get("from", "")
        msg_type: str = msg.get("type", "")

        # ── Handle STOP opt-out command ────────────────────────────────────
        if msg_type == "text":
            text_body = msg.get("text", {}).get("body", "").strip().upper()
            if text_body == "STOP":
                log.info("stop_command_received", sender=sender[:6] + "***")
                handle_stop_command.delay(sender)
                continue

        # ── Enqueue audio analysis task ────────────────────────────────────
        if msg_type == "audio":
            media_id: str = msg["audio"]["id"]
            log.info(
                "audio_message_received",
                sender=sender[:6] + "***",
                media_id=media_id,
            )
            process_audio.delay(media_id, sender)
        elif msg_type == "document":
            doc = msg.get("document", {})
            mime_type = doc.get("mime_type", "").lower()
            filename = doc.get("filename", "").lower()
            
            is_audio = (
                mime_type.startswith("audio/") or 
                mime_type in ["application/ogg", "video/mp4"] or 
                filename.endswith((".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac", ".wma"))
            )
            
            if is_audio:
                media_id: str = doc["id"]
                log.info(
                    "document_audio_received",
                    sender=sender[:6] + "***",
                    media_id=media_id,
                )
                process_audio.delay(media_id, sender)
            else:
                log.debug("non_audio_document_ignored", mime_type=mime_type, filename=filename)
        else:
            log.debug("non_audio_message_ignored", msg_type=msg_type)

    return {"status": "ok"}
