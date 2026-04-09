"""
tests/test_webhook.py — Unit tests for the FastAPI webhook server.

Covers:
  - test_verify_webhook_valid_token()
  - test_verify_webhook_invalid_token_returns_403()
  - test_receive_audio_message_enqueues_task()
  - test_receive_non_audio_message_is_ignored()
  - test_invalid_hmac_returns_401()
  - test_stop_command_triggers_delete_task()

All WhatsApp API calls and Celery tasks are mocked.
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set required env vars before importing the app
import os
os.environ.setdefault("WA_TOKEN", "test_wa_token")
os.environ.setdefault("PHONE_NUMBER_ID", "12345")
os.environ.setdefault("VERIFY_TOKEN", "test_verify_token")
os.environ.setdefault("APP_SECRET", "test_app_secret")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("R2_BUCKET", "test-bucket")
os.environ.setdefault("R2_ENDPOINT_URL", "")
os.environ.setdefault("R2_ACCESS_KEY_ID", "")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "")

from webhook.server import app

client = TestClient(app)


def _make_signature(body: bytes, secret: str = "test_app_secret") -> str:
    """Compute a valid X-Hub-Signature-256 header value."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _audio_payload(media_id: str = "media_123", sender: str = "919876543210") -> dict:
    """Return a minimal WhatsApp audio message webhook payload."""
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": sender,
                        "type": "audio",
                        "audio": {"id": media_id},
                    }]
                }
            }]
        }]
    }


def _text_payload(text: str, sender: str = "919876543210") -> dict:
    """Return a minimal WhatsApp text message webhook payload."""
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": sender,
                        "type": "text",
                        "text": {"body": text},
                    }]
                }
            }]
        }]
    }


# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_verify_webhook_valid_token():
    resp = client.get("/webhook", params={
        "hub_mode": "subscribe",
        "hub_challenge": "123456",
        "hub_verify_token": "test_verify_token",
    })
    assert resp.status_code == 200
    assert resp.text == "123456"


def test_verify_webhook_invalid_token_returns_403():
    resp = client.get("/webhook", params={
        "hub_mode": "subscribe",
        "hub_challenge": "123456",
        "hub_verify_token": "wrong_token",
    })
    assert resp.status_code == 403


@patch("webhook.server.process_audio")
def test_receive_audio_message_enqueues_task(mock_task):
    """POST with valid HMAC and audio message should enqueue process_audio."""
    payload = _audio_payload("media_abc", "919876543210")
    body = json.dumps(payload).encode()
    sig = _make_signature(body)

    resp = client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    mock_task.delay.assert_called_once_with("media_abc", "919876543210")


@patch("webhook.server.process_audio")
def test_receive_non_audio_message_is_ignored(mock_task):
    """Non-audio messages should not trigger process_audio."""
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "919876543210",
                        "type": "image",
                        "image": {"id": "img_123"},
                    }]
                }
            }]
        }]
    }
    body = json.dumps(payload).encode()
    sig = _make_signature(body)

    resp = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 200
    mock_task.delay.assert_not_called()


def test_invalid_hmac_returns_401():
    """Requests with a wrong HMAC signature must be rejected with 401."""
    payload = _audio_payload()
    body = json.dumps(payload).encode()

    resp = client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=badbadbadbad",
        },
    )
    assert resp.status_code == 401


@patch("webhook.server.handle_stop_command")
def test_stop_command_triggers_delete_task(mock_stop):
    """Sending text 'STOP' should trigger handle_stop_command task."""
    payload = _text_payload("STOP", "919876543210")
    body = json.dumps(payload).encode()
    sig = _make_signature(body)

    resp = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 200
    mock_stop.delay.assert_called_once_with("919876543210")
