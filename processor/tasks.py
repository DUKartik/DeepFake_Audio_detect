"""
processor/tasks.py — Celery worker tasks for the full audio analysis pipeline.

Pipeline per task:
  1. Check rate limit for sender
  2. Download audio from Meta CDN
  3. Convert ogg/opus → 16 kHz mono WAV via ffmpeg
  4. Check clip duration (< 3 s → low-confidence warning)
  5. Run detectors in sequence (AASIST/XLS-R, wav2vec/phase, metadata)
  6. Compute weighted ensemble score
  7. Send verdict reply via WhatsApp
  8. Write audit log to Cloudflare R2
  9. Increment rate-limit counter
  10. Clean up temp files
"""

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import structlog
from celery import Celery

import detection.model_registry as registry
from client.whatsapp import WhatsAppClient
from config import settings
from detection.ensemble import ensemble_score_cpu, ensemble_score_gpu, phase_discontinuity_score
from utils.audit import AuditLogger
from utils.exceptions import (
    AudioConversionError,
    AudioDownloadError,
    RateLimitExceededError,
)
from utils.hashing import hash_sender
from utils.rate_limiter import RateLimiter

log = structlog.get_logger(__name__)

# ── Celery app configured against Upstash Redis ───────────────────────────────
celery_app = Celery(
    "verivoice",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,   # one task at a time — models are large
    task_acks_late=True,
)


# ── Lazy singletons (initialised on first worker task) ────────────────────────
_wa_client: WhatsAppClient | None = None
_rate_limiter: RateLimiter | None = None
_audit_logger: AuditLogger | None = None


def _get_wa_client() -> WhatsAppClient:
    global _wa_client
    if _wa_client is None:
        _wa_client = WhatsAppClient(
            token=settings.wa_token,
            phone_id=settings.phone_number_id,
            fact_checker_url=settings.fact_checker_url,
        )
    return _wa_client


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            redis_url=settings.redis_url,
            daily_limit=settings.rate_limit_daily,
        )
    return _rate_limiter


def _get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(
            bucket=settings.r2_bucket,
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
        )
    return _audit_logger


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any audio file to 16 kHz mono WAV using ffmpeg.

    Args:
        input_path:  Source audio file (ogg/opus from WhatsApp).
        output_path: Destination WAV file path.

    Raises:
        AudioConversionError: If ffmpeg fails.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise AudioConversionError(
            f"ffmpeg conversion failed: {result.stderr.decode('utf-8', errors='replace')[:300]}"
        )


def _get_duration(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


@celery_app.task(
    name="processor.tasks.process_audio",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def process_audio(self, media_id: str, sender: str) -> dict:
    """Full audio analysis pipeline — Celery task entry point.

    Args:
        media_id: WhatsApp media object ID from the webhook event.
        sender:   Sender's phone number in E.164 format.

    Returns:
        dict with label, final_score, and latency_ms.
    """
    t_start = time.time()
    sender_hash = hash_sender(sender)
    log.bind(sender=sender_hash[:8], media_id=media_id)

    wa = _get_wa_client()
    limiter = _get_rate_limiter()
    auditor = _get_audit_logger()

    # ── Step 1: Rate limit check ───────────────────────────────────────────
    if not limiter.check(sender_hash):
        log.warning("rate_limit_exceeded", sender=sender_hash[:8])
        wa.send_rate_limit_message(sender)
        return {"label": "rate_limited", "final_score": -1, "latency_ms": 0}

    tmp_dir = Path(settings.tmp_dir) / sender_hash[:8]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ogg_path = str(tmp_dir / f"{media_id}.ogg")
    wav_path = str(tmp_dir / f"{media_id}.wav")

    try:
        # ── Step 2: Download audio ─────────────────────────────────────────
        log.info("downloading_audio")
        download_url = wa.get_media_url(media_id)
        audio_bytes = wa.download(download_url)
        with open(ogg_path, "wb") as f:
            f.write(audio_bytes)

        # ── Step 3: Convert to 16 kHz mono WAV ────────────────────────────
        log.info("converting_audio")
        _convert_to_wav(ogg_path, wav_path)

        # ── Step 4: Duration check ─────────────────────────────────────────
        duration = _get_duration(wav_path)
        if duration < 3.0:
            log.warning("clip_too_short", duration=duration)
            wa.send_low_confidence_warning(sender)
            return {"label": "too_short", "final_score": -1, "latency_ms": 0}

        # ── Step 5 & 6: Run detectors + ensemble ──────────────────────────
        log.info("running_detectors", device=registry.device())
        meta_detector = registry.get("meta")
        # Run meta_detector on the RAW downloaded file (ogg_path) to preserve original metadata
        meta_result = meta_detector.analyse(ogg_path)

        if registry.device() == "cuda":
            # GPU path: XLS-R + AASIST + phase
            xlsr_detector = registry.get("xlsr_aasist")
            xlsr_result = xlsr_detector.predict(wav_path)
            p_score = phase_discontinuity_score(wav_path)
            ensemble = ensemble_score_gpu(xlsr_result, p_score, meta_result)
        else:
            # CPU path: AASIST + wav2vec2 + metadata
            aasist_detector = registry.get("aasist")
            aasist_result = aasist_detector.predict(wav_path)
            wav2vec_score = 0.0
            if "wav2vec" in registry._REGISTRY:
                wav2vec_score = registry.get("wav2vec").predict(wav_path)
            ensemble = ensemble_score_cpu(aasist_result, wav2vec_score, meta_result)

        # ── Step 7: Send verdict ───────────────────────────────────────────
        total_ms = int((time.time() - t_start) * 1000)
        log.info(
            "verdict",
            label=ensemble.label,
            score=ensemble.final_score,
            latency_ms=total_ms,
        )
        wa.send_verdict(
            to=sender,
            label=ensemble.label,
            confidence_pct=ensemble.confidence_pct,
            meta_flags=ensemble.meta_flags,
        )

        # ── Step 8: Audit log ──────────────────────────────────────────────
        try:
            auditor.log_result(
                sender_hash=sender_hash,
                media_id=media_id,
                final_score=ensemble.final_score,
                label=ensemble.label,
                model_scores=ensemble.model_scores,
                meta_flags=ensemble.meta_flags,
                latency_ms=total_ms,
            )
        except Exception as exc:
            log.warning("audit_log_failed", error=str(exc))

        # ── Step 9: Increment rate-limit counter ───────────────────────────
        limiter.increment(sender_hash)

        return {
            "label": ensemble.label,
            "final_score": ensemble.final_score,
            "latency_ms": total_ms,
        }

    except (AudioDownloadError, AudioConversionError) as exc:
        log.error("pipeline_error", error=str(exc))
        wa.send_message(sender, "⚠️ Failed to process the audio. Please try again.")
        raise self.retry(exc=exc)

    finally:
        # ── Step 10: Clean up temp files ───────────────────────────────────
        for path in [ogg_path, wav_path]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        log.debug("temp_files_cleaned")


@celery_app.task(name="processor.tasks.handle_stop_command")
def handle_stop_command(sender: str) -> dict:
    """Handle STOP opt-out: delete all sender audit records.

    Args:
        sender: Raw sender phone number (will be hashed before use).

    Returns:
        dict with deleted_count.
    """
    sender_hash = hash_sender(sender)
    auditor = _get_audit_logger()
    deleted = auditor.delete_sender_records(sender_hash)
    wa = _get_wa_client()
    wa.send_message(
        sender,
        "✅ You have been unsubscribed from VeriVoice.\n"
        "All your audit records have been deleted.\n"
        "To re-subscribe, simply send an audio clip.",
    )
    log.info("stop_command_processed", sender=sender_hash[:8], deleted=deleted)
    return {"deleted_count": deleted}
