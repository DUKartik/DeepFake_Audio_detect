"""
utils/audit.py — Audit logging to Cloudflare R2 (S3-compatible).

Per PDPB compliance:
  - Only sender_hash (SHA-256) is stored — never the raw phone number.
  - R2 default encryption protects data at rest.
  - Users can request deletion via the STOP command.
"""

import json
import datetime
import structlog
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from utils.exceptions import AuditLogError

log = structlog.get_logger(__name__)


class AuditLogger:
    """Append-only JSONL audit log stored in Cloudflare R2.

    Each record is a JSON line appended to:
        audit/results/{YYYY}/{MM}/{DD}.jsonl

    Args:
        bucket:            R2 bucket name.
        endpoint_url:      R2 endpoint URL (e.g. "https://<account_id>.r2.cloudflarestorage.com").
        access_key_id:     R2 access key ID.
        secret_access_key: R2 secret access key.

    Example:
        logger = AuditLogger(bucket="verivoice-audit", ...)
        logger.log_result(sender_hash="abc123", media_id="xyz", result={...})
    """

    _KEY_PREFIX = "audit/results"
    _ADMIN_LOG_KEY = "audit/admin_access.jsonl"

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        self._bucket = bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
        )

    def _today_key(self) -> str:
        """Return the S3 key for today's audit log file."""
        now = datetime.datetime.utcnow()
        return f"{self._KEY_PREFIX}/{now.year}/{now.month:02d}/{now.day:02d}.jsonl"

    def log_result(
        self,
        sender_hash: str,
        media_id: str,
        final_score: float,
        label: str,
        model_scores: dict,
        meta_flags: list[str],
        latency_ms: int,
    ) -> None:
        """Append one JSONL record to today's audit log in R2.

        Args:
            sender_hash:  SHA-256 of the sender's phone number (no PII).
            media_id:     WhatsApp media ID for traceability.
            final_score:  Ensemble fake probability (0.0–1.0).
            label:        "fake" | "suspicious" | "real".
            model_scores: Per-model score breakdown.
            meta_flags:   List of heuristic flags raised.
            latency_ms:   End-to-end processing time.
        """
        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "sender": sender_hash[:16],   # Only first 16 chars for anonymisation
            "media_id": media_id,
            "final_score": final_score,
            "label": label,
            "model_scores": model_scores,
            "meta_flags": meta_flags,
            "latency_ms": latency_ms,
        }
        self._append_jsonl(self._today_key(), record)

    def log_admin_access(self, username: str, endpoint: str, ip: str) -> None:
        """Append an admin access event to the admin audit log.

        Args:
            username: Admin username.
            endpoint: Endpoint accessed (e.g. "/admin/stats").
            ip:       Requester IP address.
        """
        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "username": username,
            "endpoint": endpoint,
            "ip": ip,
        }
        self._append_jsonl(self._ADMIN_LOG_KEY, record)

    def _append_jsonl(self, key: str, record: dict) -> None:
        """Read existing object, append record line, re-upload.

        R2 does not natively support append — we do a read-modify-write.
        For high-volume production, replace with a streaming aggregator.
        """
        try:
            try:
                existing = self._s3.get_object(Bucket=self._bucket, Key=key)
                current = existing["Body"].read().decode("utf-8")
            except ClientError as exc:
                if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    current = ""
                else:
                    raise

            new_content = current + json.dumps(record) + "\n"
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=new_content.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            log.debug("audit_log_written", key=key, label=record.get("label"))
        except (BotoCoreError, ClientError) as exc:
            log.error("audit_log_failed", key=key, error=str(exc))
            raise AuditLogError(f"Failed to write audit log to R2: {exc}") from exc

    def delete_sender_records(self, sender_hash: str) -> int:
        """Delete all audit records for a given sender (STOP command / PDPB request).

        This iterates all JSONL files and removes lines matching the sender hash.

        Args:
            sender_hash: SHA-256 hash of the sender requesting deletion.

        Returns:
            Number of records deleted.
        """
        deleted = 0
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._KEY_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                try:
                    resp = self._s3.get_object(Bucket=self._bucket, Key=key)
                    lines = resp["Body"].read().decode("utf-8").splitlines()
                    original_count = len(lines)
                    filtered = [
                        line for line in lines
                        if sender_hash[:16] not in line
                    ]
                    if len(filtered) < original_count:
                        removed = original_count - len(filtered)
                        deleted += removed
                        self._s3.put_object(
                            Bucket=self._bucket,
                            Key=key,
                            Body=("\n".join(filtered) + "\n").encode("utf-8"),
                            ContentType="application/x-ndjson",
                        )
                        log.info("audit_records_deleted", key=key, removed=removed)
                except (BotoCoreError, ClientError) as exc:
                    log.warning("audit_delete_failed", key=key, error=str(exc))
        return deleted

    def get_recent(self, n: int = 20) -> list[dict]:
        """Return the last N audit records from today's log.

        Args:
            n: Maximum number of records to return.

        Returns:
            List of record dicts with sender truncated to 8 chars.
        """
        key = self._today_key()
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            lines = resp["Body"].read().decode("utf-8").strip().splitlines()
            records = []
            for line in lines[-n:]:
                try:
                    rec = json.loads(line)
                    # Extra anonymisation — truncate sender to 8 chars for API response
                    rec["sender"] = rec["sender"][:8]
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
            return list(reversed(records))  # most recent first
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return []
            raise AuditLogError(f"Failed to read audit log: {exc}") from exc
