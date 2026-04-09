"""
utils/hashing.py — Privacy-safe hashing utilities.

Per PDPB (India Personal Data Protection Bill):
  - Raw phone numbers must NEVER be stored or logged.
  - All sender identifiers are SHA-256 hashed before persistence.
"""

import hashlib
import hmac as _hmac


def hash_sender(phone_number: str) -> str:
    """Return the SHA-256 hex digest of a phone number.

    Args:
        phone_number: Raw E.164 phone number string (e.g. "919876543210").

    Returns:
        64-character hex string. Irreversible — the original number cannot be recovered.

    Example:
        >>> hash_sender("919876543210")
        'a3f4e...<64-hex-chars>...'
    """
    return hashlib.sha256(phone_number.encode("utf-8")).hexdigest()


def verify_hmac_sha256(secret: str, body: bytes, signature_header: str) -> bool:
    """Validate the X-Hub-Signature-256 header sent by Meta on every webhook POST.

    Args:
        secret:           The APP_SECRET from Meta Developer Console.
        body:             Raw request body bytes.
        signature_header: Value of the X-Hub-Signature-256 header (e.g. "sha256=abc123...").

    Returns:
        True if the signature is valid, False otherwise.

    Example:
        >>> verify_hmac_sha256("mysecret", b"payload", "sha256=<computed>")
        True
    """
    expected = "sha256=" + _hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    # Use constant-time comparison to prevent timing attacks
    return _hmac.compare_digest(signature_header, expected)
