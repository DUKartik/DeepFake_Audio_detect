"""
utils/exceptions.py — Custom exception hierarchy for VeriVoice.

Raising typed exceptions makes error handling explicit and testable.
"""


class VeriVoiceError(Exception):
    """Base exception for all VeriVoice errors."""


class WebhookError(VeriVoiceError):
    """Raised when webhook signature validation or parsing fails."""


class AudioDownloadError(VeriVoiceError):
    """Raised when the audio file cannot be downloaded from Meta's CDN."""


class AudioConversionError(VeriVoiceError):
    """Raised when ffmpeg fails to convert the audio to WAV."""


class ModelLoadError(VeriVoiceError):
    """Raised when an ML model cannot be loaded from disk or HuggingFace Hub."""


class InferenceError(VeriVoiceError):
    """Raised when a model inference call fails at runtime."""


class RateLimitExceededError(VeriVoiceError):
    """Raised when a sender has exceeded their daily analysis quota."""


class AuditLogError(VeriVoiceError):
    """Raised when writing to the Cloudflare R2 audit log fails."""
