"""
config.py — Environment-driven settings for VeriVoice.

All settings are loaded from environment variables (or a .env file in development).
Never hard-code secrets in this file.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for all VeriVoice services.

    Load via:
        from config import settings
    """

    # ── WhatsApp Cloud API ─────────────────────────────────────────────────
    wa_token: str = Field(..., env="WA_TOKEN", description="WhatsApp permanent access token")
    phone_number_id: str = Field(..., env="PHONE_NUMBER_ID", description="WhatsApp business phone number ID")
    verify_token: str = Field(..., env="VERIFY_TOKEN", description="Secret token used for webhook verification")
    app_secret: str = Field(..., env="APP_SECRET", description="Meta app secret for HMAC-SHA256 signature validation")

    # ── Redis (Upstash) ────────────────────────────────────────────────────
    redis_url: str = Field(..., env="REDIS_URL", description="Upstash Redis connection URL (redis://...)")

    # ── Cloudflare R2 (S3-compatible) ─────────────────────────────────────
    r2_bucket: str = Field(default="verivoice-audit", env="R2_BUCKET")
    r2_endpoint_url: str = Field(default="", env="R2_ENDPOINT_URL")
    r2_access_key_id: str = Field(default="", env="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str = Field(default="", env="R2_SECRET_ACCESS_KEY")

    # ── Model paths ────────────────────────────────────────────────────────
    aasist_model_path: str = Field(
        default="./weights/AASIST.pth",
        env="AASIST_MODEL_PATH",
        description="Path to AASIST weights (.pth). Download from github.com/clovaai/aasist",
    )
    xlsr_model_id: str = Field(
        default="facebook/wav2vec2-xls-r-300m",
        env="XLSR_MODEL_ID",
        description="HuggingFace model ID for XLS-R frontend",
    )
    use_gpu: bool = Field(
        default=True,
        env="USE_GPU",
        description="Whether to use CUDA (RTX 4050) — falls back to CPU automatically",
    )

    # ── Rate limiting ──────────────────────────────────────────────────────
    rate_limit_daily: int = Field(
        default=10,
        env="RATE_LIMIT_DAILY",
        description="Maximum analyses a single sender can request per 24-hour window",
    )

    # ── Ensemble weights — GPU deployment (XLS-R primary) ─────────────────
    weight_xlsr_aasist: float = Field(default=0.65, env="WEIGHT_XLSR_AASIST")
    weight_phase: float = Field(default=0.20, env="WEIGHT_PHASE")
    weight_meta: float = Field(default=0.15, env="WEIGHT_META")

    # ── Ensemble weights — CPU fallback (wav2vec2-base primary) ───────────
    weight_aasist_cpu: float = Field(default=0.55, env="WEIGHT_AASIST_CPU")
    weight_wav2vec_cpu: float = Field(default=0.30, env="WEIGHT_WAV2VEC_CPU")
    weight_meta_cpu: float = Field(default=0.15, env="WEIGHT_META_CPU")

    # ── Admin dashboard ────────────────────────────────────────────────────
    admin_username: str = Field(default="admin", env="ADMIN_USERNAME")
    admin_password: str = Field(default="changeme", env="ADMIN_PASSWORD")

    # ── Misc ───────────────────────────────────────────────────────────────
    tmp_dir: str = Field(default="/tmp/verivoice", env="TMP_DIR")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    fact_checker_url: str = Field(
        default="https://factchecker.in",
        env="FACT_CHECKER_URL",
        description="Link included in every verdict reply",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — import this everywhere
settings = Settings()
