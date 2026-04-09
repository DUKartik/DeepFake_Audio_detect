"""
detection/model_registry.py — Singleton that loads and caches all ML models.

Loads models once at Celery worker startup and keeps them resident in memory.
Exposes a get(name) method for lazy lookup by name string.

Weights used per deployment mode:
  CPU  → AASISTDetector, Wav2VecDetector
  GPU  → XLSRAASISTDetector (RTX 4050)
"""

import torch
import structlog

from config import settings
from utils.exceptions import ModelLoadError

log = structlog.get_logger(__name__)

_REGISTRY: dict = {}   # module-level singleton cache
_INITIALISED = False


def load_models() -> None:
    """Load and cache all ML models at worker startup.

    Call this once when the Celery worker starts. Models are cached in the
    module-level _REGISTRY dict and never reloaded during the worker lifetime.

    Raises:
        ModelLoadError: If any critical model fails to load.
    """
    global _INITIALISED
    if _INITIALISED:
        log.info("model_registry_already_loaded")
        return

    device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
    log.info("model_registry_loading", device=device)

    if device == "cuda":
        # ── GPU path: XLS-R 300M + AASIST ─────────────────────────────────
        from detection.xlsr_aasist_detector import XLSRAASISTDetector
        try:
            _REGISTRY["xlsr_aasist"] = XLSRAASISTDetector(
                xlsr_path=settings.xlsr_model_id,
                aasist_path=settings.aasist_model_path,
            )
            log.info("model_xlsr_aasist_loaded")
        except ModelLoadError as exc:
            log.warning("model_xlsr_aasist_failed", error=str(exc), falling_back_to="cpu")
            device = "cpu"  # fall through to CPU path

    if device == "cpu":
        # ── CPU path: stand-alone AASIST + wav2vec2-base ──────────────────
        from detection.aasist_detector import AASISTDetector
        from detection.wav2vec_detector import Wav2VecDetector
        try:
            _REGISTRY["aasist"] = AASISTDetector(model_path=settings.aasist_model_path)
            log.info("model_aasist_loaded")
        except ModelLoadError as exc:
            log.error("model_aasist_failed", error=str(exc))
            raise

        try:
            _REGISTRY["wav2vec"] = Wav2VecDetector()
            log.info("model_wav2vec_loaded")
        except ModelLoadError as exc:
            log.warning("model_wav2vec_failed", error=str(exc), detail="wav2vec detector disabled")

    from detection.meta_detector import MetaDetector
    _REGISTRY["meta"] = MetaDetector()
    log.info("model_meta_loaded")

    _REGISTRY["_device"] = device
    _INITIALISED = True
    log.info("model_registry_ready", models=list(_REGISTRY.keys()))


def get(name: str):
    """Return a loaded model by name.

    Args:
        name: One of "xlsr_aasist", "aasist", "wav2vec", "meta".

    Returns:
        The loaded model/detector instance.

    Raises:
        KeyError: If name not found in the registry.
    """
    if not _INITIALISED:
        load_models()
    return _REGISTRY[name]


def device() -> str:
    """Return the active compute device ("cuda" or "cpu")."""
    if not _INITIALISED:
        load_models()
    return _REGISTRY.get("_device", "cpu")
