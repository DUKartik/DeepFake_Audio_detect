"""
detection/ensemble.py — Weighted ensemble scoring logic.

Two ensemble configurations:

CPU deployment (free-tier Render.com):
  AASIST     → 0.55
  wav2vec2   → 0.30
  Metadata   → 0.15

GPU deployment (RTX 4050 local):
  XLS-R+AASIST → 0.65
  Phase score  → 0.20
  Metadata     → 0.15

Verdict thresholds (Section 4.4):
  score > 0.85 → HIGH   (fake)
  score > 0.60 → MODERATE (suspicious)
  score ≤ 0.60 → LIKELY REAL
"""

import librosa
import numpy as np
import structlog

from models.schemas import DetectResult, MetaResult, EnsembleResult

log = structlog.get_logger(__name__)

# ── CPU ensemble weights ───────────────────────────────────────────────────────
WEIGHTS_CPU = {"aasist": 0.55, "wav2vec": 0.30, "meta": 0.15}

# ── GPU ensemble weights (XLS-R primary, replaces CPU wav2vec secondary) ──────
WEIGHTS_GPU = {
    "xlsr_aasist": 0.65,   # XLS-R 300M + AASIST — primary, Hindi-aware
    "phase_score": 0.20,   # Phase discontinuity analysis (compression-robust)
    "meta": 0.15,          # Metadata heuristics
}


def phase_discontinuity_score(wav_path: str) -> float:
    """Compute phase discontinuity anomaly score from a WAV file.

    Synthetic speech has unnaturally smooth phase transitions even after
    32 kbps ogg encoding (phase is preserved in the 80–3400 Hz band).

    Args:
        wav_path: Path to a 16 kHz mono WAV file.

    Returns:
        float in [0.0, 1.0] — 1.0 indicates suspiciously smooth phase (fake).
    """
    y, _ = librosa.load(wav_path, sr=16000)
    stft = librosa.stft(y, n_fft=1024)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    variance = float(np.var(phase_diff))
    # Low variance = suspiciously smooth = higher fake probability
    score = max(0.0, min(1.0 - (variance * 35), 1.0))
    log.debug("phase_score", variance=round(variance, 5), score=round(score, 4))
    return round(score, 4)


def ensemble_score_cpu(
    aasist: DetectResult,
    wav2vec_score: float,
    meta: MetaResult,
    lang_boost: float = 0.0,
) -> EnsembleResult:
    """Compute weighted ensemble for CPU-only deployment.

    Args:
        aasist:       DetectResult from AASISTDetector.
        wav2vec_score: Raw fake probability from Wav2VecDetector.score().
        meta:          MetaResult from MetaDetector.
        lang_boost:    Optional boost (+0.05 recommended) if regional language detected.

    Returns:
        EnsembleResult with final verdict and per-model scores.

    Example:
        result = ensemble_score_cpu(aasist_result, 0.72, meta_result)
        print(result.label, result.confidence_pct)
    """
    # Normalise AASIST score so 1.0 = definitely fake regardless of original label
    a = aasist.score if aasist.label == "fake" else 1.0 - aasist.score
    w = wav2vec_score
    m = meta.score

    final = (
        WEIGHTS_CPU["aasist"] * a
        + WEIGHTS_CPU["wav2vec"] * w
        + WEIGHTS_CPU["meta"] * m
        + lang_boost
    )
    final = round(min(final, 1.0), 3)

    label, verdict = _classify(final)

    log.info(
        "ensemble_cpu",
        final_score=final,
        label=label,
        aasist=round(a, 4),
        wav2vec=round(w, 4),
        meta=round(m, 4),
    )
    return EnsembleResult(
        final_score=final,
        label=label,
        confidence_pct=int(final * 100),
        model_scores={"aasist": round(a, 4), "wav2vec": round(w, 4), "meta": round(m, 4)},
        meta_flags=meta.flags,
        verdict_text=verdict,
    )


def ensemble_score_gpu(
    xlsr_aasist: DetectResult,
    phase_score: float,
    meta: MetaResult,
    lang_boost: float = 0.0,
) -> EnsembleResult:
    """Compute weighted ensemble for RTX 4050 GPU deployment.

    Args:
        xlsr_aasist:  DetectResult from XLSRAASISTDetector.
        phase_score:  Score from phase_discontinuity_score().
        meta:         MetaResult from MetaDetector.
        lang_boost:   Optional regional language boost.

    Returns:
        EnsembleResult with final verdict and per-model scores.

    Example:
        result = ensemble_score_gpu(xlsr_result, phase_score, meta_result)
    """
    x = xlsr_aasist.score if xlsr_aasist.label == "fake" else 1.0 - xlsr_aasist.score
    p = phase_score
    m = meta.score

    final = (
        WEIGHTS_GPU["xlsr_aasist"] * x
        + WEIGHTS_GPU["phase_score"] * p
        + WEIGHTS_GPU["meta"] * m
        + lang_boost
    )
    final = round(min(final, 1.0), 3)

    label, verdict = _classify(final)

    log.info(
        "ensemble_gpu",
        final_score=final,
        label=label,
        xlsr_aasist=round(x, 4),
        phase=round(p, 4),
        meta=round(m, 4),
    )
    return EnsembleResult(
        final_score=final,
        label=label,
        confidence_pct=int(final * 100),
        model_scores={"xlsr_aasist": round(x, 4), "phase": round(p, 4), "meta": round(m, 4)},
        meta_flags=meta.flags,
        verdict_text=verdict,
    )


def _classify(score: float) -> tuple[str, str]:
    """Map a final ensemble score to a (label, verdict_text) pair.

    Args:
        score: Ensemble fake probability in [0.0, 1.0].

    Returns:
        Tuple of (label, verdict_text).
    """
    if score > 0.85:
        return "fake", "HIGH probability of deepfake"
    elif score > 0.60:
        return "suspicious", "MODERATE suspicion of manipulation"
    else:
        return "real", "Likely authentic audio"
