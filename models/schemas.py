"""
models/schemas.py — Shared dataclass schemas for all VeriVoice modules.

All inter-module communication uses these typed dataclasses — no raw dicts.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectResult:
    """Result from a single ML detector (AASIST, XLS-R, or wav2vec2).

    Attributes:
        label:      "real" or "fake"
        score:      Probability of being fake (0.0 = definitely real, 1.0 = definitely fake)
        model:      Identifier string for the model that produced this result
        latency_ms: Wall-clock inference time in milliseconds
    """

    label: str          # "real" or "fake"
    score: float        # probability of being fake (0.0–1.0)
    model: str          # identifier of the model
    latency_ms: int     # wall-clock inference time


@dataclass
class MetaResult:
    """Result from the deterministic metadata heuristics detector.

    Attributes:
        flags:      List of human-readable anomaly descriptions
        bitrate:    Detected bitrate in kbps
        codec:      Detected audio codec string
        sample_rate: Detected sample rate in Hz
        suspicious: True if any flags were raised
        score:      Normalised fake probability based on flag count (0.0–1.0)
    """

    flags: list[str] = field(default_factory=list)
    bitrate: int = 0
    codec: str = ""
    sample_rate: int = 0
    suspicious: bool = False
    score: float = 0.0

    def __post_init__(self) -> None:
        """Auto-compute score and suspicious flag from the flag list."""
        self.suspicious = len(self.flags) > 0
        # Each flag contributes 1/3; 3+ flags = score 1.0
        self.score = round(min(len(self.flags) / 3.0, 1.0), 3)


@dataclass
class EnsembleResult:
    """Final weighted ensemble result combining all detectors.

    Attributes:
        final_score:    Weighted composite fake probability (0.0–1.0)
        label:          "fake" | "suspicious" | "real"
        confidence_pct: final_score expressed as integer percentage
        model_scores:   Per-model raw scores used in the ensemble
        meta_flags:     Human-readable heuristic flags from MetaDetector
        verdict_text:   Short verdict string sent to the user
    """

    final_score: float
    label: str
    confidence_pct: int
    model_scores: dict
    meta_flags: list[str]
    verdict_text: str
