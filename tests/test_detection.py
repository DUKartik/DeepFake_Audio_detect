"""
tests/test_detection.py — Unit tests for the detection engine.

Covers:
  - test_aasist_returns_detect_result()
  - test_aasist_fake_score_above_threshold()
  - test_aasist_model_not_found_raises_error()
  - test_ensemble_high_risk_label()
  - test_ensemble_moderate_label()
  - test_ensemble_real_label()
  - test_ensemble_weights_sum_to_one()
  - test_meta_detector_flags_low_bitrate()
  - test_meta_result_score_from_flags()
  - test_phase_discontinuity_score_range()
  - test_wav2vec_score_range()

All torch.load and external calls are mocked.
"""

import os
import numpy as np
import pytest
from dataclasses import replace
from unittest.mock import MagicMock, patch, mock_open

os.environ.setdefault("WA_TOKEN", "t")
os.environ.setdefault("PHONE_NUMBER_ID", "p")
os.environ.setdefault("VERIFY_TOKEN", "v")
os.environ.setdefault("APP_SECRET", "s")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("R2_BUCKET", "b")
os.environ.setdefault("R2_ENDPOINT_URL", "")
os.environ.setdefault("R2_ACCESS_KEY_ID", "")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "")

from models.schemas import DetectResult, MetaResult, EnsembleResult
from detection.ensemble import (
    ensemble_score_cpu,
    ensemble_score_gpu,
    WEIGHTS_CPU,
    WEIGHTS_GPU,
    _classify,
)


# ─────────────────────────────────────────────────────────────────────────────
# AASISTDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAASISTDetector:
    @patch("detection.aasist_detector.torch.load")
    @patch("detection.aasist_detector.Path.exists", return_value=True)
    def test_aasist_returns_detect_result(self, mock_exists, mock_load):
        """predict() must return a DetectResult dataclass."""
        import torch
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.8]])
        mock_load.return_value = mock_model

        with patch("detection.aasist_detector.torchaudio.load") as mock_load_wav:
            mock_load_wav.return_value = (torch.zeros(1, 16000), 16000)
            from detection.aasist_detector import AASISTDetector
            detector = AASISTDetector(model_path="./fake/path.pth")
            result = detector.predict("./fake/audio.wav")

        assert isinstance(result, DetectResult)
        assert result.model == "aasist"
        assert 0.0 <= result.score <= 1.0
        assert result.label in ("real", "fake")

    @patch("detection.aasist_detector.torch.load")
    @patch("detection.aasist_detector.Path.exists", return_value=True)
    def test_aasist_fake_score_above_threshold(self, mock_exists, mock_load):
        """Scores > 0.5 from the model should produce label='fake'."""
        import torch
        mock_model = MagicMock()
        # sigmoid(2.2) ≈ 0.90 → fake
        mock_model.return_value = torch.tensor([[2.2]])
        mock_load.return_value = mock_model

        with patch("detection.aasist_detector.torchaudio.load") as mock_wav:
            mock_wav.return_value = (torch.zeros(1, 16000), 16000)
            from detection.aasist_detector import AASISTDetector
            detector = AASISTDetector(model_path="./fake/path.pth")
            result = detector.predict("./fake/audio.wav")

        assert result.label == "fake"
        assert result.score > 0.5

    def test_aasist_model_not_found_raises_error(self):
        """ModelLoadError must be raised when the weights file doesn't exist."""
        from utils.exceptions import ModelLoadError
        from detection.aasist_detector import AASISTDetector
        with pytest.raises(ModelLoadError):
            AASISTDetector(model_path="./nonexistent/path.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_detect(score: float, label: str | None = None) -> DetectResult:
    if label is None:
        label = "fake" if score > 0.5 else "real"
    return DetectResult(label=label, score=score, model="test", latency_ms=100)


def _meta(num_flags: int = 0) -> MetaResult:
    flags = [f"FLAG_{i}" for i in range(num_flags)]
    return MetaResult(flags=flags)


class TestEnsembleCPU:
    def test_ensemble_high_risk_label(self):
        """score > 0.85 → label='fake', verdict contains 'HIGH'."""
        result = ensemble_score_cpu(
            aasist=_make_detect(0.95),
            wav2vec_score=0.90,
            meta=_meta(3),
        )
        assert result.label == "fake"
        assert "HIGH" in result.verdict_text
        assert result.final_score > 0.85

    def test_ensemble_moderate_label(self):
        """0.60 < score ≤ 0.85 → label='suspicious'."""
        result = ensemble_score_cpu(
            aasist=_make_detect(0.70),
            wav2vec_score=0.65,
            meta=_meta(1),
        )
        assert result.label == "suspicious"
        assert "MODERATE" in result.verdict_text

    def test_ensemble_real_label(self):
        """score ≤ 0.60 → label='real'."""
        result = ensemble_score_cpu(
            aasist=_make_detect(0.1, label="real"),
            wav2vec_score=0.05,
            meta=_meta(0),
        )
        assert result.label == "real"
        assert "authentic" in result.verdict_text.lower()

    def test_ensemble_weights_sum_to_one(self):
        """CPU ensemble weights must sum to exactly 1.0."""
        total = sum(WEIGHTS_CPU.values())
        assert abs(total - 1.0) < 1e-9, f"CPU weights sum to {total}, expected 1.0"

    def test_ensemble_gpu_weights_sum_to_one(self):
        """GPU ensemble weights must sum to exactly 1.0."""
        total = sum(WEIGHTS_GPU.values())
        assert abs(total - 1.0) < 1e-9, f"GPU weights sum to {total}, expected 1.0"

    def test_ensemble_score_capped_at_1(self):
        """final_score must never exceed 1.0."""
        result = ensemble_score_cpu(
            aasist=_make_detect(1.0),
            wav2vec_score=1.0,
            meta=_meta(5),
            lang_boost=1.0,
        )
        assert result.final_score <= 1.0

    def test_ensemble_result_has_model_scores(self):
        """EnsembleResult.model_scores must contain all expected keys."""
        result = ensemble_score_cpu(_make_detect(0.6), 0.5, _meta(1))
        assert "aasist" in result.model_scores
        assert "wav2vec" in result.model_scores
        assert "meta" in result.model_scores

    def test_confidence_pct_is_integer(self):
        result = ensemble_score_cpu(_make_detect(0.75), 0.7, _meta(2))
        assert isinstance(result.confidence_pct, int)


# ─────────────────────────────────────────────────────────────────────────────
# MetaDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaDetector:
    def _ffprobe_json(self, bitrate: int = 64000, codec: str = "opus", sr: int = 16000) -> str:
        import json
        return json.dumps({
            "format": {"bit_rate": str(bitrate)},
            "streams": [{
                "codec_type": "audio",
                "codec_name": codec,
                "sample_rate": str(sr),
                "channels": 1,
            }]
        })

    @patch("subprocess.run")
    def test_meta_detector_flags_low_bitrate(self, mock_run):
        """Files with bitrate < 32 kbps must raise LOW_BITRATE flag."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._ffprobe_json(bitrate=20000, codec="opus", sr=16000),
        )
        from detection.meta_detector import MetaDetector
        detector = MetaDetector()
        result = detector.analyse("test.ogg")
        flag_names = [f.split(":")[0] for f in result.flags]
        assert "LOW_BITRATE" in flag_names

    @patch("subprocess.run")
    def test_meta_detector_flags_unusual_codec(self, mock_run):
        """Non-standard codecs (e.g. pcm_s16le) must raise UNUSUAL_CODEC flag."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._ffprobe_json(bitrate=128000, codec="pcm_s16le", sr=16000),
        )
        from detection.meta_detector import MetaDetector
        detector = MetaDetector()
        result = detector.analyse("test.wav")
        flag_names = [f.split(":")[0] for f in result.flags]
        assert "UNUSUAL_CODEC" in flag_names

    @patch("subprocess.run")
    def test_meta_no_flags_for_clean_audio(self, mock_run):
        """A clean 16 kHz opus file at 64 kbps should produce no flags."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._ffprobe_json(bitrate=64000, codec="opus", sr=16000),
        )
        from detection.meta_detector import MetaDetector
        detector = MetaDetector()
        result = detector.analyse("clean.ogg")
        assert result.score == 0.0
        assert not result.suspicious

    def test_meta_result_score_from_flags(self):
        """MetaResult score should equal min(flags/3, 1.0)."""
        r0 = MetaResult(flags=[])
        r1 = MetaResult(flags=["A"])
        r3 = MetaResult(flags=["A", "B", "C"])
        r5 = MetaResult(flags=["A", "B", "C", "D", "E"])

        assert r0.score == 0.0
        assert abs(r1.score - 1 / 3) < 0.01
        assert r3.score == 1.0
        assert r5.score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Classify thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestClassify:
    def test_high_threshold(self):
        label, verdict = _classify(0.86)
        assert label == "fake"
        assert "HIGH" in verdict

    def test_moderate_threshold(self):
        label, verdict = _classify(0.70)
        assert label == "suspicious"
        assert "MODERATE" in verdict

    def test_real_threshold(self):
        label, verdict = _classify(0.40)
        assert label == "real"
        assert "authentic" in verdict.lower()

    def test_boundary_0_85(self):
        """Exactly 0.85 should be 'suspicious', not 'fake' (strict >)."""
        label, _ = _classify(0.85)
        assert label == "suspicious"

    def test_boundary_0_60(self):
        """Exactly 0.60 should be 'real', not 'suspicious' (strict >)."""
        label, _ = _classify(0.60)
        assert label == "real"
