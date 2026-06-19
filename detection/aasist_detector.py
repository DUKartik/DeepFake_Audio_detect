"""
detection/aasist_detector.py — Primary deepfake detector using AASIST GNN.

AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
Networks) is the CLOVA AI model that ranked 1st on ASVspoof 2021.

Specs:
  - Parameters: ~297K (8 MB on disk)
  - Input: raw waveform at 16 kHz (no hand-crafted features)
  - Accuracy: 96.4% on ASVspoof 2021 LA eval set
  - CPU inference: ~280–350 ms per clip

Download weights:
  github.com/clovaai/aasist → pretrained/AASIST.pth
"""

import time
from pathlib import Path

import torch
import torchaudio
import structlog

from models.schemas import DetectResult
from utils.exceptions import ModelLoadError, InferenceError

log = structlog.get_logger(__name__)


class AASISTDetector:
    """Wraps the AASIST graph neural network for deepfake audio detection.

    Loads from a local .pt/.pth file. Runs entirely on CPU with torch.no_grad().

    Args:
        model_path: Path to the AASIST weights file (.pt or .pth).
        threshold:  Score threshold above which audio is labelled "fake". Default 0.5.

    Example:
        detector = AASISTDetector(model_path="./weights/AASIST.pth")
        result = detector.predict("audio.wav")
        print(result.label, result.score)
    """

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(
                f"AASIST weights not found at '{model_path}'. "
                "Download from github.com/clovaai/aasist → pretrained/AASIST.pth"
            )
        try:
            from models.AASIST import Model
            d_args = {
                "architecture": "AASIST",
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0]
            }
            self.model = Model(d_args)
            self.model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=False))
            self.model.eval()
        except Exception as exc:
            raise ModelLoadError(f"Failed to load AASIST model: {exc}") from exc

        self.threshold = threshold
        log.info("aasist_loaded", model_path=model_path, threshold=threshold)

    def predict(self, wav_path: str) -> DetectResult:
        """Run AASIST inference on a WAV file.

        Args:
            wav_path: Absolute path to a 16 kHz mono WAV file.

        Returns:
            DetectResult with label ("real" | "fake"), score (0.0–1.0),
            model identifier, and wall-clock latency.

        Raises:
            InferenceError: If the forward pass fails.
        """
        t0 = time.time()
        try:
            import soundfile as sf
            wav_data, sr = sf.read(wav_path)
            if wav_data.ndim == 1:
                waveform = torch.from_numpy(wav_data).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(wav_data).t().float()
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            with torch.no_grad():
                _, output = self.model(waveform)
                score = torch.sigmoid(output[:, 1]).item()

        except Exception as exc:
            raise InferenceError(f"AASIST inference failed: {exc}") from exc

        latency = int((time.time() - t0) * 1000)
        label = "fake" if score > self.threshold else "real"

        log.info("aasist_result", label=label, score=round(score, 4), latency_ms=latency)
        return DetectResult(
            label=label,
            score=round(score, 4),
            model="aasist",
            latency_ms=latency,
        )
