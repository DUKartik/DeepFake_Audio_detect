"""
detection/meta_detector.py — Deterministic metadata heuristics detector.

Uses ffprobe to inspect codec, bitrate, sample rate, and channel layout.
No ML involved — flags are computed deterministically.

Flags raised:
  LOW_BITRATE      — Bitrate < 32 kbps after WhatsApp re-encoding implies double compression.
  UNUSUAL_CODEC    — TTS tools often output pcm_s16le or flac, uncommon in organic clips.
  UNIFORM_SILENCE  — TTS pads with silence between sentences (> 3 sec at non-natural positions).
  SAMPLE_RATE_MISMATCH — TTS outputs at 22 kHz/44.1 kHz before WhatsApp re-encoding.
  STEREO_IDENTICAL — Voice clone models output identical L/R channels (correlation > 0.999).
"""

import json
import subprocess
import structlog
import numpy as np

from models.schemas import MetaResult
from utils.exceptions import InferenceError

log = structlog.get_logger(__name__)

# ── Thresholds (documented in Section 4.3 of VeriVoice spec) ──────────────────
_LOW_BITRATE_KBPS = 32
_ALLOWED_CODECS = {"opus", "aac", "mp3", "libopus"}
_SILENCE_THRESHOLD_DB = -50.0      # dBFS below which a frame is considered silent
_SILENCE_MIN_DURATION_S = 3.0      # seconds of contiguous silence to flag
_STEREO_CORRELATION_MAX = 0.999    # above this the two channels are identical
_EXPECTED_SAMPLE_RATES = {8000, 16000}  # expected post-WhatsApp sample rates


class MetaDetector:
    """Heuristic-only detector that inspects audio metadata via ffprobe.

    Args:
        ffprobe_path: Path to the ffprobe binary. Default "ffprobe" (must be on PATH).

    Example:
        detector = MetaDetector()
        result = detector.analyse("audio.wav")
        print(result.flags, result.score)
    """

    def __init__(self, ffprobe_path: str = "ffprobe") -> None:
        self._ffprobe = ffprobe_path

    def analyse(self, audio_path: str) -> MetaResult:
        """Run all heuristic checks on an audio file.

        Args:
            audio_path: Path to the audio file (.wav, .ogg, .opus, etc.).

        Returns:
            MetaResult with flags list, bitrate, codec, sample_rate, and score.

        Raises:
            InferenceError: If ffprobe cannot read the file.
        """
        probe = self._run_ffprobe(audio_path)
        stream = self._get_audio_stream(probe)

        flags: list[str] = []
        bitrate = int(probe.get("format", {}).get("bit_rate", 0)) // 1000
        codec = stream.get("codec_name", "").lower()
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 1))

        # ── Flag 1: Low bitrate ───────────────────────────────────────────────
        if 0 < bitrate < _LOW_BITRATE_KBPS:
            flags.append(f"LOW_BITRATE:{bitrate}kbps")
            log.debug("meta_flag_low_bitrate", bitrate=bitrate)

        # ── Flag 2: Unusual codec ─────────────────────────────────────────────
        if codec and codec not in _ALLOWED_CODECS:
            flags.append(f"UNUSUAL_CODEC:{codec}")
            log.debug("meta_flag_unusual_codec", codec=codec)

        # ── Flag 3: Sample rate mismatch ─────────────────────────────────────
        if sample_rate and sample_rate not in _EXPECTED_SAMPLE_RATES:
            flags.append(f"SAMPLE_RATE_MISMATCH:{sample_rate}Hz")
            log.debug("meta_flag_sr_mismatch", sample_rate=sample_rate)

        # ── Flags 4 & 5 require loading audio data (WAV only) ────────────────
        if audio_path.lower().endswith(".wav"):
            try:
                import soundfile as sf  # optional dep — skip if unavailable
                data, _ = sf.read(audio_path)

                # Flag 4: Uniform silence regions
                if self._has_uniform_silence(data if data.ndim == 1 else data[:, 0]):
                    flags.append("UNIFORM_SILENCE")
                    log.debug("meta_flag_uniform_silence")

                # Flag 5: Stereo channels identical (voice clone artifact)
                if channels == 2 and data.ndim == 2:
                    corr = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
                    if corr > _STEREO_CORRELATION_MAX:
                        flags.append(f"STEREO_IDENTICAL:{corr:.4f}")
                        log.debug("meta_flag_stereo_identical", corr=corr)
            except ImportError:
                log.warning("meta_soundfile_missing", detail="pip install soundfile for full heuristics")
            except Exception as exc:
                log.warning("meta_waveform_check_failed", error=str(exc))

        result = MetaResult(
            flags=flags,
            bitrate=bitrate,
            codec=codec,
            sample_rate=sample_rate,
        )
        log.info("meta_result", flags=flags, score=result.score)
        return result

    def _run_ffprobe(self, path: str) -> dict:
        """Run ffprobe and return parsed JSON output."""
        cmd = [
            self._ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise InferenceError(f"ffprobe failed: {result.stderr.strip()}")
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired as exc:
            raise InferenceError("ffprobe timed out after 10 seconds") from exc
        except json.JSONDecodeError as exc:
            raise InferenceError(f"ffprobe returned invalid JSON: {exc}") from exc

    def _get_audio_stream(self, probe: dict) -> dict:
        """Extract the first audio stream from ffprobe JSON output."""
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio":
                return stream
        return {}

    def _has_uniform_silence(self, signal: np.ndarray, sr: int = 16000) -> bool:
        """Detect contiguous silence longer than _SILENCE_MIN_DURATION_S.

        Args:
            signal: 1-D float array of audio samples.
            sr:     Sample rate of the signal.

        Returns:
            True if a suspicious silence region is found.
        """
        frame_len = int(0.02 * sr)   # 20 ms frames
        silence_frames = 0
        max_silence_frames = 0

        for i in range(0, len(signal) - frame_len, frame_len):
            frame = signal[i : i + frame_len]
            frame_rms = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-10)
            if frame_rms < _SILENCE_THRESHOLD_DB:
                silence_frames += 1
                max_silence_frames = max(max_silence_frames, silence_frames)
            else:
                silence_frames = 0

        # Convert frames back to seconds
        max_silence_s = max_silence_frames * 0.02
        return max_silence_s > _SILENCE_MIN_DURATION_S
