import sys
import os

import structlog

# Set up simple logging to see inner model logs
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()]
)

import detection.model_registry as registry
from detection.ensemble import ensemble_score_cpu, ensemble_score_gpu, phase_discontinuity_score

def test_audio(wav_path: str):
    if not os.path.exists(wav_path):
        print(f"Error: File not found at {wav_path}")
        sys.exit(1)

    print(f"Loading models...")
    registry.load_models()
    device = registry.device()
    print(f"Models loaded successfully on {device.upper()}.\n")
    
    print(f"Analysing file: {wav_path}")
    meta_detector = registry.get("meta")
    meta_result = meta_detector.analyse(wav_path)
    
    if device == "cuda":
        xlsr_detector = registry.get("xlsr_aasist")
        xlsr_result = xlsr_detector.predict(wav_path)
        p_score = phase_discontinuity_score(wav_path)
        ensemble = ensemble_score_gpu(xlsr_result, p_score, meta_result)
    else:
        aasist_detector = registry.get("aasist")
        aasist_result = aasist_detector.predict(wav_path)
        
        wav2vec_score = 0.0
        if "wav2vec" in registry._REGISTRY:
            wav2vec = registry.get("wav2vec")
            if hasattr(wav2vec, 'predict'):
                wav2vec_score = wav2vec.predict(wav_path)
                
        ensemble = ensemble_score_cpu(aasist_result, wav2vec_score, meta_result)

    print("\n" + "="*40)
    print("🎙️  VERIVOICE DETECTION RESULT  🎙️")
    print("="*40)
    print(f"Verdict:    {ensemble.label.upper()}")
    print(f"Confidence: {ensemble.confidence_pct}%")
    print(f"Score:      {ensemble.final_score:.3f}")
    print(f"Text:       {ensemble.verdict_text}")
    print(f"Meta Flags: {', '.join(ensemble.meta_flags) if ensemble.meta_flags else 'None'}")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_audio.py <path_to_16khz_mono_wav_file>")
        sys.exit(1)
        
    test_audio(sys.argv[1])
