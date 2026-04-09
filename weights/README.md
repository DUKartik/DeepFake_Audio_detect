# AASIST Model Weights

Download AASIST weights before running the system:

## Option A — Original AASIST (English, CPU)
```bash
# Clone the official repo and copy the pretrained weights
git clone https://github.com/clovaai/aasist
cp aasist/pretrained/AASIST.pth ./AASIST.pth
```

## Option B — Pre-fine-tuned deepfake XLS-R (HuggingFace)
This is a wav2vec2-large-xlsr model already fine-tuned for deepfake detection.
Use this as the starting checkpoint for Hindi fine-tuning:

```bash
python -c "
from transformers import AutoModelForAudioClassification
model = AutoModelForAudioClassification.from_pretrained(
    'Gustking/wav2vec2-large-xlsr-deepfake-audio-classification'
)
import torch
torch.save(model.state_dict(), 'xlsr_deepfake_base.pt')
"
```

## Option C — After local fine-tuning
Run `training/finetune_xlsr_hindi_local.py` to produce:
  `weights/aasist_hindi_v1.pt`

## VRAM guidance (RTX 4050 6 GB)
| Model | VRAM (FP16) |
|-------|------------|
| AASIST only | ~50 MB |
| XLS-R 300M + AASIST | ~2.2 GB total |
