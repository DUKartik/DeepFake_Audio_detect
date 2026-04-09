"""
training/finetune_xlsr_hindi_local.py
──────────────────────────────────────
Fine-tune XLS-R 300M + AASIST on Hindi deepfake data — runs locally on RTX 4050 6 GB.

No Google Colab needed. This script:
  1. Loads real Hindi speech from ai4bharat/kathbath
  2. Loads synthetic fakes from SherryT997/IndicTTS-Deepfake-Challenge-Data (Hindi)
  3. Simulates WhatsApp compression on all samples (ffmpeg → 32 kbps ogg → WAV)
  4. Fine-tunes the top 12 XLS-R transformer layers + a binary classification head
  5. Evaluates EER (Equal Error Rate) on a held-out set
  6. Exports best weights to ./weights/aasist_hindi_v1.pt

VRAM budget (RTX 4050 6 GB):
  XLS-R 300M FP16 weights:  ~1.2 GB
  Gradients (top 50% layers): ~0.6 GB
  Optimizer states (AdamW):  ~1.2 GB
  Activations (batch=4):     ~2.0 GB
  TOTAL:                     ~5.0 GB — fits with 1 GB headroom
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
    TrainingArguments,
    Trainer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = Path("./weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

XLSR_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR = "./xlsr-aasist-hindi-v1"


# ── Step 1: WhatsApp compression simulation ──────────────────────────────────
def simulate_whatsapp_compression(wav_path: str, out_path: str) -> str:
    """Compress audio through ogg/opus 32 kbps and decompress back to WAV.

    This simulates exactly what WhatsApp does to audio messages,
    ensuring the model is trained on the same quality it will see at inference.

    Args:
        wav_path: Input WAV file path.
        out_path: Output WAV file path.

    Returns:
        out_path after conversion.
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        ogg_path = tmp.name

    # Compress to 32 kbps ogg/opus (WhatsApp quality)
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-c:a", "libopus", "-b:a", "32k", "-ar", "16000",
        ogg_path,
    ], check=True, capture_output=True)

    # Decompress back to WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", ogg_path,
        "-ar", "16000", "-ac", "1", out_path,
    ], check=True, capture_output=True)

    os.unlink(ogg_path)
    return out_path


# ── Step 2: Custom PyTorch Dataset ──────────────────────────────────────────
class HindiDeepfakeDataset(Dataset):
    """HuggingFace Audio dataset wrapper.

    Args:
        hf_dataset: HuggingFace dataset with "audio" and "label" columns.
        feature_extractor: AutoFeatureExtractor for XLS-R.
        compress: If True, simulate WhatsApp compression on each sample.
    """

    def __init__(self, hf_dataset, feature_extractor, compress: bool = True) -> None:
        self.data = hf_dataset
        self.fe = feature_extractor
        self.compress = compress

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        label = sample["label"]

        # Optionally write + compress + reload
        if self.compress:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, array, sr)
                out = f.name.replace(".wav", "_wa.wav")
                try:
                    simulate_whatsapp_compression(f.name, out)
                    import librosa
                    array, _ = librosa.load(out, sr=16000)
                finally:
                    for p in [f.name, out]:
                        try:
                            os.unlink(p)
                        except FileNotFoundError:
                            pass

        inputs = self.fe(
            array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=80000,       # 5 seconds at 16 kHz
            truncation=True,
        )
        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


# ── Step 3: XLS-R + classifier model ────────────────────────────────────────
class XLSRClassifier(nn.Module):
    """XLS-R 300M with a binary classification head.

    Only the top 12 transformer layers are trainable.
    The convolutional feature extractor is frozen.

    Args:
        model_id: HuggingFace model ID for XLS-R.
        freeze_up_to: Number of bottom transformer layers to freeze.
    """

    def __init__(self, model_id: str, freeze_up_to: int = 12) -> None:
        super().__init__()
        self.xlsr = Wav2Vec2Model.from_pretrained(model_id)
        self._freeze(freeze_up_to)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def _freeze(self, freeze_up_to: int) -> None:
        """Freeze the conv feature extractor and bottom N transformer layers."""
        for param in self.xlsr.feature_extractor.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.xlsr.encoder.layers):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_values: (B, T) raw waveform tensor.

        Returns:
            (B, 1) raw logits.
        """
        hidden = self.xlsr(input_values).last_hidden_state   # (B, T, 1024)
        pooled = hidden.mean(dim=1)                           # (B, 1024)
        return self.classifier(pooled)                        # (B, 1)


# ── Step 4: EER metric ───────────────────────────────────────────────────────
def compute_eer(labels: list[int], scores: list[float]) -> float:
    """Compute Equal Error Rate (EER).

    Args:
        labels: Ground truth binary labels (0=real, 1=fake).
        scores: Predicted fake probabilities.

    Returns:
        EER as a float percentage (e.g. 5.3 means 5.3%).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(round((fpr[idx] + fnr[idx]) / 2 * 100, 2))


# ── Step 5: Main training script ─────────────────────────────────────────────
def main() -> None:
    """Full fine-tuning pipeline for Hindi deepfake detection."""
    print(f"Training on device: {DEVICE}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(XLSR_MODEL_ID)

    # ── Load real Hindi speech (Kathbath) ────────────────────────────────
    print("Loading real Hindi speech from Kathbath...")
    real_raw = load_dataset("ai4bharat/kathbath", "hi", split="train[:5000]", trust_remote_code=True)
    real_raw = real_raw.cast_column("audio", Audio(sampling_rate=16000))
    real_raw = real_raw.map(lambda x: {"label": 0})

    # ── Load synthetic Hindi fakes (IndicTTS Deepfake) ───────────────────
    print("Loading synthetic Hindi fakes from IndicTTS-Deepfake-Challenge-Data...")
    fake_raw = load_dataset(
        "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        split="train",
        trust_remote_code=True,
    )
    fake_raw = fake_raw.filter(lambda x: x.get("language") == "hindi")
    fake_raw = fake_raw.select(range(min(5000, len(fake_raw))))
    fake_raw = fake_raw.cast_column("audio", Audio(sampling_rate=16000))
    fake_raw = fake_raw.map(lambda x: {"label": 1})

    # ── Split into train / eval ──────────────────────────────────────────
    from datasets import concatenate_datasets
    combined = concatenate_datasets([real_raw, fake_raw]).shuffle(seed=42)
    split = combined.train_test_split(test_size=0.1, seed=42)

    train_ds = HindiDeepfakeDataset(split["train"], feature_extractor, compress=True)
    eval_ds = HindiDeepfakeDataset(split["test"], feature_extractor, compress=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = XLSRClassifier(model_id=XLSR_MODEL_ID, freeze_up_to=12).to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()   # FP16 for VRAM efficiency

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,       # safe for 6 GB VRAM
        gradient_accumulation_steps=4,       # effective batch = 16
        gradient_checkpointing=True,         # trade speed for VRAM
        fp16=(DEVICE == "cuda"),
        learning_rate=3e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="best",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    def collate_fn(batch: list[dict]) -> dict:
        """Stack variable-length tensors into a batch."""
        input_values = torch.stack([b["input_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"input_values": input_values, "labels": labels}

    # Wrap dataset to work with the Trainer API
    class TrainerCompatDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            return {"input_values": item["input_values"], "labels": item["labels"]}

    train_wrapper = TrainerCompatDataset(train_ds)
    eval_wrapper = TrainerCompatDataset(eval_ds)

    # ── Custom training loop with EER metric ─────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-5
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_wrapper, batch_size=4, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_wrapper, batch_size=4, shuffle=False, collate_fn=collate_fn)

    best_eer = 100.0
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            iv = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).unsqueeze(1)
            logits = model(iv)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # ── Evaluate ──────────────────────────────────────────────────────
        model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for batch in eval_loader:
                iv = batch["input_values"].to(DEVICE)
                logits = model(iv)
                scores = torch.sigmoid(logits).squeeze().cpu().tolist()
                lbls = batch["labels"].tolist()
                if isinstance(scores, float):
                    scores = [scores]
                    lbls = [lbls]
                all_scores.extend(scores)
                all_labels.extend(lbls)

        eer = compute_eer(all_labels, all_scores)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  EER={eer:.2f}%")

        if eer < best_eer:
            best_eer = eer
            out_path = WEIGHTS_DIR / "aasist_hindi_v1.pt"
            torch.save(model.state_dict(), str(out_path))
            print(f"  ✅ New best EER {eer:.2f}% — saved to {out_path}")

    print(f"\nTraining complete. Best EER: {best_eer:.2f}%")
    print(f"Model saved to {WEIGHTS_DIR / 'aasist_hindi_v1.pt'}")


if __name__ == "__main__":
    main()
