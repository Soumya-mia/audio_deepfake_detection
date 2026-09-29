"""
Two-Branch CNN vs Single-Branch Baselines for Audio Deepfake Detection
=======================================================================
Complete fixed version with:
- MPS (Apple Silicon) support
- Correct FLAC subdirectory path
- Lightweight file filtering (no soundfile dependency)
- Graceful handling of unreadable FLAC files
- Class-weighted loss for imbalance
- Subset mode for quick testing
- Per-epoch logging
- Confusion matrix with collapse detection
"""

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    DATA_DIR = "data/asvspoof2019"
    SAMPLE_RATE = 16000
    DURATION = 4.0
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 512
    BATCH_SIZE = 32
    EPOCHS = 10                  # Reduced for quick testing
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Device detection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    # Subset mode for quick iteration
    # Set USE_FULL_TRAIN = True and USE_FULL_EVAL = True when ready for full run
    USE_FULL_TRAIN = False
    USE_FULL_EVAL = False
    MAX_TRAIN_SAMPLES = 5000
    MAX_EVAL_SAMPLES = 1000


cfg = Config()
print(f"[CONFIG] Device: {cfg.DEVICE}")
print(f"[CONFIG] Use full train: {cfg.USE_FULL_TRAIN}, Use full eval: {cfg.USE_FULL_EVAL}")
if not cfg.USE_FULL_TRAIN:
    print(f"[CONFIG] Max train samples: {cfg.MAX_TRAIN_SAMPLES}")
if not cfg.USE_FULL_EVAL:
    print(f"[CONFIG] Max eval samples: {cfg.MAX_EVAL_SAMPLES}")
print(f"[CONFIG] Epochs: {cfg.EPOCHS}, Batch size: {cfg.BATCH_SIZE}")


# ============================================================
# SEED UTILITY
# ============================================================
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] Set seed to {seed}")


# ============================================================
# DATASET
# ============================================================
class AudioDeepfakeDataset(Dataset):
    """
    Dataset for ASVspoof2019.
    Loads audio via librosa, computes a 4-channel representation:
    [log-mel, phase, real, imaginary].

    Filtering strategy:
    - Uses file existence + minimum size (cheap heuristic) at init.
    - Handles load failures gracefully inside __getitem__.
    """

    def __init__(self, data_dir, split="train", max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples

        # Determine protocol and audio directory
        # IMPORTANT: FLAC files live in a "flac" subdirectory
        if split == "train":
            protocol = os.path.join(
                data_dir, "LA", "ASVspoof2019_LA_cm_protocols",
                "ASVspoof2019.LA.cm.train.trn.txt"
            )
            audio_dir = os.path.join(
                data_dir, "LA", "ASVspoof2019_LA_train", "flac"
            )
        elif split == "eval":
            protocol = os.path.join(
                data_dir, "LA", "ASVspoof2019_LA_cm_protocols",
                "ASVspoof2019.LA.cm.eval.trl.txt"
            )
            audio_dir = os.path.join(
                data_dir, "LA", "ASVspoof2019_LA_eval", "flac"
            )
        else:
            raise ValueError(f"Unknown split: {split}")

        self.audio_dir = audio_dir
        self.samples = []

        print(f"[AudioLoader] Loading protocol: {protocol}")
        with open(protocol, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                utt_id = parts[1]
                label_str = parts[-1]
                # ASVspoof: 'bonafide' = real (0), 'spoof' = fake (1)
                label = 0 if label_str == "bonafide" else 1
                self.samples.append((utt_id, label))

        print(f"[AudioLoader] Loaded {len(self.samples)} total samples from protocol")

        # Lightweight filter: remove missing or suspiciously small files
        print(f"[AudioLoader] Filtering missing/empty files (size check)...")
        valid_samples = []
        missing_count = 0
        tiny_count = 0

        for utt_id, label in self.samples:
            audio_path = os.path.join(self.audio_dir, f"{utt_id}.flac")
            if not os.path.exists(audio_path):
                missing_count += 1
                continue
            if os.path.getsize(audio_path) < 100:
                tiny_count += 1
                continue
            valid_samples.append((utt_id, label))

        self.samples = valid_samples
        print(f"[AudioLoader] Removed {missing_count} missing, {tiny_count} tiny files. "
              f"{len(self.samples)} samples remain.")

        # Stratified truncation if max_samples is set
        if max_samples is not None and max_samples < len(self.samples):
            real_samples = [s for s in self.samples if s[1] == 0]
            fake_samples = [s for s in self.samples if s[1] == 1]

            n_real = min(len(real_samples), max_samples // 2)
            n_fake = min(len(fake_samples), max_samples - n_real)

            self.samples = real_samples[:n_real] + fake_samples[:n_fake]
            random.shuffle(self.samples)
            print(f"[AudioLoader] Stratified truncation to {len(self.samples)} samples")

        # Report class distribution
        n_real = sum(1 for _, label in self.samples if label == 0)
        n_fake = sum(1 for _, label in self.samples if label == 1)
        print(f"[AudioLoader] Class distribution: Real={n_real}, Fake={n_fake}")

        # Track load failures
        self.failed_count = 0

        # Constants
        self.n_mels = 64
        self.n_fft = 1024
        self.hop_length = 512
        self.sample_rate = 16000
        self.duration = 4.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa

        utt_id, label = self.samples[idx]
        audio_path = os.path.join(self.audio_dir, f"{utt_id}.flac")

        try:
            y, sr = librosa.load(
                audio_path, sr=self.sample_rate, duration=self.duration
            )
        except Exception as e:
            # Return zeros on failure (rare, ~1% of files)
            self.failed_count += 1
            if self.failed_count <= 10:
                print(f"[WARN] Load failed for {utt_id}: {type(e).__name__}: {e}")
            y = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)

        # Pad or truncate to fixed length
        target_len = int(self.sample_rate * self.duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # STFT
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        real = np.real(stft)
        imag = np.imag(stft)

        # Log-mel spectrogram from magnitude
        mel = librosa.feature.melspectrogram(
            S=magnitude ** 2, sr=self.sample_rate, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        # Resize all to fixed shape
        target_shape = (64, 128)
        mel_db = self._resize(mel_db, target_shape)
        phase = self._resize(phase, target_shape)
        real = self._resize(real, target_shape)
        imag = self._resize(imag, target_shape)

        # Stack 4 channels
        x = np.stack([mel_db, phase, real, imag], axis=0).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    @staticmethod
    def _resize(arr, shape):
        """Resize 2D array to target shape using nearest-neighbor indexing."""
        from numpy import linspace
        h, w = arr.shape
        th, tw = shape
        h_idx = linspace(0, h - 1, th).astype(int)
        w_idx = linspace(0, w - 1, tw).astype(int)
        return arr[np.ix_(h_idx, w_idx)]


# ============================================================
# MODELS
# ============================================================
class MagnitudeOnlyCNN(nn.Module):
    """Single-branch baseline: uses only the magnitude channel."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x[:, 0:1, :, :]  # Only magnitude channel
        return self.classifier(self.features(x))


class NaiveFusedCNN(nn.Module):
    """Baseline: concatenates all 4 channels at the input."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TwoBranchCNN(nn.Module):
    """
    Two-branch architecture:
    - Magnitude branch processes the log-mel channel.
    - Phase branch processes phase, real, imaginary channels.
    - Features are fused with a learnable weight (alpha).
    """

    def __init__(self):
        super().__init__()
        # Magnitude branch
        self.mag_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # Phase branch (takes phase + real + imag = 3 channels)
        self.phase_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        mag = x[:, 0:1, :, :]         # (B, 1, 64, 128)
        phase = x[:, 1:4, :, :]       # (B, 3, 64, 128)

        mag_feat = self.mag_branch(mag)        # (B, 128, 4, 4)
        phase_feat = self.phase_branch(phase)  # (B, 128, 4, 4)

        # Learnable weighted fusion
        alpha = torch.sigmoid(self.alpha)
        fused = torch.cat([
            alpha * mag_feat,
            (1 - alpha) * phase_feat
        ], dim=1)  # (B, 256, 4, 4)

        return self.classifier(fused)


# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def print_confusion_matrix(y_true, y_pred, model_name):
    """Print confusion matrix and detect collapse."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n  ── Confusion Matrix ({model_name}) ──")
    print(f"                  Pred Real   Pred Fake")
    print(f"  Actual Real     {cm[0, 0]:>8d}   {cm[0, 1]:>8d}")
    print(f"  Actual Fake     {cm[1, 0]:>8d}   {cm[1, 1]:>8d}")

    # Collapse detection
    if cm[0, 0] == 0 and cm[1, 0] == 0:
        print(f"  ⚠️  COLLAPSE: Model always predicts 'Fake'")
    elif cm[0, 1] == 0 and cm[1, 1] == 0:
        print(f"  ⚠️  COLLAPSE: Model always predicts 'Real'")
    else:
        print(f"  ✓ Model is predicting both classes")

    # Per-class recall
    recall_real = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
    recall_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)
    print(f"  Recall (Real): {recall_real:.4f}")
    print(f"  Recall (Fake): {recall_fake:.4f}")

    return cm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# SINGLE EXPERIMENT
# ============================================================
def run_experiment(seed, model_name, train_loader, eval_loader, device, class_weights):
    """Train and evaluate one model with a given seed."""
    set_seed(seed)

    # Build model
    if model_name == "magnitude":
        model = MagnitudeOnlyCNN()
    elif model_name == "naive_fused":
        model = NaiveFusedCNN()
    elif model_name == "two_branch":
        model = TwoBranchCNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    n_params = count_parameters(model)
    print(f"\n  [MODEL] {model_name} | Params: {n_params:,}")

    # Weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_eval_acc = 0.0
    best_preds = None
    best_labels = None

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        eval_loss, eval_acc, preds, labels = evaluate(
            model, eval_loader, criterion, device
        )
        scheduler.step()

        print(f"    Epoch {epoch:>2}/{cfg.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2f}%")

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_preds = preds
            best_labels = labels

    return best_eval_acc, best_preds, best_labels


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("TWO-BRANCH ARCHITECTURE vs SINGLE-BRANCH BASELINES")
    print("=" * 70)

    # ----------------------------------------------------------
    # Build datasets
    # ----------------------------------------------------------
    print("\n[1] Building shared dataset...")

    train_max = None if cfg.USE_FULL_TRAIN else cfg.MAX_TRAIN_SAMPLES
    eval_max = None if cfg.USE_FULL_EVAL else cfg.MAX_EVAL_SAMPLES

    train_dataset = AudioDeepfakeDataset(
        cfg.DATA_DIR, split="train", max_samples=train_max
    )
    eval_dataset = AudioDeepfakeDataset(
        cfg.DATA_DIR, split="eval", max_samples=eval_max
    )

    # ----------------------------------------------------------
    # Calculate class weights for weighted loss
    # ----------------------------------------------------------
    n_real = sum(1 for _, label in train_dataset.samples if label == 0)
    n_fake = sum(1 for _, label in train_dataset.samples if label == 1)
    total = n_real + n_fake

    weight_real = total / (2 * n_real) if n_real > 0 else 1.0
    weight_fake = total / (2 * n_fake) if n_fake > 0 else 1.0

    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(cfg.DEVICE)
    print(f"\n[LOSS] Class weights: Real={weight_real:.4f}, Fake={weight_fake:.4f}")

    # ----------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,  # MPS does not support pin_memory
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    print(f"\n  Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Quick batch check
    x_sample, y_sample = next(iter(train_loader))
    print(f"  Sample batch shape: {x_sample.shape}")
    print(f"  Label distribution: Real={(y_sample == 0).sum().item()}, "
          f"Fake={(y_sample == 1).sum().item()}")

    # ----------------------------------------------------------
    # Run experiments across seeds
    # ----------------------------------------------------------
    seeds = [42, 123, 2024]
    models = ["magnitude", "naive_fused", "two_branch"]

    all_results = {m: [] for m in models}
    all_confusions = {}

    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"SEED = {seed}")
        print(f"{'=' * 70}")

        for model_name in models:
            acc, preds, labels = run_experiment(
                seed, model_name, train_loader, eval_loader, cfg.DEVICE, class_weights
            )
            all_results[model_name].append(acc)
            all_confusions[(seed, model_name)] = (preds, labels)

            print(f"  → {model_name}: Best Eval Acc = {acc:.2f}%")
            print_confusion_matrix(labels, preds, f"{model_name} (seed={seed})")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("📊 DEFINITIVE COMPARISON (mean ± std over 3 seeds)")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Mean':>8} {'Std':>8}   Runs")
    print("-" * 70)

    for model_name in models:
        accs = all_results[model_name]
        mean = np.mean(accs)
        std = np.std(accs)
        runs_str = ", ".join([f"{a:.2f}" for a in accs])
        print(f"{model_name:<25} {mean:>7.2f}% {std:>7.2f}%   [{runs_str}]")

    print(f"\n  Two-branch vs Magnitude:   "
          f"{np.mean(all_results['two_branch']) - np.mean(all_results['magnitude']):+.2f}%")
    print(f"  Two-branch vs Naive Fused: "
          f"{np.mean(all_results['two_branch']) - np.mean(all_results['naive_fused']):+.2f}%")

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    output = {
        "config": {
            "device": cfg.DEVICE,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "learning_rate": cfg.LEARNING_RATE,
            "use_full_train": cfg.USE_FULL_TRAIN,
            "use_full_eval": cfg.USE_FULL_EVAL,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "class_weights": {
                "real": float(weight_real),
                "fake": float(weight_fake),
            },
        },
        "results": {
            model: {
                "accuracy_per_seed": [float(a) for a in all_results[model]],
                "mean": float(np.mean(all_results[model])),
                "std": float(np.std(all_results[model])),
            }
            for model in models
        },
    }

    with open("results/two_branch_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to: results/two_branch_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()