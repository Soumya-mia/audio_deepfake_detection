"""
ablation_study.py - Quantify the contribution of phase features.

Compares 3 models:
1. Magnitude-only (Log-Mel channel only)
2. Phase-only (Phase + Phase Diff + Group Delay)
3. Fused (all 4 channels) - the proposed model
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from src.preprocessing.audio_loader import AudioLoader
from src.feature_extraction.build_sequence_dataset import extract_sequence_features


class AblationCNN(nn.Module):
    """Same architecture as CNN+BiLSTM but with configurable input channels."""
    def __init__(self, in_channels=4, num_classes=2, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super(AblationCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 8, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 3, 1, 2)
        batch, time, channels, freq = x.shape
        x = x.reshape(batch, time, channels * freq)
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn * lstm_out).sum(dim=1)
        return self.classifier(context)


def build_dataset(loader, samples, n_per_class=500):
    """Build balanced dataset from a list of (path, label) samples."""
    real = [(p, l) for p, l in samples if l == 'bonafide'][:n_per_class]
    fake = [(p, l) for p, l in samples if l == 'spoof'][:n_per_class]
    balanced = real + fake
    np.random.shuffle(balanced)
    
    X_list, y_list = [], []
    for i, (path, label) in enumerate(balanced):
        sig, sr = loader.load_audio(path)
        X_list.append(extract_sequence_features(sig, sr))
        y_list.append(0 if label == 'bonafide' else 1)
        if (i + 1) % 200 == 0:
            print(f"      Processed {i+1}/{len(balanced)}")
    
    return np.array(X_list), np.array(y_list)


def train_model(X_train, y_train, X_test, y_test, in_channels, config_name):
    """Train one ablation model and return accuracy."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=32, shuffle=False,
    )
    
    model = AblationCNN(in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"    Training {config_name}...")
    for epoch in range(20):  # 20 epochs for speed
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(by.numpy())
    
    return accuracy_score(all_labels, all_preds)


if __name__ == "__main__":
    print("=" * 70)
    print("ABLATION STUDY: Contribution of Phase Features")
    print("=" * 70)
    
    loader = AudioLoader("data/asvspoof2019/")
    
    # ==========================================
    # Load data ONCE (shared across all experiments)
    # ==========================================
    print("\n[1] Building shared dataset...")
    print("  Loading TRAIN split...")
    train_samples = loader.get_dataset('train')
    X_train_full, y_train = build_dataset(loader, train_samples, n_per_class=500)
    print(f"  Train tensor: {X_train_full.shape}")
    
    print("\n  Loading EVAL split (unseen attacks)...")
    eval_samples = loader.get_dataset('dev')
    X_test_full, y_test = build_dataset(loader, eval_samples, n_per_class=200)
    print(f"  Eval tensor: {X_test_full.shape}")
    
    # ==========================================
    # Experiment 1: Magnitude-only (channel 0)
    # ==========================================
    print("\n" + "=" * 70)
    print("[2] EXPERIMENT 1: Magnitude-only (Log-Mel)")
    print("=" * 70)
    X_train_mag = X_train_full[:, 0:1, :, :]  # Keep channel 0 only
    X_test_mag = X_test_full[:, 0:1, :, :]
    acc_mag = train_model(X_train_mag, y_train, X_test_mag, y_test, 
                          in_channels=1, config_name="Magnitude-only")
    print(f"  ✅ Magnitude-only accuracy: {acc_mag*100:.2f}%")
    
    # ==========================================
    # Experiment 2: Phase-only (channels 1,2,3)
    # ==========================================
    print("\n" + "=" * 70)
    print("[3] EXPERIMENT 2: Phase-only (Phase + Phase Diff + Group Delay)")
    print("=" * 70)
    X_train_ph = X_train_full[:, 1:4, :, :]  # Keep channels 1,2,3
    X_test_ph = X_test_full[:, 1:4, :, :]
    acc_ph = train_model(X_train_ph, y_train, X_test_ph, y_test,
                         in_channels=3, config_name="Phase-only")
    print(f"  ✅ Phase-only accuracy: {acc_ph*100:.2f}%")
    
    # ==========================================
    # Experiment 3: Fused (all 4 channels)
    # ==========================================
    print("\n" + "=" * 70)
    print("[4] EXPERIMENT 3: Fused (Magnitude + Phase)")
    print("=" * 70)
    acc_fused = train_model(X_train_full, y_train, X_test_full, y_test,
                            in_channels=4, config_name="Fused")
    print(f"  ✅ Fused accuracy: {acc_fused*100:.2f}%")
    
    # ==========================================
    # Final Comparison Table
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Model':<35} {'Accuracy':>10}")
    print("-" * 70)
    print(f"{'Magnitude-only':<35} {acc_mag*100:>9.2f}%")
    print(f"{'Phase-only':<35} {acc_ph*100:>9.2f}%")
    print(f"{'Fused (Proposed)':<35} {acc_fused*100:>9.2f}%")
    print("-" * 70)
    print(f"{'Phase contribution:':<35} {(acc_fused - acc_mag)*100:>+9.2f}%")
    print("=" * 70)
    
    # Save results
    with open('results/ablation_study.txt', 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Magnitude-only:     {acc_mag*100:.2f}%\n")
        f.write(f"Phase-only:         {acc_ph*100:.2f}%\n")
        f.write(f"Fused (Proposed):   {acc_fused*100:.2f}%\n")
        f.write(f"Phase contribution: {(acc_fused - acc_mag)*100:+.2f}%\n")
    print("\n💾 Results saved to: results/ablation_study.txt")