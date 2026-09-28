"""
phase_diagnostic.py - Investigate the phase-only generalization asymmetry.

Trains phase-only model for 50 epochs, evaluating on BOTH dev (seen attacks)
and eval (unseen attacks) at every 10-epoch checkpoint.
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


class PhaseOnlyCNN(nn.Module):
    """Same architecture but takes 3 phase channels as input."""
    def __init__(self, in_channels=3, num_classes=2, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super(PhaseOnlyCNN, self).__init__()
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


def build_dataset(loader, samples, n_per_class, tag=""):
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
            print(f"      {tag}: {i+1}/{len(balanced)}")
    return np.array(X_list), np.array(y_list)


def evaluate(model, loader_, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in loader_:
            preds = model(bx.to(device)).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(by.numpy())
    return accuracy_score(labels_all, preds_all)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE-ONLY DIAGNOSTIC: Is the 50% a real phenomenon?")
    print("=" * 70)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    loader = AudioLoader("data/asvspoof2019/")
    
    # Build datasets
    print("\n[1] Building datasets...")
    print("  Train:")
    X_train_full, y_train = build_dataset(loader, loader.get_dataset('train'), 500, "TRAIN")
    print("  Dev:")
    X_dev_full, y_dev = build_dataset(loader, loader.get_dataset('dev'), 200, "DEV")
    print("  Eval:")
    X_eval_full, y_eval = build_dataset(loader, loader.get_dataset('eval'), 200, "EVAL")
    
    # Extract phase channels only (1, 2, 3)
    X_train_phase = X_train_full[:, 1:4, :, :]
    X_dev_phase = X_dev_full[:, 1:4, :, :]
    X_eval_phase = X_eval_full[:, 1:4, :, :]
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_phase), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_dev_phase), torch.LongTensor(y_dev)),
        batch_size=32, shuffle=False,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_eval_phase), torch.LongTensor(y_eval)),
        batch_size=32, shuffle=False,
    )
    
    # Initialize model
    model = PhaseOnlyCNN(in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print("\n[2] Training for 50 epochs with checkpoint evaluations...")
    
    history = []  # (epoch, train_loss, dev_acc, eval_acc)
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            dev_acc = evaluate(model, dev_loader, device) * 100
            eval_acc = evaluate(model, eval_loader, device) * 100
            history.append((epoch + 1, avg_loss, dev_acc, eval_acc))
            print(f"Epoch {epoch+1:2d}/50 | Loss: {avg_loss:.4f} | Dev: {dev_acc:5.2f}% | Eval: {eval_acc:5.2f}%")
    
    # ==========================================
    # Summary Table
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 PHASE-ONLY LEARNING CURVE")
    print("=" * 70)
    print(f"{'Epoch':>6} {'Loss':>10} {'DEV Acc':>10} {'EVAL Acc':>10}")
    print("-" * 70)
    for epoch, loss, dev_acc, eval_acc in history:
        print(f"{epoch:>6} {loss:>10.4f} {dev_acc:>9.2f}% {eval_acc:>9.2f}%")
    print("=" * 70)
    
    # Save
    with open('results/phase_diagnostic.txt', 'w') as f:
        f.write("PHASE-ONLY DIAGNOSTIC RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Epoch':>6} {'Loss':>10} {'DEV Acc':>10} {'EVAL Acc':>10}\n")
        for epoch, loss, dev_acc, eval_acc in history:
            f.write(f"{epoch:>6} {loss:>10.4f} {dev_acc:>9.2f}% {eval_acc:>9.2f}%\n")
    
    print("\n💾 Results saved to: results/phase_diagnostic.txt")
    
    # ==========================================
    # Interpret the result
    # ==========================================
    final_dev = history[-1][2]
    final_eval = history[-1][3]
    
    print("\n" + "=" * 70)
    print("🧠 INTERPRETATION")
    print("=" * 70)
    
    if final_dev > 80 and final_eval > 80:
        print("✅ RESULT: Phase features ARE discriminative on both seen and unseen attacks")
        print("   → The 50% earlier was UNDERTRAINING, not a real phenomenon")
        print("   → Fusion with magnitude should improve cross-attack accuracy")
    elif final_dev < 60 and final_eval > 80:
        print("🔥 MAJOR FINDING: Phase features are ATTACK-AGNOSTIC but SPEAKER-SENSITIVE")
        print("   → Phase overfits to speaker identity (fails on dev: different speakers)")
        print("   → But captures universal synthesis anomalies (succeeds on eval: same logic)")
        print("   → This is a PUBLISHABLE, novel result")
    elif final_dev < 60 and final_eval < 60:
        print("⚠️ RESULT: Phase features are NOT discriminative at all")
        print("   → The 86.5% on eval was likely statistical noise in the small sample")
        print("   → We should expand the eval sample size to verify")
    else:
        print("❓ MIXED RESULT — needs further investigation")
    print("=" * 70)