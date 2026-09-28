"""
ablation_v2.py - FAIR ablation study: 50 epochs for all 3 models.
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


def train_and_evaluate(X_train, y_train, X_test, y_test, in_channels, config_name, epochs=50):
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
    
    print(f"    Training {config_name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}")
    
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
    print("FAIR ABLATION STUDY (50 epochs for all models)")
    print("=" * 70)
    
    loader = AudioLoader("data/asvspoof2019/")
    
    print("\n[1] Building shared dataset...")
    X_train_full, y_train = build_dataset(loader, loader.get_dataset('train'), 500, "TRAIN")
    X_eval_full, y_eval = build_dataset(loader, loader.get_dataset('eval'), 200, "EVAL")
    print(f"  Train: {X_train_full.shape}, Eval: {X_eval_full.shape}")
    
    # 50 epochs for fair comparison
    EPOCHS = 50
    
    print("\n" + "=" * 70)
    print("[2] Magnitude-only (Log-Mel)")
    print("=" * 70)
    acc_mag = train_and_evaluate(
        X_train_full[:, 0:1], y_train,
        X_eval_full[:, 0:1], y_eval,
        in_channels=1, config_name="Magnitude-only", epochs=EPOCHS
    )
    print(f"  ✅ Magnitude-only: {acc_mag*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("[3] Phase-only (Phase + Phase Diff + Group Delay)")
    print("=" * 70)
    acc_ph = train_and_evaluate(
        X_train_full[:, 1:4], y_train,
        X_eval_full[:, 1:4], y_eval,
        in_channels=3, config_name="Phase-only", epochs=EPOCHS
    )
    print(f"  ✅ Phase-only: {acc_ph*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("[4] Fused (all 4 channels)")
    print("=" * 70)
    acc_fused = train_and_evaluate(
        X_train_full, y_train,
        X_eval_full, y_eval,
        in_channels=4, config_name="Fused", epochs=EPOCHS
    )
    print(f"  ✅ Fused: {acc_fused*100:.2f}%")
    
    # ==========================================
    # Summary Table
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 FAIR ABLATION STUDY RESULTS (50 epochs)")
    print("=" * 70)
    print(f"{'Model':<35} {'Accuracy':>10}")
    print("-" * 70)
    print(f"{'Magnitude-only':<35} {acc_mag*100:>9.2f}%")
    print(f"{'Phase-only':<35} {acc_ph*100:>9.2f}%")
    print(f"{'Fused (Proposed)':<35} {acc_fused*100:>9.2f}%")
    print("-" * 70)
    print(f"{'Phase contribution to fusion:':<35} {(acc_fused - acc_mag)*100:>+9.2f}%")
    print(f"{'Phase vs Magnitude alone:':<35} {(acc_ph - acc_mag)*100:>+9.2f}%")
    print("=" * 70)
    
    with open('results/ablation_v2.txt', 'w') as f:
        f.write("FAIR ABLATION STUDY (50 epochs)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Magnitude-only:     {acc_mag*100:.2f}%\n")
        f.write(f"Phase-only:         {acc_ph*100:.2f}%\n")
        f.write(f"Fused (Proposed):   {acc_fused*100:.2f}%\n")
    print("\n💾 Saved to: results/ablation_v2.txt")