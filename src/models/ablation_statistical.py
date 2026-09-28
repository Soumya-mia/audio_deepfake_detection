"""
ablation_statistical.py - Run ablation 3x with different seeds for statistical significance.
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
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
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
    return np.array(X_list), np.array(y_list)


def train_once(X_train, y_train, X_test, y_test, in_channels, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    
    for epoch in range(30):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx.to(device)).argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(by.numpy())
    return accuracy_score(labels_all, preds_all) * 100


if __name__ == "__main__":
    print("=" * 70)
    print("STATISTICAL ABLATION (3 seeds)")
    print("=" * 70)
    
    loader = AudioLoader("data/asvspoof2019/")
    
    print("\nBuilding dataset...")
    X_train_full, y_train = build_dataset(loader, loader.get_dataset('train'), 500, "TRAIN")
    X_eval_full, y_eval = build_dataset(loader, loader.get_dataset('eval'), 200, "EVAL")
    print(f"Train: {X_train_full.shape}, Eval: {X_eval_full.shape}")
    
    seeds = [42, 123, 2024]
    results = {'magnitude': [], 'phase': [], 'fused': []}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED = {seed}")
        print(f"{'='*70}")
        
        # Magnitude
        acc_m = train_once(X_train_full[:, 0:1], y_train, X_eval_full[:, 0:1], y_eval, 1, seed)
        print(f"  Magnitude: {acc_m:.2f}%")
        results['magnitude'].append(acc_m)
        
        # Phase
        acc_p = train_once(X_train_full[:, 1:4], y_train, X_eval_full[:, 1:4], y_eval, 3, seed)
        print(f"  Phase:     {acc_p:.2f}%")
        results['phase'].append(acc_p)
        
        # Fused
        acc_f = train_once(X_train_full, y_train, X_eval_full, y_eval, 4, seed)
        print(f"  Fused:     {acc_f:.2f}%")
        results['fused'].append(acc_f)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 STATISTICAL SUMMARY (mean ± std over 3 seeds)")
    print("=" * 70)
    for name, accs in results.items():
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"  {name.capitalize():<12} {mean:.2f}% ± {std:.2f}%   runs: {[f'{a:.2f}' for a in accs]}")
    
    # Phase contribution
    mag_mean = np.mean(results['magnitude'])
    fused_mean = np.mean(results['fused'])
    print(f"\n  Phase contribution (Fused - Magnitude): {fused_mean - mag_mean:+.2f}%")
    
    # Save
    with open('results/ablation_statistical.txt', 'w') as f:
        f.write("STATISTICAL ABLATION (3 seeds)\n")
        f.write("=" * 50 + "\n")
        for name, accs in results.items():
            f.write(f"{name}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  ({accs})\n")
        f.write(f"\nPhase contribution: {fused_mean - mag_mean:+.2f}%\n")
    print("\n💾 Saved to results/ablation_statistical.txt")