"""
cnn_bilstm.py - CNN + BiLSTM with Attention for audio deepfake detection.
Tested on the OFFICIAL ASVspoof 2019 EVAL split (unseen attacks A07-A19).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocessing.audio_loader import AudioLoader
from src.feature_extraction.build_sequence_dataset import extract_sequence_features


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
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


if __name__ == "__main__":
    print("=" * 60)
    print("CNN + BiLSTM - Official ASVspoof EVAL Split (Unseen Attacks)")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    loader = AudioLoader("data/asvspoof2019/")
    
    # ==========================================
    # TRAIN (attacks A01-A06)
    # ==========================================
    print("\n[1/3] Loading TRAIN split (attacks A01-A06)...")
    train_samples = loader.get_dataset('train')
    real_train = [(p, l) for p, l in train_samples if l == 'bonafide'][:500]
    fake_train = [(p, l) for p, l in train_samples if l == 'spoof'][:500]
    balanced_train = real_train + fake_train
    np.random.shuffle(balanced_train)
    
    X_train_list, y_train_list = [], []
    for i, (path, label) in enumerate(balanced_train):
        sig, sr = loader.load_audio(path)
        X_train_list.append(extract_sequence_features(sig, sr))
        y_train_list.append(0 if label == 'bonafide' else 1)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(balanced_train)}")
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    print(f"  Train tensor: {X_train.shape}")
    
    # ==========================================
    # EVAL (attacks A07-A19 - UNSEEN!)
    # ==========================================
    print("\n[2/3] Loading EVAL split (UNSEEN attacks A07-A19)...")
    eval_samples = loader.get_dataset('eval')
    real_eval = [(p, l) for p, l in eval_samples if l == 'bonafide'][:200]
    fake_eval = [(p, l) for p, l in eval_samples if l == 'spoof'][:200]
    balanced_eval = real_eval + fake_eval
    np.random.shuffle(balanced_eval)
    
    X_test_list, y_test_list = [], []
    for i, (path, label) in enumerate(balanced_eval):
        sig, sr = loader.load_audio(path)
        X_test_list.append(extract_sequence_features(sig, sr))
        y_test_list.append(0 if label == 'bonafide' else 1)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(balanced_eval)}")
    
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    print(f"  Eval tensor: {X_test.shape}")
    
    # ==========================================
    # TRAINING
    # ==========================================
    print("\n[3/3] Training...")
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=32, shuffle=False,
    )
    
    model = CNNBiLSTM().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\nTraining for 30 epochs...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/30 | Loss: {total_loss/len(train_loader):.4f}")
    
    # ==========================================
    # EVALUATION ON UNSEEN ATTACKS
    # ==========================================
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds = model(bx.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(by.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ HONEST Accuracy on UNSEEN attacks (A07-A19): {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    cm = confusion_matrix(all_labels, all_preds)
    print("📉 Confusion Matrix:")
    print(f"              Pred Real  Pred Fake")
    print(f"Actual Real   {cm[0,0]:<9}  {cm[0,1]:<9}")
    print(f"Actual Fake   {cm[1,0]:<9}  {cm[1,1]:<9}")