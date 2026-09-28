"""
train_cnn_2d.py - Train a 2D CNN on Spectrograms (Time vs Frequency).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing.audio_loader import AudioLoader

print("1. Loading audio and generating spectrograms...")

loader = AudioLoader("data/asvspoof2019/")
train_data = loader.get_dataset('train')

# Balance dataset (500 Real, 500 Fake)
real_files = [(p, l) for p, l in train_data if l == 'bonafide'][:500]
fake_files = [(p, l) for p, l in train_data if l == 'spoof'][:500]
balanced_data = real_files + fake_files
np.random.shuffle(balanced_data)

X_spectrograms = []
y_labels = []

for audio_path, label in balanced_data:
    signal, sr = loader.load_audio(audio_path)
    
    # Compute Mel-Spectrogram (Log scale)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel_spec)  # Convert to dB
    
    # Pad/truncate to exactly 128 time steps
    if log_mel.shape[1] < 128:
        pad = 128 - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
    else:
        log_mel = log_mel[:, :128]
    
    X_spectrograms.append(log_mel)
    y_labels.append(0 if label == 'bonafide' else 1)

X = np.array(X_spectrograms)  # Shape: (1000, 64, 128)
y = np.array(y_labels)

print(f"Spectrograms shape: {X.shape}")  # (1000, 64, 128)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
# 2D CNN expects: (Batch, Channels, Height, Width) -> (Batch, 1, 64, 128)
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
y_test_t = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

# ==========================================
# TODO: Define the 2D CNN Architecture
# ==========================================
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        # HINT: Use Conv2d instead of Conv1d.
        # Input: (batch, 1, 64, 128)
        # YOU write the layers here!
        # Example: self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Then pooling: self.pool = nn.MaxPool2d(2, 2)
        # After a few layers, flatten and use Linear layers.
        pass

    def forward(self, x):
        # YOU write the forward pass here!
        # x -> conv -> relu -> pool -> conv -> relu -> pool -> flatten -> fc -> output
        return x

# ==========================================
# The rest is provided for you
# ==========================================
model = SpectrogramCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n2. Training the 2D CNN on Spectrograms...")
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\n✅ 2D CNN Test Accuracy: {100 * correct / total:.2f}%")