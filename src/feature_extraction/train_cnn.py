"""
train_cnn.py - Train a 1D Convolutional Neural Network (CNN) on FFT features.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

print("1. Loading features...")
X = np.load('X_features_balanced.npy')  # Shape: (1000, 256)
y = np.load('y_labels_balanced.npy')    # Shape: (1000,)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 3. Convert to PyTorch tensors
# CNN expects input shape: (Batch, Channels, Length)
# We have 1 channel (the FFT magnitude), and Length = 256.
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # (800, 1, 256)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)    # (200, 1, 256)
y_test_t = torch.LongTensor(y_test)

# 4. Create DataLoaders (for batching)
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Define the 1D CNN Architecture
class SimpleCNN1D(nn.Module):
    def __init__(self):
        super(SimpleCNN1D, self).__init__()
        # Conv1d: input_channels=1, output_channels=32, kernel_size=3
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Reduces length by half
        
        # After two pools: 256 -> 128 -> 64
        self.fc1 = nn.Linear(64 * 64, 128)  # 64 channels * 64 length
        self.fc2 = nn.Linear(128, 2)        # 2 output classes (Real/Fake)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 128)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, 64)
        x = x.view(x.size(0), -1)                # Flatten: (batch, 64*64)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                          # (batch, 2)
        return x

# 6. Initialize the model, loss function, and optimizer
model = SimpleCNN1D()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n2. Training the 1D CNN...")
# 7. Training loop
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

# 8. Evaluation on test set
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

accuracy = 100 * correct / total
print(f"\n✅ CNN Test Accuracy: {accuracy:.2f}%")

# 9. Simple Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n📉 Confusion Matrix:")
print("              Predicted Real  Predicted Fake")
print(f"Actual Real   {cm[0,0]:<12}  {cm[0,1]:<12}")
print(f"Actual Fake   {cm[1,0]:<12}  {cm[1,1]:<12}")