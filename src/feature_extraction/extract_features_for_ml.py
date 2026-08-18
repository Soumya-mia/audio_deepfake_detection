"""
extract_features_for_ml.py - Extract balanced FFT features (500 Real, 500 Fake).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.preprocessing.audio_loader import AudioLoader
import random

# 1. Load the dataset
loader = AudioLoader("data/asvspoof2019/")
train_data = loader.get_dataset('train')

print(f"Total files in dataset: {len(train_data)}")

# 2. Separate real and fake files
real_files = []
fake_files = []

for audio_path, label in train_data:
    if label == 'bonafide':
        real_files.append((audio_path, label))
    else:
        fake_files.append((audio_path, label))

print(f"Real files found: {len(real_files)}")
print(f"Fake files found: {len(fake_files)}")

# 3. Take 500 from each (or the minimum available)
sample_size = 500
selected_real = real_files[:sample_size]
selected_fake = fake_files[:sample_size]

# 4. Combine and shuffle
balanced_data = selected_real + selected_fake
random.shuffle(balanced_data)  # Mix them up so the order doesn't matter

print(f"Balanced dataset size: {len(balanced_data)} (500 Real + 500 Fake)")

# 5. Extract features
X_features = []
y_labels = []

for i, (audio_path, label) in enumerate(balanced_data):
    # Load audio
    signal, sr = loader.load_audio(audio_path)
    
    # Compute FFT magnitude
    fft_result = np.fft.rfft(signal)
    magnitude = np.abs(fft_result)
    
    # Normalize to exactly 256 features
    downsample_factor = len(magnitude) // 256
    if downsample_factor > 0:
        magnitude = magnitude[::downsample_factor]
    else:
        magnitude = np.pad(magnitude, (0, 256 - len(magnitude)))
    
    X_features.append(magnitude[:256])
    y_labels.append(0 if label == 'bonafide' else 1)  # 0=Real, 1=Fake
    
    if i % 100 == 0:
        print(f"Processed {i+1}/{len(balanced_data)} files...")

# 6. Convert to numpy arrays
X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"\nSuccessfully processed {len(X_features)} samples!")
print(f"Feature vector shape: {X_features.shape}")
print(f"Labels shape: {y_labels.shape}")

print(f"\nClass distribution (Balanced):")
print(f"  Real (0): {np.sum(y_labels == 0)}")
print(f"  Fake (1): {np.sum(y_labels == 1)}")

# 7. Save the data
np.save('X_features_balanced.npy', X_features)
np.save('y_labels_balanced.npy', y_labels)
print("\nFeatures saved to: X_features_balanced.npy and y_labels_balanced.npy")