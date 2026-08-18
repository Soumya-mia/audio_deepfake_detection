"""
dsp_features.py - Step 1: Just compute FFT for one audio file.
"""

import sys
sys.path.insert(0, '.')   # <-- FIX: tells Python to look in current folder

import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.audio_loader import AudioLoader

# 1. Build the loader and get the shopping list
loader = AudioLoader("data/asvspoof2019/")
train_data = loader.get_dataset('train')

# 2. Pick the first audio file
audio_path, label = train_data[0]
print(f"Audio file: {audio_path}")
print(f"Label: {label}")

# 3. Load the audio signal
signal, sr = loader.load_audio(audio_path)
print(f"Signal length (samples): {len(signal)}")

# 4. Compute the FFT (Fast Fourier Transform)
fft_result = np.fft.rfft(signal)

# 5. Print the result
print(f"FFT output shape: {fft_result.shape}")
print(f"First 5 frequency coefficients: {fft_result[:5]}")

# 6. Plot the frequency spectrum (Bonus)
magnitude = np.abs(fft_result)
frequencies = np.fft.rfftfreq(len(signal), d=1/sr)

plt.figure(figsize=(10,4))
plt.plot(frequencies, magnitude)
plt.title(f"Frequency Spectrum of {label} audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.grid(True)
plt.show()