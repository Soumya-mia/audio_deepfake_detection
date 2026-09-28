"""
dsp_features.py - Step 2: Compute the Spectrogram (STFT).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.audio_loader import AudioLoader

# 1. Load the audio file
loader = AudioLoader("data/asvspoof2019/")
train_data = loader.get_dataset('train')
audio_path, label = train_data[0]
for audio_path, label in train_data:
    if label == 'spoof':
        print(f"Found a spoof file: {audio_path}")
        signal, sr = loader.load_audio(audio_path)
        break
signal, sr = loader.load_audio(audio_path)

print(f"Label: {label}")
print(f"Duration: {len(signal)/sr:.2f} seconds")

# 2. Compute Short-Time Fourier Transform (Spectrogram)
# This breaks the audio into tiny windows (25ms) and runs FFT on each.
n_fft = 512          # Window size (512 samples = 32ms at 16kHz)
hop_length = 160     # Step size (10ms overlap)
stft_result = np.abs(  # Convert to magnitude (remove complex numbers)
    np.fft.rfft(
        np.array([signal[i:i+n_fft] for i in range(0, len(signal)-n_fft, hop_length)])
    ).T
)

# 3. Plot the Spectrogram (Time vs Frequency)
plt.figure(figsize=(12, 6))
plt.imshow(stft_result, 
           aspect='auto', 
           origin='lower', 
           extent=[0, len(signal)/sr, 0, sr/2],  # x-axis: time, y-axis: frequency
           cmap='inferno')
plt.colorbar(label='Magnitude (dB)')
plt.title(f'Spectrogram of {label} audio')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 8000)  # Focus on human speech range
plt.tight_layout()
plt.show()
