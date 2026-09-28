"""
build_sequence_dataset.py - Build a time-series dataset for CNN+BiLSTM.
Preserves the time axis (unlike Day 3 which averaged it away).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
from src.preprocessing.audio_loader import AudioLoader

def extract_sequence_features(signal, sr, n_mels=64, max_time_frames=128):
    """
    Extract a 2D feature tensor for one audio file.
    
    Returns:
        feature_tensor: shape (n_mels + phase_channels, max_time_frames)
        Each channel represents one type of feature:
        - Channel 0: Log-Mel Spectrogram (magnitude)
        - Channel 1: Phase Spectrum
        - Channel 2: Phase Difference
        - Channel 3: Group Delay
    """
    # 1. Log-Mel Spectrogram (magnitude)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # 2. STFT for phase features
    stft = librosa.stft(signal, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    phase_diff = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    group_delay = -np.diff(phase, axis=0)
    
    # 3. Downsample phase features to match n_mels bins
    def resize_2d(arr, target_bins, target_time):
        """Downsample or pad a 2D array to (target_bins, target_time)."""
        # Frequency axis
        if arr.shape[0] > target_bins:
            step = arr.shape[0] // target_bins
            arr = arr[::step][:target_bins]
        elif arr.shape[0] < target_bins:
            pad = target_bins - arr.shape[0]
            arr = np.pad(arr, ((0, pad), (0, 0)))
        # Time axis
        if arr.shape[1] > target_time:
            arr = arr[:, :target_time]
        elif arr.shape[1] < target_time:
            pad = target_time - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)))
        return arr
    
    phase_r = resize_2d(phase, n_mels, max_time_frames)
    phase_diff_r = resize_2d(phase_diff, n_mels, max_time_frames)
    group_delay_r = resize_2d(group_delay, n_mels, max_time_frames)
    
    # 4. Ensure all channels have same shape (n_mels, max_time_frames)
    log_mel_r = resize_2d(log_mel, n_mels, max_time_frames)
    
    # 5. Stack into a 3D tensor: (channels, n_mels, time_frames)
    feature_tensor = np.stack([log_mel_r, phase_r, phase_diff_r, group_delay_r], axis=0)
    
    return feature_tensor.astype(np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("Building Time-Series Dataset for CNN+BiLSTM")
    print("=" * 60)
    
    loader = AudioLoader("data/asvspoof2019/")
    train_data = loader.get_dataset('train')
    
    # Balance: 500 Real + 500 Fake
    real_files = [(p, l) for p, l in train_data if l == 'bonafide'][:500]
    fake_files = [(p, l) for p, l in train_data if l == 'spoof'][:500]
    balanced = real_files + fake_files
    np.random.shuffle(balanced)
    
    X_list = []
    y_list = []
    
    print(f"Processing {len(balanced)} audio files...")
    for i, (path, label) in enumerate(balanced):
        signal, sr = loader.load_audio(path)
        features = extract_sequence_features(signal, sr)
        X_list.append(features)
        y_list.append(0 if label == 'bonafide' else 1)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(balanced)}")
    
    X = np.array(X_list)   # Shape: (1000, 4, 64, 128)
    y = np.array(y_list)   # Shape: (1000,)
    
    print(f"\n✅ Dataset built successfully!")
    print(f"   X shape: {X.shape}  (samples, channels, mel_bins, time_frames)")
    print(f"   y shape: {y.shape}")
    print(f"   Class distribution: Real={np.sum(y==0)}, Fake={np.sum(y==1)}")
    
    np.save('X_sequence.npy', X)
    np.save('y_sequence.npy', y)
    print("\n💾 Saved to X_sequence.npy and y_sequence.npy")