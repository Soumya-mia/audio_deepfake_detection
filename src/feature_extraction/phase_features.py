"""
phase_features.py - Extract phase-aware features for deepfake detection.
This is the CORE NOVELTY of the project.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
import matplotlib.pyplot as plt
from src.preprocessing.audio_loader import AudioLoader


class PhaseFeatureExtractor:
    """
    Extract phase-based features:
    - Phase Spectrum (raw phase angles)
    - Phase Difference (temporal continuity between frames)
    - Group Delay (derivative of phase w.r.t. frequency)
    
    These features capture synthesis artifacts that magnitude-only
    features (like Mel-spectrograms) completely miss.
    """
    
    def __init__(self, sr=16000, n_fft=512, hop_length=160):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        print(f"[PhaseFeatureExtractor] Initialized with n_fft={n_fft}, hop={hop_length}")
    
    def compute_stft(self, signal):
        """Compute complex STFT (preserves both magnitude and phase)."""
        return librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
    
    def compute_phase_spectrum(self, signal):
        """Extract phase angles from STFT."""
        stft = self.compute_stft(signal)
        return np.angle(stft)
    
    def compute_phase_difference(self, signal):
        """
        Compute phase difference between consecutive time frames.
        Synthesized speech often has unnatural phase jumps.
        """
        phase = self.compute_phase_spectrum(signal)
        # Difference along time axis (axis=1)
        phase_diff = np.diff(phase, axis=1)
        # Wrap to [-pi, pi] using complex exponential trick
        phase_diff = np.angle(np.exp(1j * phase_diff))
        return phase_diff
    
    def compute_group_delay(self, signal):
        """
        Compute group delay: negative derivative of phase w.r.t. frequency.
        Captures phase continuity artifacts.
        """
        phase = self.compute_phase_spectrum(signal)
        # Derivative along frequency axis (axis=0)
        group_delay = -np.diff(phase, axis=0)
        return group_delay
    
    def extract_all_phase_features(self, signal):
        """Extract all three phase features as a dictionary."""
        return {
            'phase_spectrum': self.compute_phase_spectrum(signal),
            'phase_difference': self.compute_phase_difference(signal),
            'group_delay': self.compute_group_delay(signal)
        }
    
    def compute_phase_statistics(self, signal):
        """
        Compute statistical summaries of phase features.
        These numerical features can be fed to ML models.
        
        For each of the 3 phase features, we extract 6 statistics:
        mean, std, median, Q1, Q3, range.
        
        Returns: A fixed-length vector of 18 dimensions (3 features * 6 stats).
        """
        features = self.extract_all_phase_features(signal)
        stats_vector = []
        
        for name, feat in features.items():
            # Flatten the 2D array and compute statistics
            flat = feat.flatten()
            stats_vector.extend([
                np.mean(flat),                    # Average phase value
                np.std(flat),                     # Variability of phase
                np.median(flat),                  # Middle value
                np.percentile(flat, 25),          # Quartile 1
                np.percentile(flat, 75),          # Quartile 3
                np.max(flat) - np.min(flat),      # Range (max - min)
            ])
        
        return np.array(stats_vector)  # 18 dimensions


# ==========================================
# Test Block
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("Phase Feature Extraction - Statistical Analysis")
    print("=" * 70)
    
    loader = AudioLoader("data/asvspoof2019/")
    train_data = loader.get_dataset('train')
    
    # Collect 50 real and 50 fake samples
    print("\nCollecting samples...")
    real_signals, fake_signals = [], []
    sr = 16000
    
    for path, label in train_data:
        if label == 'bonafide' and len(real_signals) < 50:
            sig, sr = loader.load_audio(path)
            real_signals.append(sig)
        elif label == 'spoof' and len(fake_signals) < 50:
            sig, _ = loader.load_audio(path)
            fake_signals.append(sig)
        if len(real_signals) >= 50 and len(fake_signals) >= 50:
            break
    
    print(f"Collected {len(real_signals)} real samples and {len(fake_signals)} fake samples.")
    
    extractor = PhaseFeatureExtractor(sr=sr)
    
    # Compute statistics for all samples
    print("\nComputing phase statistics (this may take ~30 seconds)...")
    real_stats = np.array([extractor.compute_phase_statistics(s) for s in real_signals])
    fake_stats = np.array([extractor.compute_phase_statistics(s) for s in fake_signals])
    
    print(f"\nReal phase statistics shape: {real_stats.shape}")
    print(f"Fake phase statistics shape: {fake_stats.shape}")
    
    # Build feature names for the table
    feature_names = []
    for name in ['phase_spectrum', 'phase_difference', 'group_delay']:
        for stat in ['mean', 'std', 'median', 'q25', 'q75', 'range']:
            feature_names.append(f"{name}_{stat}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 Feature Comparison (Real vs Fake)")
    print("=" * 80)
    print(f"{'Feature':<40} {'Real Mean':>12} {'Fake Mean':>12} {'Difference':>12}")
    print("-" * 80)
    
    large_diffs = []
    for i, name in enumerate(feature_names):
        real_val = real_stats[:, i].mean()
        fake_val = fake_stats[:, i].mean()
        diff = abs(real_val - fake_val)
        marker = " ⭐" if diff > 0.5 else ""
        if diff > 0.5:
            large_diffs.append((name, real_val, fake_val, diff))
        print(f"{name:<40} {real_val:>12.4f} {fake_val:>12.4f} {diff:>12.4f}{marker}")
    
    print("\n" + "=" * 80)
    print(f"⭐ Features with large difference (>0.5): {len(large_diffs)}")
    if large_diffs:
        print("These are the discriminative phase features to feed into the model:")
        for name, r, f, d in large_diffs:
            print(f"  - {name}: real={r:.4f}, fake={f:.4f}, diff={d:.4f}")
    else:
        print("No single feature shows dramatic difference. Combined features will matter.")
        print("This is expected for subtle phase artifacts—the CNN will learn combinations.")
    print("=" * 80)
    
    # ==========================================
    # Visualization: Side-by-side comparison
    # ==========================================
    print("\nGenerating visualization...")
    
    # Get one representative real and fake sample
    real_sample, fake_sample = None, None
    for path, label in train_data:
        if label == 'bonafide' and real_sample is None:
            real_sample, _ = loader.load_audio(path)
        elif label == 'spoof' and fake_sample is None:
            fake_sample, _ = loader.load_audio(path)
        if real_sample is not None and fake_sample is not None:
            break
    
    real_features = extractor.extract_all_phase_features(real_sample)
    fake_features = extractor.extract_all_phase_features(fake_sample)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, (name, feat) in enumerate(real_features.items()):
        axes[0, i].imshow(feat, aspect='auto', cmap='viridis')
        axes[0, i].set_title(f'REAL: {name}', fontsize=11)
        axes[0, i].set_xlabel('Time Frame')
        axes[0, i].set_ylabel('Frequency Bin')
    
    for i, (name, feat) in enumerate(fake_features.items()):
        axes[1, i].imshow(feat, aspect='auto', cmap='viridis')
        axes[1, i].set_title(f'FAKE: {name}', fontsize=11)
        axes[1, i].set_xlabel('Time Frame')
        axes[1, i].set_ylabel('Frequency Bin')
    
    plt.suptitle("Phase Feature Comparison: Real vs Fake (The Core Novelty)", fontsize=14)
    plt.tight_layout()
    
    import os
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/phase_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Phase comparison plot saved to: results/phase_comparison.png")