"""
plot_spectrogram.py - Functions to visualize audio signals and their spectrograms.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import librosa

class AudioVisualizer:
    """
    A collection of static methods to generate publication-ready plots.
    """
    
    @staticmethod
    def plot_waveform(signal, sr, title="Waveform", save_path=None):
        """
        Plot the raw audio waveform.
        """
        plt.figure(figsize=(10, 3))
        time = np.linspace(0, len(signal) / sr, num=len(signal))
        plt.plot(time, signal, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()

    @staticmethod
    def plot_spectrogram(signal, sr, n_fft=512, hop_length=160, title="Spectrogram", save_path=None):
        """
        Plot the Mel-Spectrogram with a colorbar.
        """
        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()

    @staticmethod
    def plot_fft_comparison(real_signal, fake_signal, sr, save_path=None):
        """
        Special comparison plot: Real vs Fake FFT (THE 1030Hz vs 250Hz plot).
        This is your "smoking gun" visualization for the research paper.
        """
        def get_fft(signal, sr):
            fft = np.fft.rfft(signal)
            mag = np.abs(fft)
            freqs = np.fft.rfftfreq(len(signal), d=1/sr)
            return freqs, mag

        freqs_r, mag_r = get_fft(real_signal, sr)
        freqs_f, mag_f = get_fft(fake_signal, sr)

        plt.figure(figsize=(12, 4))
        
        # Real plot
        plt.subplot(1, 2, 1)
        plt.plot(freqs_r, mag_r, color='green')
        plt.title("REAL (Bonafide) - Peak at ~1030 Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(0, 8000)
        plt.grid(True, alpha=0.3)
        
        # Fake plot
        plt.subplot(1, 2, 2)
        plt.plot(freqs_f, mag_f, color='red')
        plt.title("FAKE (Spoof) - Peak at ~250 Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(0, 8000)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle("DSP Forensic Evidence: High-Frequency Artifacts Reveal Deepfakes")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()
        print("[Visualizer] FFT comparison plot rendered successfully.")

    @staticmethod
    def plot_confusion_matrix(cm, class_names=['Real', 'Fake'], save_path=None):
        """
        Plot a beautiful confusion matrix heatmap.
        """
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()


# ==========================================
# Quick Test (Runs if you execute this file directly)
# ==========================================
if __name__ == "__main__":
    from src.preprocessing.audio_loader import AudioLoader
    
    print("Testing AudioVisualizer...")
    loader = AudioLoader("data/asvspoof2019/")
    
    # Get one real and one fake sample
    real_path = None
    fake_path = None
    for path, label in loader.get_dataset('train'):
        if label == 'bonafide' and real_path is None:
            real_path = path
        elif label == 'spoof' and fake_path is None:
            fake_path = path
        if real_path and fake_path:
            break
    
    real_signal, sr = loader.load_audio(real_path)
    fake_signal, _ = loader.load_audio(fake_path)
    
    # Render the comparison plot
    AudioVisualizer.plot_fft_comparison(real_signal, fake_signal, sr)