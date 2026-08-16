"""
audio_loader.py - Load and parse ASVspoof 2019 LA dataset.
This is the first step in our DSP + Deep Learning pipeline.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


class AudioLoader:
    """
    Load audio files and labels from ASVspoof 2019 LA dataset.
    Input: Path to dataset folder (e.g., "data/asvspoof2019/")
    Output: Audio signals with their labels ('bonafide' or 'spoof')
    """
    
    def __init__(self, data_dir: str, sr: int = 16000):
        """
        Initialize the loader.
        :param data_dir: Root directory of the dataset (contains 'LA/' subfolder).
        :param sr: Target sample rate (ASVspoof uses 16 kHz).
        """
        self.data_dir = Path(data_dir)
        self.sr = sr
        print(f"[AudioLoader] Initialized with data_dir: {self.data_dir}")
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load a single audio file (FLAC) and return the signal and sample rate.
        """
        signal, sr = librosa.load(filepath, sr=self.sr)
        return signal, sr
    
    def parse_label_file(self, label_path: str) -> Dict[str, str]:
        """
        Parse ASVspoof .trn label file (located in the CM protocols folder).
        Format: speaker_id utterance_id spoof_type tag
        Returns: {utterance_id: tag} where tag is 'bonafide' or 'spoof'.
        """
        labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    utterance_id = parts[1]   # e.g., LA_T_0000001
                    tag = parts[-1]           # <-- FIXED: always takes the last column!
                    labels[utterance_id] = tag
        print(f"[AudioLoader] Loaded {len(labels)} labels from {label_path}")
        return labels
    
    def get_dataset(self, split: str = 'train') -> List[Tuple[str, str]]:
        
        # Map split names to folder names and label file names (inside CM protocols folder)
        split_map = {
            'train': ('ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
            'dev':   ('ASVspoof2019_LA_dev',   'ASVspoof2019.LA.cm.dev.trn.txt'),
            'eval':  ('ASVspoof2019_LA_eval',  'ASVspoof2019.LA.cm.eval.trn.txt')
        }
        
        folder, label_file = split_map[split]
        audio_dir = self.data_dir / 'LA' / folder / 'flac'
        # CORRECT PATH: uses your actual folder name
        label_path = self.data_dir / 'LA' / 'ASVspoof2019_LA_cm_protocols' / label_file
        
        labels = self.parse_label_file(str(label_path))
        
        dataset = []
        for utterance_id, label in labels.items():
            audio_path = audio_dir / f"{utterance_id}.flac"
            if audio_path.exists():
                dataset.append((str(audio_path), label))
            else:
                print(f"[Warning] Audio file not found: {audio_path}")
        
        print(f"[AudioLoader] Loaded {len(dataset)} samples for {split} split")
        return dataset


# ==========================================
# Test the AudioLoader (only runs when executed directly)
# ==========================================
if __name__ == "__main__":
    print("Testing AudioLoader...")
    
    # Create an instance of the loader
    loader = AudioLoader(data_dir="data/asvspoof2019/")
    
    # Load training data
    train_data = loader.get_dataset('train')
    
    if len(train_data) > 0:
        # Print the first sample
        audio_path, label = train_data[0]
        signal, sr = loader.load_audio(audio_path)
        print(f"\nFirst sample:")
        print(f"  Path: {audio_path}")
        print(f"  Label: {label}")
        print(f"  Duration: {len(signal)/sr:.2f} seconds")
        print(f"  Signal shape: {signal.shape}")
    else:
        print("No training data found. Make sure the dataset is downloaded and extracted.")