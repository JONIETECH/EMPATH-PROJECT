import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
import mne
from sklearn.model_selection import train_test_split
import torch.optim as optim
from joblib import dump, load

try:
    import pywt
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False
    print("Warning: PyWavelets not available. Some features will be disabled.")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augment_eeg_signal(signal, sfreq):
    """Augment EEG signal with noise and shift"""
    signal += np.random.normal(0, 0.02, signal.shape)  # Add noise
    shift = np.random.randint(0, int(sfreq * 0.15))  # Time shift
    signal = np.roll(signal, shift)
    return signal

def extract_features(raw):
    """Extract comprehensive features from EEG data"""
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    features = []
    
    for i in range(data.shape[0]):
        signal_data = augment_eeg_signal(data[i], sfreq)
        # Basic statistical features
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        power = np.sum(signal_data**2) / len(signal_data)
        
        # Frequency domain features
        peak_freq = np.argmax(np.abs(np.fft.fft(signal_data))) * sfreq / len(signal_data)
        
        # Hjorth parameters
        hjorth_mobility = np.sqrt(np.var(np.diff(signal_data)) / np.var(signal_data))
        
        # Add wavelet features if available
        if HAS_WAVELETS:
            # Add your wavelet feature extraction here
            pass
        
        features.extend([mean_val, std_val, power, peak_freq, hjorth_mobility])
    
    return np.array(features)

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ---------------------- CONTRASTIVE LEARNING MODEL ----------------------
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(ContrastiveModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.input_dim = input_dim

    def forward(self, x):
        return self.fc(x)

    def save_model(self, path='model.joblib'):
        """Save the model using joblib"""
        model_dict = {
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
        }
        dump(model_dict, path)
    
    @classmethod
    def load_model(cls, path='model.joblib'):
        """Load the model using joblib"""
        model_dict = load(path)
        model = cls(input_dim=model_dict['input_dim'])
        model.load_state_dict(model_dict['state_dict'])
        return model 