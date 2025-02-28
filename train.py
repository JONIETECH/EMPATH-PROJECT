from model import ContrastiveModel, CustomDataset, train_model, evaluate_model, plot_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import mne
import numpy as np
import pywt

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
        
        features.extend([mean_val, std_val, power, peak_freq, hjorth_mobility])
    
    return np.array(features)

def process_edf_file(file_path, augment=True):
    """Process a single EDF file and extract features"""
    # Read EEG file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Basic preprocessing
    raw.filter(1, 40, fir_design='firwin')
    
    # Extract comprehensive features
    features = extract_features(raw)
    
    return features

def load_dataset(dataset_type):
    """
    Load specific dataset type
    dataset_type: one of ['CMPS', 'SCWT', 'TMCT', 'Music']
    """
    base_path = "Dataset"
    dataset_paths = {
        'CMPS': os.path.join(base_path, "Complex Mathematical Problem solving (CMPS)"),
        'SCWT': os.path.join(base_path, "Stroop Colour Word Test(SCWT)"),
        'TMCT': os.path.join(base_path, "Trier Mental Challenge Test (TMCT)"),
        'Music': os.path.join(base_path, "Participants Listening to Relaxing Music")
    }
    
    if dataset_type not in dataset_paths:
        raise ValueError(f"Dataset type must be one of {list(dataset_paths.keys())}")
    
    path = dataset_paths[dataset_type]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
        
    # Load and process all EDF files in the directory
    all_data = []
    labels = []
    total_files = len([f for f in os.listdir(path) if f.endswith('.edf')])
    processed_files = 0
    
    for file in os.listdir(path):
        if file.endswith('.edf'):
            file_path = os.path.join(path, file)
            processed_files += 1
            print(f"Processing file {processed_files}/{total_files}: {file}")
            try:
                features = process_edf_file(file_path)
                all_data.append(features)
                
                # Assign labels based on dataset type and file number
                if dataset_type == 'CMPS':
                    # For CMPS: 1-11 are non-stress, 12-22 are stress
                    file_num = int(''.join(filter(str.isdigit, file.split('(')[1])))
                    labels.append(1 if file_num >= 12 else 0)
                elif dataset_type == 'SCWT':
                    # For SCWT: odd numbers are non-stress, even numbers are stress
                    file_num = int(''.join(filter(str.isdigit, file.split('(')[1])))
                    labels.append(1 if file_num % 2 == 0 else 0)
                elif dataset_type == 'TMCT':
                    # For TMCT: first half are non-stress, second half are stress
                    file_num = int(''.join(filter(str.isdigit, file.split('(')[1])))
                    labels.append(1 if file_num > 10 else 0)
                else:  # Music
                    # For Music: first half are non-stress, second half are stress
                    file_num = int(''.join(filter(str.isdigit, file.split('(')[1])))
                    labels.append(1 if file_num > 10 else 0)
            except Exception as e:
                print(f"⚠️ Error processing {file}: {str(e)}")
                print("Skipping this file and continuing...")
                continue
    
    if not all_data:
        raise FileNotFoundError(f"No valid EDF files found in {path}")
        
    # Convert to numpy arrays and pad if necessary
    max_features = max(len(f) for f in all_data)
    all_data = [np.pad(f, (0, max_features - len(f)), mode='constant') for f in all_data]
    X = np.vstack(all_data)
    y = np.array(labels)
    
    # Create a DataFrame with features and labels
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = y
    
    return df

def main():
    # Select which dataset to use
    dataset_type = 'CMPS'  # Change this to use different datasets
    print(f"Loading {dataset_type} dataset...")
    data = load_dataset(dataset_type)
    
    # Prepare features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Convert to float32 to match model dtype
    X = X.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False)
    
    # Initialize model and training components
    model = ContrastiveModel(X_train.shape[1])
    # Move model to CPU and set dtype
    model = model.to('cpu').float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, scheduler)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    best_model = ContrastiveModel.load_model('best_model.joblib')
    best_model.to(device)
    y_true, y_pred = evaluate_model(best_model, test_loader)
    plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main() 