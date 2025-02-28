import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
import mne
from sklearn.model_selection import train_test_split
import torch.optim as optim
from joblib import dump, load

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # Add any other parameters you want to save
        }
        dump(model_dict, path)
    
    @classmethod
    def load_model(cls, path='model.joblib'):
        """Load the model using joblib"""
        model_dict = load(path)
        model = cls(input_dim=model_dict['input_dim'])
        model.load_state_dict(model_dict['state_dict'])
        return model

# ---------------------- TRAINING FUNCTION ----------------------
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=50):
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            # Ensure tensors are float32
            x_batch = x_batch.float()
            y_batch = y_batch.long()  # CrossEntropyLoss expects long
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        # Save model if it has the best loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_model('best_model.joblib')
            print(f"Saved best model with loss: {best_loss:.4f}")

# ---------------------- EVALUATION ----------------------
def evaluate_model(model, test_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.float()
            outputs = model(x_batch)
            predictions = torch.argmax(outputs, dim=1).cpu()
            y_pred.extend(predictions.numpy())
            y_true.extend(y_batch.numpy())
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Optimized Contrastive Model")
    plt.show()
