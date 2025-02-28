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

# Load and preprocess your data
# Assuming your dataset is in a CSV file in the dataset folder
data_path = "/c:/Users/Uncle Shimmy/Desktop/ml code/EMPATH-PROJECT/dataset/data.csv"
data = pd.read_csv(data_path)

# Assuming the last column is the label and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    def forward(self, x):
        return self.fc(x)

model = ContrastiveModel(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # L2 Regularization
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ---------------------- TRAINING FUNCTION ----------------------
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=50):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------------- EVALUATION ----------------------
def evaluate_model(model, test_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
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

train_model(model, train_loader, criterion, optimizer, scheduler)
y_true, y_pred = evaluate_model(model, test_loader)
plot_confusion_matrix(y_true, y_pred)
