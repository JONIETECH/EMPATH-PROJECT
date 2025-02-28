try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pywt
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from torch.nn.utils import weight_norm
    from torch.utils.data import DataLoader, Dataset
    import mne
    from sklearn.model_selection import train_test_split
    import torch.optim as optim
    from model import ContrastiveModel
except ModuleNotFoundError as e:
    print(f"Error: {e}. Please install the missing module and try again.")
    exit()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming X_train, train_loader, and test_loader are already defined
model = ContrastiveModel(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

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