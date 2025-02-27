from IPython import get_ipython
from IPython.display import display
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
from torch.nn.utils import weight_norm  # used for weight normalization.
from torch.utils.data import DataLoader, Dataset
import mne
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mne
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import weight_norm  # used for weight normalization.


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

model = ContrastiveModel(X_train.shape[1]).cuda()
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
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
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

train_model(model, train_loader, criterion, optimizer, scheduler)

# ---------------------- EVALUATION ----------------------
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.cuda()
        outputs = model(x_batch)
        predictions = torch.argmax(outputs, dim=1).cpu()
        y_pred.extend(predictions.numpy())
        y_true.extend(y_batch.numpy())

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
