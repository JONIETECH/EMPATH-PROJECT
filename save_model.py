import torch

# Assuming 'model' is your trained model
model_path = "contrastive_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
