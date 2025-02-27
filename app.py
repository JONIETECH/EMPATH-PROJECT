from flask import Flask, request, jsonify
import torch
import numpy as np
from model import ContrastiveModel

app = Flask(__name__)

# Load the model
input_dim = 128  # Replace with the actual input dimension
model = ContrastiveModel(input_dim)
model.load_state_dict(torch.load("contrastive_model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input'])
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
