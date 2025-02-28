from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
from model import ContrastiveModel
import os
import mne
import tempfile

app = Flask(__name__)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
input_dim = 128  # Replace with the actual input dimension
model = None

def load_model():
    global model
    model = ContrastiveModel(input_dim)
    model.load_state_dict(torch.load("contrastive_model.pth", map_location=device))
    model.to(device)
    model.eval()

@app.before_first_request
def initialize():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

def process_eeg_file(file_path):
    # Read EEG file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # Basic preprocessing
    raw.filter(l_freq=1, h_freq=50)  # Bandpass filter
    
    # Extract features (example: power spectral density)
    psd, freqs = raw.psd(fmin=1, fmax=50, n_fft=256)
    
    # Convert to input format expected by model
    features = psd.mean(axis=0)  # Average across channels
    
    # Ensure features match input_dim
    if len(features) > input_dim:
        features = features[:input_dim]
    elif len(features) < input_dim:
        features = np.pad(features, (0, input_dim - len(features)))
    
    return features

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400

        if not file.filename.endswith('.edf'):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file format. Please upload an EDF file.'
            }), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp_file:
            file.save(temp_file.name)
            features = process_eeg_file(temp_file.name)

        # Remove temporary file
        os.unlink(temp_file.name)

        # Make prediction
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return jsonify({
            'status': 'success',
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['input'])
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
