from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import torch
import numpy as np
from model import ContrastiveModel
import os
import mne
import tempfile

app = FastAPI(title="EEG Stress Detection")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.edf'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload an EDF file."
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            features = process_eeg_file(temp_file.name)

        # Remove temporary file
        os.unlink(temp_file.name)

        # Make prediction
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return JSONResponse({
            'status': 'success',
            'prediction': prediction
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(data: dict):
    try:
        input_data = np.array(data['input'])
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return {
            'status': 'success',
            'prediction': prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
