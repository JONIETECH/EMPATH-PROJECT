from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from pathlib import Path
import os
import mne
import tempfile
from app.model import ContrastiveModel, extract_features
from joblib import load

# Create FastAPI app
app = FastAPI(title="EEG Stress Detection")

# Get the absolute path to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

print(f"Base Directory: {BASE_DIR}")
print(f"Templates Directory: {BASE_DIR / 'templates'}")
print(f"Static Directory: {BASE_DIR / 'static'}")

# Verify directories exist
if not (BASE_DIR / 'templates').exists():
    raise RuntimeError(f"Templates directory not found at {BASE_DIR / 'templates'}")
if not (BASE_DIR / 'static').exists():
    raise RuntimeError(f"Static directory not found at {BASE_DIR / 'static'}")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
input_dim = 128
model = None

def load_model():
    global model
    try:
        model_path = BASE_DIR / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = ContrastiveModel.load_model(str(model_path))
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("Warning: Model failed to load")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    print("Accessing home route")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.get("/health")
async def health_check():
    try:
        model_path = BASE_DIR / "best_model.joblib"
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_file_exists": model_path.exists()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def process_edf_file(file_path):
    """Process a single EDF file and extract features"""
    try:
        # Read EEG file using MNE
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Basic preprocessing
        raw.filter(1, 40, fir_design='firwin')
        
        # Extract features using the same function as training
        features = extract_features(raw)
        
        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing EEG file: {str(e)}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload an EDF file.")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file using our feature extraction
            features = process_edf_file(temp_file.name)
            
            # Remove temporary file
            os.unlink(temp_file.name)

            # Ensure features match expected input dimension
            if len(features) > input_dim:
                features = features[:input_dim]
            elif len(features) < input_dim:
                features = np.pad(features, (0, input_dim - len(features)))

            # Make prediction
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            return JSONResponse({
                'status': 'success',
                'prediction': prediction,
                'message': 'High Stress' if prediction == 1 else 'Low Stress'
            })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Rest of your routes remain the same... 