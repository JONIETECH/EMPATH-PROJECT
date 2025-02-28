from fastapi import FastAPI, File, UploadFile, HTTPException
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
from .model import ContrastiveModel
from joblib import load

# Create FastAPI app
app = FastAPI(title="EEG Stress Detection")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
input_dim = 128
model = None

def load_model():
    global model
    try:
        model_path = BASE_DIR / "model.joblib"
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Rest of your routes remain the same... 