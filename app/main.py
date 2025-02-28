from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.background import BackgroundTasks
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

# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-render-app-name.onrender.com",  # Update this with your Render domain
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
model_loaded = False
model_loading = False

def load_model():
    global model, model_loaded, model_loading
    try:
        if model_loaded:
            return True
        if model_loading:
            return False
        
        model_loading = True
        model_path = BASE_DIR / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = ContrastiveModel.load_model(str(model_path))
        model.to(device)
        model.eval()
        model_loaded = True
        model_loading = False
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model_loading = False
        return False

@app.on_event("startup")
async def startup_event():
    print("Starting up FastAPI application...")
    # Start model loading in background
    BackgroundTasks().add_task(load_model)

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
            "model_loaded": model_loaded,
            "model_loading": model_loading,
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
        
        # Print number of channels for debugging
        print(f"Number of EEG channels: {len(raw.ch_names)}")
        
        # Extract features using the same function as training
        features = extract_features(raw)
        
        # Print feature shape for debugging
        print(f"Extracted features shape: {features.shape}")
        
        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing EEG file: {str(e)}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few moments."
            )
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload an EDF file.")

        # Create a unique temporary file name
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_{os.urandom(8).hex()}.edf")
        
        try:
            # Save the uploaded file
            content = await file.read()
            with open(temp_file_path, 'wb') as f:
                f.write(content)

            # Process the file using our feature extraction
            features = process_edf_file(temp_file_path)

            # Print shapes for debugging
            print(f"Original features shape: {features.shape}")
            
            # Reshape features to match model input
            features = features.reshape(-1)  # Flatten the array
            
            # Calculate number of features per channel (5 features: mean, std, power, peak_freq, hjorth)
            features_per_channel = 5
            
            # Take only the features we need (input_dim // features_per_channel channels)
            num_channels_needed = input_dim // features_per_channel
            features = features[:num_channels_needed * features_per_channel]
            
            # Pad if we don't have enough features
            if len(features) < input_dim:
                features = np.pad(features, (0, input_dim - len(features)))
            elif len(features) > input_dim:
                features = features[:input_dim]

            print(f"Final features shape: {features.shape}")

            # Make prediction
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            print(f"Input tensor shape: {input_tensor.shape}")
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            return JSONResponse({
                'status': 'success',
                'prediction': prediction,
                'message': 'High Stress' if prediction == 1 else 'Low Stress'
            })

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")

    except Exception as e:
        # Log the error for debugging
        print(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Make sure app is available at module level
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Rest of your routes remain the same... 