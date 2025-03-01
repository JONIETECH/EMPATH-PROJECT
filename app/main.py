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
        "*",  # Allow all origins in production
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
input_dim = 155  # Number of features (31 channels * 5 features per channel)

def load_model():
    global model, model_loaded, model_loading
    try:
        if model_loaded:
            return True
        if model_loading:
            return False
        
        model_loading = True
        print("Starting model loading...")
        model_path = BASE_DIR / "best_model.joblib"
        print(f"Looking for model at: {model_path}")
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        print("Loading model from file...")
        model = ContrastiveModel.load_model(str(model_path))
        model.to(device)
        model.eval()  # Set to evaluation mode
        model_loaded = True
        model_loading = False
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model_loading = False
        return False

@app.on_event("startup")
async def startup_event():
    print("Starting up FastAPI application...")
    # Load model immediately
    if not load_model():
        print("WARNING: Failed to load model during startup!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    print("Accessing home route")
    try:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "app_name": "EEG Stress Detection"
            }
        )
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        # Log more details about the error
        import traceback
        print(traceback.format_exc())
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.head("/")
async def head_root():
    return HTMLResponse(content="")

@app.get("/health")
async def health_check():
    try:
        model_path = BASE_DIR / "best_model.joblib"
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "model_loading": model_loading,
            "model_file_exists": model_path.exists(),
            "model_path": str(model_path),
            "base_dir": str(BASE_DIR),
            "device": str(device)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def process_edf_file(file_path):
    """Process a single EDF file and extract features"""
    try:
        print(f"Processing file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")

        # Read EEG file using MNE
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            raise ValueError(f"Failed to read EDF file: {str(e)}")
        
        print(f"EDF file loaded successfully. Info: {raw.info}")
        print(f"Channel names: {raw.ch_names}")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        
        if len(raw.ch_names) < 1:
            raise ValueError("No EEG channels found in the file")

        # Basic preprocessing
        try:
            raw.filter(1, 40, fir_design='firwin')
        except Exception as e:
            raise ValueError(f"Failed to filter EEG data: {str(e)}")
        
        # Print number of channels for debugging
        print(f"Number of EEG channels: {len(raw.ch_names)}")
        
        # Extract features using the same function as training
        try:
            features = extract_features(raw)
        except Exception as e:
            raise ValueError(f"Failed to extract features: {str(e)}")
        
        # Print feature shape for debugging
        print(f"Extracted features shape: {features.shape}")
        
        return features
    except Exception as e:
        print(f"Error in process_edf_file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise ValueError(f"Error processing EEG file: {str(e)}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Please try again in a few moments."
            )

        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        print(f"\n=== Starting Analysis ===")
        print(f"File name: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        if not file.filename.endswith('.edf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload an EDF file.")

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_{os.urandom(8).hex()}.edf")
        
        try:
            # Save the uploaded file
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")

            with open(temp_file_path, 'wb') as f:
                f.write(content)

            if not os.path.exists(temp_file_path):
                raise HTTPException(status_code=500, detail="Failed to save uploaded file")

            # Process the file
            features = process_edf_file(temp_file_path)
            
            if features is None:
                raise HTTPException(status_code=400, detail="Failed to extract features from file")

            # Print shapes for debugging
            print(f"\n=== Feature Processing ===")
            print(f"Original features shape: {features.shape}")
            
            # Reshape features to match model input
            features = features.reshape(-1)  # Flatten the array
            print(f"Flattened features shape: {features.shape}")
            
            # Calculate number of features per channel
            features_per_channel = 5
            num_channels_needed = input_dim // features_per_channel
            print(f"Number of channels needed: {num_channels_needed}")
            features = features[:num_channels_needed * features_per_channel]
            
            # Pad if we don't have enough features
            if len(features) < input_dim:
                print(f"Padding features from {len(features)} to {input_dim}")
                features = np.pad(features, (0, input_dim - len(features)))
            elif len(features) > input_dim:
                print(f"Truncating features from {len(features)} to {input_dim}")
                features = features[:input_dim]

            print(f"\n=== Model Input ===")
            print(f"Final features shape: {features.shape}")
            print(f"Feature statistics:")
            print(f"- Mean: {features.mean():.4f}")
            print(f"- Std: {features.std():.4f}")
            print(f"- Min: {features.min():.4f}")
            print(f"- Max: {features.max():.4f}")

            # Make prediction
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            print(f"\n=== Model Prediction ===")
            print(f"Input tensor shape: {input_tensor.shape}")
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                print(f"Raw output: {output.numpy()}")
                print(f"Prediction: {prediction}")

            print("\n=== Analysis Complete ===")
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
        import traceback
        error_trace = traceback.format_exc()
        print(f"Full error traceback:\n{error_trace}")
        # Return a more detailed error response
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": {
                    "error": str(e) or "Unknown error occurred",
                    "traceback": error_trace,
                    "type": type(e).__name__
                }
            }
        )

# Make sure app is available at module level
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Rest of your routes remain the same... 