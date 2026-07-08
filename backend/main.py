import os
import sys
# Inject project root path into Python path to resolve imports properly when run from different subfolders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import json

from utils.helper import load_config
from pipeline.predict_pipeline import UnifiedPredictionPipeline

app = FastAPI(
    title="Fused Tire Diagnostics API Backend",
    description="FastAPI backend serving tire wear severity classification, RUL predictions, and alignment heuristics.",
    version="1.0.0"
)

# Enable CORS for React frontend running on localhost (typically 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs folder to serve Grad-CAM and IG heatmaps statically
os.makedirs("outputs/heatmaps", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load prediction pipeline
try:
    pipeline = UnifiedPredictionPipeline()
except Exception as e:
    print(f"Error initializing UnifiedPredictionPipeline: {str(e)}")
    pipeline = None

# Scenario presets mapping
PRESET_SCENARIOS = {
    "New Tire Baseline": {
        "expected_life": 60000,
        "km_driven": 2000,
        "camber": 0.0,
        "road_condition": "Smooth",
        "weather_condition": "Dry",
        "brand": "Michelin",
        "size": "205/55R16",
        "retreaded": "No"
    },
    "Over-inflated City Commute": {
        "expected_life": 50000,
        "km_driven": 15000,
        "camber": -0.2,
        "road_condition": "Smooth",
        "weather_condition": "Humid",
        "brand": "Continental",
        "size": "225/65R17",
        "retreaded": "No"
    },
    "Under-inflated Highway Driving": {
        "expected_life": 55000,
        "km_driven": 35000,
        "camber": 0.5,
        "road_condition": "Smooth",
        "weather_condition": "Cold",
        "brand": "Bridgestone",
        "size": "245/40R18",
        "retreaded": "No"
    },
    "Rough Off-Road Terrain": {
        "expected_life": 40000,
        "km_driven": 18000,
        "camber": -1.5,
        "road_condition": "Off-road",
        "weather_condition": "Rainy",
        "brand": "Goodyear",
        "size": "195/65R15",
        "retreaded": "Yes"
    }
}

@app.get("/api/presets")
def get_presets():
    """
    Get vehicle presets profiles.
    """
    return PRESET_SCENARIOS

@app.post("/api/predict")
async def predict_tire_diagnostics(
    image: UploadFile = File(...),
    sensor_data: str = Form(None)
):
    """
    Unified diagnostics prediction endpoint.
    Accepts a tire tread photo and tabular sensor readings (encoded as a JSON string).
    Returns combined classification, RUL estimation, and alignment diagnostics with uncertainty.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline is not loaded on backend.")

    # Save uploaded image file to a temporary location
    temp_dir = "outputs"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, f"uploaded_{image.filename}")
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")

    # Parse tabular sensor data if provided
    tabular_dict = None
    if sensor_data:
        try:
            tabular_dict = json.loads(sensor_data)
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise HTTPException(status_code=400, detail=f"Invalid JSON string in 'sensor_data': {str(e)}")

    # Run unified prediction pipeline
    try:
        results = pipeline.predict(temp_image_path, tabular_data=tabular_dict)
        
        # Map file paths to public static URLs
        if results.get("explanation_heatmap_path"):
            gc_rel = os.path.relpath(results["explanation_heatmap_path"], start=".")
            results["explanation_heatmap_url"] = f"http://localhost:8000/{gc_rel}"
        else:
            results["explanation_heatmap_url"] = ""
            
        if results.get("explanation_ig_path"):
            ig_rel = os.path.relpath(results["explanation_ig_path"], start=".")
            results["explanation_ig_url"] = f"http://localhost:8000/{ig_rel}"
        else:
            results["explanation_ig_url"] = ""
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")
    finally:
        pass

@app.get("/health")
def health_check():
    """
    Liveness diagnostic check.
    """
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}
