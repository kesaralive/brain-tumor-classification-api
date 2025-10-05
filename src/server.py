
"""
FastAPI Brain Tumor Classification API
Endpoints for uploading MRI images and getting predictions
"""
import os 

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import  List
import uvicorn
from pathlib import Path
import shutil
import io
from PIL import Image
import base64
import numpy as np
from datetime import datetime
import uuid


from contextlib import asynccontextmanager
from classes.BrainTumorPredictor import BrainTumorPredictor
from schemas.ErrorResponse import ErrorResponse
from schemas.HealthResponse import HealthResponse
from schemas.PredictionResponse import PredictionResponse



@asynccontextmanager
async def lifespan(app:FastAPI):
    """Run on API startup"""
    print("="*60)
    print("Brain Tumor Classification API")
    print("="*60)
    print(f"Available models: {list(MODELS_CONFIG.keys())}")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print("="*60)
    yield
    """Run on API shutdown"""
    print("\nShutting down API...")
    # Clear model cache
    model_cache.clear()


app = FastAPI(
    title="Brain Tumor Classification API",
    description="Deep Learning API for Brain Tumor Detection and Classification from MRI Scans",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model configurations
MODELS_CONFIG = {
    "ResNet50": {
        "path": "./models/best_ResNet50.pth",
        "description": "ResNet-50 based classifier - Good balance of speed and accuracy"
    },
    "EfficientNet": {
        "path": "./models/best_EfficientNet.pth",
        "description": "EfficientNet-B0 - Fast and accurate"
    },
    "CustomCNN": {
        "path": "./models/best_CustomCNN.pth",
        "description": "Custom CNN architecture - Lightweight and fast"
    },
    "ViT": {
        "path": "./models/best_ViT.pth",
        "description": "Vision Transformer - Highest accuracy but slower"
    }
}



CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Create directories
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# Cache for loaded models to avoid reloading
model_cache = {}


def get_or_load_model(model_type: str) -> BrainTumorPredictor:
    """Load model from cache or create new instance"""
    if model_type not in model_cache:
        if model_type not in MODELS_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Model type '{model_type}' not found. Available models: {list(MODELS_CONFIG.keys())}"
            )
        
        model_path = MODELS_CONFIG[model_type]["path"]
        
        # Check if model file exists
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}. Please train the model first."
            )
        
        try:
            model_cache[model_type] = BrainTumorPredictor(
                model_path=model_path,
                model_type=model_type,
                class_names=CLASS_NAMES
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )
    
    return model_cache[model_type]


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file"""
    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: 10MB, got: {file_size / 1024 / 1024:.2f}MB"
        )


def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file and return path"""
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return save_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )

def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()




@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API information"""
    return {
        "status": "online",
        "available_models": list(MODELS_CONFIG.keys()),
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_models": list(MODELS_CONFIG.keys()),
        "version": "1.0.0"
    }

@app.get("/models")
async def list_models():
    """List all available models with descriptions"""
    models_info = []
    for model_name, config in MODELS_CONFIG.items():
        model_exists = Path(config["path"]).exists()
        models_info.append({
            "name": model_name,
            "description": config["description"],
            "path": config["path"],
            "available": model_exists,
            "loaded_in_cache": model_name in model_cache
        })
    
    return {
        "success": True,
        "total_models": len(MODELS_CONFIG),
        "models": models_info
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="MRI scan image file"),
    model_type: str = Query(
        "ResNet50",
        description="Model type to use for prediction",
        enum=list(MODELS_CONFIG.keys())
    ),
    include_visualization: bool = Query(
        False,
        description="Include Grad-CAM visualization in response"
    )
):
    """
    Predict brain tumor type from MRI scan
    
    **Parameters:**
    - **file**: MRI image file (jpg, jpeg, png)
    - **model_type**: Model to use (ResNet50, EfficientNet, CustomCNN, ViT)
    - **include_visualization**: Generate Grad-CAM visualization
    
    **Returns:**
    - Prediction results with confidence scores
    """
    start_time = datetime.now()
    image_path = None
    viz_path = None
    
    try:
        # Validate image
        validate_image(file)
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        image_id = image_path.stem
        
        # Load model
        predictor = get_or_load_model(model_type)
        
        # Make prediction
        if include_visualization:
            viz_path = RESULTS_DIR / f"{image_id}_visualization.png"
            results = predictor.visualize_prediction(
                image_path=str(image_path),
                save_path=str(viz_path),
                show_gradcam=True
            )
        else:
            results = predictor.predict(str(image_path))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = PredictionResponse(
            success=True,
            prediction=results['predicted_class'],
            confidence=round(results['confidence'], 2),
            probabilities={k: round(v, 2) for k, v in results['probabilities'].items()},
            model_used=model_type,
            processing_time=round(processing_time, 3),
            image_id=image_id,
            visualization_url=f"/visualization/{image_id}" if include_visualization else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Cleanup uploaded file
        if image_path and image_path.exists():
            try:
                image_path.unlink()
            except:
                pass

@app.get("/visualization/{image_id}")
async def get_visualization(image_id: str):
    """Get Grad-CAM visualization image"""
    viz_path = RESULTS_DIR / f"{image_id}_visualization.png"
    
    if not viz_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Visualization not found. Generate it first using include_visualization=true"
        )
    
    return FileResponse(viz_path, media_type="image/png")


@app.delete("/clear-cache")
async def clear_model_cache():
    """Clear loaded models from memory"""
    global model_cache
    cleared_models = list(model_cache.keys())
    model_cache = {}

    return {
        "success": True,
        "message": "Model cache cleared",
        "cleared_models": cleared_models
    }

@app.delete("/clear-results")
async def clear_results():
    """Clear all generated visualization files"""
    deleted_count = 0
    for file in RESULTS_DIR.glob("*.png"):
        try:
            file.unlink()
            deleted_count += 1
        except:
            pass
    
    return {
        "success": True,
        "message": f"Deleted {deleted_count} result files"
    }

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

def start():
    uvicorn.run("src.server:app", host="0.0.0.0", port=int(
        os.getenv("FASTAPI_PORT", "8081")), reload=True)
