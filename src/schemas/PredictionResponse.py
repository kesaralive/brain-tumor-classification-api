from pydantic import BaseModel
from typing import Optional, Dict

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    processing_time: float
    image_id: str
    visualization_url: Optional[str] = None