from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    available_models: List[str]
    version: str
