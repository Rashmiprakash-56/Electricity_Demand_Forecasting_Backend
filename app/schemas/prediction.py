from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    forecast_horizon: int = 24

class PredictionResponse(BaseModel):
    predictions: List[float]
    timestamps: List[str]
    model_id: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None

class ExplainabilityRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    reference_data: Optional[Any] = None
    current_data: Optional[Any] = None