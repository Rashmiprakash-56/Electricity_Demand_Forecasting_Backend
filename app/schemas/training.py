from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class TrainingRequest(BaseModel):
    model_type: str
    data_path: str
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingResponse(BaseModel):
    job_id: int
    status: str
    message: str

class TrainingStatus(BaseModel):
    job_id: int
    status: str
    model_type: str
    metrics: Optional[Dict[str, float]] = None
    mlflow_run_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None