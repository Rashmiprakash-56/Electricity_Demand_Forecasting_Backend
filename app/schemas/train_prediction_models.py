from pydantic import BaseModel
from typing import List, Any, Dict

class PredictRequest(BaseModel):
    filter_date: str

class PredictResponse(BaseModel):
    filter_date: str
    mape : float
    DemandDetails : List[dict]
    shap_values: Dict[str, Any]
    PredictedDate : str

class TrainRequest(BaseModel):
    custom_params : Dict
