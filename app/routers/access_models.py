from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.services.preprocessing import DataSplitter
from app.services.train_model import process_data, train_model, predict_and_explain
from app.schemas.train_prediction_models import PredictRequest,PredictResponse,TrainRequest
import traceback
from app.utils.common_func import shap_global_importance,shap_waterfall_format
from datetime import datetime
import traceback
import uuid

router = APIRouter()
training_status = {} # inmemory storage

def train_model_background(task_id: str, X_train, y_train, custom_params):
    """Background task for model training"""
    try:
        training_status[task_id] = {
            "status": "training",
            "progress": 0,
            "message": "Training started",
            "started_at": datetime.now().isoformat()
        }
        train_model(X_train_val=X_train, y_train_val=y_train, custom_params=custom_params)
        
        training_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Model training completed successfully",
            "started_at": training_status[task_id]["started_at"],
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        training_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}",
            "error": tb,
            "started_at": training_status[task_id].get("started_at"),
            "failed_at": datetime.now().isoformat()
        }


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def train(background_tasks: BackgroundTasks, request: TrainRequest):
    """Start model training as a background task"""
    try:
        custom_params = request.custom_params
        df = process_data()
        data_splitter = DataSplitter(df=df)
        X_train, y_train = data_splitter.load_train_val()

        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Add training to background tasks
        background_tasks.add_task(
            train_model_background,
            task_id=task_id,
            X_train=X_train,
            y_train=y_train,
            custom_params=custom_params
        )
        
        # Initialize status
        training_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Training queued",
            "created_at": datetime.now().isoformat()
        }

        return {
            "message": "Model training started in background",
            "task_id": task_id
        }

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=tb
        )


@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Get the current status of a training task"""
    if task_id not in training_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training task not found"
        )
    
    return training_status[task_id]


@router.get("/train/history")
async def get_training_history():
    """Get all training tasks history"""
    return {
        "tasks": [
            {"task_id": tid, **info} 
            for tid, info in training_status.items()
        ]
    }

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK
)
async def predict(request: PredictRequest):
    try:
        df = process_data()

        data_splitter = DataSplitter(df=df)
        X_test, y_test = data_splitter.load_test(filter_date=request.filter_date)

        predictions, y_true, shap_values, mape = predict_and_explain(
            X_test=X_test,
            y_test=y_test
        )
        
        shap_serialized = {
                "local": shap_waterfall_format(
                    shap_values.values.tolist(),
                    shap_values.base_values.tolist(),
                    predictions,
                    shap_values.feature_names
                ),
                "global": shap_global_importance(
                    shap_values.values.tolist(),
                    shap_values.feature_names
                )
            }
        actual_demand = y_true.values.tolist()
        predicted_demand = predictions.tolist()
        DemandDetails = []
        for i in range(24):
            DemandDetails.append({
                'hour' : i+1,
                'actual_demand' : actual_demand[i],
                'predicted_demand' : predicted_demand[i],
                'diff' :  actual_demand[i]- predicted_demand[i],
                'hourly_mape' : abs(actual_demand[i] - predicted_demand[i]) / actual_demand[i] * 100 if actual_demand[i] != 0 else 0
            })


        return PredictResponse(
            filter_date=request.filter_date,
            DemandDetails=DemandDetails,
            mape=mape,
            shap_values=shap_serialized,
            PredictedDate= request.filter_date
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= traceback.format_exc()
        )