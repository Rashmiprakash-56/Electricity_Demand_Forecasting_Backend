import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import access_models
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from app.core.config import settings
from app.utils.data_fetcher import fetch_dataset_from_hub
import mlflow
###
from app.database import create_db_and_table,get_async_session,User
from app.user import auth_backend,current_active_user , fastapi_users
from app.schemas.user import UserCreate,UserRead,UserUpdate
from app.core.logger import get_logger

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app : FastAPI):
    # Download dataset files from Hugging Face Hub (skips if already present)
    fetch_dataset_from_hub()
    await create_db_and_table()
    yield

# Initialize MLflow
try:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
except:
    log.info("Warning: Could not connect to MLflow. Make sure MLflow server is running.")

app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="API for training and predicting electricity demand using multiple ML models",
    version="1.0.0", 
    lifespan=lifespan
)

# CORS configuration
_cors_origins = ["http://localhost:5174","http://localhost:8888"]
if settings.FRONTEND_URL:
    _cors_origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"])
app.include_router(fastapi_users.get_register_router(UserRead,UserCreate), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"])
app.include_router(fastapi_users.get_users_router(UserRead,UserUpdate), prefix="/users", tags=["users"])

app.include_router(access_models.router,prefix="/model",tags=["ML"])


@app.get("/")
async def root():
    return {
        "message": "Electricity Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
    log.info("Server has started")