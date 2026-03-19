from typing import List
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
def _fix_db_url(url: str | None) -> str | None:
    """Ensure the DATABASE_URL uses the asyncpg driver for async SQLAlchemy."""
    if not url:
        return url
    url = url.replace("postgresql+psycopg://", "postgresql+asyncpg://")
    url = url.replace("postgresql://", "postgresql+asyncpg://")
    # Avoid double-replacing (e.g. already asyncpg)
    url = url.replace("postgresql+asyncpg+asyncpg://", "postgresql+asyncpg://")
    return url

class Settings():
    # Database
    DATABASE_URL: str = _fix_db_url(os.getenv("DATABASE_URL"))
    
    # JWT
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    ALGORITHM: str = "HS256" 
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGIN", "")
    
    # Hugging Face Hub 
    HF_MODEL_REPO: str = os.getenv("HF_MODEL_REPO", "")   
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_DATASET_REPO: str = os.getenv("HF_DATASET_REPO", "Rashmiprakash78/demand_and_weather")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")   
    MLFLOW_EXPERIMENT_NAME: str =  os.getenv("MLFLOW_EXPERIMENT_NAME", "")  
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Directories
    DATA_DIR = PROJECT_ROOT / "data"
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

    # Ensure required directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Data paths
    ENERGY_DATA_PATH = DATA_DIR / "energy_dataset.csv"
    WEATHER_DATA_PATH = DATA_DIR / "weather_features.csv"

    # Artifact paths
    TRAINED_MODEL_PATH = ARTIFACT_DIR / "trained_model.pkl"
    ENCODER_PATH = ARTIFACT_DIR / "encoder.pkl"
    EXPLAINER_PATH = ARTIFACT_DIR / "explainer.pkl"
    
    # Model settings
    SUPPORTED_MODELS: List[str] = ["xgboost"]
    
    #split size
    TEST_SIZE = 0.2
    
    class Config:
        env_file = ".env"
    

class ModelConfig():
    RANDOM_SEED = 42
    N_SPLITS = 5
    NUM_BOOST_ROUND = 500
    EARLY_STOPPING = 30
    N_TRIALS = 50

    org_weather_cat_cols = ['weather_id', 'weather_main', 'weather_description','weather_icon']

    numeric_energy_df_col =[ 'generation biomass',
        'generation fossil brown coal/lignite',
        'generation fossil coal-derived gas', 'generation fossil gas',
        'generation fossil hard coal', 'generation fossil oil',
        'generation fossil oil shale', 'generation fossil peat',
        'generation geothermal', 'generation hydro pumped storage aggregated',
        'generation hydro pumped storage consumption',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir', 'generation marine',
        'generation nuclear', 'generation other', 'generation other renewable',
        'generation solar', 'generation waste', 'generation wind offshore',
        'generation wind onshore', 'forecast solar day ahead',
        'forecast wind offshore eday ahead', 'forecast wind onshore day ahead',
        'total load forecast', 'total load actual', 'price day ahead',
        'price actual']

    numeric_weather_col = [ 'temp', 'temp_min', 'temp_max', 'pressure',
        'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h',
        'clouds_all']

    comparision_cols = ['datetime','total load forecast']

    non_target_col = ['total load forecast', 'price day ahead','price actual','generation biomass',
                    'generation fossil brown coal/lignite', 'generation fossil gas',
                    'generation fossil hard coal', 'generation fossil oil',
                    'generation hydro pumped storage consumption',
                    'generation hydro run-of-river and poundage',
                    'generation hydro water reservoir', 'generation nuclear',
                    'generation other', 'generation other renewable', 'generation solar',
                    'generation waste', 'generation wind onshore',
                    'forecast solar day ahead',
    ] # these cols will be dropped
    
    target_col='total load actual'


settings = Settings()
model_config = ModelConfig()