from sklearn.metrics import mean_absolute_percentage_error
from app.services.preprocessing import (DataPreprocessor,
                                        DemandPreprocessor,
                                        WeatherPreprocessor,
                                        DataSplitter,
                                        TimeTargetEncoder)
from app.core.config import settings,model_config
from app.prediction_models.xgboost import XGBoostModel
import pandas as pd
import joblib
from app.core.logger import get_logger
from app.core.logger import get_logger
from app.utils.storage import save_to_hub_or_local, load_from_hub_or_local

log = get_logger(__name__)

# ── In-memory cache for processed data ────────────────────────────────
_data_cache: pd.DataFrame | None = None

def clear_data_cache():
    """Clear the cached processed DataFrame (e.g. if data files change)."""
    global _data_cache
    _data_cache = None
    log.info("Data cache cleared")


def save_encoder(encoder, path:str):
    save_to_hub_or_local(encoder, "encoder.pkl", path)
    log.info("Encoder Saved")
    
def load_encoder(path: str):
    encoder = load_from_hub_or_local("encoder.pkl", path)
    log.info("Encoder Loaded")
    return encoder

def process_data():
    global _data_cache

    if _data_cache is not None:
        log.info("Using cached processed data")
        return _data_cache.copy()

    log.info("Processing data from CSV files (first time — will be cached)")
    data_processor  = DataPreprocessor()
    demand_processor = DemandPreprocessor(num_cols=model_config.numeric_energy_df_col)
    weather_processor = WeatherPreprocessor(num_cols=model_config.numeric_weather_col)


    energy_df = data_processor.load_data(settings.ENERGY_DATA_PATH)
    processed_energy_df = demand_processor.process_generation_demand(energy_df)
    log.info('Demand Data Processed')

    weather_df = data_processor.load_data(settings.WEATHER_DATA_PATH)
    processed_weather_df = weather_processor.prepare_weather(weather_df)
    log.info('Weather Data Processed')

    processed_dataset = pd.merge(processed_energy_df,processed_weather_df, on='datetime', how='inner')

    processed_dataset = processed_dataset.drop(columns=model_config.non_target_col)

    final_df = data_processor.preprocess(processed_dataset,target_col='total load actual')
    log.info("Data Processing Complete")

    _data_cache = final_df.copy()
    log.info("Processed data cached in memory")

    return final_df

    
def train_model(X_train_val, y_train_val,custom_params):
    weather_cat_cols = [  
        col for col in X_train_val.columns 
        if any(org_col in col for org_col in model_config.org_weather_cat_cols)
    ]

    encoder = TimeTargetEncoder(
        cols=weather_cat_cols,
        time_col="datetime",
        n_splits=5,
        smoothing=10
    )

    xgb_model = XGBoostModel(encoder=encoder,custom_params=custom_params)
    
    xgb_model.train(
        X_train=X_train_val,
        y_train=y_train_val
    )
    
    save_encoder(encoder, settings.ENCODER_PATH)
    xgb_model.save()


def predict_and_explain(X_test, y_test, model:str='xgboost'):

    encoder = load_encoder(settings.ENCODER_PATH)
    xgb_model = XGBoostModel(encoder=encoder)
    xgb_model.load()
    
    prediction, shap_value = xgb_model.predict(X_test, return_shap=True)
    mape = mean_absolute_percentage_error(y_test, prediction)*100
    
    return prediction, y_test, shap_value, mape