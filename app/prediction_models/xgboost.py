import pickle
import xgboost as xgb
import numpy as np
import joblib
from typing import Dict, Tuple
import optuna
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from app.core.config import settings,model_config
from app.core.logger import get_logger
from app.utils.storage import save_to_hub_or_local, load_from_hub_or_local

log = get_logger(__name__)

class XGBoostModel:
    """
    XGBoost regression model with hyperparameter tuning and SHAP explanations.
    """
    
    def __init__(self, encoder,custom_params:Dict = None):
        self.encoder = encoder
        self.is_trained = False
        self.explainer = None
        self.model = None
        self.model_param = {
            "objective": "reg:squarederror",
            "eval_metric": "mape",
            "booster": "gbtree",
            "eta": 0.3,
            "max_depth": 5,
            "subsample": 0.4,
            "colsample_bytree": 0.7,
            "lambda": 1,
            "alpha": 1,
            "seed": 42,    
            "verbosity": 0,
        }
        self.best_params = {}
        self.custom_params = custom_params
    
    def xgb_objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        """
        X_train_val = self.X_train_val
        y_train_val = self.y_train_val
        custom_params = self.custom_params

        params = {
            # Fixed parameters from the payload
            "objective": custom_params.get("objective", "reg:squarederror"),
            "eval_metric": custom_params.get("eval_metric", "mape"),
            "booster": custom_params.get("booster", "gbtree"),
            "seed": 42,
            "verbosity": 0,
            
            # Tunable parameters mapped from the frontend ranges
            "eta": trial.suggest_float(
                "eta", 
                custom_params["eta_min"], 
                custom_params["eta_max"], 
                log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 
                custom_params["max_depth_min"], 
                custom_params["max_depth_max"]
            ),
            "subsample": trial.suggest_float(
                "subsample", 
                custom_params["subsample_min"], 
                custom_params["subsample_max"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 
                custom_params["colsample_bytree_min"], 
                custom_params["colsample_bytree_max"]
            ),
            "lambda": trial.suggest_float(
                "lambda", 
                custom_params["lambda_min"], 
                custom_params["lambda_max"], 
                log=True
            ),
            "alpha": trial.suggest_float(
                "alpha", 
                custom_params["alpha_min"], 
                custom_params["alpha_max"], 
                log=True
            ),
        }
        
        num_boost_round = trial.suggest_int(
            "n_estimators", 
            custom_params["n_estimators_min"], 
            custom_params["n_estimators_max"]
        )

        
        tscv = TimeSeriesSplit(n_splits=model_config.N_SPLITS)
        fold_mapes = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            X_train = X_train_val.iloc[train_idx]
            X_val = X_train_val.iloc[val_idx]
            
            y_train = y_train_val.iloc[train_idx]
            y_val = y_train_val.iloc[val_idx]
            
            # Use a fresh encoder for each fold's validation
            from copy import deepcopy
            fold_encoder = deepcopy(self.encoder)
            
            # Encode categorical columns
            X_train_enc = fold_encoder.fit_transform(X_train, y_train).drop(
                columns=['datetime'], errors='ignore'
            )
            X_val_enc = fold_encoder.transform(X_val).drop(
                columns=['datetime'], errors='ignore'
            )
            
            dtrain = xgb.DMatrix(X_train_enc, label=y_train)
            dvalid = xgb.DMatrix(X_val_enc, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=custom_params.get('early_stopping_rounds',30),
                verbose_eval=False,
            )
            
            preds = model.predict(dvalid)
            mape = mean_absolute_percentage_error(y_val, preds)
            fold_mapes.append(mape)
        
        return float(np.mean(fold_mapes))
    
    def get_best_parameters(self) -> Dict:
        """
        Use Optuna to find the best hyperparameters for the model.
        """
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            custom_params = self.custom_params
            
            pruner = None
            if self.custom_params.get('n_trials',5) >= 10:
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

            log.info('Getting best Parameters for XGBoost')
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
            )

            study.optimize(
                self.xgb_objective,
                n_trials=custom_params.get('n_trials',5),
            )
            
            self.best_params = study.best_params
            
            for key, val in study.best_params.items():
                self.model_param[key] = val
            
            return self.model_param
        except Exception as e:
            log.error(f"Error in xgb_objective: {type(e).__name__}: {str(e)}")
            import traceback
            # log.error(traceback.format_exc())
            raise  # Re-raise so Optuna knows it failed

    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the XGBoost model with hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary containing training metrics
        """
        self.X_train_val = X_train
        self.y_train_val = y_train
        
        self.get_best_parameters()

        X_train_enc = (
            self.encoder
            .fit_transform(X_train, y_train)
            .drop(columns=["datetime"], errors="ignore")
        )
        
        n_estimators = self.best_params.get("n_estimators", 100)
        params = self.best_params.copy()
        params.pop("n_estimators", None)
        
        model = xgb.XGBRegressor(
            **params,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state= 42
        )
        log.info("Training XGBoost with best parameters")
        model.fit(X_train_enc, y_train)

        self.explainer = shap.TreeExplainer(model)
        
        self.model = model
        self.is_trained = True
        
        log.info("XGBoost Model Trained")
        return {"status": "trained", "best_params": self.best_params}
    
    def predict(self, X: np.ndarray, return_shap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            return_shap: Whether to return SHAP values
            
        Returns:
            Tuple of (predictions, shap_values) if return_shap=True
            Otherwise just predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained. Call train() first or load a trained model.")
        
        X_pred = (
            self.encoder
            .transform(X)   
            .drop(columns=["datetime"], errors="ignore")
        )
        
        predictions = self.model.predict(X_pred)
        log.info("XGBoost Prediction complete")
        
        if return_shap and self.explainer is not None:
            shap_values = self.explainer(X_pred)
            return predictions, shap_values
        
        return predictions
    
    def save(self):
        """
        Save the trained model and explainer to HF Hub directly or local fallback.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model. Call train() first.")
        
        # Save model
        save_to_hub_or_local(self.model, "trained_model.pkl", settings.TRAINED_MODEL_PATH)
        log.info(f"XGBoost Model saved")
        
        # Save explainer
        if self.explainer is not None:
            save_to_hub_or_local(self.explainer, "explainer.pkl", settings.EXPLAINER_PATH, use_pickle=True)
            log.info(f"XGBoost Explainer saved")
    
    def load(self):
        """
        Load a trained model and explainer from HF Hub directly or local fallback.
        """
        # Load model
        self.model = load_from_hub_or_local("trained_model.pkl", settings.TRAINED_MODEL_PATH)
        log.info("XGBoost Model Loaded")
        
        try:
            self.explainer = load_from_hub_or_local("explainer.pkl", settings.EXPLAINER_PATH, use_pickle=True)
        except FileNotFoundError:
            self.explainer = None
        
        self.is_trained = True