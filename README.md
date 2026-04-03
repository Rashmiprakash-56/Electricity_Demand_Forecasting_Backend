---
title: Electricity Demand Forecasting API
emoji: ⚡
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

# Electricity Demand Forecasting - Backend

This is the FastAPI backend for the **Electricity Demand Forecasting** project. It handles data processing pipelines, asynchronous machine learning model training, hyperparameter optimization, and prediction generation with SHAP explainability.

## Key Features
- **Background Training Tasks:** Asynchronous training of XGBoost models via Optuna for automated hyperparameter search space exploration. It relies on `TimeSeriesSplit` to prevent look-ahead bias associated with time-series data.
- **Model Registry & Deployment:** Integrates with `huggingface_hub` to smoothly pull datasets and push the best trained models/SHAP explainers directly to the Hugging Face Hub.
- **AI Interpretability:** Deploys a `TreeExplainer` providing local feature contributions (for Waterfall charts) and global insights to maintain transparency over predictions.
- **Data Integration:** Evaluates weather data alongside historical generation profiles, using custom preprocessors like `TimeTargetEncoder`.
- **Secure API:** Implements robust JWT-based authentication (`fastapi-users`) limiting unauthorized access to ML infrastructure orchestration.

## Technology Stack
- **Framework:** FastAPI (Python)
- **Machine Learning Core:** XGBoost (`XGBRegressor`), Optuna, SHAP (`TreeExplainer`), Scikit-Learn.
- **Database & Auth:** PostgreSQL/SQLite managed by SQLAlchemy and Asyncpg. `fastapi-users` for the authentication flow.
- **ML Ops:** MLflow (experiment tracking), Hugging Face Hub client.

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
1. Navigate to the `backend` directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API Server

Ensure your `.env` file is appropriately populated with database credentials, Hugging Face read/write tokens, and the JWT secret before starting the server. A sample `.env.example` is provided for guidance.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```
Once active, interactive API documentation (Swagger UI) is available at `http://localhost:7860/docs`.
