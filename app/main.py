"""
FastAPI application entry point for MP190.

Provides:
- GET /health: Health check returning {"status": "ok"}
- POST /predict: Inference endpoint validating client rules and logging JSON records
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.logging_config import setup_logger

logger = setup_logger("mp190-api")

app = FastAPI(
    title="MP190 - Kochi Clean Air Cell Prediction Service",
    description="MLOps service for predicting tomorrow's PM2.5 levels at Kochi air stations.",
    version="2.0.0",
)

MODEL_PATH = Path("models/model_v2.joblib")
_model = None


def get_model():
    """Load model artifact lazily if available."""
    global _model
    if _model is None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    return _model


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint"
)
def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", service="mp190-api")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate PM2.5 next day prediction"
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate next-day PM2.5 prediction from validated station measurements.
    Logs structured JSON line to stdout for every prediction.
    """
    model = get_model()

    # If model is not yet trained/available, generate baseline estimate
    if model is not None:
        try:
            input_df = pd.DataFrame([request.model_dump()])
            prediction_val = float(model.predict(input_df)[0])
        except Exception as e:
            logger.error(f"Prediction execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference error occurred"
            )
    else:
        # Fallback heuristic for scaffolding/testing before model training
        raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Prediction model is unavailable"
    )

    # Structured JSON log line per prediction (stdout only)
    logger.info(
        "Prediction generated",
        extra={
            "extra_fields": {
                "event": "prediction",
                "model_version": "2.0",
                "prediction": prediction_val,
                "station_zone": request.station_zone,
                "day_type": request.day_type,
                "status": "success",
            }
        }
    )

    return PredictionResponse(
        prediction=prediction_val,
        status="success",
        model_version="2.0",
    )
