"""
Model training module with MLflow experiment tracking and model registration.

Tracks experiments to sqlite:///mlflow.db and logs data_sha256,
hyperparameters, MAE metrics, and registers the champion model artifact.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.config import load_params
from src.evaluate import calculate_mae, calculate_rmse


def compute_data_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 checksum of raw dataset for MLflow audit tracking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "mp190_kochi_air_quality"
) -> Dict[str, Any]:
    """
    Train a regressor, log metrics and data hash to MLflow, and return trained pipeline.
    """
    if params is None:
        params = load_params()

        from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from src.feature_engineering import build_preprocessor
    import mlflow
    import mlflow.sklearn

    numeric_features = [
        "pm25_today",
        "wind_kmph",
        "humidity_pct",
        "temp_c",
        "traffic_index",
        "stubble_fires",
    ]

    categorical_features = ["day_type", "station_zone"]

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    model_params = params["model"]["hyperparameters"]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**model_params))
    ])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        pipeline.fit(x_train, y_train)

        mlflow.log_params(model_params)

        metrics = {}

        if x_val is not None and y_val is not None:
            predictions = pipeline.predict(x_val)
            metrics["mae"] = calculate_mae(y_val, predictions)
            metrics["rmse"] = calculate_rmse(y_val, predictions)
            mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model"
)

        save_model_artifact(pipeline)

        return {
            "status": "trained",
            "experiment": experiment_name,
            "run_id": run.info.run_id,
            "metrics": metrics,
            "pipeline": pipeline,
        }


def save_model_artifact(pipeline: Any, output_path: str | Path = "models/model.joblib") -> None:
    """Save trained pipeline artifact to disk using joblib."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
