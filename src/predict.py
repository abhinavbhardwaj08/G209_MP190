"""
Batch prediction module for MP190 holdout dataset.

Generates final predictions using the registered champion model
and formats predictions/holdout_predictions.csv with record_id,prediction.
"""

from pathlib import Path
from typing import Optional
import joblib
import pandas as pd


def generate_holdout_predictions(
    model_path: str | Path = "models/model.joblib",
    holdout_path: str | Path = "data/raw/holdout_features.csv",
    output_path: str | Path = "predictions/holdout_predictions.csv"
) -> pd.DataFrame:
    """
    Load champion model and produce holdout predictions in exact required schema:
    record_id,prediction.
    """
    model_file = Path(model_path)
    holdout_file = Path(holdout_path)
    out_file = Path(output_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_file}")
    if not holdout_file.exists():
        raise FileNotFoundError(f"Holdout features not found at {holdout_file}")

    model = joblib.load(model_file)
    holdout_df = pd.read_csv(holdout_file)

    if "record_id" not in holdout_df.columns:
        raise ValueError("Holdout features missing record_id column")

    predictions = model.predict(holdout_df)
    results_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions,
    })

    out_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_file, index=False)
    return results_df
