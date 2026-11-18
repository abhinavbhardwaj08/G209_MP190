"""
Evaluation metrics module for MP190.

Computes Mean Absolute Error (MAE) as the primary evaluation metric,
along with supplementary regression metrics and group-wise error breakdowns.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_r2(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute coefficient of determination (R2 score)."""
    return float(r2_score(y_true, y_pred))


def evaluate_by_group(
    df: pd.DataFrame,
    group_column: str,
    y_true_col: str,
    y_pred_col: str
) -> Dict[str, float]:
    """
    Compute MAE separately for each unique value of the group column.
    Useful for station_zone breakdown required in findings.yaml.
    """
    group_errors: Dict[str, float] = {}
    for group_val, subset in df.groupby(group_column):
        mae = calculate_mae(subset[y_true_col], subset[y_pred_col])
        group_errors[str(group_val)] = mae
    return group_errors
