"""
Unit tests for model metrics and retraining decision rules.

All tests utilize synthetic in-memory arrays and do not require external files.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import calculate_mae, calculate_r2, calculate_rmse, evaluate_by_group
from src.retrain import evaluate_promotion_rule


def test_calculate_mae():
    """Verify Mean Absolute Error computation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |10-12|=2, |20-18|=2, |30-33|=3 -> (2+2+3)/3 = 2.3333
    mae = calculate_mae(y_true, y_pred)
    assert pytest.approx(mae, 0.001) == 7.0 / 3.0


def test_calculate_rmse():
    """Verify Root Mean Squared Error computation."""
    y_true = np.array([0.0, 3.0])
    y_pred = np.array([4.0, 3.0])
    # 4^2 + 0^2 = 16 -> sqrt(16/2) = sqrt(8) approx 2.8284
    rmse = calculate_rmse(y_true, y_pred)
    assert pytest.approx(rmse, 0.001) == np.sqrt(8.0)


def test_calculate_r2():
    """Verify R2 score returns 1.0 for perfect predictions."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert calculate_r2(y_true, y_pred) == 1.0


def test_evaluate_by_group():
    """Verify group-wise MAE calculation across station zones."""
    df = pd.DataFrame({
        "station_zone": ["Central", "Central", "Industrial"],
        "y_true": [10.0, 20.0, 50.0],
        "y_pred": [12.0, 18.0, 55.0],
    })
    group_mae = evaluate_by_group(df, "station_zone", "y_true", "y_pred")
    assert "Central" in group_mae
    assert "Industrial" in group_mae
    assert pytest.approx(group_mae["Central"], 0.01) == 2.0
    assert pytest.approx(group_mae["Industrial"], 0.01) == 5.0


def test_evaluate_promotion_rule_promote():
    """v2 promoted when MAE is > 10% lower than v1."""
    # mae_v1 = 10.0, mae_v2 = 8.5 -> (10 - 8.5)/10 = 15% improvement >= 10%
    result = evaluate_promotion_rule(mae_v1=10.0, mae_v2=8.5, improvement_threshold=0.10)
    assert result["decision"] == "promote"
    assert result["relative_improvement"] == 0.15


def test_evaluate_promotion_rule_keep():
    """v2 kept (not promoted) when MAE improvement is less than 10%."""
    # mae_v1 = 10.0, mae_v2 = 9.5 -> (10 - 9.5)/10 = 5% improvement < 10%
    result = evaluate_promotion_rule(mae_v1=10.0, mae_v2=9.5, improvement_threshold=0.10)
    assert result["decision"] == "keep"
