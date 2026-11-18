"""
Model retraining and promotion workflow for MP190.

Retrains v2 using chronological split (first 70% of later.csv added to train.csv)
and compares v1 vs v2 on the remaining 30% to apply the 10% promotion rule.
"""

from typing import Any, Dict, Tuple
import pandas as pd

from src.evaluate import calculate_mae


def split_later_data(
    later_df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort later.csv chronologically and split into train (first 70%) and check (last 30%).
    """
    df_sorted = later_df.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_part = df_sorted.iloc[:split_idx].copy()
    check_part = df_sorted.iloc[split_idx:].copy()
    return train_part, check_part


def evaluate_promotion_rule(
    mae_v1: float,
    mae_v2: float,
    improvement_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Determine if v2 qualifies for promotion:
    v2 promoted only if mae_v2 is at least 10% lower than mae_v1.
    """
    if mae_v1 <= 0:
        return {"decision": "keep", "relative_improvement": 0.0}

    relative_improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if relative_improvement >= improvement_threshold else "keep"

    return {
        "mae_v1": mae_v1,
        "mae_v2": mae_v2,
        "relative_improvement": relative_improvement,
        "threshold": improvement_threshold,
        "decision": decision,
    }
