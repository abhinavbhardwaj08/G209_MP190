"""
Data drift monitoring module using Population Stability Index (PSI).

Computes 10-bin PSI for all numeric features between baseline and production
data distributions to detect covariate shift.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) using 10 quantile bins from reference.

    Args:
        reference: Baseline series (e.g. from train.csv).
        current: Monitoring series (e.g. from later.csv).
        num_bins: Number of bins (10 per assignment specifications).
        epsilon: Small constant to prevent division by zero or log(0).

    Returns:
        PSI value as float.
    """
    ref_clean = np.asarray(reference[~np.isnan(reference)])
    cur_clean = np.asarray(current[~np.isnan(current)])

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Determine bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges = np.unique(bin_edges)  # handle tied percentiles

    if len(bin_edges) < 2:
        return 0.0

    # Ensure edge coverage
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequencies
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    cur_pct = (cur_counts / len(cur_clean)) + epsilon

    # Normalize back to sum = 1
    ref_pct /= np.sum(ref_pct)
    cur_pct /= np.sum(cur_pct)

    # PSI calculation: sum((current - reference) * ln(current / reference))
    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_val)


def calculate_all_psi(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_columns: List[str],
    num_bins: int = 10
) -> Dict[str, float]:
    """Calculate PSI for all specified numeric columns."""
    results: Dict[str, float] = {}
    for col in numeric_columns:
        if col in train_df.columns and col in later_df.columns:
            results[col] = calculate_psi(train_df[col], later_df[col], num_bins=num_bins)
    return results


def find_most_drifted_feature(psi_results: Dict[str, float]) -> Tuple[str, float]:
    """Identify the feature with the highest PSI value."""
    if not psi_results:
        return ("none", 0.0)
    most_drifted = max(psi_results.items(), key=lambda item: item[1])
    return most_drifted
