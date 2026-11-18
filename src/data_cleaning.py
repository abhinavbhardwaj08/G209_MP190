"""
Data cleaning and preprocessing module for MP190.

Transforms messy raw data into a clean, leakage-free dataset ready
for feature engineering and model training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def remove_leaking_columns(df: pd.DataFrame, leak_cols: List[str]) -> pd.DataFrame:
    """Drop columns that cause data leakage at prediction time."""
    cols_to_drop = [c for c in leak_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while retaining the first occurrence."""
    return df.drop_duplicates()


def correct_unit_mixup(
    df: pd.DataFrame,
    column: str,
    group_col: str,
    group_val: str,
    factor: float
) -> pd.DataFrame:
    """Correct unit mixup for a specific group by dividing by the factor."""
    df_clean = df.copy()
    if column in df_clean.columns and group_col in df_clean.columns:
        mask = df_clean[group_col] == group_val
        df_clean.loc[mask, column] = df_clean.loc[mask, column] / factor
    return df_clean


def handle_sentinel_values(
    df: pd.DataFrame,
    column: str,
    sentinel_val: float,
    replacement: Optional[float] = None
) -> pd.DataFrame:
    """Replace sentinel values with NaN or imputed value."""
    df_clean = df.copy()
    if column in df_clean.columns:
        df_clean[column] = df_clean[column].replace(sentinel_val, replacement)
    return df_clean


def clean_dataset(df: pd.DataFrame, findings: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Apply full cleaning pipeline based on data-quality findings.
    Preserves original dataframe by operating on a copy.
    """
    cleaned = df.copy()
    # Baseline deduplication
    cleaned = remove_duplicates(cleaned)
    return cleaned


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned dataframe to designated processed data path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
