"""
Data cleaning and preprocessing module for MP190.

Transforms messy raw data into a clean, leakage-free dataset ready
for feature engineering and model training:
1. Removes 39 duplicate records.
2. Corrects wind_kmph unit mixup in Residential zone (divided by 1000).
3. Corrects 8 target_typos in pm25_next_day (divided by 10).
4. Drops leaking column pm25_measured_next_day.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while retaining the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)


def correct_unit_mixup(
    df: pd.DataFrame,
    column: str = "wind_kmph",
    group_col: str = "station_zone",
    group_val: str = "Residential",
    factor: float = 1000.0
) -> pd.DataFrame:
    """
    Correct unit mixup for a specific group by dividing by the exact conversion factor.
    Example: Residential wind recorded in m/h converted to km/h by dividing by 1000.
    """
    df_clean = df.copy()
    if column in df_clean.columns and group_col in df_clean.columns:
        mask = df_clean[group_col] == group_val
        df_clean.loc[mask, column] = df_clean.loc[mask, column] / factor
    return df_clean


def correct_target_typos(
    df: pd.DataFrame,
    target_column: str = "pm25_next_day",
    threshold: float = 500.0,
    factor: float = 10.0
) -> pd.DataFrame:
    """
    Correct target typographical errors where numbers were typed 10x too large.
    Records above the threshold (e.g. 500 ug/m3) are divided by factor (10.0).
    """
    df_clean = df.copy()
    if target_column in df_clean.columns:
        mask = df_clean[target_column] > threshold
        df_clean.loc[mask, target_column] = df_clean.loc[mask, target_column] / factor
    return df_clean


def remove_leaking_columns(
    df: pd.DataFrame,
    leak_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Drop columns that cause data leakage at prediction time."""
    if leak_cols is None:
        leak_cols = ["pm25_measured_next_day"]
    cols_to_drop = [c for c in leak_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


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
    Apply full cleaning pipeline based on verified empirical findings:
    1. Deduplication (removes 39 extra duplicate rows).
    2. Unit mixup correction (wind_kmph in Residential / 1000).
    3. Target typos correction (pm25_next_day > 500 / 10).
    4. Target leakage removal (drop pm25_measured_next_day).
    """
    cleaned = df.copy()

    # Step 1: Remove exact duplicates
    cleaned = remove_duplicates(cleaned)

    # Step 2: Correct unit mixup
    cleaned = correct_unit_mixup(
        cleaned,
        column="wind_kmph",
        group_col="station_zone",
        group_val="Residential",
        factor=1000.0
    )

    # Step 3: Correct target typos
    cleaned = correct_target_typos(
        cleaned,
        target_column="pm25_next_day",
        threshold=500.0,
        factor=10.0
    )

    # Step 4: Drop target leakage columns
    cleaned = remove_leaking_columns(cleaned, ["pm25_measured_next_day"])

    return cleaned


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned dataframe to designated processed data path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_cleaning_pipeline(
    raw_path: str | Path = "data/raw/train.csv",
    output_path: str | Path = "data/processed/train_cleaned.csv"
) -> pd.DataFrame:
    """
    Execute the end-to-end cleaning workflow on the raw training dataset
    and save the cleaned artifact to data/processed/train_cleaned.csv.
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_file}")

    raw_df = pd.read_csv(raw_file)
    cleaned_df = clean_dataset(raw_df)
    save_processed_data(cleaned_df, output_path)
    return cleaned_df


if __name__ == "__main__":
    cleaned = run_cleaning_pipeline()
    print(f"Cleaning complete. Processed dataset saved with shape: {cleaned.shape}")
