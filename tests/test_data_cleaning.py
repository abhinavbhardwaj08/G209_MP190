"""
Unit tests for data cleaning operations.

All unit tests utilize self-contained synthetic in-memory DataFrames
and do not require external CSV files.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_dataset,
    correct_target_typos,
    correct_unit_mixup,
    handle_sentinel_values,
    remove_duplicates,
    remove_leaking_columns,
)


def test_remove_duplicates():
    """Verify exact duplicate rows are removed, retaining unique records."""
    df = pd.DataFrame({
        "record_id": ["A1", "A2", "A1"],
        "val": [10.0, 20.0, 10.0],
    })
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2
    assert list(cleaned["record_id"]) == ["A1", "A2"]


def test_remove_leaking_columns():
    """Verify specified leakage columns are removed from dataset."""
    df = pd.DataFrame({
        "pm25_today": [30.0, 40.0],
        "pm25_measured_next_day": [35.0, 42.0],
        "pm25_next_day": [35.0, 42.0],
    })
    cleaned = remove_leaking_columns(df, ["pm25_measured_next_day"])
    assert "pm25_measured_next_day" not in cleaned.columns
    assert "pm25_today" in cleaned.columns
    assert "pm25_next_day" in cleaned.columns


def test_correct_unit_mixup():
    """Verify values for Residential group are correctly divided by factor 1000."""
    df = pd.DataFrame({
        "station_zone": ["Central", "Residential", "Industrial"],
        "wind_kmph": [14.0, 15000.0, 16.0],
    })
    cleaned = correct_unit_mixup(df, "wind_kmph", "station_zone", "Residential", 1000.0)
    assert cleaned.loc[cleaned["station_zone"] == "Residential", "wind_kmph"].values[0] == 15.0
    assert cleaned.loc[cleaned["station_zone"] == "Central", "wind_kmph"].values[0] == 14.0
    assert cleaned.loc[cleaned["station_zone"] == "Industrial", "wind_kmph"].values[0] == 16.0


def test_correct_target_typos():
    """Verify targets exceeding threshold (> 500) are divided by factor 10."""
    df = pd.DataFrame({
        "pm25_next_day": [150.0, 2100.0, 850.0, 220.0],
    })
    cleaned = correct_target_typos(df, target_column="pm25_next_day", threshold=500.0, factor=10.0)
    assert cleaned["pm25_next_day"].tolist() == [150.0, 210.0, 85.0, 220.0]


def test_handle_sentinel_values():
    """Verify sentinel numbers are replaced with None/NaN."""
    df = pd.DataFrame({
        "temp_c": [25.0, -999.0, 30.0],
    })
    cleaned = handle_sentinel_values(df, "temp_c", -999.0, replacement=None)
    assert pd.isna(cleaned.loc[1, "temp_c"])
    assert cleaned.loc[0, "temp_c"] == 25.0


def test_clean_dataset_pipeline():
    """Verify composite cleaning pipeline applies deduplication, mixup, typos, and leak drop."""
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R2"],  # 1 duplicate
        "station_zone": ["Residential", "Residential", "Residential"],
        "wind_kmph": [14000.0, 12000.0, 12000.0],  # unit mixup
        "pm25_measured_next_day": [100.0, 110.0, 110.0],  # leakage
        "pm25_next_day": [1000.0, 110.0, 110.0],  # 1 typo (>500)
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2  # Deduplicated
    assert "pm25_measured_next_day" not in cleaned.columns  # Leak dropped
    assert cleaned.loc[0, "wind_kmph"] == 14.0  # Unit mixup fixed
    assert cleaned.loc[0, "pm25_next_day"] == 100.0  # Typo fixed


def test_cleaned_dataset_file_integrity():
    """Verify processed training dataset on disk satisfies all quality guarantees."""
    cleaned_path = Path("data/processed/train_cleaned.csv")
    if not cleaned_path.exists():
        pytest.skip("Cleaned dataset not yet generated on disk")

    df = pd.read_csv(cleaned_path)
    assert len(df) == 1000  # Exactly 1039 - 39 duplicates
    assert len(df.columns) == 11  # Leakage column dropped
    assert "pm25_measured_next_day" not in df.columns
    assert df.duplicated().sum() == 0  # No duplicate rows
    assert df["pm25_next_day"].max() <= 350.0  # No typos remaining (> 500)
    assert df[df["station_zone"] == "Residential"]["wind_kmph"].max() <= 35.0  # Unit mixup resolved
