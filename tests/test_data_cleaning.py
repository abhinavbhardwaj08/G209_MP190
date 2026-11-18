"""
Unit tests for data cleaning operations.

All tests utilize self-contained synthetic in-memory DataFrames
and do not require external CSV files.
"""

import pandas as pd
import pytest

from src.data_cleaning import (
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
    """Verify values for a specific group are correctly scaled by the factor."""
    df = pd.DataFrame({
        "station_zone": ["Central", "Residential", "Central"],
        "wind_kmph": [10.0, 10000.0, 15.0],
    })
    # Suppose Residential was in metres/hr instead of km/hr (factor 1000)
    cleaned = correct_unit_mixup(df, "wind_kmph", "station_zone", "Residential", 1000.0)
    assert cleaned.loc[cleaned["station_zone"] == "Residential", "wind_kmph"].values[0] == 10.0
    assert cleaned.loc[cleaned["station_zone"] == "Central", "wind_kmph"].values[0] == 10.0


def test_handle_sentinel_values():
    """Verify sentinel numbers are replaced with None/NaN."""
    df = pd.DataFrame({
        "temp_c": [25.0, -999.0, 30.0],
    })
    cleaned = handle_sentinel_values(df, "temp_c", -999.0, replacement=None)
    assert pd.isna(cleaned.loc[1, "temp_c"])
    assert cleaned.loc[0, "temp_c"] == 25.0
