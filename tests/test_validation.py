"""
Unit tests for Pydantic schema validation rules.

Verifies boundary constraints on temp_c and wind_kmph.
"""

import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest


def test_prediction_request_valid():
    """Verify valid request instantiates without errors."""
    req = PredictionRequest(
        pm25_today=35.0,
        wind_kmph=10.0,
        humidity_pct=65.0,
        temp_c=30.0,
        traffic_index=100.0,
        stubble_fires=2.0,
        day_type="weekday",
        station_zone="Central",
    )
    assert req.temp_c == 30.0
    assert req.wind_kmph == 10.0


def test_temperature_boundary():
    """Verify temp_c = 70 is allowed, but > 70 is rejected."""
    # Boundary: 70.0 is allowed
    req_70 = PredictionRequest(
        pm25_today=35.0,
        wind_kmph=10.0,
        humidity_pct=65.0,
        temp_c=70.0,
        traffic_index=100.0,
        stubble_fires=2.0,
        day_type="weekday",
        station_zone="Central",
    )
    assert req_70.temp_c == 70.0

    # Above 70 is rejected
    with pytest.raises(ValidationError):
        PredictionRequest(
            pm25_today=35.0,
            wind_kmph=10.0,
            humidity_pct=65.0,
            temp_c=70.01,
            traffic_index=100.0,
            stubble_fires=2.0,
            day_type="weekday",
            station_zone="Central",
        )


def test_wind_speed_boundary():
    """Verify wind_kmph = 0 is allowed, but < 0 is rejected."""
    # Boundary: 0.0 is allowed
    req_0 = PredictionRequest(
        pm25_today=35.0,
        wind_kmph=0.0,
        humidity_pct=65.0,
        temp_c=30.0,
        traffic_index=100.0,
        stubble_fires=2.0,
        day_type="weekday",
        station_zone="Central",
    )
    assert req_0.wind_kmph == 0.0

    # Below 0 is rejected
    with pytest.raises(ValidationError):
        PredictionRequest(
            pm25_today=35.0,
            wind_kmph=-0.1,
            humidity_pct=65.0,
            temp_c=30.0,
            traffic_index=100.0,
            stubble_fires=2.0,
            day_type="weekday",
            station_zone="Central",
        )


# --- Unit tests for src/data_validation.py detection algorithms ---
import pandas as pd
from src.data_validation import (
    check_leaking_column,
    check_unit_mixup,
    check_sentinel_value,
    check_duplicate_rows,
    check_stuck_sensor,
    check_target_typos,
    check_spelling_variants,
)


def test_check_leaking_column():
    """Verify leaking column detection when missing from holdout."""
    df = pd.DataFrame({"feat1": [1, 2], "leak_col": [3, 4], "pm25_next_day": [10, 20]})
    res = check_leaking_column(df, holdout_columns=["feat1"])
    assert res["present"] is True
    assert res["column"] == "leak_col"


def test_check_unit_mixup():
    """Verify unit mixup factor detection."""
    df = pd.DataFrame({
        "station_zone": ["Central"] * 5 + ["Residential"] * 5,
        "wind_kmph": [10.0] * 5 + [10000.0] * 5,
    })
    res = check_unit_mixup(df, numeric_columns=["wind_kmph"], group_column="station_zone")
    assert res["present"] is True
    assert res["column"] == "wind_kmph"
    assert res["group_value"] == "Residential"
    assert res["factor"] == 1000


def test_check_sentinel_value():
    """Verify sentinel value detector."""
    # Isolated sentinel value of -999
    df = pd.DataFrame({"temp_c": [20.0, 21.0, 22.0, -999.0, -999.0, -999.0, -999.0, -999.0]})
    res = check_sentinel_value(df, numeric_columns=["temp_c"])
    assert res["present"] is True
    assert res["column"] == "temp_c"
    assert res["value"] == -999.0
    assert res["count"] == 5


def test_check_duplicate_rows():
    """Verify duplicate row counting."""
    df = pd.DataFrame({"a": [1, 2, 1], "b": [3, 4, 3]})
    res = check_duplicate_rows(df)
    assert res["present"] is True
    assert res["count"] == 1


def test_check_stuck_sensor():
    """Verify stuck sensor detection on constant run of dates."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "temp_c": [25.0] * 20})
    res = check_stuck_sensor(df, date_column="date", numeric_columns=["temp_c"], min_streak_days=14)
    assert res["present"] is True
    assert res["column"] == "temp_c"
    assert res["start_date"] == "2026-01-01"


def test_check_target_typos():
    """Verify target typo detection for 10x outliers."""
    targets = [50.0] * 30 + [55.0] * 30 + [500.0, 520.0]  # 2 outliers ~10x
    df = pd.DataFrame({"pm25_next_day": targets})
    res = check_target_typos(df, target_column="pm25_next_day")
    assert res["present"] is True
    assert res["count"] == 2


def test_check_spelling_variants():
    """Verify spelling variant detection."""
    df = pd.DataFrame({"day_type": ["weekday", "Weekday", "weekday "]})
    res = check_spelling_variants(df, categorical_columns=["day_type"])
    assert res["present"] is True
    assert res["column"] == "day_type"
    assert res["true_levels"] == 1
