"""
Data validation module for detecting the seven data-quality issues in MP190.

Issues investigated per Student Handbook Section 7:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_columns: Optional[List[str]] = None,
    target_column: str = "pm25_next_day"
) -> Dict[str, Any]:
    """
    Identify potential feature leakage: columns present in training data
    that are absent from holdout_features.csv or contain future information.
    """
    result: Dict[str, Any] = {"present": False, "column": None}
    if holdout_columns is not None:
        missing_in_holdout = [
            c for c in train_df.columns
            if c not in holdout_columns and c != target_column
        ]
        if missing_in_holdout:
            result["present"] = True
            result["column"] = missing_in_holdout[0]
            return result

    # Fallback heuristic: check for suspicious column names containing 'measured' or 'next_day'
    for col in train_df.columns:
        if col != target_column and ("measured_next_day" in col or "next_day" in col):
            return {"present": True, "column": col}

    return result


def check_unit_mixup(
    df: pd.DataFrame,
    numeric_columns: List[str],
    group_column: str = "station_zone"
) -> Dict[str, Any]:
    """
    Detect unit mixups where one group has values scaled by a conversion factor
    (e.g., 10, 12, 60, 100, 1000).
    """
    candidate_factors = [10, 12, 60, 100, 1000]
    result: Dict[str, Any] = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }

    if group_column not in df.columns:
        return result

    groups = df[group_column].dropna().unique()
    if len(groups) < 2:
        return result

    for col in numeric_columns:
        if col not in df.columns or col == "record_id":
            continue

        group_means = df.groupby(group_column)[col].mean()

        for g1 in group_means.index:
            for g2 in group_means.index:
                if g1 == g2:
                    continue
                m1 = group_means[g1]
                m2 = group_means[g2]
                if m2 <= 0:
                    continue
                ratio = m1 / m2
                for factor in candidate_factors:
                    if 0.80 * factor <= ratio <= 1.20 * factor:
                        return {
                            "present": True,
                            "column": col,
                            "group_value": str(g1),
                            "factor": factor,
                        }

    return result


def check_sentinel_value(
    df: pd.DataFrame,
    numeric_columns: List[str],
    candidates: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Detect sentinel values (e.g., -1, -999, 999, 9999) used in place of missing data.
    """
    if candidates is None:
        candidates = [-999.0, -1.0, 999.0, 9999.0, -99.0, 99.0]

    for col in numeric_columns:
        if col not in df.columns or col in ["record_id", "date"]:
            continue

        col_series = df[col].dropna()
        for cand in candidates:
            cand_count = int((col_series == cand).sum())
            if cand_count >= 5:
                other_vals = col_series[col_series != cand]
                if len(other_vals) > 0 and (cand < other_vals.min() or cand > other_vals.max()):
                    return {
                        "present": True,
                        "column": col,
                        "value": cand,
                        "count": cand_count,
                    }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }


def check_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Count duplicate records in the dataset (extra copies to be removed).
    """
    extra_copies = int(df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
    }


def check_stuck_sensor(
    df: pd.DataFrame,
    date_column: str = "date",
    numeric_columns: Optional[List[str]] = None,
    min_streak_days: int = 14
) -> Dict[str, Any]:
    """
    Detect sensors producing constant repeating values over consecutive days
    (e.g., stuck reading lasting 2 or more weeks per handbook).
    """
    if numeric_columns is None:
        numeric_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["record_id", "date"]
        ]

    if date_column not in df.columns:
        return {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None,
        }

    df_sorted = df.sort_values(date_column).copy()

    for col in numeric_columns:
        if col not in df_sorted.columns:
            continue

        streaks = (df_sorted[col] != df_sorted[col].shift(1)).cumsum()
        grouped = df_sorted.groupby(streaks).agg(
            start_date=(date_column, "first"),
            end_date=(date_column, "last"),
            count=(col, "count"),
        )

        for _, row in grouped.iterrows():
            try:
                days = (pd.to_datetime(row["end_date"]) - pd.to_datetime(row["start_date"])).days
                if days >= min_streak_days and row["count"] >= min_streak_days:
                    return {
                        "present": True,
                        "column": col,
                        "start_date": str(row["start_date"]),
                        "end_date": str(row["end_date"]),
                    }
            except Exception:
                pass

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


def check_target_typos(
    df: pd.DataFrame,
    target_column: str = "pm25_next_day",
    factor: float = 10.0
) -> Dict[str, Any]:
    """
    Detect targets typed with an extra zero (10x too large) sitting far above distribution.
    Identified by a large empty gap below the anomalous cluster.
    """
    if target_column not in df.columns:
        return {"present": False, "count": None}

    targets = df[target_column].dropna().sort_values().values
    if len(targets) < 10:
        return {"present": False, "count": None}

    diffs = np.diff(targets)
    gap_threshold = np.mean(diffs) + 5 * np.std(diffs)

    typo_candidates = []
    for i in range(len(diffs) - 1, max(len(diffs) - 20, 0), -1):
        if diffs[i] > gap_threshold and targets[i + 1] > 2 * targets[i]:
            typo_candidates = targets[i + 1:]
            break

    if len(typo_candidates) > 0:
        return {
            "present": True,
            "count": int(len(typo_candidates)),
        }

    return {
        "present": False,
        "count": None,
    }


def check_spelling_variants(
    df: pd.DataFrame,
    categorical_columns: List[str],
    reference_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Detect inconsistent spellings / casing / trailing spaces in categorical columns.
    """
    for col in categorical_columns:
        if col not in df.columns:
            continue

        raw_levels = df[col].dropna().unique()
        clean_levels = df[col].dropna().astype(str).str.strip().str.lower().unique()

        if len(raw_levels) > len(clean_levels):
            return {
                "present": True,
                "column": col,
                "true_levels": int(len(clean_levels)),
            }

        if reference_df is not None and col in reference_df.columns:
            ref_levels = set(reference_df[col].dropna().unique())
            extra_in_raw = set(raw_levels) - ref_levels
            if extra_in_raw:
                return {
                    "present": True,
                    "column": col,
                    "true_levels": int(len(ref_levels)),
                }

    return {
        "present": False,
        "column": None,
        "true_levels": None,
    }


def run_all_validation_checks(
    train_df: pd.DataFrame,
    holdout_columns: Optional[List[str]] = None,
    reference_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Execute all seven data quality checks and return findings structured
    to mirror findings.yaml.
    """
    numeric_cols = [
        "pm25_today", "wind_kmph", "humidity_pct", "temp_c",
        "traffic_index", "stubble_fires"
    ]
    categorical_cols = ["day_type", "station_zone"]

    return {
        "leaking_column": check_leaking_column(train_df, holdout_columns),
        "unit_mixup": check_unit_mixup(train_df, numeric_cols, group_column="station_zone"),
        "sentinel_value": check_sentinel_value(train_df, numeric_cols),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, categorical_cols, reference_df),
    }
