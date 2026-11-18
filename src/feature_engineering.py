"""
Feature engineering module for MP190 air quality prediction.

Extracts date-derived features and builds reusable scikit-learn transformers
ensuring zero data leakage across train, test, and inference.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_date_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Extract calendar features from date string if present.
    """
    df_out = df.copy()
    if date_column in df_out.columns:
        dt = pd.to_datetime(df_out[date_column], errors="coerce")
        df_out["month"] = dt.dt.month
        df_out["day_of_week"] = dt.dt.dayofweek
        df_out["day_of_year"] = dt.dt.dayofyear
        df_out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    return df_out


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Construct a reusable scikit-learn ColumnTransformer for numerical scaling
    and categorical one-hot encoding with imputation.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
