"""Reusable sklearn preprocessing pipeline for the Adult Income MLP.

One pipeline object is fit on the TRAINING split only, serialized with joblib,
and reloaded unchanged by the training script and the FastAPI service, so every
row (train, val, test, /predict input) goes through identical transforms:

    select + clean      keep only FEATURE_COLS; None / pd.NA / '?' / '' -> NaN
    numeric columns     median imputation -> StandardScaler
    categorical columns constant 'Unknown' imputation -> OneHotEncoder

Protected attributes (sex, race) and the excluded columns (education, fnlwgt)
are dropped in the first step, so they can never reach the model even if the
caller passes a full ``adult_income`` row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from shared.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MISSING_CATEGORY,
    NUMERIC_COLS,
    TARGET_CLASSES,
    TARGET_LABEL,
    TARGET_NAME,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREPROCESSOR_PATH = REPO_ROOT / "models" / "preprocessor.joblib"

# String values treated as missing in categorical inputs (the raw UCI marker
# and blank form fields), in addition to None / NaN / pd.NA.
_MISSING_TOKENS = ["?", ""]

PathLike = Union[str, Path]


def _select_and_clean(X) -> pd.DataFrame:
    """Keep only FEATURE_COLS and normalize missing values to NaN.

    Module-level (not a lambda) so the fitted pipeline can be pickled.
    """
    X = pd.DataFrame(X)
    absent = [c for c in FEATURE_COLS if c not in X.columns]
    if absent:
        raise ValueError(f"missing feature columns: {absent}")
    X = X[FEATURE_COLS].copy()
    for c in NUMERIC_COLS:
        # errors="raise": a non-numeric value is a bad request, not a missing one.
        X[c] = pd.to_numeric(X[c], errors="raise").astype("float64")
    for c in CATEGORICAL_COLS:
        col = X[c].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
        X[c] = col.mask(col.isna() | col.isin(_MISSING_TOKENS), np.nan)
    return X


def _select_feature_names_out(transformer, input_features):
    return np.asarray(FEATURE_COLS, dtype=object)


def build_preprocessor() -> Pipeline:
    """Return a new, UNFITTED preprocessing pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("select", FunctionTransformer(
            _select_and_clean, validate=False,
            feature_names_out=_select_feature_names_out)),
        ("columns", ColumnTransformer([
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ])),
    ])


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split ``adult_income`` rows into (feature frame, 0/1 target array).

    Uses ``income_label`` when present, otherwise derives it from ``income``.
    """
    if TARGET_LABEL in df.columns:
        y = df[TARGET_LABEL]
    elif TARGET_NAME in df.columns:
        y = (df[TARGET_NAME] == TARGET_CLASSES[1]).astype(int)
    else:
        raise ValueError(f"need a '{TARGET_LABEL}' or '{TARGET_NAME}' column")
    if y.isna().any():
        raise ValueError("target contains missing values")
    return df[FEATURE_COLS].copy(), y.to_numpy(dtype=np.int64)


def fit_preprocessor(train_df: pd.DataFrame) -> Pipeline:
    """Fit a new preprocessor on training rows ONLY.

    ``train_df`` must be ``adult_income`` rows including the ``split`` column,
    and every row must be ``split == 'train'``; this guards against leaking
    validation/test statistics (means, medians, categories) into the pipeline.
    """
    if "split" not in train_df.columns:
        raise ValueError("train_df needs the 'split' column to verify train-only fitting")
    splits = set(train_df["split"].unique())
    if splits != {"train"}:
        raise ValueError(f"preprocessor must be fit on train rows only, got splits {sorted(splits)}")
    return build_preprocessor().fit(train_df)


def transform_features(pre: Pipeline, df) -> np.ndarray:
    """Apply a fitted preprocessor; return a float32 matrix ready for torch."""
    return pre.transform(df).astype(np.float32)


def save_preprocessor(pre: Pipeline, path: PathLike = DEFAULT_PREPROCESSOR_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, path)
    return path


def load_preprocessor(path: PathLike = DEFAULT_PREPROCESSOR_PATH) -> Pipeline:
    # joblib uses pickle: only load files this project produced.
    return joblib.load(Path(path))
