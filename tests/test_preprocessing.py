"""Offline tests for the Adult Income preprocessing pipeline.

Uses a small hand-built frame shaped like ``adult_income`` rows -- no network,
no Supabase, no downloaded data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.preprocessing import (
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    split_xy,
    transform_features,
)
from shared.features import (
    CATEGORICAL_COLS,
    EXCLUDED_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    PROTECTED_COLS,
)


def _row(**overrides):
    row = {
        "id": 1, "age": 30, "workclass": "Private", "fnlwgt": 100_000,
        "education": "Bachelors", "education_num": 13,
        "marital_status": "Never-married", "occupation": "Sales",
        "relationship": "Not-in-family", "race": "White", "sex": "Female",
        "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
        "native_country": "United-States", "income": "<=50K", "income_label": 0,
        "split": "train",
    }
    row.update(overrides)
    return row


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = [
        _row(id=1, age=20, hours_per_week=20),
        _row(id=2, age=30, occupation=None, workclass=None),
        _row(id=3, age=40, marital_status="Married-civ-spouse", relationship="Husband",
             sex="Male", income=">50K", income_label=1, capital_gain=5000),
        _row(id=4, age=60, occupation="Exec-managerial", native_country=None,
             income=">50K", income_label=1),
        # Validation / test rows with extreme values that must NOT influence fitting.
        _row(id=5, age=90, hours_per_week=99, split="val"),
        _row(id=6, age=17, occupation="Armed-Forces", split="test"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def train(frame) -> pd.DataFrame:
    return frame[frame["split"] == "train"]


def _feature_names(pre) -> list:
    return list(pre.get_feature_names_out())


def test_feature_contract():
    assert set(NUMERIC_COLS).isdisjoint(CATEGORICAL_COLS)
    for col in list(EXCLUDED_COLS) + PROTECTED_COLS:
        assert col not in FEATURE_COLS


def test_fit_uses_train_rows_only(train):
    pre = fit_preprocessor(train)
    num = pre.named_steps["columns"].named_transformers_["num"]
    ages = NUMERIC_COLS.index("age")
    assert num.named_steps["scale"].mean_[ages] == pytest.approx(train["age"].mean())
    assert num.named_steps["impute"].statistics_[ages] == pytest.approx(train["age"].median())
    # 'Armed-Forces' appears only in the test row, so it must not be a learned category.
    assert "cat__occupation_Armed-Forces" not in _feature_names(pre)


def test_fit_rejects_non_train_rows(frame, train):
    with pytest.raises(ValueError, match="train rows only"):
        fit_preprocessor(frame)
    with pytest.raises(ValueError, match="split"):
        fit_preprocessor(train.drop(columns="split"))


def test_protected_and_excluded_columns_never_reach_the_model(train):
    pre = fit_preprocessor(train)
    names = _feature_names(pre)
    for col in list(EXCLUDED_COLS) + PROTECTED_COLS:
        assert f"num__{col}" not in names, col
        assert not any(n.startswith(f"cat__{col}_") for n in names), col
    # Changing a protected attribute must not change the transformed features.
    a = train.iloc[[0]]
    b = a.assign(sex="Male", race="Black", education="Doctorate", fnlwgt=1)
    np.testing.assert_array_equal(transform_features(pre, a), transform_features(pre, b))


def test_missing_categories_become_unknown(train):
    pre = fit_preprocessor(train)
    names = _feature_names(pre)
    assert "cat__occupation_Unknown" in names
    assert "cat__workclass_Unknown" in names
    assert "cat__native_country_Unknown" in names
    # None, pd.NA, '?' and '' at predict time all map to the Unknown column.
    rows = pd.DataFrame([_row(occupation=v) for v in (None, pd.NA, "?", "")])
    out = transform_features(pre, rows)
    assert (out[:, names.index("cat__occupation_Unknown")] == 1).all()


def test_missing_numeric_imputed_with_train_median(train):
    pre = fit_preprocessor(train)
    out = transform_features(pre, pd.DataFrame([_row(age=None)]))
    assert not np.isnan(out).any()
    num = pre.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    i = NUMERIC_COLS.index("age")
    expected = (train["age"].median() - num.mean_[i]) / num.scale_[i]
    assert out[0, _feature_names(pre).index("num__age")] == pytest.approx(expected, rel=1e-5)


def test_unseen_category_is_ignored(train):
    pre = fit_preprocessor(train)
    out = transform_features(pre, pd.DataFrame([_row(workclass="Not-a-real-class")]))
    names = _feature_names(pre)
    workclass_cols = [i for i, n in enumerate(names) if n.startswith("cat__workclass_")]
    assert (out[0, workclass_cols] == 0).all()


def test_output_shape_and_dtype(frame, train):
    pre = fit_preprocessor(train)
    out = transform_features(pre, frame)
    assert out.dtype == np.float32
    assert out.shape == (len(frame), len(_feature_names(pre)))
    assert not np.isnan(out).any()


def test_missing_feature_column_raises(train):
    pre = fit_preprocessor(train)
    with pytest.raises(ValueError, match="missing feature columns"):
        pre.transform(train.drop(columns="occupation"))


def test_split_xy(train):
    X, y = split_xy(train)
    assert list(X.columns) == FEATURE_COLS
    np.testing.assert_array_equal(y, train["income_label"].to_numpy())
    # Falls back to deriving the label from the income text.
    _, y2 = split_xy(train.drop(columns="income_label"))
    np.testing.assert_array_equal(y2, y)


def test_save_load_roundtrip(tmp_path, frame, train):
    pre = fit_preprocessor(train)
    path = save_preprocessor(pre, tmp_path / "models" / "preprocessor.joblib")
    assert path.exists()
    reloaded = load_preprocessor(path)
    np.testing.assert_array_equal(
        transform_features(pre, frame), transform_features(reloaded, frame))
