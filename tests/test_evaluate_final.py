"""Offline tests for the final evaluation of the selected run.

A small synthetic Adult-shaped frame and a tiny trained MLP -- no Supabase, no
real data, no files under models/.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import api.evaluate_final as ef
from api.calibration import CALIBRATOR_FILENAME, TemperatureScaler
from api.persist_runs import RUNS_COLUMNS
from api.train import TrainConfig, build_run_row, prepare_splits, train_model
from shared.features import FEATURE_COLS, TARGET_CLASSES


def _quiet(_msg: str) -> None:
    pass


def _adult_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Adult-shaped rows whose label depends on education_num and hours_per_week."""
    rng = np.random.default_rng(seed)
    edu = rng.integers(1, 17, n)
    hours = rng.integers(10, 80, n)
    score = (edu - 10) / 3 + (hours - 40) / 20 + rng.normal(0, 0.7, n)
    split = np.array(["train"] * (n // 2) + ["val"] * (n // 4) + ["test"] * (n - n // 2 - n // 4))
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "split": split,
        "income_label": (score > 0.3).astype(int),
        "age": rng.integers(17, 90, n),
        "education_num": edu,
        "capital_gain": rng.choice([0, 0, 0, 5000], n),
        "capital_loss": np.zeros(n, dtype=int),
        "hours_per_week": hours,
        "workclass": rng.choice(["Private", "Self-emp-inc", "State-gov", None], n),
        "marital_status": rng.choice(["Never-married", "Married-civ-spouse", "Divorced"], n),
        "occupation": rng.choice(["Sales", "Exec-managerial", "Craft-repair"], n),
        "relationship": rng.choice(["Husband", "Wife", "Not-in-family"], n),
        "native_country": rng.choice(["United-States", "Mexico"], n),
    })


@pytest.fixture(scope="module")
def pipeline():
    """(cfg, train result, preprocessor, {split: raw rows}) from a tiny trained run."""
    df = _adult_frame()
    data = prepare_splits(df)
    cfg = TrainConfig(name="tiny", hidden_sizes=[16, 8], activation="gelu", dropout=0.0,
                      learning_rate=0.01, weight_decay=0.0, epochs=5, batch_size=32, seed=0)
    result = train_model(cfg, data.X["train"], data.y["train"], data.X["val"], data.y["val"],
                         log=_quiet)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, part in frames.items():
        assert part["income_label"].nunique() == 2, f"{s} needs both classes"
    return cfg, result, data.preprocessor, frames


def _fitted(pipeline):
    _, result, pre, frames = pipeline
    calibrator, report = ef.fit_calibration(result.model, pre, frames["val"])
    return calibrator, report


# ---------------------------------------------------------------------------
# calibration uses validation only
# ---------------------------------------------------------------------------
def test_calibrator_fits_on_validation_rows_only(pipeline):
    _, result, pre, frames = pipeline
    for bad in (frames["test"], frames["train"], pd.concat([frames["val"], frames["test"]])):
        with pytest.raises(ValueError, match="'val'"):
            ef.fit_calibration(result.model, pre, bad)

    calibrator, report = ef.fit_calibration(result.model, pre, frames["val"])
    assert calibrator.is_fitted
    assert calibrator.n_fit == len(frames["val"]) == report["n"]
    assert report["fitted_on_split"] == "val" and report["method"] == "temperature"
    for key in ("brier_before", "brier_after", "ece_before", "ece_after", "bins_before", "bins_after"):
        assert key in report
    json.dumps(report, allow_nan=False)


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def test_confusion_matrix_layout_is_tn_fp_fn_tp():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1]
    assert ef.confusion_matrix_2x2(y_true, y_pred) == [[2, 1], [1, 1]]  # [[TN, FP], [FN, TP]]
    assert all(isinstance(v, int) for r in ef.confusion_matrix_2x2(y_true, y_pred) for v in r)


def test_per_class_metrics_hand_computed():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1]  # TN=1 FP=1 FN=1 TP=2
    pc = ef.per_class_metrics(y_true, y_pred)
    assert list(pc) == TARGET_CLASSES
    assert pc["<=50K"] == pytest.approx({"label": 0, "precision": 0.5, "recall": 0.5,
                                         "f1": 0.5, "support": 2})
    assert pc[">50K"] == pytest.approx({"label": 1, "precision": 2 / 3, "recall": 2 / 3,
                                        "f1": 2 / 3, "support": 3})


def test_threshold_stays_fixed_at_half(pipeline):
    assert ef.DECISION_THRESHOLD == 0.5
    np.testing.assert_array_equal(ef.predict_labels([0.49999, 0.5, 0.51, 0.0]), [0, 1, 1, 0])
    assert "threshold" not in inspect.signature(ef.evaluate_test).parameters
    assert not any("threshold" in opt for opt in ef.build_parser()._option_string_actions)

    _, result, pre, frames = pipeline
    calibrator, _ = _fitted(pipeline)
    evaluation, preds = ef.evaluate_test(result.model, pre, calibrator, frames["test"])
    assert evaluation["threshold"] == 0.5 and evaluation["metrics"]["threshold"] == 0.5
    np.testing.assert_array_equal(preds["predicted_label"], (preds["prob_calibrated"] >= 0.5).astype(int))
    # temperature scaling never moves a row across 0.5
    np.testing.assert_array_equal(preds["predicted_label"], (preds["prob_uncalibrated"] >= 0.5).astype(int))


# ---------------------------------------------------------------------------
# test evaluation
# ---------------------------------------------------------------------------
def test_evaluate_test_requires_fitted_calibrator_and_test_rows(pipeline):
    _, result, pre, frames = pipeline
    with pytest.raises(RuntimeError, match="validation"):
        ef.evaluate_test(result.model, pre, TemperatureScaler(), frames["test"])
    calibrator, _ = _fitted(pipeline)
    with pytest.raises(ValueError, match="'test'"):
        ef.evaluate_test(result.model, pre, calibrator, frames["val"])


def test_evaluate_test_contents(pipeline):
    _, result, pre, frames = pipeline
    calibrator, _ = _fitted(pipeline)
    evaluation, preds = ef.evaluate_test(result.model, pre, calibrator, frames["test"])
    n = len(frames["test"])

    m = evaluation["metrics"]
    for key in ("loss", "loss_calibrated", "accuracy", "precision", "recall", "f1", "roc_auc"):
        assert isinstance(m[key], float)
    (tn, fp), (fn, tp) = evaluation["confusion_matrix"]
    assert tn + fp + fn + tp == n
    assert m["accuracy"] == pytest.approx((tn + tp) / n)
    assert evaluation["per_class"]["<=50K"]["support"] == tn + fp
    assert evaluation["per_class"][">50K"]["support"] == fn + tp
    assert evaluation["per_class"][">50K"]["precision"] == pytest.approx(m["precision"])

    cal = evaluation["calibration"]
    for key in ("brier_before", "brier_after", "ece_before", "ece_after"):
        assert 0.0 <= cal[key] <= 1.0
    assert sum(b["count"] for b in cal["bins_before"]) == n

    assert len(preds) == n
    assert list(preds["adult_income_id"]) == list(frames["test"]["id"])
    assert preds["prob_calibrated"].between(0, 1).all()
    json.dumps(evaluation, allow_nan=False)


def test_test_evaluation_refits_nothing(pipeline, monkeypatch):
    _, result, pre, frames = pipeline
    calibrator, _ = _fitted(pipeline)
    weights = {k: v.clone() for k, v in result.model.state_dict().items()}
    temperature = calibrator.temperature
    X_before = ef.transform_features(pre, frames["test"])

    def refit(*_args, **_kwargs):
        raise AssertionError("something was refit during test evaluation")

    for cls in (Pipeline, ColumnTransformer):
        monkeypatch.setattr(cls, "fit", refit)
        monkeypatch.setattr(cls, "fit_transform", refit)
    monkeypatch.setattr(TemperatureScaler, "fit", refit)

    ef.evaluate_test(result.model, pre, calibrator, frames["test"])
    ef.permutation_importance(result.model, pre, calibrator, frames["test"], n_repeats=2)

    assert all(torch.equal(weights[k], v) for k, v in result.model.state_dict().items())
    assert not result.model.training
    assert calibrator.temperature == temperature
    np.testing.assert_array_equal(ef.transform_features(pre, frames["test"]), X_before)


def test_test_rows_are_loaded_only_after_calibrator_is_saved(pipeline, tmp_path, monkeypatch):
    _, result, pre, frames = pipeline
    events = []
    real_fit = TemperatureScaler.fit

    def spy_fit(self, logits, y):
        events.append(("fit", len(y)))
        return real_fit(self, logits, y)

    def load_test_df():
        assert (tmp_path / CALIBRATOR_FILENAME).exists(), "calibrator must be saved before test"
        events.append(("load_test", None))
        return frames["test"]

    monkeypatch.setattr(TemperatureScaler, "fit", spy_fit)
    out = ef.run_final_evaluation(result.model, pre, frames["val"], load_test_df, tmp_path,
                                  log=_quiet)

    assert events == [("fit", len(frames["val"])), ("load_test", None)]  # one fit, on val
    for name in (CALIBRATOR_FILENAME, ef.CALIBRATION_FILENAME, ef.EVALUATION_FILENAME,
                 ef.IMPORTANCE_FILENAME, ef.PREDICTIONS_FILENAME):
        assert (tmp_path / name).exists(), name
    saved = TemperatureScaler.load(tmp_path / CALIBRATOR_FILENAME)
    assert saved.temperature == out["calibrator"].temperature
    assert json.loads((tmp_path / ef.EVALUATION_FILENAME).read_text())["split"] == "test"


# ---------------------------------------------------------------------------
# permutation importance
# ---------------------------------------------------------------------------
def test_permutation_importance_reports_original_features(pipeline):
    _, result, pre, frames = pipeline
    calibrator, _ = _fitted(pipeline)
    imp = ef.permutation_importance(result.model, pre, calibrator, frames["test"], n_repeats=3, seed=7)

    names = [f["feature"] for f in imp["features"]]
    assert sorted(names) == sorted(FEATURE_COLS) and len(names) == 10
    assert not any("__" in n for n in names)  # no one-hot names like cat__occupation_Sales
    assert [f["rank"] for f in imp["features"]] == list(range(1, 11))
    means = [f["importance_mean"] for f in imp["features"]]
    assert means == sorted(means, reverse=True)
    assert all(len(f["importances"]) == 3 for f in imp["features"])
    assert imp["split"] == "test" and imp["scoring"] == "roc_auc"
    json.dumps(imp, allow_nan=False)

    again = ef.permutation_importance(result.model, pre, calibrator, frames["test"], n_repeats=3, seed=7)
    assert again == imp  # seeded, so reproducible


def test_permutation_importance_rejects_train_rows(pipeline):
    _, result, pre, frames = pipeline
    calibrator, _ = _fitted(pipeline)
    with pytest.raises(ValueError, match="held-out"):
        ef.permutation_importance(result.model, pre, calibrator, frames["train"], n_repeats=1)


# ---------------------------------------------------------------------------
# final runs row
# ---------------------------------------------------------------------------
def test_final_run_row_matches_runs_schema(pipeline, tmp_path):
    cfg, result, pre, frames = pipeline
    base = build_run_row(cfg, result, Path("models/tiny/model.pt"),
                         Path("models/tiny/preprocessor.joblib"))
    out = ef.run_final_evaluation(result.model, pre, frames["val"], lambda: frames["test"],
                                  tmp_path, log=_quiet)
    row = ef.build_final_run_row(base, out["calibration"], out["evaluation"],
                                 out["permutation_importance"])

    assert list(row) == list(RUNS_COLUMNS)
    assert row["is_best"] is True
    assert row["calibration_method"] == "temperature"
    assert row["val_loss"] == base["val_loss"] and row["best_epoch"] == base["best_epoch"]
    assert row["test_accuracy"] == out["evaluation"]["metrics"]["accuracy"]
    assert row["ece_after"] == out["evaluation"]["calibration"]["ece_after"]  # headline = test
    assert row["calibration_bins"]["val"]["n"] == len(frames["val"])
    assert row["calibration_bins"]["test"]["n"] == len(frames["test"])
    assert row["confusion_matrix"] == out["evaluation"]["confusion_matrix"]
    assert set(row["per_class_metrics"]) == set(TARGET_CLASSES)
    assert len(row["permutation_importance"]["features"]) == 10
    assert row["checkpoint_path"] == "models/tiny/model.pt"
    json.dumps(row, allow_nan=False)


def test_already_evaluated_run_is_refused(tmp_path):
    ef.check_not_yet_evaluated(tmp_path, {"test_metrics": {}})  # fresh run: fine
    with pytest.raises(ValueError, match="already been evaluated"):
        ef.check_not_yet_evaluated(tmp_path, {"test_metrics": {"accuracy": 0.8}})
    (tmp_path / ef.EVALUATION_FILENAME).write_text("{}")
    with pytest.raises(ValueError, match="already been evaluated"):
        ef.check_not_yet_evaluated(tmp_path, {"test_metrics": {}})
