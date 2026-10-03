"""Offline tests for the config-driven training pipeline.

Tiny synthetic tensors and hand-built rows only -- no Supabase, no real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from api.model import MLP, load_checkpoint, make_activation, predict_proba, save_checkpoint
from api.train import (
    BestCheckpoint,
    TrainConfig,
    build_run_row,
    evaluate,
    load_config,
    prepare_splits,
    set_seed,
    train_model,
)
from shared.features import NUMERIC_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_CONFIG = {
    "name": "tiny",
    "hidden_sizes": [16, 8],
    "activation": "relu",
    "dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "epochs": 15,
    "batch_size": 32,
    "seed": 0,
}


def _cfg(**overrides) -> TrainConfig:
    return TrainConfig(**{**VALID_CONFIG, **overrides})


def _tiny_data(seed: int = 0):
    """300 rows, 6 features, label = (x0 + x1 > 0); 200 train / 100 val."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X[:200], y[:200], X[200:], y[200:]


def _quiet(_msg: str) -> None:
    pass


def _n_hidden_layers(model: nn.Module) -> int:
    return sum(isinstance(m, nn.Linear) for m in model.modules()) - 1  # minus output layer


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def test_mlp_has_at_least_two_hidden_layers():
    assert _n_hidden_layers(MLP(10, [16, 8])) == 2
    assert _n_hidden_layers(MLP(10, [32, 16, 8])) == 3
    with pytest.raises(ValueError, match="at least 2 hidden layers"):
        MLP(10, [16])
    assert MLP(10, [16, 8])(torch.zeros(4, 10)).shape == (4,)  # one logit per row


@pytest.mark.parametrize("name, cls", [("relu", nn.ReLU), ("gelu", nn.GELU), ("GELU", nn.GELU)])
def test_activation_selection(name, cls):
    assert isinstance(make_activation(name), cls)
    model = MLP(10, [8, 8], activation=name)
    assert sum(isinstance(m, cls) for m in model.modules()) == 2  # one per hidden layer


def test_unknown_activation_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        make_activation("swishy")


def test_seed_makes_initialization_deterministic():
    def build(seed):
        set_seed(seed)
        return MLP(10, [16, 8], "relu", 0.1)

    a, b, c = build(0), build(0), build(1)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    model = MLP(6, [8, 4], "gelu", 0.2)
    path = save_checkpoint(tmp_path / "run" / "model.pt", model, config={"name": "x"})
    loaded, meta = load_checkpoint(path)
    X = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
    np.testing.assert_allclose(predict_proba(model, X), predict_proba(loaded, X))
    assert meta["activation"] == "gelu"
    assert meta["hidden_sizes"] == [8, 4]
    assert meta["config"] == {"name": "x"}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({
        **VALID_CONFIG, "hidden_sizes": [128, 64], "activation": "GELU",
        "learning_rate": "1e-3", "early_stopping_patience": 3,
    }))
    cfg = load_config(path)
    assert cfg.hidden_sizes == [128, 64]
    assert cfg.activation == "gelu"
    assert cfg.learning_rate == pytest.approx(0.001)
    assert cfg.early_stopping_patience == 3
    assert cfg.architecture == "mlp"


def test_baseline_config_is_valid():
    cfg = load_config(REPO_ROOT / "api" / "configs" / "baseline.yaml")
    assert len(cfg.hidden_sizes) >= 2


@pytest.mark.parametrize("change", [
    {"hiden_sizes": [16, 8]},          # typo -> unknown key
    {"learning_rate": None},           # removed below -> missing key
    {"hidden_sizes": [16]},            # only one hidden layer
    {"activation": "swishy"},
    {"dropout": 1.0},
    {"learning_rate": 0},
    {"name": "../escape"},
])
def test_load_config_rejects_bad_configs(tmp_path, change):
    raw = {**VALID_CONFIG, **change}
    raw = {k: v for k, v in raw.items() if v is not None}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


# ---------------------------------------------------------------------------
# training loop + best checkpoint
# ---------------------------------------------------------------------------
def test_training_runs_on_tiny_dataset():
    result = train_model(_cfg(), *_tiny_data(), log=_quiet)
    h = result.history
    assert 1 <= len(h) <= 15
    for key in ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_roc_auc"):
        assert key in h[0]
    assert h[-1]["train_loss"] < h[0]["train_loss"]
    assert result.best_val_metrics["accuracy"] > 0.8


def test_training_is_reproducible():
    h1 = train_model(_cfg(epochs=5, dropout=0.2), *_tiny_data(), log=_quiet).history
    h2 = train_model(_cfg(epochs=5, dropout=0.2), *_tiny_data(), log=_quiet).history
    assert h1 == h2


def test_best_checkpoint_keeps_lowest_val_loss_weights():
    model = MLP(4, [4, 4])
    tracker = BestCheckpoint(patience=2)
    for epoch, val_loss in enumerate([1.0, 0.8, 0.9, 0.85], start=1):
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float(epoch))  # weights "are" the epoch number
        tracker.update(epoch, {"loss": val_loss}, model)
        if epoch < 4:
            assert not tracker.should_stop
    assert tracker.best_epoch == 2
    assert tracker.best_loss == 0.8
    # The saved state is a copy of epoch 2's weights, not the live (epoch 4) ones.
    assert all(torch.all(v == 2.0) for v in tracker.best_state.values())
    assert tracker.should_stop  # epochs 3 and 4 did not improve; patience = 2


def test_best_checkpoint_without_patience_never_stops():
    tracker = BestCheckpoint(patience=None)
    model = MLP(4, [4, 4])
    for epoch in range(1, 11):
        tracker.update(epoch, {"loss": float(epoch)}, model)  # gets worse every epoch
    assert tracker.best_epoch == 1
    assert not tracker.should_stop


def test_train_model_returns_best_epoch_weights():
    X_tr, y_tr, X_val, y_val = _tiny_data()
    result = train_model(_cfg(epochs=10), X_tr, y_tr, X_val, y_val, log=_quiet)
    val_losses = [h["val_loss"] for h in result.history]
    assert result.best_epoch == int(np.argmin(val_losses)) + 1
    assert evaluate(result.model, X_val, y_val)["loss"] == pytest.approx(min(val_losses))


# ---------------------------------------------------------------------------
# splits / preprocessing
# ---------------------------------------------------------------------------
def _adult_row(id_, split, age, occupation="Sales", label=0):
    return {
        "id": id_, "split": split, "income_label": label, "age": age,
        "education_num": 10, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
        "workclass": "Private", "marital_status": "Never-married", "occupation": occupation,
        "relationship": "Not-in-family", "native_country": "United-States",
    }


def _adult_frame():
    return pd.DataFrame([
        _adult_row(1, "train", 20),
        _adult_row(2, "train", 30, label=1),
        _adult_row(3, "train", 40),
        _adult_row(4, "val", 90, label=1),                          # extreme val value
        _adult_row(5, "test", 17, occupation="Armed-Forces"),       # test-only category
    ])


def test_prepare_splits_fits_preprocessor_on_train_rows_only():
    data = prepare_splits(_adult_frame())
    num = data.preprocessor.named_steps["columns"].named_transformers_["num"]
    assert num.named_steps["scale"].mean_[NUMERIC_COLS.index("age")] == pytest.approx(30.0)
    assert "cat__occupation_Armed-Forces" not in data.preprocessor.get_feature_names_out()
    assert {s: len(data.y[s]) for s in ("train", "val", "test")} == {"train": 3, "val": 1, "test": 1}
    np.testing.assert_array_equal(data.y["train"], [0, 1, 0])
    np.testing.assert_array_equal(data.ids["test"], [5])
    assert data.X["train"].shape[1] == data.X["test"].shape[1]


def test_prepare_splits_rejects_bad_split_columns():
    df = _adult_frame()
    with pytest.raises(ValueError, match="no rows"):
        prepare_splits(df[df["split"] != "test"])
    with pytest.raises(ValueError, match="no split"):
        prepare_splits(df.assign(split=[None, "train", "train", "val", "test"]))
    with pytest.raises(ValueError, match="unexpected split"):
        prepare_splits(df.assign(split=["train", "train", "train", "val", "holdout"]))


# ---------------------------------------------------------------------------
# runs row
# ---------------------------------------------------------------------------
RUNS_COLUMNS = {
    "name", "architecture", "hidden_sizes", "activation", "dropout", "learning_rate",
    "weight_decay", "epochs", "best_epoch", "batch_size", "seed", "config",
    "train_loss", "val_loss", "val_accuracy", "val_roc_auc",
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc",
    "train_metrics", "val_metrics", "test_metrics", "checkpoint_path", "preprocessor_path",
}


def test_build_run_row_matches_runs_table():
    cfg = _cfg(epochs=3)
    X_tr, y_tr, X_val, y_val = _tiny_data()
    result = train_model(cfg, X_tr, y_tr, X_val, y_val, log=_quiet)
    ckpt, pre = Path("models/tiny/model.pt"), Path("models/tiny/preprocessor.joblib")

    row = build_run_row(cfg, result, ckpt, pre)
    assert set(row) == RUNS_COLUMNS
    assert row["best_epoch"] == result.best_epoch
    assert row["val_loss"] == result.best_val_metrics["loss"]
    assert row["test_accuracy"] is None and row["test_metrics"] == {}  # test untouched
    assert row["checkpoint_path"] == "models/tiny/model.pt"
    json.dumps(row, allow_nan=False)  # must be valid JSON for Supabase

    test_m = evaluate(result.model, X_val, y_val)  # stand-in for a test split
    row = build_run_row(cfg, result, ckpt, pre, test_metrics=test_m)
    assert row["test_accuracy"] == test_m["accuracy"]
    assert row["test_metrics"] == test_m
