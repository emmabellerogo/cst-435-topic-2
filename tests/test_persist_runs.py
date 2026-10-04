"""Offline tests for building the controlled-experiment rows of the runs table.

Hand-built run.json files in tmp_path only -- no Supabase, no files under models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.persist_runs import (
    FINAL_EVALUATION_COLUMNS,
    RUNS_COLUMNS,
    build_controlled_rows,
    complete_run_row,
    main,
    validate_run_row,
    validation_winner,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = REPO_ROOT / "db" / "migrations" / "001_init.sql"


def _schema_runs_columns():
    """Column names of ``create table runs`` in 001_init.sql, minus generated ones."""
    body = SCHEMA.read_text().split("create table if not exists runs (", 1)[1].split("\n);", 1)[0]
    cols = []
    for line in body.splitlines():
        line = line.split("--", 1)[0].strip()
        if line:
            cols.append(line.split()[0])
    return [c for c in cols if c not in ("id", "created_at")]


def _val_row(name, val_loss=0.316, hidden_sizes=(64, 32), activation="relu"):
    """A validation-only row, shaped like api.train.build_run_row output."""
    metrics = {"loss": val_loss, "accuracy": 0.85, "precision": 0.73, "recall": 0.61,
               "f1": 0.66, "roc_auc": 0.907}
    config = {"name": name, "hidden_sizes": list(hidden_sizes), "activation": activation,
              "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 30,
              "batch_size": 256, "seed": 42, "early_stopping_patience": 5, "architecture": "mlp"}
    return {
        "name": name, "architecture": "mlp", "hidden_sizes": list(hidden_sizes),
        "activation": activation, "dropout": 0.1, "learning_rate": 0.001,
        "weight_decay": 0.0001, "epochs": 30, "best_epoch": 18, "batch_size": 256, "seed": 42,
        "config": config, "train_loss": 0.298, "val_loss": val_loss, "val_accuracy": 0.85,
        "val_roc_auc": 0.907, "test_accuracy": None, "test_precision": None,
        "test_recall": None, "test_f1": None, "test_roc_auc": None,
        "train_metrics": {**metrics, "loss": 0.298}, "val_metrics": metrics, "test_metrics": {},
        "checkpoint_path": f"models/{name}/model.pt",
        "preprocessor_path": f"models/{name}/preprocessor.joblib",
    }


def _final_row(name="gelu"):
    """The selected run after api.evaluate_final."""
    test = {"loss": 0.32, "loss_calibrated": 0.319, "accuracy": 0.85, "precision": 0.72,
            "recall": 0.6, "f1": 0.65, "roc_auc": 0.905, "threshold": 0.5, "n": 7327}
    return complete_run_row({
        **_val_row(name, val_loss=0.3156, activation="gelu"),
        "test_accuracy": 0.85, "test_precision": 0.72, "test_recall": 0.6, "test_f1": 0.65,
        "test_roc_auc": 0.905, "test_metrics": test,
        "calibration_method": "temperature", "ece_before": 0.02, "ece_after": 0.01,
        "brier_before": 0.104, "brier_after": 0.103,
        "calibration_bins": {"n_bins": 10, "val": {}, "test": {}},
        "confusion_matrix": [[5200, 360], [700, 1067]],
        "per_class_metrics": {"<=50K": {"support": 5560}, ">50K": {"support": 1767}},
        "permutation_importance": {"features": []},
        "is_best": True,
    })


def _write_runs(models_dir: Path, rows):
    for row in rows:
        d = models_dir / row["name"]
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(row))


def _write_comparison(path: Path, winner="gelu", names=("gelu", "deep", "baseline")):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"winner": winner, "test_set_evaluated": False,
                                "runs": [{"name": n} for n in names]}))
    return path


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def test_runs_columns_match_schema_file():
    assert list(RUNS_COLUMNS) == _schema_runs_columns()
    assert set(FINAL_EVALUATION_COLUMNS) <= set(RUNS_COLUMNS)


def test_validation_only_row_is_completed_with_nulls():
    row = {k: v for k, v in _val_row("baseline").items() if not k.startswith("test_")}
    full = complete_run_row(row)
    assert list(full) == list(RUNS_COLUMNS)
    assert full["is_best"] is False
    assert full["test_metrics"] == {}
    assert all(full[c] is None for c in FINAL_EVALUATION_COLUMNS)


def test_unknown_column_is_rejected():
    with pytest.raises(ValueError, match="not in runs"):
        complete_run_row({**_val_row("baseline"), "val_precision": 0.7})


def test_best_row_without_final_evaluation_is_invalid():
    row = complete_run_row(_val_row("gelu"))
    with pytest.raises(ValueError, match="final evaluation"):
        validate_run_row({**row, "is_best": True})


def test_bad_confusion_matrix_is_invalid():
    with pytest.raises(ValueError, match="TN, FP"):
        validate_run_row({**_final_row(), "confusion_matrix": [[1, 2, 3], [4, 5, 6]]})


# ---------------------------------------------------------------------------
# the controlled set
# ---------------------------------------------------------------------------
def test_only_gelu_is_marked_best(tmp_path):
    _write_runs(tmp_path, [_val_row("baseline"), _val_row("deep", hidden_sizes=(128, 64, 32)),
                           _final_row("gelu")])
    rows = build_controlled_rows(tmp_path, ["baseline", "gelu", "deep"], best="gelu")

    assert [r["name"] for r in rows] == ["baseline", "deep", "gelu"]  # best inserted last
    assert {r["name"]: r["is_best"] for r in rows} == {"baseline": False, "deep": False, "gelu": True}
    for r in rows:
        assert list(r) == list(RUNS_COLUMNS)
        if r["name"] != "gelu":
            assert r["test_metrics"] == {} and r["test_accuracy"] is None
            assert r["calibration_method"] is None and r["permutation_importance"] is None
    json.dumps(rows, allow_nan=False)


def test_non_selected_run_with_test_results_is_refused(tmp_path):
    deep = {**_val_row("deep"), "test_accuracy": 0.85}
    _write_runs(tmp_path, [_val_row("baseline"), deep, _final_row("gelu")])
    with pytest.raises(ValueError, match="only the selected run"):
        build_controlled_rows(tmp_path, ["baseline", "gelu", "deep"], best="gelu")


def test_second_best_row_is_refused(tmp_path):
    _write_runs(tmp_path, [_final_row("baseline"), _final_row("gelu")])
    with pytest.raises(ValueError, match="is_best"):
        build_controlled_rows(tmp_path, ["baseline", "gelu"], best="gelu")


def test_best_run_must_be_final_evaluated(tmp_path):
    _write_runs(tmp_path, [_val_row("baseline"), _val_row("gelu")])
    with pytest.raises(ValueError, match="api.evaluate_final"):
        build_controlled_rows(tmp_path, ["baseline", "gelu"], best="gelu")


def test_best_must_be_one_of_the_runs(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        build_controlled_rows(tmp_path, ["baseline", "deep"], best="gelu")


def test_validation_winner_read_from_comparison(tmp_path):
    path = _write_comparison(tmp_path / "experiments" / "controlled_comparison.json")
    assert validation_winner(path) == "gelu"
    path.write_text(json.dumps({"winner": "gelu", "test_set_evaluated": True, "runs": []}))
    with pytest.raises(ValueError, match="validation-only"):
        validation_winner(path)


def test_cli_dry_run_writes_rows_without_supabase(tmp_path, monkeypatch):
    import api.db

    def no_supabase(*_a, **_k):
        raise AssertionError("dry run must not touch Supabase")

    monkeypatch.setattr(api.db, "insert_training_run", no_supabase)
    monkeypatch.setattr(api.db, "fetch_run_names", no_supabase)

    models = tmp_path / "models"
    _write_runs(models, [_val_row("baseline"), _val_row("deep"), _final_row("gelu")])
    comparison = _write_comparison(models / "experiments" / "controlled_comparison.json")
    main(["--comparison", str(comparison), "--models-dir", str(models)])

    rows = json.loads((comparison.parent / "runs_rows.json").read_text())
    assert [r["name"] for r in rows] == ["deep", "baseline", "gelu"]
    assert [r["is_best"] for r in rows] == [False, False, True]
