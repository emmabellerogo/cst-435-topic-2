"""Offline tests for the controlled-comparison runner.

Tiny YAML configs and hand-built rows only -- no Supabase, no real data.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

import api.run_experiments as rx
from api.run_experiments import (
    CONTROLLED_FIELDS,
    SELECTION_CRITERION,
    build_parser,
    check_controls,
    format_summary_table,
    load_experiment_configs,
    plan_runs,
    rank_runs,
)
from api.train import prepare_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "api" / "configs"
EXPERIMENT_CONFIGS = [CONFIG_DIR / f for f in ("baseline.yaml", "gelu.yaml", "deep.yaml")]

TINY = {
    "name": "a",
    "hidden_sizes": [8, 4],
    "activation": "relu",
    "dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "epochs": 3,
    "batch_size": 4,
    "seed": 0,
}


def _write(tmp_path: Path, filename: str, **overrides) -> Path:
    path = tmp_path / filename
    path.write_text(yaml.safe_dump({**TINY, **overrides}))
    return path


def _summary(name, val_loss, val_roc_auc=0.9, **extra):
    return {
        "name": name, "hidden_sizes": [64, 32], "activation": "relu", "dropout": 0.1,
        "best_epoch": 10, "val_loss": val_loss, "val_accuracy": 0.85,
        "val_precision": 0.7, "val_recall": 0.6, "val_f1": 0.65, "val_roc_auc": val_roc_auc,
        **extra,
    }


# ---------------------------------------------------------------------------
# loading configs
# ---------------------------------------------------------------------------
def test_load_experiment_configs_reads_all(tmp_path):
    paths = [_write(tmp_path, "a.yaml", name="a"),
             _write(tmp_path, "b.yaml", name="b", activation="gelu")]
    configs = load_experiment_configs(paths)
    assert [c.name for c in configs] == ["a", "b"]
    assert [c.activation for c in configs] == ["relu", "gelu"]


def test_project_experiment_configs_are_controlled():
    configs = {c.name: c for c in load_experiment_configs(EXPERIMENT_CONFIGS)}
    assert set(configs) == {"baseline", "gelu", "deep"}
    check_controls(list(configs.values()))  # must not raise

    def differing(a, b):
        da, db = asdict(configs[a]), asdict(configs[b])
        return {k for k in da if da[k] != db[k]} - {"name"}

    assert differing("baseline", "gelu") == {"activation"}
    assert differing("baseline", "deep") == {"hidden_sizes"}
    assert configs["deep"].hidden_sizes == [128, 64, 32]


def test_load_experiment_configs_needs_two(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        load_experiment_configs([_write(tmp_path, "a.yaml")])


def test_duplicate_config_names_rejected(tmp_path):
    a = _write(tmp_path, "a.yaml", name="same")
    b = _write(tmp_path, "b.yaml", name="same", activation="gelu")
    with pytest.raises(ValueError, match="'same' is used by both"):
        load_experiment_configs([a, b])
    with pytest.raises(ValueError, match="is used by both"):
        load_experiment_configs([a, a])  # the same file listed twice


def test_summary_folder_name_reserved(tmp_path):
    paths = [_write(tmp_path, "a.yaml"), _write(tmp_path, "e.yaml", name="experiments")]
    with pytest.raises(ValueError, match="reserved"):
        load_experiment_configs(paths)


@pytest.mark.parametrize("field, value", [
    ("seed", 1), ("batch_size", 8), ("epochs", 4),
    ("learning_rate", 0.02), ("weight_decay", 0.1), ("early_stopping_patience", 2),
])
def test_check_controls_rejects_mismatched_controls(tmp_path, field, value):
    assert field in CONTROLLED_FIELDS
    configs = load_experiment_configs([
        _write(tmp_path, "a.yaml", name="a"),
        _write(tmp_path, "b.yaml", name="b", **{field: value}),
    ])
    with pytest.raises(ValueError, match=field):
        check_controls(configs)


# ---------------------------------------------------------------------------
# overwrite behaviour
# ---------------------------------------------------------------------------
def _fake_saved_run(output_dir: Path, cfg, config_dict=None):
    out = output_dir / cfg.name
    out.mkdir(parents=True)
    (out / "model.pt").write_bytes(b"")
    (out / "run.json").write_text(json.dumps({"config": config_dict or asdict(cfg)}))


def test_plan_runs_never_silently_overwrites(tmp_path):
    configs = load_experiment_configs([_write(tmp_path, "a.yaml", name="a"),
                                       _write(tmp_path, "b.yaml", name="b")])
    out = tmp_path / "models"
    assert plan_runs(configs, out) == {"a": "train", "b": "train"}

    _fake_saved_run(out, configs[0])
    with pytest.raises(ValueError, match="already exists"):
        plan_runs(configs, out)
    assert plan_runs(configs, out, overwrite=True) == {"a": "train", "b": "train"}
    assert plan_runs(configs, out, reuse_existing=True) == {"a": "reuse", "b": "train"}
    with pytest.raises(ValueError, match="not both"):
        plan_runs(configs, out, overwrite=True, reuse_existing=True)


def test_plan_runs_refuses_to_reuse_a_run_from_another_config(tmp_path):
    configs = load_experiment_configs([_write(tmp_path, "a.yaml", name="a"),
                                       _write(tmp_path, "b.yaml", name="b")])
    out = tmp_path / "models"
    _fake_saved_run(out, configs[0], config_dict={**asdict(configs[0]), "dropout": 0.5})
    with pytest.raises(ValueError, match="different config"):
        plan_runs(configs, out, reuse_existing=True)


# ---------------------------------------------------------------------------
# ranking + formatting
# ---------------------------------------------------------------------------
def test_rank_runs_orders_by_validation_loss():
    ranked = rank_runs([
        _summary("mid", 0.32),
        _summary("best", 0.30, val_roc_auc=0.80),  # lower loss wins despite lower AUC
        _summary("worst", 0.35, val_roc_auc=0.99),
    ])
    assert [s["name"] for s in ranked] == ["best", "mid", "worst"]
    assert [s["rank"] for s in ranked] == [1, 2, 3]


def test_rank_runs_tie_breaks_and_missing_losses():
    ranked = rank_runs([
        _summary("nan", float("nan")),
        _summary("tie_low_auc", 0.30, val_roc_auc=0.85),
        _summary("tie_high_auc", 0.30, val_roc_auc=0.90),
        _summary("none", None),
    ])
    assert [s["name"] for s in ranked][:2] == ["tie_high_auc", "tie_low_auc"]
    assert {s["name"] for s in ranked[2:]} == {"nan", "none"}
    with pytest.raises(ValueError, match="finite validation loss"):
        rank_runs([_summary("only_nan", float("nan"))])


def test_format_summary_table():
    ranked = rank_runs([
        _summary("baseline", 0.31634, hidden_sizes=[64, 32]),
        _summary("gelu", 0.31201, activation="gelu", val_roc_auc=None),
    ])
    text = format_summary_table(ranked)
    lines = text.splitlines()
    for header in ("rank", "config", "hidden_sizes", "activation", "dropout", "best_epoch",
                   "val_loss", "val_acc", "val_prec", "val_recall", "val_f1", "val_auc"):
        assert header in lines[0]
    assert set(lines[1].replace(" ", "")) == {"-"}
    assert lines[2].startswith("1") and "gelu" in lines[2] and lines[2].endswith("<- winner")
    assert "baseline" in lines[3] and "winner" not in lines[3]
    assert "[64, 32]" in lines[3] and "0.3163" in lines[3] and "0.31634" not in text
    assert "n/a" in lines[2]  # missing AUC
    assert f"Selection criterion: {SELECTION_CRITERION}" in text
    assert "Winner: gelu (val_loss 0.3120)" in text
    assert "Test set: not evaluated." in text


# ---------------------------------------------------------------------------
# the runner never evaluates test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--evaluate-test", "--write-run"])
def test_cli_has_no_test_or_supabase_flags(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["a.yaml", "b.yaml", flag])


def _adult_frame() -> pd.DataFrame:
    rows = []
    for i, (split, age, label) in enumerate(
        [("train", 20 + 3 * k, k % 2) for k in range(12)]
        + [("val", 25 + 5 * k, k % 2) for k in range(6)]
        + [("test", 30 + 4 * k, k % 2) for k in range(3)],
        start=1,
    ):
        rows.append({
            "id": i, "split": split, "income_label": label, "age": age,
            "education_num": 9 + label * 4, "capital_gain": 0, "capital_loss": 0,
            "hours_per_week": 40, "workclass": "Private", "marital_status": "Never-married",
            "occupation": "Sales", "relationship": "Not-in-family",
            "native_country": "United-States",
        })
    return pd.DataFrame(rows)


def test_main_trains_and_summarizes_without_touching_test(tmp_path, monkeypatch):
    def prepare_without_test(df):
        data = prepare_splits(df)
        del data.X["test"], data.y["test"], data.ids["test"]  # any test access -> KeyError
        return data

    monkeypatch.setattr(rx, "load_adult_income", _adult_frame)
    monkeypatch.setattr(rx, "prepare_splits", prepare_without_test)
    paths = [_write(tmp_path, "a.yaml", name="a"),
             _write(tmp_path, "b.yaml", name="b", activation="gelu")]
    out = tmp_path / "models"

    rx.main([*map(str, paths), "--output-dir", str(out)])

    for name in ("a", "b"):
        row = json.loads((out / name / "run.json").read_text())
        assert row["test_metrics"] == {}
        assert all(row[k] is None for k in row if k.startswith("test_") and k != "test_metrics")
        assert (out / name / "model.pt").exists()
    summary = json.loads((out / "experiments" / "controlled_comparison.json").read_text())
    assert summary["test_set_evaluated"] is False
    assert summary["winner"] == summary["runs"][0]["name"]
    assert summary["controlled"]["seed"] == 0
    assert not any("test" in k for run in summary["runs"] for k in run)
    assert (out / "experiments" / "controlled_comparison.csv").exists()

    # A second run without a flag must refuse rather than overwrite.
    with pytest.raises(SystemExit, match="already exists"):
        rx.main([*map(str, paths), "--output-dir", str(out)])
