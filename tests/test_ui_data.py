"""Offline tests for the UI's data helpers: CSV merge/template and the
evaluation-results loader (Supabase rows or committed files)."""
from __future__ import annotations

import base64
import io
import json

import pandas as pd
import pytest

from tests.ui_fakes import CONFUSION, SCHEMA, batch_response, runs_rows, write_model_files

import perf_data  # noqa: E402
import tab_concepts  # noqa: E402
import tab_score_csv  # noqa: E402
import tab_score_row  # noqa: E402

FEATURE_NAMES = [f["name"] for f in SCHEMA["features"]]


# -- Score CSV -----------------------------------------------------------------
def test_template_csv_has_exactly_the_schema_columns():
    df = pd.read_csv(io.BytesIO(tab_score_csv.template_csv(SCHEMA)))
    assert list(df.columns) == FEATURE_NAMES
    assert "sex" not in df.columns and "race" not in df.columns
    assert len(df) == 1
    assert df.loc[0, "workclass"] != "Unknown"


def test_merge_predictions_appends_in_row_order():
    df = pd.DataFrame({"age": ["30", "40", "50"], "sex": ["Male", "Female", "Male"]})
    resp = batch_response(3)
    resp["predictions"] = list(reversed(resp["predictions"]))  # order must not matter
    out = tab_score_csv.merge_predictions(df, resp)
    assert list(out["age"]) == ["30", "40", "50"]
    assert list(out["predicted_income"]) == [">50K", "<=50K", ">50K"]
    assert list(out["probability_over_50k"]) == [0.9, 0.1, 0.9]
    assert list(out["predicted_label"]) == [1, 0, 1]


def test_merge_predictions_rejects_count_mismatch():
    with pytest.raises(ValueError):
        tab_score_csv.merge_predictions(pd.DataFrame({"age": ["30"]}), batch_response(2))


def test_read_preview_keeps_blank_cells_as_text():
    df = tab_score_csv.read_preview(b" age ,workclass\n30,\n")
    assert list(df.columns) == ["age", "workclass"]
    assert df.loc[0, "workclass"] == ""


# -- Score a Row -----------------------------------------------------------------
def test_form_fields_are_the_ten_schema_features_without_protected_attributes():
    schema = {**SCHEMA, "features": SCHEMA["features"] + [
        {"name": "sex", "kind": "categorical", "categories": ["Female", "Male"]}]}
    names = [f["name"] for f in tab_score_row.input_fields(schema)]
    assert names == FEATURE_NAMES
    assert len(names) == 10


def test_example_profile_only_uses_values_the_schema_allows():
    fields = {f["name"]: f for f in SCHEMA["features"]}
    assert tab_score_row.example_value(fields["workclass"]) == "Private"
    narrow = {**fields["age"], "maximum": 40}
    assert tab_score_row.example_value(narrow) == narrow["default"]  # 45 is out of range
    odd = {**fields["occupation"], "categories": ["Sales"]}
    assert tab_score_row.example_value(odd) == "Sales"


# -- Concepts ------------------------------------------------------------------
def test_hand_built_xor_network_is_exact():
    out = tab_concepts.hand_built_xor()["out"][:, 0]
    assert list(out) == [0.0, 1.0, 1.0, 0.0]


def test_gelu_matches_known_values():
    assert tab_concepts.gelu(0.0) == pytest.approx(0.0)
    assert tab_concepts.gelu(1.0) == pytest.approx(0.8413447, abs=1e-6)
    assert float(tab_concepts.gelu(-0.75)) == pytest.approx(-0.1700, abs=1e-3)


def test_layer_shapes_for_the_served_model():
    shapes = tab_concepts.layer_shapes(84, [64, 32])
    assert list(shapes["weight W"]) == ["84 × 64", "64 × 32", "32 × 1"]
    assert int(shapes["parameters"].sum()) == 84 * 64 + 64 + 64 * 32 + 32 + 32 + 1


def test_xor_training_reduces_loss():
    losses, probs = tab_concepts.train_xor("GELU", hidden=4, lr=0.5, epochs=3000, seed=0)
    assert losses[-1] < losses[0]
    assert probs.shape == (4, 1)


# -- evaluation results --------------------------------------------------------
def test_bundle_from_files(tmp_path):
    b = perf_data.from_model_files(write_model_files(tmp_path))
    assert b["source"] == "files"
    assert b["selected"]["name"] == "gelu"
    assert b["confusion_matrix"] == CONFUSION
    assert b["test"]["accuracy"] == 0.8540
    assert [e["name"] for e in b["experiments"]] == ["gelu", "deep", "baseline"]
    assert [e["selected"] for e in b["experiments"]] == [True, False, False]
    assert b["importance"][0]["feature"] == "marital_status"
    assert b["calibration"]["temperature"] == 1.0238


def test_bundle_from_supabase_rows_matches_files(tmp_path):
    a = perf_data.from_runs_rows(runs_rows())
    b = perf_data.from_model_files(write_model_files(tmp_path))
    assert a["source"] == "supabase" and a["selected"]["run_id"] == 3
    for key in ("confusion_matrix", "per_class", "importance"):
        assert a[key] == b[key]
    assert [e["name"] for e in a["experiments"]] == [e["name"] for e in b["experiments"]]
    assert a["calibration"]["test"]["ece_after"] == b["calibration"]["test"]["ece_after"]


def test_runs_rows_need_exactly_one_best():
    rows = runs_rows()
    for r in rows:
        r["is_best"] = True
    with pytest.raises(perf_data.PerformanceDataError):
        perf_data.from_runs_rows(rows)


def test_supabase_failure_falls_back_to_files(tmp_path, monkeypatch):
    def fail(url, key):
        raise RuntimeError("permission denied for table runs")
    monkeypatch.setattr(perf_data, "fetch_runs_rows", fail)
    b = perf_data.load_performance("https://x.supabase.co", "anon", str(write_model_files(tmp_path)))
    assert b["source"] == "files"
    assert any("permission denied" in w for w in b["warnings"])


def test_no_source_raises_clear_error(tmp_path):
    with pytest.raises(perf_data.PerformanceDataError) as exc:
        perf_data.load_performance(None, None, str(tmp_path))
    assert "not configured" in str(exc.value)


# -- learning curves and split comparison (real committed files) ----------------
REAL_MODELS = perf_data.REPO_ROOT / "models"


def test_history_for_the_selected_run_matches_its_stored_best_epoch():
    run = json.loads((REAL_MODELS / "gelu" / "run.json").read_text())
    h = perf_data.load_history(REAL_MODELS, "gelu", run["best_epoch"], run["val_loss"])
    assert h["available"], h["detail"]
    assert [e["epoch"] for e in h["epochs"]] == list(range(1, len(h["epochs"]) + 1))
    best = h["epochs"][run["best_epoch"] - 1]
    assert best["val_loss"] == run["val_loss"] and best["train_loss"] == run["train_loss"]
    assert all(e["train_accuracy"] is not None and e["val_accuracy"] is not None for e in h["epochs"])


def test_history_is_reported_unavailable_not_invented(tmp_path):
    missing = perf_data.load_history(tmp_path, "gelu", 18, 0.3156)
    assert not missing["available"] and missing["epochs"] == []
    assert "no per-epoch history" in missing["detail"]

    (tmp_path / "gelu").mkdir()
    (tmp_path / "gelu" / "history.json").write_text(json.dumps(
        [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.5, "train_accuracy": 0.8, "val_accuracy": 0.79}]))
    stale = perf_data.load_history(tmp_path, "gelu", 1, 0.3156)  # val loss disagrees
    assert not stale["available"] and "does not match" in stale["detail"]

    (tmp_path / "gelu" / "history.json").write_text(json.dumps([{"epoch": 1, "train_loss": 0.4}]))
    partial = perf_data.load_history(tmp_path, "gelu", 1, None)
    assert not partial["available"]


def test_split_metrics_are_the_stored_train_val_test_values():
    run = json.loads((REAL_MODELS / "gelu" / "run.json").read_text())
    b = perf_data.load_performance(None, None, str(REAL_MODELS))
    assert b["selected"]["name"] == "gelu" and b["history"]["available"]
    for split in ("train", "val", "test"):
        stored = run[f"{split}_metrics"]
        for k in perf_data.SPLIT_METRIC_KEYS:
            assert b["splits"][split][k] == stored[k]
    assert b["splits"]["test"]["loss_calibrated"] == run["test_metrics"]["loss_calibrated"]
    # the comparison does not disturb the existing evaluation results
    assert b["test"]["accuracy"] == run["test_accuracy"]
    assert b["calibration"]["temperature"] == pytest.approx(1.0238, abs=1e-4)


def test_split_metrics_from_supabase_rows():
    rows = runs_rows()
    sel = next(r for r in rows if r["is_best"])
    sel["train_metrics"] = {"loss": 0.29, "accuracy": 0.86}
    b = perf_data.from_runs_rows(rows)
    assert b["splits"]["train"]["accuracy"] == 0.86
    assert b["splits"]["val"]["loss"] == sel["val_metrics"]["loss"]
    assert b["splits"]["test"]["accuracy"] == sel["test_metrics"]["accuracy"]
    assert "train_metrics" in perf_data.RUN_COLUMNS.split(",")


def test_split_comparison_frame_has_one_column_per_split():
    import tab_performance

    df = tab_performance.split_comparison_frame(
        {"train": {"accuracy": 0.86}, "val": {"accuracy": 0.85}, "test": {}})
    assert list(df.columns) == ["metric", "train", "validation", "test"]
    acc = df[df["metric"] == "accuracy"].iloc[0]
    assert (acc["train"], acc["validation"]) == (0.86, 0.85) and acc["test"] is None


# -- Bias Audit: direct anon read --------------------------------------------------
def _jwt(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{body}.signature"


def test_ui_refuses_non_public_supabase_keys():
    perf_data.check_public_key(_jwt({"role": "anon", "iss": "supabase"}))
    perf_data.check_public_key("sb_publishable_abc")
    for bad in (_jwt({"role": "service_role"}), "sb_secret_abc"):
        with pytest.raises(perf_data.UnsafeKeyError):
            perf_data.check_public_key(bad)
        with pytest.raises(perf_data.UnsafeKeyError):  # refused before any network call
            perf_data.fetch_fairness_view("https://x.supabase.co", bad, 3)


def test_direct_frame_copies_rates_and_derives_counts():
    import tab_bias_audit
    from tests.ui_fakes import AUDIT

    df = tab_bias_audit.direct_frame(AUDIT["groups"])
    female = df[df["group"] == "Female"].iloc[0]
    assert female["labeled test rows"] == 2429
    assert female["actually >50K"] == 155 + 110
    assert female["predicted >50K"] == 155 + 58
    assert female["FPR (view)"] == 0.0268 and female["FNR (view)"] == 0.4151  # copied as is


def test_compare_sources_reports_differences():
    import copy

    import tab_bias_audit
    from tests.ui_fakes import AUDIT

    assert tab_bias_audit.compare_sources(AUDIT["groups"], AUDIT["groups"]) == []
    other = copy.deepcopy(AUDIT["groups"])
    other[0]["fnr"] = 0.5
    diffs = tab_bias_audit.compare_sources(AUDIT["groups"], other)
    assert diffs == ["Female fnr: /audit 0.4151 vs direct 0.5"]


# -- Concepts: clean tables and gradient shapes ---------------------------------
def test_xor_table_holds_plain_python_ints():
    df = tab_concepts.xor_table(tab_concepts.hand_built_xor())
    for col in df.columns:
        assert all(type(v) is int for v in df[col].tolist()), col
    assert "np." not in df.to_string()
    assert df["output h1 − 2·h2"].tolist() == df["XOR target"].tolist() == [0, 1, 1, 0]


def test_gradient_shapes_match_weight_shapes():
    grads = tab_concepts.gradient_shapes(84, [64, 32])
    weights = tab_concepts.layer_shapes(84, [64, 32])
    assert list(grads["layer"]) == list(reversed(weights["layer"]))  # output first
    for g, w in zip(grads["weight gradient"], reversed(list(weights["weight W"]))):
        assert g.endswith(f"= {w}")
    assert grads["error signal δ"].iloc[0] == "δ3 = ∂L/∂z: B × 1"


# -- Score CSV: limits and config --------------------------------------------------
def test_csv_limits_match_the_api():
    from api import main

    assert tab_score_csv.MAX_UPLOAD_BYTES == main.MAX_UPLOAD_BYTES
    assert tab_score_csv.MAX_ROWS == main.MAX_BATCH_ROWS
    assert tab_score_csv.limit_problem(main.MAX_UPLOAD_BYTES, main.MAX_BATCH_ROWS) is None
    assert "MB" in tab_score_csv.limit_problem(main.MAX_UPLOAD_BYTES + 1, 1)
    assert "rows" in tab_score_csv.limit_problem(100, main.MAX_BATCH_ROWS + 1)


def test_streamlit_upload_limit_is_5_mb_wherever_the_app_starts():
    import tomllib

    root = perf_data.REPO_ROOT
    copies = [root / ".streamlit" / "config.toml", root / "ui" / ".streamlit" / "config.toml"]
    assert copies[0].read_text() == copies[1].read_text()
    for path in copies:
        assert tomllib.loads(path.read_text())["server"]["maxUploadSize"] == tab_score_csv.MAX_UPLOAD_MB


def test_template_csv_is_the_valid_example_profile():
    df = pd.read_csv(io.BytesIO(tab_score_csv.template_csv(SCHEMA)), dtype=str)
    row = df.iloc[0].to_dict()
    expected = {k: str(v) for k, v in tab_score_row.EXAMPLE_PROFILE.items()}
    assert row == expected


def test_experiment_chart_clips_bars_to_the_zoomed_axis():
    import tab_performance
    from tests.ui_fakes import EXPERIMENTS

    spec = tab_performance.experiments_chart(
        [{**e, "selected": e["name"] == "gelu"} for e in EXPERIMENTS]).to_dict()
    lo, hi = spec["encoding"]["y"]["scale"]["domain"]
    assert 0 < lo < min(e["val_loss"] for e in EXPERIMENTS) <= max(e["val_loss"] for e in EXPERIMENTS) < hi
    # unclipped bars extend to 0, outside the domain, and collapse the plot
    assert spec["mark"]["clip"] is True
