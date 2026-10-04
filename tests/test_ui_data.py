"""Offline tests for the UI's data helpers: CSV merge/template and the
evaluation-results loader (Supabase rows or committed files)."""
from __future__ import annotations

import io

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
