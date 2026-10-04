"""End-to-end UI tests with Streamlit's AppTest, fully offline.

The FastAPI service is replaced by tests.ui_fakes.FakeApi (patched over
requests.request) and the evaluation results come from fixture files, so these
tests need no network, no Render and no Supabase.
"""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.ui_fakes import API_URL, UI_DIR, FakeApi, FakeResponse, write_model_files

import api_client  # noqa: E402

APP = str(UI_DIR / "app.py")
EXPECTED_TABS = ["Concepts", "Score a Row", "Score CSV", "Model Performance", "Bias Audit", "Model Card"]


@pytest.fixture
def run_app(monkeypatch, tmp_path):
    """Return a function that runs the app against a FakeApi; returns (app, api)."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.setenv("INCOME_INSIGHT_MODELS_DIR", str(write_model_files(tmp_path / "models")))

    def go(routes=None):
        st.cache_data.clear()
        api = FakeApi(routes)
        monkeypatch.setattr(api_client.requests, "request", api)
        at = AppTest.from_file(APP, default_timeout=60)
        at.secrets["API_URL"] = API_URL
        at.run()
        return at, api
    return go


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_app_has_the_six_required_tabs(run_app):
    at, _ = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == EXPECTED_TABS


def test_score_row_form_has_ten_inputs_and_no_protected_attributes(run_app):
    at, _ = run_app()
    labels = [w.label for w in at.number_input if w.key and w.key.startswith("row_")]
    labels += [w.label for w in at.selectbox if w.key and w.key.startswith("row_")]
    assert len(labels) == 10
    assert not any("sex" in lbl.lower() or "race" in lbl.lower() for lbl in labels)


def test_predict_button_calls_api_and_shows_result(run_app):
    at, api = run_app()
    at.button(key="predict_row").click().run()
    assert not at.exception
    sent = [c for c in api.calls if c["path"] == "/predict"]
    assert len(sent) == 1
    features = sent[0]["json"]["features"]
    assert set(features) == {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week",
                             "workclass", "marital_status", "occupation", "relationship",
                             "native_country"}
    assert features["workclass"] == "Private" and features["age"] == 45  # the example profile
    values = {m.label: m.value for m in at.metric}
    assert values["Predicted income"] == ">50K"
    assert values["Probability of >50K"] == "83.0%"


def test_predict_validation_error_is_shown_not_raised(run_app):
    detail = {"message": "1 invalid feature value(s); nothing was scored or logged", "n_errors": 1,
              "errors": [{"row": 1, "column": "age", "value": 200, "error": "must be between 17 and 90"}]}
    at, _ = run_app({("POST", "/predict"): lambda **kw: FakeResponse(422, {"detail": detail})})
    at.button(key="predict_row").click().run()
    assert not at.exception
    assert "nothing was scored or logged" in _all_text(at)


def test_bias_audit_shows_the_api_values_unchanged(run_app):
    at, _ = run_app()
    assert not at.exception
    values = [m.value for m in at.metric if m.label.startswith(("False-positive", "False-negative"))]
    assert values == ["0.0268", "0.4151", "0.0985", "0.3804"]


def test_model_performance_reads_the_stored_results(run_app):
    at, _ = run_app()
    values = {m.label: m.value for m in at.metric}
    assert values["Accuracy"] == "0.8540"
    assert values["ROC-AUC"] == "0.9060"
    assert values["Recall (>50K)"] == "0.6144"
    text = _all_text(at)
    assert "FN = 676" in text
    assert "validation loss" in text


def test_app_survives_when_the_api_is_down(run_app):
    import requests

    def down(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    at, _ = run_app({key: down for key in [("GET", "/healthz"), ("GET", "/version"),
                                            ("GET", "/schema"), ("GET", "/audit")]})
    assert not at.exception
    assert [t.label for t in at.tabs] == EXPECTED_TABS
    assert "Could not reach the API" in _all_text(at)


# -- Score a Row: buttons and logging status ------------------------------------
def _form_value(at, name):
    for w in list(at.number_input) + list(at.selectbox):
        if w.key == f"row_{name}":
            return w.value
    raise KeyError(name)


def test_reset_and_example_buttons_refill_the_form(run_app):
    at, _ = run_app()
    at.button(key="row_reset").click().run()
    assert not at.exception
    assert _form_value(at, "age") == 37 and _form_value(at, "hours_per_week") == 40  # medians
    assert _form_value(at, "workclass") == "Federal-gov"  # first known category
    at.button(key="row_example").click().run()
    assert _form_value(at, "age") == 45 and _form_value(at, "workclass") == "Private"
    assert _form_value(at, "relationship") == "Husband"


@pytest.mark.parametrize("logged, expected", [(True, "Logged to Supabase as request ffffffffffff"),
                                               (False, "Not logged.")])
def test_prediction_shows_class_probability_and_logging_status(run_app, logged, expected):
    from tests.ui_fakes import PREDICT

    body = {**PREDICT, "proba": 0.815338791378, "logged": logged}
    at, _ = run_app({("POST", "/predict"): lambda **kw: FakeResponse(200, body)})
    at.button(key="predict_row").click().run()
    values = {m.label: m.value for m in at.metric}
    assert values["Predicted income"] == ">50K"
    assert values["Probability of >50K"] == "81.5%"
    assert expected in _all_text(at)


# -- Model Performance: curves and split comparison ------------------------------
def test_performance_says_history_is_unavailable_instead_of_drawing_curves(run_app):
    at, _ = run_app()  # the fixture files have no history.json
    text = _all_text(at)
    assert "Per-epoch history is unavailable" in text
    assert "Learning curves" in [h.value.split(" (")[0] for h in at.subheader]


def test_performance_shows_curves_and_split_table_from_the_real_files(run_app, monkeypatch):
    from tests.ui_fakes import REPO_ROOT

    monkeypatch.setenv("INCOME_INSIGHT_MODELS_DIR", str(REPO_ROOT / "models"))
    at, _ = run_app()
    assert not at.exception
    text = _all_text(at)
    assert "Per-epoch history is unavailable" not in text
    assert "models/gelu/history.json" in text
    subheaders = [h.value for h in at.subheader]
    assert "Train / validation / test comparison (gelu)" in subheaders
    split_table = next(d.value for d in at.dataframe
                       if list(d.value.columns) == ["metric", "train", "validation", "test"])
    assert split_table["test"].notna().all()
    # the existing evaluation sections are still there
    for name in ("Confusion matrix", "Per-class metrics", "Calibration",
                 "Experiment comparison (validation set)", "Permutation importance (test set)"):
        assert name in subheaders


# -- Bias Audit: direct anon read ------------------------------------------------
def test_bias_audit_direct_read_not_configured_is_explained(run_app):
    at, _ = run_app()
    assert "direct read was skipped" in _all_text(at)
    assert "Possible mitigations (not implemented)" in [h.value for h in at.subheader]


def test_bias_audit_direct_read_matches_audit(monkeypatch, tmp_path):
    import perf_data
    from tests.ui_fakes import AUDIT, runs_rows

    calls = []

    def fake_view(url, key, run_id):
        calls.append((url, key, run_id))
        return [dict(g, run_id=run_id) for g in AUDIT["groups"]]

    monkeypatch.setattr(perf_data, "fetch_fairness_view", fake_view)
    monkeypatch.setattr(perf_data, "fetch_runs_rows", lambda url, key: runs_rows())
    monkeypatch.setenv("INCOME_INSIGHT_MODELS_DIR", str(write_model_files(tmp_path / "models")))
    st.cache_data.clear()
    monkeypatch.setattr(api_client.requests, "request", FakeApi())
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["API_URL"] = API_URL
    at.secrets["SUPABASE_URL"] = "https://example.supabase.co"
    at.secrets["SUPABASE_ANON_KEY"] = "sb_publishable_test"
    at.run()
    assert not at.exception
    assert calls == [("https://example.supabase.co", "sb_publishable_test", 3)]
    assert any("matches the `/audit` numbers exactly" in s.value for s in at.success)
    direct = next(d.value for d in at.dataframe if "labeled test rows" in d.value.columns)
    assert list(direct["labeled test rows"]) == [2429, 4898]
    # the SQL values from /audit are still shown unchanged
    values = [m.value for m in at.metric if m.label.startswith(("False-positive", "False-negative"))]
    assert values == ["0.0268", "0.4151", "0.0985", "0.3804"]


# -- Concepts --------------------------------------------------------------------
def test_concepts_separates_training_and_inference(run_app):
    at, _ = run_app()
    latex = [l.value.strip("$\n") for l in at.latex]
    assert r"q = \sigma(z)" in latex
    assert r"p = \sigma\!\left(\frac{z}{T}\right)" in latex
    assert any(r"\frac{1}{B}(q - y)" in l for l in latex)
    assert not any("(p - y)" in l for l in latex)
    xor = next(d.value for d in at.dataframe if "XOR target" in d.value.columns)
    assert "np." not in xor.to_string()
