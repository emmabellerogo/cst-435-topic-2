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
