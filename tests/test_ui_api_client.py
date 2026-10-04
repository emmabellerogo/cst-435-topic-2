"""Offline tests for ui/api_client.py: the UI's side of the FastAPI contract."""
from __future__ import annotations

import pytest
import requests

from tests.ui_fakes import API_URL, PREDICT, FakeApi, FakeResponse

import api_client  # noqa: E402  (importable once tests.ui_fakes put ui/ on sys.path)
from api_client import ApiClient, ApiError  # noqa: E402

FEATURES = {"age": 45, "education_num": 13, "capital_gain": 0, "capital_loss": 0,
            "hours_per_week": 45, "workclass": "Private", "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial", "relationship": "Husband",
            "native_country": "United-States"}


@pytest.fixture
def fake(monkeypatch):
    def install(routes=None):
        api = FakeApi(routes)
        monkeypatch.setattr(api_client.requests, "request", api)
        return api
    return install


def test_predict_sends_contract_payload_without_run_id(fake):
    api = fake()
    resp = ApiClient(API_URL).predict(FEATURES)
    assert resp == PREDICT
    call = api.calls[-1]
    assert (call["method"], call["path"]) == ("POST", "/predict")
    assert call["json"] == {"features": FEATURES}  # run_id is optional and left out
    assert "sex" not in call["json"]["features"] and "race" not in call["json"]["features"]


def test_predict_batch_uses_multipart_field_named_file(fake):
    api = fake({("POST", "/predict_batch"): lambda **kw: FakeResponse(200, {"n_rows": 0})})
    ApiClient(API_URL).predict_batch("people.csv", b"age\n30\n")
    files = api.calls[-1]["files"]
    assert list(files) == ["file"]
    name, content, mime = files["file"]
    assert (name, content, mime) == ("people.csv", b"age\n30\n", "text/csv")


def test_row_errors_from_422_are_exposed(fake):
    detail = {"message": "2 invalid feature value(s); nothing was scored or logged", "n_errors": 2,
              "errors": [{"row": 1, "column": "age", "value": 200, "error": "must be between 17 and 90"},
                         {"row": 2, "column": "workclass", "value": "Pirate", "error": "unknown category"}]}
    fake({("POST", "/predict"): lambda **kw: FakeResponse(422, {"detail": detail})})
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL).predict(FEATURES)
    err = exc.value
    assert err.status_code == 422
    assert "nothing was scored" in err.message
    assert [e["column"] for e in err.errors] == ["age", "workclass"]
    assert err.n_errors == 2


def test_missing_columns_from_batch_422(fake):
    detail = {"message": "the CSV is missing required columns", "missing_columns": ["age", "occupation"]}
    fake({("POST", "/predict_batch"): lambda **kw: FakeResponse(422, {"detail": detail})})
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL).predict_batch("x.csv", b"a\n1\n")
    assert exc.value.missing_columns == ["age", "occupation"]


def test_pydantic_validation_list_is_readable(fake):
    detail = [{"loc": ["body", "features", "age"], "msg": "Field required", "input": None}]
    fake({("POST", "/predict"): lambda **kw: FakeResponse(422, {"detail": detail})})
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL).predict({})
    assert exc.value.errors[0]["column"] == "features.age"
    assert exc.value.errors[0]["error"] == "Field required"


def test_503_explains_service_problem(fake):
    detail = {"message": "Supabase request failed while logging the prediction", "error": "boom"}
    fake({("POST", "/predict"): lambda **kw: FakeResponse(503, {"detail": detail})})
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL).predict(FEATURES)
    assert exc.value.status_code == 503
    assert "temporarily unavailable" in exc.value.message
    assert "logging the prediction" in exc.value.message


def test_healthz_503_still_returns_body(fake):
    body = {"status": "degraded", "model_loader": False, "supabase": True, "run_loaded": False,
            "run_id": None, "detail": "ArtifactError: missing model artifacts"}
    fake({("GET", "/healthz"): lambda **kw: FakeResponse(503, body)})
    assert ApiClient(API_URL).healthz() == body


def test_connection_error_becomes_api_error(monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(api_client.requests, "request", boom)
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL).schema()
    assert "Could not reach the API" in exc.value.message


def test_timeout_mentions_render_wake_up(monkeypatch):
    def slow(*a, **kw):
        raise requests.exceptions.ReadTimeout("slow")
    monkeypatch.setattr(api_client.requests, "request", slow)
    with pytest.raises(ApiError) as exc:
        ApiClient(API_URL, timeout=5).version()
    assert "free tier" in exc.value.message


def test_non_json_success_is_an_error(fake):
    fake({("GET", "/schema"): lambda **kw: FakeResponse(200, None)})
    with pytest.raises(ApiError):
        ApiClient(API_URL).schema()
