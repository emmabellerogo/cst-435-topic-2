"""Offline tests for the serving API (the selected GELU run).

The real committed artifacts in models/gelu/ are loaded read-only; every
Supabase call goes to the in-memory FakeDB from conftest.py.
"""
from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from api import serving
from api.main import app
from api.model import load_checkpoint
from api.preprocessing import load_preprocessor
from shared.features import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_BOUNDS, NUMERIC_COLS
from tests.conftest import GELU_DIR, GELU_RUN_ID

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reference_prediction.json"
REFERENCE = json.loads(FIXTURE.read_text())["features"]
YOUNG = {**REFERENCE, "age": 23, "education_num": 9, "hours_per_week": 30,
         "marital_status": "Never-married", "relationship": "Own-child",
         "occupation": "Other-service"}
UNKNOWNS = {**REFERENCE, "workclass": "?", "occupation": "", "native_country": "Mexico"}


def _csv(rows, columns=None) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows, columns=columns or FEATURE_COLS).to_csv(buf, index=False)
    return buf.getvalue().encode()


def _upload(client, content: bytes):
    return client.post("/predict_batch", files={"file": ("rows.csv", content, "text/csv")})


def _independent_proba(features: dict) -> float:
    """Calibrated P(>50K) computed straight from the files, without api.serving."""
    pre = load_preprocessor(GELU_DIR / "preprocessor.joblib")
    model, _ = load_checkpoint(GELU_DIR / "model.pt")
    temperature = json.loads((GELU_DIR / "calibrator.json").read_text())["temperature"]
    X = pre.transform(pd.DataFrame([features])).astype(np.float32)
    with torch.no_grad():
        logit = float(model(torch.as_tensor(X))[0])
    return 1.0 / (1.0 + math.exp(-logit / temperature))


# ---------------------------------------------------------------------------
# /healthz and /version
# ---------------------------------------------------------------------------
def test_healthz_reports_artifacts_and_run(client):
    body = client.get("/healthz").json()
    assert body == {"status": "ok", "model_loader": True, "supabase": True,
                    "run_loaded": True, "run_id": GELU_RUN_ID, "detail": None}


def test_healthz_degraded_when_supabase_unreachable(client):
    client.fake.fail.add("ping")
    resp = client.get("/healthz")
    assert resp.status_code == 200  # the model can still predict once Supabase is back
    assert resp.json()["status"] == "degraded" and resp.json()["supabase"] is False


def test_version_identifies_gelu_run_without_secrets(client, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "super-secret-service-key")
    resp = client.get("/version")
    body = resp.json()
    assert body["project"] == "Income Insight"
    assert body["run_name"] == "gelu" and body["run_id"] == GELU_RUN_ID
    m = body["model"]
    assert m["activation"] == "gelu" and m["hidden_sizes"] == [64, 32]
    assert m["calibration_method"] == "temperature" and m["threshold"] == 0.5
    assert m["temperature"] == json.loads((GELU_DIR / "calibrator.json").read_text())["temperature"]
    assert m["n_inputs"] == 10
    assert "torch_version" in body and "sklearn_version" in body
    assert "super-secret" not in resp.text


# ---------------------------------------------------------------------------
# /schema
# ---------------------------------------------------------------------------
def test_schema_has_exactly_the_10_model_features(client):
    body = client.get("/schema").json()
    names = [f["name"] for f in body["features"]]
    assert names == FEATURE_COLS and len(names) == 10
    for protected in ("sex", "race"):
        assert protected not in names
        assert protected not in body["categories"]
        assert protected not in body["numeric_features"] + body["categorical_features"]

    encoder = (load_preprocessor(GELU_DIR / "preprocessor.joblib")
               .named_steps["columns"].named_transformers_["cat"].named_steps["onehot"])
    fitted = {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL_COLS, encoder.categories_)}
    for f in body["features"]:
        if f["name"] in NUMERIC_COLS:
            assert (f["kind"], f["type"]) == ("numeric", "integer")
            assert (f["minimum"], f["maximum"]) == NUMERIC_BOUNDS[f["name"]]
            assert f["minimum"] <= f["default"] <= f["maximum"]
        else:
            assert (f["kind"], f["type"]) == ("categorical", "string")
            assert f["categories"] == fitted[f["name"]]  # derived from the preprocessor
    # keys the current Streamlit form reads
    assert body["numeric_features"] == NUMERIC_COLS
    assert body["categorical_features"] == CATEGORICAL_COLS
    assert body["categories"] == fitted
    assert body["target_classes"] == ["<=50K", ">50K"]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
def test_predict_valid_row(client):
    resp = client.post("/predict", json={"features": REFERENCE})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] in (0, 1) and 0.0 <= body["proba"] <= 1.0
    assert body["label"] == int(body["proba"] >= 0.5)
    assert body["income"] == ["<=50K", ">50K"][body["label"]]
    assert (body["run_id"], body["run_name"]) == (GELU_RUN_ID, "gelu")
    assert body["threshold"] == 0.5 and body["calibration_method"] == "temperature"
    assert body["logged"] is True


def test_predict_matches_independent_pipeline(client):
    for features in (REFERENCE, YOUNG):
        proba = client.post("/predict", json={"features": features}).json()["proba"]
        assert proba == pytest.approx(_independent_proba(features), abs=1e-6)


def test_predict_frozen_reference_probability(client):
    ref = json.loads(FIXTURE.read_text())
    if ref["expected_proba"] is None:
        pytest.skip("no snapshot yet: run `python -m tests.make_reference_prediction` once")
    body = client.post("/predict", json={"features": ref["features"]}).json()
    assert abs(body["proba"] - ref["expected_proba"]) <= ref["tolerance"]
    assert body["label"] == ref["expected_label"]


@pytest.mark.parametrize("proba, label", [(0.5, 1), (0.49999, 0), (0.50001, 1), (0.0, 0), (1.0, 1)])
def test_predict_uses_fixed_threshold_half(client, monkeypatch, proba, label):
    monkeypatch.setattr(serving, "predict_proba", lambda art, records: np.array([proba]))
    body = client.post("/predict", json={"features": REFERENCE}).json()
    assert (body["proba"], body["label"], body["threshold"]) == (proba, label, 0.5)
    assert client.fake.predictions[-1]["predicted_label"] == label


def test_predict_does_not_accept_a_threshold(client):
    resp = client.post("/predict", json={"features": REFERENCE, "threshold": 0.3})
    assert resp.status_code == 422


def test_predict_logs_expected_payload(client):
    body = client.post("/predict", json={"features": REFERENCE}).json()
    assert client.fake.predictions == [{
        "id": 1,
        "request_hash": serving.request_hash(REFERENCE),
        "predicted_label": body["label"],
        "predicted_proba": body["proba"],
        "served_by_run_id": GELU_RUN_ID,
        "adult_income_id": None,  # arbitrary user input: no ground-truth link
    }]
    assert body["request_hash"] == client.fake.predictions[0]["request_hash"]


def test_request_hash_is_deterministic(client):
    shuffled = dict(reversed(list(REFERENCE.items())))
    assert serving.request_hash(shuffled) == serving.request_hash(REFERENCE)
    assert serving.request_hash({**REFERENCE, "age": 46}) != serving.request_hash(REFERENCE)
    assert len(serving.request_hash(REFERENCE)) == 64

    as_strings = {**REFERENCE, "age": "45", "native_country": " United-States "}
    h1 = client.post("/predict", json={"features": REFERENCE}).json()["request_hash"]
    h2 = client.post("/predict", json={"features": as_strings}).json()["request_hash"]
    assert h1 == h2 == serving.request_hash(REFERENCE)


@pytest.mark.parametrize("features, column", [
    ({**REFERENCE, "sex": "Male"}, "sex"),                  # protected attribute: not an input
    ({**REFERENCE, "race": "White"}, "race"),
    ({k: v for k, v in REFERENCE.items() if k != "occupation"}, "occupation"),
    ({**REFERENCE, "occupation": "Astronaut"}, "occupation"),
    ({**REFERENCE, "age": 200}, "age"),
    ({**REFERENCE, "age": 45.5}, "age"),
    ({**REFERENCE, "capital_gain": "lots"}, "capital_gain"),
])
def test_predict_rejects_bad_rows(client, features, column):
    resp = client.post("/predict", json={"features": features})
    assert resp.status_code == 422
    assert column in resp.text
    assert client.fake.predictions == []


def test_predict_rejects_a_different_run_id(client):
    assert client.post("/predict", json={"features": REFERENCE, "run_id": 1}).status_code == 409
    assert client.post("/predict", json={"features": REFERENCE,
                                         "run_id": GELU_RUN_ID}).status_code == 200


def test_predict_logging_failure_is_a_clear_503(client):
    client.fake.fail.add("insert_predictions")
    resp = client.post("/predict", json={"features": REFERENCE})
    assert resp.status_code == 503
    assert "logging the prediction" in resp.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------
def test_supabase_down_at_startup_then_recovers(fake_db):
    fake_db.fail.add("fetch_best_runs")
    with TestClient(app) as c:
        health = c.get("/healthz").json()
        assert health["model_loader"] is True and health["run_loaded"] is False
        assert health["status"] == "degraded"
        assert c.post("/predict", json={"features": REFERENCE}).status_code == 503

        fake_db.fail.clear()
        resp = c.post("/predict", json={"features": REFERENCE})  # re-reads the run lazily
        assert resp.status_code == 200 and resp.json()["run_id"] == GELU_RUN_ID


def test_startup_rejects_runs_row_that_does_not_match(fake_db):
    fake_db.runs[GELU_RUN_ID]["config"] = {**fake_db.runs[GELU_RUN_ID]["config"], "activation": "relu"}
    with TestClient(app) as c:
        health = c.get("/healthz").json()
        assert health["run_loaded"] is False and "config" in health["detail"]
        assert c.post("/predict", json={"features": REFERENCE}).status_code == 503
    assert fake_db.predictions == []


def test_missing_artifacts_make_the_service_unready(fake_db, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 503 and resp.json()["model_loader"] is False
        assert c.get("/schema").status_code == 503
        assert c.post("/predict", json={"features": REFERENCE}).status_code == 503


@pytest.mark.parametrize("edit, message", [
    (lambda d: _edit_json(d / "run.json", name="deep"), "expected 'gelu'"),
    (lambda d: _edit_json(d / "calibrator.json", temperature=2.0), "temperature"),
    (lambda d: (d / "calibrator.json").unlink(), "missing"),
])
def test_load_artifacts_rejects_non_gelu_files(tmp_path, edit, message):
    model_dir = tmp_path / "gelu"
    shutil.copytree(GELU_DIR, model_dir)
    edit(model_dir)
    with pytest.raises(serving.ArtifactError, match=message):
        serving.load_artifacts(model_dir)


def _edit_json(path: Path, **changes) -> None:
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


# ---------------------------------------------------------------------------
# /predict_batch
# ---------------------------------------------------------------------------
def test_predict_batch_scores_every_row(client):
    resp = _upload(client, _csv([REFERENCE, YOUNG, UNKNOWNS]))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_rows"] == len(body["predictions"]) == body["logged"] == 3
    assert [p["row"] for p in body["predictions"]] == [1, 2, 3]
    assert (body["run_id"], body["threshold"]) == (GELU_RUN_ID, 0.5)
    for p in body["predictions"]:
        assert 0.0 <= p["proba"] <= 1.0 and p["label"] == int(p["proba"] >= 0.5)
    assert body["n_predicted_positive"] == sum(p["label"] for p in body["predictions"])
    assert len(client.fake.predictions) == 3
    assert all(r["served_by_run_id"] == GELU_RUN_ID and r["adult_income_id"] is None
               for r in client.fake.predictions)

    single = client.post("/predict", json={"features": REFERENCE}).json()
    assert body["predictions"][0]["proba"] == pytest.approx(single["proba"], abs=1e-6)
    assert body["predictions"][0]["request_hash"] == single["request_hash"]


def test_predict_batch_reads_supabase_before_scoring(client):
    before = client.fake.run_reads
    _upload(client, _csv([REFERENCE]))
    _upload(client, _csv([REFERENCE]))
    assert client.fake.run_reads == before + 2  # one runs read per batch request

    client.fake.fail.add("fetch_run")
    resp = _upload(client, _csv([REFERENCE]))
    assert resp.status_code == 503
    assert "confirming the active run" in resp.json()["detail"]["message"]
    assert len(client.fake.predictions) == 2  # nothing scored or logged on failure


def test_predict_batch_refuses_when_run_no_longer_best(client):
    client.fake.runs[GELU_RUN_ID]["is_best"] = False
    resp = _upload(client, _csv([REFERENCE]))
    assert resp.status_code == 503 and "is_best" in resp.text
    assert client.fake.predictions == []


def test_predict_batch_ignores_extra_columns_including_protected(client):
    rows = [{**REFERENCE, "sex": "Female", "race": "White", "income": ">50K"}]
    resp = _upload(client, _csv(rows, columns=FEATURE_COLS + ["sex", "race", "income"]))
    assert resp.status_code == 200
    assert resp.json()["ignored_columns"] == ["sex", "race", "income"]
    assert set(client.fake.predictions[0]) == {"id", "request_hash", "predicted_label",
                                               "predicted_proba", "served_by_run_id",
                                               "adult_income_id"}


@pytest.mark.parametrize("content, message", [
    (_csv([{k: v for k, v in REFERENCE.items() if k != "occupation"}],
          columns=[c for c in FEATURE_COLS if c != "occupation"]), "missing required columns"),
    (b"age,education_num\n1,2\n3,4,5,6\n", "could not parse"),  # ragged row
    (b"", "empty"),
    (",".join(FEATURE_COLS).encode() + b"\n", "no data rows"),
    (b"\xff\xfe\x00a\x00g\x00e", "UTF-8"),
    (_csv([{**REFERENCE, "age": "abc"}]), "invalid feature value"),
    (_csv([REFERENCE, {**REFERENCE, "hours_per_week": 500}]), "invalid feature value"),
])
def test_predict_batch_rejects_bad_files_clearly(client, content, message):
    resp = _upload(client, content)
    assert resp.status_code == 422
    assert message in resp.json()["detail"]["message"]
    assert client.fake.predictions == []


def test_predict_batch_reports_row_and_column_of_bad_values(client):
    resp = _upload(client, _csv([REFERENCE, {**REFERENCE, "age": "abc"}]))
    errors = resp.json()["detail"]["errors"]
    assert errors == [{"row": 2, "column": "age", "value": "abc", "error": "missing or not a number"}]


def test_predict_batch_missing_columns_are_listed(client):
    cols = [c for c in FEATURE_COLS if c not in ("age", "relationship")]
    detail = _upload(client, _csv([REFERENCE], columns=cols)).json()["detail"]
    assert detail["missing_columns"] == ["age", "relationship"]
    assert detail["required_columns"] == FEATURE_COLS


def test_predict_batch_requires_a_file(client):
    assert client.post("/predict_batch").status_code == 422


def test_predict_batch_logging_failure_is_a_clear_503(client):
    client.fake.fail.add("insert_predictions")
    resp = _upload(client, _csv([REFERENCE, YOUNG]))
    assert resp.status_code == 503
    assert "logging the batch predictions" in resp.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# /audit
# ---------------------------------------------------------------------------
def test_audit_passes_sql_view_rows_through(client):
    # Rates deliberately inconsistent with the counts: if the API recomputed
    # FPR/FNR in Python, these values would change.
    view_rows = [
        {"run_id": GELU_RUN_ID, "group_value": "Female", "n": 100, "tp": 10, "fp": 5,
         "tn": 80, "fn": 5, "fpr": 0.123, "fnr": 0.456},
        {"run_id": GELU_RUN_ID, "group_value": "Male", "n": 200, "tp": 50, "fp": 20,
         "tn": 100, "fn": 30, "fpr": 0.789, "fnr": None},
    ]
    client.fake.audit_rows[GELU_RUN_ID] = view_rows
    body = client.get("/audit").json()
    assert body["run_id"] == GELU_RUN_ID and body["run_name"] == "gelu"
    assert (body["attribute"], body["split"], body["source"]) == ("sex", "test", "v_fairness_audit")
    assert body["groups"] == [{k: v for k, v in r.items() if k != "run_id"} for r in view_rows]
    assert body["note"] is None
    assert client.fake.calls[f"fetch_fairness_audit:{GELU_RUN_ID}"] == 1  # scoped to the served run


def test_audit_with_no_view_rows_explains_why(client):
    body = client.get("/audit").json()
    assert body["groups"] == [] and "held-out" in body["note"]


def test_audit_supabase_failure_is_a_clear_503(client):
    client.fake.fail.add("fetch_fairness_audit")
    resp = client.get("/audit")
    assert resp.status_code == 503
    assert "v_fairness_audit" in resp.json()["detail"]["message"]
