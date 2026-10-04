"""Request-schema tests for the prediction endpoints."""
from __future__ import annotations


def test_predict_rejects_missing_fields(client):
    # Missing 'features' -> 422 Unprocessable Entity from Pydantic validation.
    resp = client.post("/predict", json={"run_id": 1})
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client):
    resp = client.post("/predict", json={"run_id": "not-an-int", "features": {}})
    assert resp.status_code == 422


def test_audit_takes_no_python_grouping(client):
    # The old endpoint grouped logged predictions by any feature in Python. The
    # audit now comes only from v_fairness_audit (grouped by sex in SQL), so a
    # 'by' parameter changes nothing.
    resp = client.get("/audit", params={"by": "occupation"})
    assert resp.status_code == 200
    assert resp.json()["attribute"] == "sex"
