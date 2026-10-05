"""Live Supabase checks of the serving API.

Skipped automatically unless real Supabase credentials are present, so the
suite (and CI, which has no secrets) still passes offline.

1. ``test_live_startup_reads_gelu_run_and_audit_view`` is read-only. It starts
   the app against the real project and confirms startup found the GELU run and
   that /audit reads the SQL view.

2. ``test_live_predict_writes_a_predictions_row`` WRITES to the live database,
   so it needs a second, explicit opt-in (RUN_LIVE_WRITE_TESTS=1). It sends one
   POST /predict with the frozen reference profile
   (tests/fixtures/reference_prediction.json), then reads the predictions table
   back with the service-role client and checks that exactly one new row for
   that request exists with the returned label, probability and run id, and
   with adult_income_id NULL (so it can never enter the fairness audit). The
   row is left in place: it is indistinguishable from a prediction made in the
   app, and deleting it would be a second write.

Run them yourself (never commit the keys):

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest -rs tests/test_supabase_roundtrip.py
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... RUN_LIVE_WRITE_TESTS=1 \\
        pytest -rs tests/test_supabase_roundtrip.py
"""
from __future__ import annotations

import json
import os

import pytest

from tests.conftest import GELU_RUN_ID, REPO_ROOT, has_supabase_creds

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)

REFERENCE = json.loads((REPO_ROOT / "tests" / "fixtures" / "reference_prediction.json").read_text())


def test_live_startup_reads_gelu_run_and_audit_view():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        health = c.get("/healthz").json()
        assert health["status"] == "ok", health
        assert health["run_id"] == GELU_RUN_ID

        version = c.get("/version").json()
        assert version["run_name"] == "gelu" and version["run_id"] == GELU_RUN_ID

        audit = c.get("/audit")
        assert audit.status_code == 200
        assert audit.json()["source"] == "v_fairness_audit"


@pytest.mark.skipif(os.environ.get("RUN_LIVE_WRITE_TESTS") != "1",
                    reason="Writes one predictions row; set RUN_LIVE_WRITE_TESTS=1 to run.")
def test_live_predict_writes_a_predictions_row():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    def rows_for(request_hash: str):
        return (
            db.get_client().table("predictions")
            .select("id,request_hash,predicted_label,predicted_proba,served_by_run_id,"
                    "adult_income_id,created_at", count="exact")
            .eq("request_hash", request_hash).eq("served_by_run_id", GELU_RUN_ID)
            .order("id", desc=True).limit(1).execute()
        )

    with TestClient(app) as c:
        assert c.get("/healthz").json()["status"] == "ok"

        # The hash depends only on the 10 feature values, so it can be read before the write.
        from api import serving
        expected_hash = serving.request_hash(REFERENCE["features"])
        before = rows_for(expected_hash)

        resp = c.post("/predict", json={"features": REFERENCE["features"]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["logged"] is True
        assert body["request_hash"] == expected_hash
        assert body["run_id"] == GELU_RUN_ID
        assert body["proba"] == pytest.approx(REFERENCE["expected_proba"], abs=REFERENCE["tolerance"])

        after = rows_for(expected_hash)

    # Exactly one more row for this request (unless someone scored the same profile in
    # the app at the same moment, which would only make the count larger).
    assert after.count >= (before.count or 0) + 1
    newest = after.data[0]
    assert before.data == [] or newest["id"] > before.data[0]["id"]
    assert newest["predicted_label"] == body["label"]
    assert newest["predicted_proba"] == pytest.approx(body["proba"], abs=1e-9)
    assert newest["served_by_run_id"] == GELU_RUN_ID
    assert newest["adult_income_id"] is None
