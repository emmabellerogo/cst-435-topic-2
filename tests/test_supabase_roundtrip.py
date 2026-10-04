"""Live Supabase check of the serving API (read-only).

Starts the app against the real project and confirms startup found the GELU
run and that /audit reads the SQL view. It does NOT call /predict, so it never
writes a prediction row. Skipped automatically unless real Supabase credentials
are present, so the suite still passes offline. Run it yourself with:

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... pytest tests/test_supabase_roundtrip.py
"""
from __future__ import annotations

import pytest

from tests.conftest import GELU_RUN_ID, has_supabase_creds

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)


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
