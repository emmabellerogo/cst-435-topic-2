"""Shared pytest fixtures.

The API tests run WITHOUT any cloud access: every Supabase function the API
uses (api.db.*) is replaced by an in-memory FakeDB. They do use the real,
committed model artifacts in models/gelu/ (read-only). The live Supabase test
is skipped unless real credentials are present, so `pytest` stays offline.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GELU_DIR = REPO_ROOT / "models" / "gelu"
GELU_RUN_ID = 3  # the gelu row's id in the live runs table


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def gelu_runs_row() -> dict:
    """What db.fetch_run(3) returns live: the persisted gelu row's serving columns."""
    run = json.loads((GELU_DIR / "run.json").read_text())
    from api.db import SERVED_RUN_COLS

    row = {c: run.get(c) for c in SERVED_RUN_COLS.split(",")}
    row.update(id=GELU_RUN_ID, created_at="2026-10-03T00:00:00+00:00")
    return row


class FakeDB:
    """In-memory stand-in for the api.db functions the service calls.

    ``fail`` holds function names that should raise (simulated outage);
    ``calls`` counts every call by function name.
    """

    def __init__(self):
        self.runs = {GELU_RUN_ID: gelu_runs_row()}
        self.predictions: list = []
        self.audit_rows: dict = {GELU_RUN_ID: []}
        self.fail: set = set()
        self.calls: Counter = Counter()

    def _enter(self, name: str) -> None:
        self.calls[name] += 1
        if name in self.fail:
            raise ConnectionError(f"simulated Supabase outage in {name}")

    @property
    def run_reads(self) -> int:
        return self.calls["fetch_run"] + self.calls["fetch_best_runs"]

    def ping(self) -> bool:
        self.calls["ping"] += 1
        return "ping" not in self.fail

    def fetch_run(self, run_id):
        self._enter("fetch_run")
        row = self.runs.get(run_id)
        return dict(row) if row else None

    def fetch_best_runs(self):
        self._enter("fetch_best_runs")
        return [dict(r) for r in self.runs.values() if r.get("is_best")]

    def insert_predictions(self, rows):
        self._enter("insert_predictions")
        stored = []
        for r in rows:
            stored.append({"id": len(self.predictions) + 1, **r})
            self.predictions.append(stored[-1])
        return stored

    def fetch_fairness_audit(self, run_id):
        self._enter("fetch_fairness_audit")
        self.calls[f"fetch_fairness_audit:{run_id}"] += 1
        return [dict(r) for r in self.audit_rows.get(run_id, [])]


@pytest.fixture
def fake_db(monkeypatch):
    """Patch every api.db function the API uses with a fresh FakeDB."""
    if not (GELU_DIR / "model.pt").exists():
        pytest.skip("models/gelu/ artifacts are not present")
    from api import db

    fake = FakeDB()
    for name in ("ping", "fetch_run", "fetch_best_runs", "insert_predictions",
                 "fetch_fairness_audit"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    monkeypatch.delenv("SERVED_RUN_ID", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    return fake


@pytest.fixture
def client(fake_db):
    """A TestClient whose startup loaded models/gelu/ and read run 3 from the FakeDB."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        c.fake = fake_db  # expose for assertions
        yield c
