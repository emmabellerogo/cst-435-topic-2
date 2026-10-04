"""Supabase persistence helpers for the API tier.

All Supabase access for the model service is funneled through this module. The
FastAPI app calls these functions; it never talks to Supabase directly, so the
tests can replace each function with an in-memory fake. The Streamlit UI NEVER
imports this module -- it does its own read-only queries with the anon key.

Tables (db/migrations/001_init.sql): adult_income, runs, predictions, and the
v_fairness_audit view.

Environment variables (set locally in a .env, and in the Render dashboard):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> the service-role key (server-side only, secret!)
"""
from __future__ import annotations

import os
from typing import List, Optional

from supabase import Client, create_client

from shared.features import FEATURE_COLS, TARGET_LABEL

_client: Optional[Client] = None


def get_client() -> Client:
    """Lazily create and cache a Supabase client."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def ping() -> bool:
    """Return True if the Supabase client can reach the runs table."""
    try:
        get_client().table("runs").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# adult_income + training runs (used by api/train.py and api/persist_runs.py)
# ---------------------------------------------------------------------------
# Only what training needs: no sex/race, which are not model inputs.
_ADULT_TRAINING_COLS = ",".join(["id", "split", TARGET_LABEL, *FEATURE_COLS])


def fetch_adult_income(page_size: int = 1000, split: Optional[str] = None) -> List[dict]:
    """Return the training columns of adult_income rows, ordered by id.

    ``split`` ('train' | 'val' | 'test') restricts the read to one split; the
    default reads every row. PostgREST caps each response (1000 rows by default
    on Supabase), so page through with range() until an empty page comes back.
    """
    client = get_client()
    rows: List[dict] = []
    while True:
        query = client.table("adult_income").select(_ADULT_TRAINING_COLS)
        if split is not None:
            query = query.eq("split", split)
        resp = query.order("id").range(len(rows), len(rows) + page_size - 1).execute()
        if not resp.data:
            return rows
        rows.extend(resp.data)


def insert_training_run(row: dict) -> dict:
    """Insert one completed training run (built by api.train.build_run_row)."""
    resp = get_client().table("runs").insert(row).execute()
    return resp.data[0]


def fetch_run_names() -> List[str]:
    """Names of every row already in ``runs`` (used to refuse duplicate inserts)."""
    resp = get_client().table("runs").select("name").execute()
    return [r["name"] for r in resp.data]


# ---------------------------------------------------------------------------
# serving (used by api/main.py)
# ---------------------------------------------------------------------------
# What the API needs to confirm a runs row describes the model it loaded, plus
# the headline metrics /version reports. Never the large JSONB metric columns.
SERVED_RUN_COLS = (
    "id,name,is_best,architecture,hidden_sizes,activation,dropout,best_epoch,config,"
    "calibration_method,checkpoint_path,preprocessor_path,"
    "val_loss,test_accuracy,test_roc_auc,created_at"
)


def fetch_run(run_id: int) -> Optional[dict]:
    """One runs row by id, or None."""
    resp = (
        get_client().table("runs").select(SERVED_RUN_COLS)
        .eq("id", run_id).limit(1).execute()
    )
    return resp.data[0] if resp.data else None


def fetch_best_runs() -> List[dict]:
    """Every runs row flagged is_best (the unique index allows at most one)."""
    resp = get_client().table("runs").select(SERVED_RUN_COLS).eq("is_best", True).execute()
    return resp.data


def insert_predictions(rows: List[dict]) -> List[dict]:
    """Insert prediction-log rows in one request; returns the stored rows."""
    if not rows:
        return []
    resp = get_client().table("predictions").insert(rows).execute()
    return resp.data


def fetch_fairness_audit(run_id: int) -> List[dict]:
    """The SQL-computed FPR/FNR rows of v_fairness_audit for one run, by group."""
    resp = (
        get_client().table("v_fairness_audit")
        .select("run_id,group_value,n,tp,fp,tn,fn,fpr,fnr")
        .eq("run_id", run_id).order("group_value").execute()
    )
    return resp.data
