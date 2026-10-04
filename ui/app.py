"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

A THIN client for the Income Insight API:
  * every prediction is an HTTPS call to the FastAPI service on Render
    (no torch / sklearn / model code here),
  * fairness rates come from GET /audit exactly as the SQL view computed them,
  * the finished evaluation results (confusion matrix, calibration, experiment
    comparison, permutation importance) are read read-only from the Supabase
    `runs` table with the anon key, or from the committed files in models/.

Six tabs: Concepts, Score a Row, Score CSV, Model Performance, Bias Audit, Model Card.

Configuration (st.secrets on Streamlit Cloud, or environment variables locally):
    API_URL              -> the Render base URL, e.g. https://<your-api>.onrender.com
    SUPABASE_URL         -> https://<ref>.supabase.co         (optional)
    SUPABASE_ANON_KEY    -> the public anon key, never the service key (optional)

Run locally from the repo root:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the sibling modules importable however the app is started
# (streamlit run, Streamlit Cloud, or streamlit.testing.AppTest).
_UI_DIR = str(Path(__file__).resolve().parent)
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

import tab_bias_audit  # noqa: E402
import tab_concepts  # noqa: E402
import tab_model_card  # noqa: E402
import tab_performance  # noqa: E402
import tab_score_csv  # noqa: E402
import tab_score_row  # noqa: E402
import ui_services as svc  # noqa: E402
from api_client import ApiError  # noqa: E402

TAB_NAMES = ["Concepts", "Score a Row", "Score CSV", "Model Performance", "Bias Audit", "Model Card"]

st.set_page_config(page_title="Income Insight", page_icon="💼", layout="wide")


def render_sidebar(url) -> None:
    with st.sidebar:
        st.header("Service status")
        if not url:
            st.warning("API_URL is not set.")
            return
        st.caption(f"API: {url}")
        try:
            health = svc.get_health(url)
        except ApiError as e:
            st.error(e.message)
        else:
            ok = health.get("status") == "ok"
            (st.success if ok else st.warning)(f"API status: {health.get('status', 'unknown')}")
            st.write(
                f"Model loaded: {'yes' if health.get('model_loader') else 'no'}  \n"
                f"Supabase reachable: {'yes' if health.get('supabase') else 'no'}  \n"
                f"Served run: {health.get('run_id') if health.get('run_loaded') else 'not confirmed'}"
            )
            if health.get("detail"):
                st.caption(f"Detail: {health['detail']}")
        version = svc.try_version(url)
        if version:
            st.write(
                f"Run: **{version.get('run_name')}** (id {version.get('run_id')})  \n"
                f"API version: {version.get('api_version')}  \n"
                f"Build: {version.get('git_sha')}"
            )
        st.caption("A sleeping Render service can take about a minute to answer the first request.")
        if st.button("Refresh data from the API", key="refresh"):
            st.cache_data.clear()
            st.rerun()


st.title("💼 Income Insight")
st.caption(
    "Predicts whether a person's annual income is above \\$50K from 10 census features. "
    "Streamlit (this UI) → FastAPI on Render (frozen GELU MLP) → Supabase Postgres."
)

api_url = svc.api_url()
render_sidebar(api_url)

tabs = st.tabs(TAB_NAMES)
with tabs[0]:
    tab_concepts.render(api_url)
with tabs[1]:
    tab_score_row.render(api_url)
with tabs[2]:
    tab_score_csv.render(api_url)
with tabs[3]:
    tab_performance.render()
with tabs[4]:
    tab_bias_audit.render(api_url)
with tabs[5]:
    tab_model_card.render(api_url)
