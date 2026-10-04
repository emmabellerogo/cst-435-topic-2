"""Shared helpers for the Streamlit tabs: settings, cached API reads, error display."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

import perf_data
from api_client import ApiClient, ApiError

# Shown in the UI and used for guards. These are never sent to the model.
PROTECTED_ATTRIBUTES = ("sex", "race")

FIELD_LABELS = {
    "age": "Age (years)",
    "education_num": "Education level (1-16)",
    "capital_gain": "Capital gain ($)",
    "capital_loss": "Capital loss ($)",
    "hours_per_week": "Hours worked per week",
    "workclass": "Work class",
    "marital_status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Relationship (role in household)",
    "native_country": "Native country",
}


def label(name: str) -> str:
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# settings: st.secrets on Streamlit Cloud, environment variables locally
# ---------------------------------------------------------------------------
def _secrets_available() -> bool:
    # Reading st.secrets with no secrets.toml prints an error box in the app,
    # so check quietly first (local runs may use environment variables only).
    loader = getattr(st.secrets, "load_if_toml_exists", None)
    if loader is None:
        return True
    try:
        return bool(loader())
    except Exception:
        return False


def get_setting(name: str) -> Optional[str]:
    value = None
    if _secrets_available():
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
    if not value:
        value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def api_url() -> Optional[str]:
    url = get_setting("API_URL")
    return url.rstrip("/") if url else None


def client(url: str) -> ApiClient:
    return ApiClient(url)


# ---------------------------------------------------------------------------
# cached reads (exceptions are not cached, so a failed call is retried next run)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_health(url: str) -> dict:
    return ApiClient(url).healthz()


@st.cache_data(ttl=300, show_spinner=False)
def get_version(url: str) -> dict:
    return ApiClient(url).version()


@st.cache_data(ttl=300, show_spinner=False)
def get_schema(url: str) -> dict:
    return ApiClient(url).schema()


@st.cache_data(ttl=300, show_spinner=False)
def get_audit(url: str) -> dict:
    return ApiClient(url).audit()


@st.cache_data(ttl=600, show_spinner=False)
def _performance(supabase_url: Optional[str], anon_key: Optional[str], models_dir: str) -> dict:
    return perf_data.load_performance(supabase_url, anon_key, models_dir)


def get_performance() -> dict:
    """The finished evaluation results (raises perf_data.PerformanceDataError)."""
    return _performance(get_setting("SUPABASE_URL"), get_setting("SUPABASE_ANON_KEY"),
                        str(perf_data.default_models_dir()))


@st.cache_data(ttl=300, show_spinner=False)
def _fairness_view(supabase_url: str, anon_key: str, run_id: int) -> list:
    return perf_data.fetch_fairness_view(supabase_url, anon_key, run_id)


def get_fairness_view(run_id: int) -> Optional[list]:
    """v_fairness_audit rows read directly with the anon key; None when not configured.

    Raises perf_data.UnsafeKeyError for a non-public key, or the client's error.
    """
    supabase_url, anon_key = get_setting("SUPABASE_URL"), get_setting("SUPABASE_ANON_KEY")
    if not (supabase_url and anon_key):
        return None
    return _fairness_view(supabase_url, anon_key, int(run_id))


def try_performance() -> Optional[dict]:
    try:
        return get_performance()
    except perf_data.PerformanceDataError:
        return None


def try_version(url: Optional[str]) -> Optional[dict]:
    if not url:
        return None
    try:
        return get_version(url)
    except ApiError:
        return None


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------
def show_api_error(err: ApiError, what: str = "The request") -> None:
    st.error(f"{what} failed. {err.message}")
    if err.missing_columns:
        st.write("Missing required columns: " + ", ".join(f"`{c}`" for c in err.missing_columns))
    rows = err.errors
    if rows:
        shown = len(rows)
        total = err.n_errors
        note = f" (showing the first {shown} of {total})" if total > shown else ""
        st.write(f"Problems found{note}:")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def no_api_url_message() -> None:
    st.warning(
        "API_URL is not configured, so this tab cannot reach the model. "
        "Add it to ui/.streamlit/secrets.toml (see secrets.toml.example)."
    )


def fmt(value, spec: str = ".4f", missing: str = "n/a") -> str:
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)
