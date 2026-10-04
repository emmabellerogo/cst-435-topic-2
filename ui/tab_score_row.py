"""Score a Row tab: a form built from GET /schema, sent to POST /predict.

The fields, their order, numeric limits, training-median defaults and the
allowed category values all come from /schema (categories are read from the
fitted preprocessor on the API side), so nothing about the input contract is
hard-coded here. Sex and race are never asked for: they are not model inputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import ui_services as svc
from api_client import ApiClient, ApiError

KEY_PREFIX = "row_"
RESULT_KEY = "row_last_prediction"

# The example request from the API contract in the handoff document. Used only
# to pre-fill the form; any value the live /schema does not allow is skipped.
EXAMPLE_PROFILE = {
    "age": 45,
    "education_num": 13,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 45,
    "workclass": "Private",
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "native_country": "United-States",
}


def input_fields(schema: dict) -> List[dict]:
    """The schema's features, minus anything that must never be a model input."""
    return [f for f in schema.get("features", []) if f.get("name") not in svc.PROTECTED_ATTRIBUTES]


def default_value(field: dict):
    if field.get("kind") == "numeric":
        return int(field["default"]) if field.get("default") is not None else int(field["minimum"])
    cats = field.get("categories") or []
    known = [c for c in cats if c != "Unknown"]
    return (known or cats or [None])[0]


def example_value(field: dict):
    """The example profile's value if the schema allows it, else the schema default."""
    value = EXAMPLE_PROFILE.get(field["name"])
    if field.get("kind") == "numeric":
        lo, hi = field.get("minimum"), field.get("maximum")
        if isinstance(value, (int, float)) and (lo is None or value >= lo) and (hi is None or value <= hi):
            return int(value)
    elif value in (field.get("categories") or []):
        return value
    return default_value(field)


def _key(field: dict) -> str:
    return KEY_PREFIX + field["name"]


def _fill(fields: List[dict], use_example: bool) -> None:
    for f in fields:
        st.session_state[_key(f)] = example_value(f) if use_example else default_value(f)
    st.session_state.pop(RESULT_KEY, None)


def collect_features(fields: List[dict], state) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for f in fields:
        value = state[_key(f)]
        out[f["name"]] = int(value) if f.get("kind") == "numeric" else value
    return out


def probability_chart(proba: float, threshold: float) -> alt.Chart:
    base = pd.DataFrame({"label": ["P(>50K)"], "start": [0.0], "end": [1.0]})
    fill = pd.DataFrame({"label": ["P(>50K)"], "start": [0.0], "end": [proba]})
    thr = pd.DataFrame({"x": [threshold]})
    track = alt.Chart(base).mark_bar(color="#e6e6e6", size=28).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
        x2="end:Q", y=alt.Y("label:N", title=None),
    )
    bar = alt.Chart(fill).mark_bar(color="#2b7bb9" if proba >= threshold else "#8a8a8a", size=28).encode(
        x="start:Q", x2="end:Q", y="label:N",
    )
    rule = alt.Chart(thr).mark_rule(color="#c0392b", strokeWidth=2).encode(x="x:Q")
    return (track + bar + rule).properties(height=70)


def show_result(resp: dict) -> None:
    proba = float(resp["proba"])
    threshold = float(resp.get("threshold", 0.5))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted income", resp["income"])
    c2.metric("Probability of >50K", f"{proba:.1%}")
    c3.metric("Decision threshold", f"{threshold:.2f}")
    st.altair_chart(probability_chart(proba, threshold), use_container_width=True)
    st.caption(
        f"The model predicts >50K when the calibrated probability is at least {threshold:.2f} "
        "(the red line). The probability is an estimate learned from 1994 U.S. Census data, "
        "not a fact about this person."
    )
    st.caption(
        f"Served by run {resp.get('run_id')} ({resp.get('run_name')}), "
        f"calibration: {resp.get('calibration_method')}. "
        + ("Logged to Supabase as request " + str(resp.get("request_hash", ""))[:12] + "…"
           if resp.get("logged") else "Not logged.")
    )
    with st.expander("Full API response"):
        st.json(resp)


def render(url: Optional[str]) -> None:
    st.header("Score a Row")
    st.write(
        "Enter one person's details and the API returns a prediction from the deployed model. "
        "The form is built from the API's `/schema`, so the fields, limits and allowed values "
        "always match the model being served. Sex and race are not asked for because the "
        "model does not use them."
    )
    if not url:
        svc.no_api_url_message()
        return
    try:
        schema = svc.get_schema(url)
    except ApiError as e:
        svc.show_api_error(e, "Loading the input form from /schema")
        return

    leaked = [f["name"] for f in schema.get("features", []) if f.get("name") in svc.PROTECTED_ATTRIBUTES]
    if leaked:
        st.warning(f"/schema lists {leaked} as inputs; they are left out of this form. "
                   "Please report this to the backend owner.")

    fields = input_fields(schema)
    if not fields:
        st.error("/schema returned no input fields.")
        return
    for f in fields:  # first visit: start from the example profile
        if _key(f) not in st.session_state:
            st.session_state[_key(f)] = example_value(f)

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load example profile", key="row_example", on_click=_fill, args=(fields, True))
    b2.button("Reset to defaults", key="row_reset", on_click=_fill, args=(fields, False),
              help="Numeric fields go to the training-set medians from /schema.")

    numeric = [f for f in fields if f.get("kind") == "numeric"]
    categorical = [f for f in fields if f.get("kind") != "numeric"]

    st.markdown("**Numbers**")
    cols = st.columns(max(len(numeric), 1))
    for col, f in zip(cols, numeric):
        col.number_input(
            svc.label(f["name"]),
            min_value=int(f["minimum"]), max_value=int(f["maximum"]), step=1, key=_key(f),
            help=f"`{f['name']}`, whole number from {f['minimum']} to {f['maximum']}. "
                 f"Training median: {f.get('default')}.",
        )
    st.markdown("**Categories**")
    cols = st.columns(3)
    for i, f in enumerate(categorical):
        cols[i % 3].selectbox(
            svc.label(f["name"]), f.get("categories") or [], key=_key(f),
            help=f"`{f['name']}`: {len(f.get('categories') or [])} values learned in training.",
        )

    if st.button("Predict income", type="primary", key="predict_row"):
        features = collect_features(fields, st.session_state)
        try:
            with st.spinner("Asking the model…"):
                resp = ApiClient(url).predict(features)
        except ApiError as e:
            st.session_state.pop(RESULT_KEY, None)
            svc.show_api_error(e, "The prediction")
        else:
            st.session_state[RESULT_KEY] = resp

    if RESULT_KEY in st.session_state:
        show_result(st.session_state[RESULT_KEY])
