"""Score CSV tab: upload a CSV, send it to POST /predict_batch, show and download
the scored rows.

The API does all validation (required columns, numeric ranges, known
categories) and rejects the whole file if any value is bad, so nothing is
partly scored or logged. This tab shows the API's row-by-row errors and only
adds a local preview and a heads-up about missing columns before sending.
"""
from __future__ import annotations

import hashlib
import io
from typing import List, Optional

import pandas as pd
import streamlit as st

import ui_services as svc
from api_client import ApiClient, ApiError

RESULT_KEY = "csv_last_result"
PREDICTION_COLUMNS = ["predicted_income", "probability_over_50k", "predicted_label", "request_hash"]


def required_columns(schema: dict) -> List[str]:
    return [f["name"] for f in schema.get("features", []) if f.get("name") not in svc.PROTECTED_ATTRIBUTES]


def template_csv(schema: dict) -> bytes:
    """A header with the required columns plus one example row of defaults."""
    row = {}
    for f in schema.get("features", []):
        if f.get("name") in svc.PROTECTED_ATTRIBUTES:
            continue
        if f.get("kind") == "numeric":
            row[f["name"]] = f.get("default") if f.get("default") is not None else f.get("minimum")
        else:
            cats = [c for c in (f.get("categories") or []) if c != "Unknown"] or (f.get("categories") or [""])
            row[f["name"]] = cats[0]
    return pd.DataFrame([row], columns=list(row)).to_csv(index=False).encode("utf-8")


def read_preview(content: bytes) -> pd.DataFrame:
    """Read the CSV the same way the API does: every cell as text, blanks kept."""
    df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def merge_predictions(df: pd.DataFrame, resp: dict) -> pd.DataFrame:
    """Append the API's predictions to the uploaded rows.

    The API numbers rows from 1 in file order; the counts must match exactly.
    """
    preds = pd.DataFrame(resp.get("predictions") or [])
    if len(preds) != len(df):
        raise ValueError(f"the API returned {len(preds)} predictions for {len(df)} rows")
    preds = preds.sort_values("row").reset_index(drop=True)
    if list(preds["row"]) != list(range(1, len(df) + 1)):
        raise ValueError("the API's row numbers do not match the uploaded rows")
    out = df.reset_index(drop=True).copy()
    out["predicted_income"] = preds["income"].to_numpy()
    out["probability_over_50k"] = preds["proba"].astype(float).to_numpy()
    out["predicted_label"] = preds["label"].astype(int).to_numpy()
    out["request_hash"] = preds["request_hash"].to_numpy()
    return out


def _file_id(name: str, content: bytes) -> str:
    return f"{name}:{hashlib.sha256(content).hexdigest()}"


def render(url: Optional[str]) -> None:
    st.header("Score CSV")
    st.write(
        "Upload a CSV with one person per row. The API checks every value, scores every row "
        "with the same frozen model as **Score a Row**, and logs each prediction. If any value "
        "is invalid, the whole file is rejected and nothing is scored, so fix the listed rows "
        "and upload again."
    )
    if not url:
        svc.no_api_url_message()
        return
    try:
        schema = svc.get_schema(url)
    except ApiError as e:
        svc.show_api_error(e, "Loading the column list from /schema")
        return

    required = required_columns(schema)
    st.markdown("**Required columns** (other columns are allowed and ignored): "
                + ", ".join(f"`{c}`" for c in required))
    st.caption("Limits: 5 MB and 10,000 rows per file. Numbers must be whole numbers within the "
               "ranges shown on Score a Row; categories must match the allowed values exactly.")
    st.download_button("Download a template CSV", template_csv(schema),
                       file_name="income_insight_template.csv", mime="text/csv",
                       key="csv_template")

    upload = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
    if upload is None:
        st.session_state.pop(RESULT_KEY, None)
        return
    content = upload.getvalue()
    file_id = _file_id(upload.name, content)

    try:
        preview = read_preview(content)
    except Exception as e:  # noqa: BLE001 - show any parse problem to the user
        st.error(f"This file could not be read as a CSV: {e}")
        return
    st.write(f"**{upload.name}**: {len(preview):,} rows, {len(preview.columns)} columns")
    st.dataframe(preview.head(20), use_container_width=True, hide_index=True)
    if len(preview) > 20:
        st.caption("Showing the first 20 rows.")

    missing = [c for c in required if c not in preview.columns]
    if missing:
        st.warning("This file is missing required columns, so the API will reject it: "
                   + ", ".join(f"`{c}`" for c in missing))
    present_protected = [c for c in svc.PROTECTED_ATTRIBUTES if c in preview.columns]
    if present_protected:
        st.info(f"Columns {present_protected} are in the file but are not model inputs; "
                "the API ignores them.")

    if st.button("Score this file", type="primary", key="score_csv"):
        try:
            with st.spinner(f"Scoring {len(preview):,} rows…"):
                resp = ApiClient(url).predict_batch(upload.name, content)
        except ApiError as e:
            st.session_state.pop(RESULT_KEY, None)
            svc.show_api_error(e, "Scoring the file")
        else:
            st.session_state[RESULT_KEY] = {"file_id": file_id, "resp": resp}

    result = st.session_state.get(RESULT_KEY)
    if not result or result["file_id"] != file_id:
        return
    resp = result["resp"]
    try:
        scored = merge_predictions(preview, resp)
    except ValueError as e:
        st.error(f"The API response could not be matched to the file: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", f"{resp['n_rows']:,}")
    c2.metric("Predicted >50K", f"{resp['n_predicted_positive']:,}")
    share = resp["n_predicted_positive"] / resp["n_rows"] if resp["n_rows"] else 0.0
    c3.metric("Share predicted >50K", f"{share:.1%}")
    c4.metric("Logged to Supabase", f"{resp['logged']:,}")
    if resp.get("ignored_columns"):
        st.caption("Ignored columns: " + ", ".join(resp["ignored_columns"]))
    st.caption(f"Run {resp.get('run_id')} ({resp.get('run_name')}), threshold "
               f"{resp.get('threshold')}, calibration: {resp.get('calibration_method')}.")

    st.dataframe(
        scored, use_container_width=True, hide_index=True,
        column_config={
            "probability_over_50k": st.column_config.ProgressColumn(
                "probability_over_50k", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )
    base = upload.name.rsplit(".", 1)[0]
    st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}_scored.csv", mime="text/csv", key="csv_download")
