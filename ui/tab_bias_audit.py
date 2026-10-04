"""Bias Audit tab: false-positive and false-negative rates by sex, from GET /audit.

The rates are computed in SQL by the Supabase view v_fairness_audit and passed
through unchanged by the API. This tab displays them as received and never
recomputes FPR or FNR.
"""
from __future__ import annotations

from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import ui_services as svc
from api_client import ApiError

COLUMNS = ["group_value", "n", "tp", "fp", "tn", "fn", "fpr", "fnr"]


def groups_frame(groups: List[dict]) -> pd.DataFrame:
    """The API's group rows, in the API's own column names and values."""
    return pd.DataFrame([{c: g.get(c) for c in COLUMNS} for g in groups], columns=COLUMNS)


def rates_chart(df: pd.DataFrame) -> alt.Chart:
    long = df.melt(id_vars="group_value", value_vars=["fpr", "fnr"], var_name="rate", value_name="value")
    long["rate"] = long["rate"].map({"fpr": "False-positive rate (FPR)", "fnr": "False-negative rate (FNR)"})
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("group_value:N", title=None),
        y=alt.Y("value:Q", title="rate", axis=alt.Axis(format="%")),
        color=alt.Color("group_value:N", title=None, legend=None),
        column=alt.Column("rate:N", title=None),
        tooltip=["group_value", "rate", alt.Tooltip("value:Q", format=".4f")],
    ).properties(width=220, height=260)


def comparison_sentences(df: pd.DataFrame) -> List[str]:
    """Plain-language comparison of the API's values (only reads them, never recomputes)."""
    if len(df) != 2 or df["fpr"].isna().any() or df["fnr"].isna().any():
        return []
    out = []
    for col, name, meaning in (
        ("fpr", "false-positive rate",
         "people who really earn ≤50K are more often wrongly labeled >50K"),
        ("fnr", "false-negative rate",
         "people who really earn >50K are more often missed (labeled ≤50K)"),
    ):
        hi = df.loc[df[col].idxmax()]
        lo = df.loc[df[col].idxmin()]
        if hi[col] == lo[col]:
            out.append(f"Both groups have the same {name} ({hi[col]:.1%}).")
        else:
            out.append(f"**{hi['group_value']}** has the higher {name}: {hi[col]:.1%} vs "
                       f"{lo[col]:.1%} for {lo['group_value']}. In {hi['group_value'].lower()} "
                       f"rows, {meaning}.")
    return out


def render(url: Optional[str]) -> None:
    st.header("Bias Audit")
    st.write(
        "Does the model make different kinds of mistakes for different groups? This page "
        "compares women and men on the labeled **test** rows, where the true income is known. "
        "The rates are computed in SQL by the Supabase view `v_fairness_audit` and returned by "
        "the API's `/audit` endpoint; they are shown here exactly as received."
    )
    if not url:
        svc.no_api_url_message()
        return
    try:
        audit = svc.get_audit(url)
    except ApiError as e:
        svc.show_api_error(e, "Loading the audit from /audit")
        return

    groups = audit.get("groups") or []
    st.caption(f"Run {audit.get('run_id')} ({audit.get('run_name')}), attribute: "
               f"`{audit.get('attribute')}`, split: `{audit.get('split')}`, source: "
               f"`{audit.get('source')}`.")
    if not groups:
        st.info(audit.get("note") or "The audit view has no rows for this run yet.")
        return

    df = groups_frame(groups)
    st.subheader("Rates by group")
    cols = st.columns(len(df))
    for col, (_, g) in zip(cols, df.iterrows()):
        col.markdown(f"**{g['group_value']}** ({int(g['n']):,} test rows)")
        col.metric("False-positive rate (FPR)", svc.fmt(g["fpr"]))
        col.metric("False-negative rate (FNR)", svc.fmt(g["fnr"]))
    st.altair_chart(rates_chart(df), use_container_width=False)

    st.subheader("Counts from the view")
    st.dataframe(
        df.rename(columns={"group_value": "group", "n": "n (rows)", "tp": "TP", "fp": "FP",
                           "tn": "TN", "fn": "FN", "fpr": "FPR", "fnr": "FNR"}),
        hide_index=True, use_container_width=True,
        column_config={"FPR": st.column_config.NumberColumn(format="%.4f"),
                       "FNR": st.column_config.NumberColumn(format="%.4f")},
    )
    st.markdown(
        "- **FPR = FP / (FP + TN)**: of the people who really earn ≤50K, the share the model "
        "wrongly labels >50K.\n"
        "- **FNR = FN / (FN + TP)**: of the people who really earn >50K, the share the model "
        "misses.\n"
        "- TP/FP/TN/FN are true positives, false positives, true negatives and false "
        "negatives, with >50K as the positive class."
    )

    st.subheader("What this means")
    for sentence in comparison_sentences(df):
        st.markdown(f"- {sentence}")
    st.write(
        "The model never sees `sex` or `race`: they are not in the input form, the CSV columns "
        "or the model's features, and are kept in the database only for this audit. Leaving them "
        "out does **not** guarantee fair outcomes. Other features can act as **proxies**: "
        "`relationship` (Husband/Wife) and `marital_status` are closely tied to sex, and "
        "`occupation` and hours worked differ between groups in this 1994 data. A model can "
        "therefore treat groups differently without ever being told the group."
    )
    st.write(
        "The groups also start from different base rates: in this data a larger share of men "
        "than women earn >50K. A model that is equally accurate overall can still produce "
        "different error rates when base rates differ, which is why both FPR and FNR are "
        "reported instead of a single accuracy number."
    )
    st.caption("This audit covers one attribute (sex) on one test split at one threshold (0.5). "
               "It describes the model's behavior; it is not a certification that the model is fair.")
