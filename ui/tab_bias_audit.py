"""Bias Audit tab: false-positive and false-negative rates by sex.

Two read paths to the same SQL view, v_fairness_audit:
  1. GET /audit -- the API reads the view and passes its rows through.
  2. A direct, read-only SELECT on the view with the public anon key (001_init.sql
     grants anon SELECT on this aggregate view; it holds counts per group only).

The view only counts predictions linked to labeled held-out (test) rows, so
FPR/FNR never involve the unlabeled predictions people make in this app. This
tab displays FPR and FNR as received and never recomputes them; the only
arithmetic is base rates and predicted-positive rates from the view's counts.
"""
from __future__ import annotations

from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import ui_services as svc
from api_client import ApiError
from perf_data import FAIRNESS_VIEW, UnsafeKeyError

COLUMNS = ["group_value", "n", "tp", "fp", "tn", "fn", "fpr", "fnr"]


def groups_frame(groups: List[dict]) -> pd.DataFrame:
    """The API's group rows, in the API's own column names and values."""
    return pd.DataFrame([{c: g.get(c) for c in COLUMNS} for g in groups], columns=COLUMNS)


def direct_frame(rows: List[dict]) -> pd.DataFrame:
    """The anon-key view rows plus counts derived from them (FPR/FNR are copied, not recomputed)."""
    out = []
    for r in rows:
        n, tp, fp, fn = (int(r.get(k) or 0) for k in ("n", "tp", "fp", "fn"))
        out.append({
            "group": r.get("group_value"),
            "labeled test rows": n,
            "actually >50K": tp + fn,
            "base rate (>50K)": (tp + fn) / n if n else None,
            "predicted >50K": tp + fp,
            "predicted-positive rate": (tp + fp) / n if n else None,
            "FPR (view)": r.get("fpr"),
            "FNR (view)": r.get("fnr"),
        })
    return pd.DataFrame(out)


def compare_sources(api_groups: List[dict], view_rows: List[dict]) -> List[str]:
    """Differences between the /audit rows and the direct view rows (empty = identical)."""
    a = {g.get("group_value"): g for g in api_groups}
    b = {r.get("group_value"): r for r in view_rows}
    diffs = []
    for group in sorted(set(a) | set(b), key=str):
        if group not in a or group not in b:
            diffs.append(f"group {group!r} is only in {'/audit' if group in a else 'the direct read'}")
            continue
        for k in ("n", "tp", "fp", "tn", "fn", "fpr", "fnr"):
            x, y = a[group].get(k), b[group].get(k)
            if (x is None) != (y is None) or (x is not None and abs(float(x) - float(y)) > 1e-9):
                diffs.append(f"{group} {k}: /audit {x} vs direct {y}")
    return diffs


def base_rate_sentence(df: pd.DataFrame) -> Optional[str]:
    """Compare the groups' base rates using the view's own counts."""
    if len(df) != 2 or not {"tp", "fn", "n"} <= set(df.columns):
        return None
    rates = {g["group_value"]: (g["tp"] + g["fn"]) / g["n"] for _, g in df.iterrows() if g["n"]}
    if len(rates) != 2:
        return None
    (g1, r1), (g2, r2) = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    return (f"Base rates differ: {r1:.1%} of {g1} test rows really earn >50K, compared with "
            f"{r2:.1%} of {g2} rows.")


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
    audit = None
    if not url:
        svc.no_api_url_message()
    else:
        try:
            audit = svc.get_audit(url)
        except ApiError as e:
            svc.show_api_error(e, "Loading the audit from /audit")

    df = None
    if audit is not None:
        groups = audit.get("groups") or []
        st.caption(f"Run {audit.get('run_id')} ({audit.get('run_name')}), attribute: "
                   f"`{audit.get('attribute')}`, split: `{audit.get('split')}`, source: "
                   f"`{audit.get('source')}` via the API's `/audit`.")
        if groups:
            df = groups_frame(groups)
            render_api_rates(df)
        else:
            st.info(audit.get("note") or "The audit view has no rows for this run yet.")

    render_direct_read(audit)
    if df is not None:
        render_interpretation(df)


def render_api_rates(df: pd.DataFrame) -> None:
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


def render_direct_read(audit: Optional[dict]) -> None:
    st.subheader("Breakdown by sex, read directly from Supabase")
    st.write(
        f"The same `{FAIRNESS_VIEW}` view, queried straight from this page with the public "
        "**anon** key (read-only), without going through the API. The view returns one row of "
        "counts per group, never individual records, and only counts predictions linked to a "
        "labeled test row. Predictions made on **Score a Row** or **Score CSV** have no true "
        "label, so they are not part of these numbers."
    )
    run_id = (audit or {}).get("run_id")
    if run_id is None:
        perf = svc.try_performance()
        run_id = (perf or {}).get("selected", {}).get("run_id")
    if run_id is None:
        st.info("The served run id is unknown (the API and the Supabase runs table were both "
                "unavailable), so the direct read was skipped.")
        return
    try:
        rows = svc.get_fairness_view(run_id)
    except UnsafeKeyError as e:
        st.error(f"Direct read refused: {e}. Replace it with the anon key in the app's secrets.")
        return
    except Exception as e:  # noqa: BLE001 - any client/permission error is shown, not raised
        st.warning(f"The direct read of `{FAIRNESS_VIEW}` failed ({type(e).__name__}: {e}). "
                   "The anon role needs SELECT on this view, which db/migrations/001_init.sql "
                   "grants; check that the migration's grants were applied.")
        return
    if rows is None:
        st.info("SUPABASE_URL and SUPABASE_ANON_KEY are not configured for this app, so the "
                "direct read was skipped. Add the public anon key (never the service-role key) "
                "to the secrets to enable it.")
        return
    if not rows:
        st.info(f"`{FAIRNESS_VIEW}` returned no rows for run {run_id}.")
        return

    direct = direct_frame(rows)
    st.dataframe(
        direct, hide_index=True, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.4f")
                       for c in ("base rate (>50K)", "predicted-positive rate",
                                 "FPR (view)", "FNR (view)")},
    )
    st.caption(f"Source: Supabase `{FAIRNESS_VIEW}` (anon key, SELECT only), run {run_id}, "
               "test split. FPR and FNR are the view's values; base rate = (TP + FN) / n and "
               "predicted-positive rate = (TP + FP) / n are computed here from its counts.")
    if audit and audit.get("groups"):
        diffs = compare_sources(audit["groups"], rows)
        if diffs:
            st.warning("The direct read and `/audit` disagree: " + "; ".join(diffs))
        else:
            st.success("The direct read matches the `/audit` numbers exactly.")


def render_interpretation(df: pd.DataFrame) -> None:
    st.subheader("What this means")
    for sentence in comparison_sentences(df):
        st.markdown(f"- {sentence}")
    base = base_rate_sentence(df)
    if base:
        st.markdown(f"- {base}")
    st.write(
        "The model never sees `sex` or `race`: they are not in the input form, the CSV columns "
        "or the model's features, and are kept in the database only for this audit. Leaving them "
        "out does **not** guarantee fair outcomes. Other features can act as **proxies**: "
        "`relationship` (Husband/Wife) and `marital_status` are closely tied to sex, and "
        "`occupation` and hours worked differ between groups in this 1994 data. A model can "
        "therefore treat groups differently without ever being told the group."
    )
    st.write(
        "Different base rates also matter. A model that is equally accurate overall can still "
        "produce different error rates when the groups start from different base rates, which "
        "is why both FPR and FNR are reported instead of a single accuracy number. Here, women "
        "who really earn >50K are missed more often (higher FNR), while men who earn ≤50K are "
        "more often wrongly predicted >50K (higher FPR)."
        if _women_higher_fnr_men_higher_fpr(df) else
        "Different base rates also matter. A model that is equally accurate overall can still "
        "produce different error rates when the groups start from different base rates, which "
        "is why both FPR and FNR are reported instead of a single accuracy number."
    )

    st.subheader("Possible mitigations (not implemented)")
    st.markdown(
        "None of these has been applied: the numbers above describe the model as it is served. "
        "Each option trades some overall accuracy or simplicity for smaller gaps, and would "
        "have to be tuned on the validation split and re-checked on test.\n"
        "- **Group-aware thresholds (post-processing).** Choose a separate cut-off per group "
        "on validation data so that FNR (equal opportunity) or both FPR and FNR (equalized "
        "odds) are closer. This needs `sex` at decision time, which raises legal and ethical "
        "questions of its own.\n"
        "- **Reweighting the training data (pre-processing).** Give each (sex, income) "
        "combination equal total weight in the loss, so under-represented cases such as women "
        "earning >50K count more during training.\n"
        "- **A fairness penalty during training (in-processing).** Add a term to the loss that "
        "grows with the FNR or FPR gap between groups.\n"
        "- **Testing the proxies.** Retrain without `relationship` (or with it merged) and "
        "measure how both accuracy and the gaps change.\n"
        "- **Better evidence.** Report confidence intervals and repeat over several seeds and "
        "splits before drawing conclusions from one test split."
    )
    st.caption("This audit covers one attribute (sex) on one test split at one threshold (0.5). "
               "It describes the model's behavior; it is not a certification that the model is fair.")


def _women_higher_fnr_men_higher_fpr(df: pd.DataFrame) -> bool:
    g = {r["group_value"]: r for _, r in df.iterrows()}
    if set(g) != {"Female", "Male"} or df[["fpr", "fnr"]].isna().any().any():
        return False
    return g["Female"]["fnr"] > g["Male"]["fnr"] and g["Male"]["fpr"] > g["Female"]["fpr"]
