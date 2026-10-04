"""Model Card tab: a one-page summary of what the model is, how it was built,
how well it works, where it is unfair and what it must not be used for.

Live values come from /version, /schema and /audit and from the stored
evaluation results (perf_data). Facts that no endpoint exposes (the dataset
size and split, the excluded columns) come from the backend handoff and are
kept in DATASET_FACTS / EXCLUDED_FEATURES below so they are easy to check.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

import ui_services as svc
from api_client import ApiError

# From the backend handoff (db/load.py output); no API endpoint returns these.
DATASET_FACTS = {
    "name": "UCI Adult Income (extracted from the 1994 U.S. Census database)",
    "rows": 48_842,
    "train": 34_188,
    "val": 7_327,
    "test": 7_327,
    "split_rule": "fixed 70/15/15 split with seed 42, stratified on income and sex, "
                  "assigned once when the data was loaded into Supabase",
}
EXCLUDED_FEATURES = [
    ("education", "duplicates education_num (the same information as a number)"),
    ("fnlwgt", "a census sampling weight, not a characteristic of the person"),
    ("sex", "protected attribute; kept in Supabase only for the fairness audit"),
    ("race", "protected attribute; kept in Supabase only for the fairness audit"),
]


def _schema_or_none(url: Optional[str]) -> Optional[dict]:
    if not url:
        return None
    try:
        return svc.get_schema(url)
    except ApiError:
        return None


def _audit_or_none(url: Optional[str]) -> Optional[dict]:
    if not url:
        return None
    try:
        return svc.get_audit(url)
    except ApiError:
        return None


def render(url: Optional[str]) -> None:
    st.header("Model Card")
    version = svc.try_version(url) or {}
    model = version.get("model") or {}
    perf = svc.try_performance()
    schema = _schema_or_none(url)
    audit = _audit_or_none(url)

    missing = [name for name, v in (("/version", version), ("evaluation results", perf),
                                    ("/schema", schema), ("/audit", audit)) if not v]
    if missing:
        st.caption("Some live values could not be loaded (" + ", ".join(missing)
                   + "); those parts are shown as n/a.")

    sel = (perf or {}).get("selected") or {}
    hidden = model.get("hidden_sizes") or sel.get("hidden_sizes")
    activation = model.get("activation") or sel.get("activation")
    n_enc = model.get("n_encoded_features")

    # -- at a glance -------------------------------------------------------------
    st.subheader("Model at a glance")
    glance = {
        "Selected model": f"{version.get('run_name') or sel.get('name') or 'n/a'} "
                          f"(Supabase run id {version.get('run_id') or sel.get('run_id') or 'n/a'})",
        "Architecture": (f"PyTorch MLP: {n_enc or 'n/a'} inputs → "
                         + " → ".join(str(h) for h in (hidden or [])) + " → 1 logit, "
                         f"{str(activation or 'n/a').upper()} activations, dropout "
                         f"{model.get('dropout', sel.get('dropout', 'n/a'))}"),
        "Best epoch": str(model.get("best_epoch") or sel.get("best_epoch") or "n/a"),
        "Output": f"calibrated P(income > $50K) = sigmoid(logit / T), "
                  f"T ≈ {svc.fmt(model.get('temperature'))}; label >50K when P ≥ "
                  f"{model.get('threshold', 0.5)}",
        "Model version": f"API {version.get('api_version', 'n/a')}, build "
                         f"{version.get('git_sha', 'n/a')}, torch {version.get('torch_version', 'n/a')}, "
                         f"scikit-learn {version.get('sklearn_version', 'n/a')}",
    }
    st.table(pd.DataFrame({"": list(glance), "value": list(glance.values())}).set_index(""))

    # -- purpose -------------------------------------------------------------------
    st.subheader("Purpose and intended use")
    st.write(
        "Income Insight is a CST-435 course project. It demonstrates a complete deep-learning "
        "product: real data in a database, a trained and calibrated neural network served by an "
        "API, and a web interface with a fairness audit. It is intended for **learning and "
        "demonstration**: exploring how census features relate to income in the Adult dataset "
        "and how to evaluate and audit a classifier."
    )
    st.subheader("Inappropriate uses")
    st.markdown(
        "Do **not** use this model to make or support decisions about real people, including:\n"
        "- hiring, promotion, pay or performance decisions;\n"
        "- lending, credit, insurance, housing or rental decisions;\n"
        "- eligibility for benefits, services or programs;\n"
        "- estimating a specific person's actual income.\n\n"
        "The data is from 1994, the \\$50K threshold is not adjusted for inflation, and the "
        "fairness audit shows unequal error rates between groups."
    )

    # -- data ------------------------------------------------------------------------
    st.subheader("Dataset")
    st.write(
        f"{DATASET_FACTS['name']}, **{DATASET_FACTS['rows']:,} rows** stored in the Supabase "
        f"`adult_income` table. Split: **{DATASET_FACTS['train']:,} train / "
        f"{DATASET_FACTS['val']:,} validation / {DATASET_FACTS['test']:,} test**, a "
        f"{DATASET_FACTS['split_rule']}. The target is whether income is >50K (positive class) "
        "or ≤50K."
    )

    st.subheader("Features")
    if schema:
        feats = [f for f in schema.get("features", []) if f.get("name") not in svc.PROTECTED_ATTRIBUTES]
        st.dataframe(pd.DataFrame([
            {"feature": f["name"], "type": f.get("kind"),
             "allowed values": (f"{f.get('minimum')}–{f.get('maximum')}" if f.get("kind") == "numeric"
                                else f"{len(f.get('categories') or [])} categories")}
            for f in feats]), hide_index=True, use_container_width=True)
    else:
        st.write("Feature list unavailable (/schema could not be reached).")
    st.markdown("**Excluded from the model's inputs**")
    st.dataframe(pd.DataFrame(EXCLUDED_FEATURES, columns=["column", "why it is excluded"]),
                 hide_index=True, use_container_width=True)
    st.write(
        "Preprocessing (fitted on the training split only, then frozen): numeric features get "
        "median imputation and standard scaling; categorical features get missing values "
        "replaced by `Unknown` and one-hot encoding"
        + (f". The 10 raw features become {n_enc} model inputs." if n_enc else ".")
    )

    # -- training & selection ----------------------------------------------------------
    st.subheader("Training and model selection")
    names = ", ".join(f"`{e['name']}`" for e in (perf or {}).get("experiments", [])) or "n/a"
    st.write(
        f"Three configurations ({names}) were trained under matched controls (same split, seed, "
        "learning rate, batch size, epoch budget, weight decay and early stopping) with AdamW "
        "and binary cross-entropy. Each run kept the epoch with the lowest validation loss. The "
        "configuration with the lowest **validation loss** was selected; the test set was not "
        "used for any choice. Temperature scaling was then fitted on the validation set, and "
        "the test set was evaluated once."
    )

    # -- metrics -------------------------------------------------------------------
    st.subheader("Final test metrics")
    if perf:
        test = perf["test"]
        cm = perf.get("confusion_matrix")
        cal = (perf.get("calibration") or {}).get("test") or {}
        metric_rows = [
            ("Accuracy", test.get("accuracy")), ("ROC-AUC", test.get("roc_auc")),
            ("Precision (>50K)", test.get("precision")), ("Recall (>50K)", test.get("recall")),
            ("F1 (>50K)", test.get("f1")),
            ("ECE before → after calibration",
             f"{svc.fmt(cal.get('ece_before'))} → {svc.fmt(cal.get('ece_after'))}"),
            ("Brier before → after calibration",
             f"{svc.fmt(cal.get('brier_before'))} → {svc.fmt(cal.get('brier_after'))}"),
        ]
        st.table(pd.DataFrame([{"metric": m, "value": v if isinstance(v, str) else svc.fmt(v)}
                               for m, v in metric_rows]).set_index("metric"))
        if cm:
            (tn, fp), (fn, tp) = cm
            st.caption(f"Confusion matrix [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]. "
                       f"The {fn:,} false negatives make >50K the harder class.")
    else:
        st.write("Evaluation results unavailable.")

    # -- importance --------------------------------------------------------------------
    st.subheader("What the model relies on (permutation importance)")
    imp = (perf or {}).get("importance") or []
    if imp:
        top = sorted(imp, key=lambda f: f.get("rank", 99))[:3]
        st.markdown("\n".join(
            f"{i}. **{f['feature']}**: ROC-AUC drops by about {f['importance_mean']:.4f} when shuffled"
            for i, f in enumerate(top, start=1)))
        st.write(
            "Importance is measured on the 10 original features (all one-hot columns of a "
            "categorical feature are shuffled together). It shows what the model uses, not what "
            "causes income. The top feature, `marital_status`, is also closely tied to sex, "
            "which matters for the fairness findings below."
            if top and top[0]["feature"] == "marital_status" else
            "Importance is measured on the 10 original features (all one-hot columns of a "
            "categorical feature are shuffled together). It shows what the model uses, not what "
            "causes income."
        )
    else:
        st.write("Permutation importance unavailable.")

    # -- fairness ------------------------------------------------------------------------
    st.subheader("Fairness findings")
    groups = (audit or {}).get("groups") or []
    if groups:
        st.table(pd.DataFrame([{"group": g.get("group_value"), "test rows": g.get("n"),
                                "FPR": svc.fmt(g.get("fpr")), "FNR": svc.fmt(g.get("fnr"))}
                               for g in groups]).set_index("group"))
        st.write(
            "Rates are from the SQL view `v_fairness_audit` (see the Bias Audit tab). The model "
            "does not use sex or race as inputs, yet error rates differ between women and men, "
            "because features such as relationship, marital status and occupation act as proxies."
        )
    else:
        st.write("Audit results unavailable.")

    # -- limitations -------------------------------------------------------------------
    st.subheader("Limitations")
    st.markdown(
        "- **Old data**: 1994 census records; incomes, jobs and the meaning of \\$50K have "
        "changed since.\n"
        "- **Class imbalance**: >50K is the minority class and has many false negatives.\n"
        "- **Unequal errors**: false-positive and false-negative rates differ by sex; proxies "
        "for protected attributes remain in the inputs.\n"
        "- **Calibration is overall, not per group**: a probability can be well calibrated on "
        "average and still be off for one group.\n"
        "- **Narrow comparison**: three configurations, one seed; the differences between them "
        "are small.\n"
        "- **Fixed threshold**: 0.5 was not tuned, so precision and recall reflect that choice.\n"
        "- **Population**: mostly U.S.-born respondents; other countries have few examples."
    )

    st.subheader("What is logged")
    st.write(
        "Each prediction stores a SHA-256 hash of the 10 inputs, the predicted label, the "
        "probability and the run id in the Supabase `predictions` table. The raw inputs and any "
        "protected attributes are not stored with user predictions."
    )
