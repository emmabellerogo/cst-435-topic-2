"""Model Performance tab: the finished, frozen evaluation of the selected run.

Every number is read from the evaluation results (Supabase `runs`, or the
committed files in models/) by perf_data.py. Nothing is retrained, re-selected
or re-evaluated here; the only arithmetic is formatting for display.
"""
from __future__ import annotations

from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import ui_services as svc
from perf_data import PerformanceDataError

CLASS_NAMES = ["<=50K", ">50K"]
SPLIT_LABELS = {"train": "train", "val": "validation", "test": "test"}
SPLIT_METRICS = [("loss", "BCE loss (uncalibrated, lower is better)"), ("accuracy", "accuracy"),
                 ("precision", "precision (>50K)"), ("recall", "recall (>50K)"),
                 ("f1", "F1 (>50K)"), ("roc_auc", "ROC-AUC")]


def confusion_frame(cm: List[List[int]]) -> pd.DataFrame:
    """[[TN, FP], [FN, TP]] -> long frame for a heatmap."""
    names = [["TN (true negative)", "FP (false positive)"], ["FN (false negative)", "TP (true positive)"]]
    rows = []
    for i, actual in enumerate(CLASS_NAMES):
        for j, predicted in enumerate(CLASS_NAMES):
            count = int(cm[i][j])
            rows.append({"actual": actual, "predicted": predicted, "count": count,
                         "cell": names[i][j], "label": f"{names[i][j].split(' ')[0]}: {count:,}"})
    return pd.DataFrame(rows)


def confusion_chart(cm: List[List[int]]) -> alt.Chart:
    df = confusion_frame(cm)
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=CLASS_NAMES, title="Predicted class"),
        y=alt.Y("actual:N", sort=CLASS_NAMES, title="Actual class"),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["cell", "count"],
    )
    text = base.mark_text(fontSize=15).encode(
        text=alt.Text("label:N"),
        color=alt.condition(alt.datum.count > float(df["count"].max()) / 2,
                            alt.value("white"), alt.value("black")),
    )
    return (rect + text).properties(height=260)


def reliability_frame(cal_split: dict) -> pd.DataFrame:
    rows = []
    for which, key in (("before calibration", "bins_before"), ("after calibration", "bins_after")):
        for b in cal_split.get(key) or []:
            if b.get("count"):
                rows.append({"version": which, "mean predicted probability": b["mean_predicted"],
                             "observed share >50K": b["fraction_positive"], "rows in bin": b["count"]})
    return pd.DataFrame(rows)


def reliability_chart(df: pd.DataFrame) -> alt.Chart:
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
    lines = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("mean predicted probability:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed share >50K:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("version:N", title=None),
        tooltip=["version", "mean predicted probability", "observed share >50K", "rows in bin"],
    )
    return (diag + lines).properties(height=320)


def curves_frame(epochs: List[dict], metric: str) -> pd.DataFrame:
    """Long frame of train and validation values of one metric, by epoch."""
    rows = []
    for e in epochs:
        for split in ("train", "val"):
            value = e.get(f"{split}_{metric}")
            if value is not None:
                rows.append({"epoch": int(e["epoch"]), "split": SPLIT_LABELS[split], metric: value})
    return pd.DataFrame(rows)


def curve_chart(df: pd.DataFrame, metric: str, title: str, best_epoch: Optional[int]) -> alt.Chart:
    lines = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("epoch:Q", title="epoch", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("split:N", title=None,
                        scale=alt.Scale(domain=["train", "validation"], range=["#2b7bb9", "#e67e22"])),
        tooltip=["epoch", "split", alt.Tooltip(f"{metric}:Q", format=".4f")],
    )
    if best_epoch is None:
        return lines.properties(height=260)
    rule = alt.Chart(pd.DataFrame({"epoch": [best_epoch]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="epoch:Q")
    return (lines + rule).properties(height=260)


def split_comparison_frame(splits: dict) -> pd.DataFrame:
    """One row per metric, one column per split, from the stored metrics only."""
    rows = []
    for key, name in SPLIT_METRICS:
        rows.append({"metric": name, **{SPLIT_LABELS[s]: (splits.get(s) or {}).get(key)
                                         for s in ("train", "val", "test")}})
    return pd.DataFrame(rows, columns=["metric", "train", "validation", "test"])


def experiments_frame(experiments: List[dict]) -> pd.DataFrame:
    rows = []
    for rank, e in enumerate(experiments, start=1):
        rows.append({
            "rank (val loss)": rank,
            "config": e["name"] + (" ✅ selected" if e["selected"] else ""),
            "hidden sizes": str(e["hidden_sizes"]),
            "activation": e["activation"],
            "dropout": e["dropout"],
            "best epoch": e["best_epoch"],
            "val loss": e["val_loss"],
            "val accuracy": e["val_accuracy"],
            "val precision": e["val_precision"],
            "val recall": e["val_recall"],
            "val F1": e["val_f1"],
            "val ROC-AUC": e["val_roc_auc"],
        })
    return pd.DataFrame(rows)


def experiments_chart(experiments: List[dict]) -> alt.Chart:
    """Validation loss per config on a zoomed y-axis.

    The axis does not start at 0 (the losses differ only in the 4th decimal),
    so the bars must be clipped: unclipped, each bar still extends down to 0,
    far outside the plot, and Streamlit's fit-to-width sizing then collapses the
    plot area to nothing.
    """
    chart_df = pd.DataFrame({"config": [e["name"] for e in experiments],
                             "validation loss": [e["val_loss"] for e in experiments],
                             "selected": [e["selected"] for e in experiments]})
    lo = min(chart_df["validation loss"]) * 0.995
    hi = max(chart_df["validation loss"]) * 1.003
    return alt.Chart(chart_df).mark_bar(clip=True).encode(
        x=alt.X("config:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("validation loss:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
        color=alt.condition(alt.datum.selected, alt.value("#2b7bb9"), alt.value("#b0b0b0")),
        tooltip=["config", alt.Tooltip("validation loss:Q", format=".4f")],
    ).properties(height=240)


def accuracy_note(experiments: List[dict]) -> Optional[str]:
    """Say so when the run with the best validation accuracy is not the selected one."""
    scored = [e for e in experiments if e.get("val_accuracy") is not None]
    if not scored:
        return None
    top = max(scored, key=lambda e: e["val_accuracy"])
    sel = next((e for e in experiments if e["selected"]), None)
    if sel is None or top is sel or sel.get("val_accuracy") is None:
        return None
    return (f"`{top['name']}` had slightly higher validation accuracy "
            f"({top['val_accuracy']:.4f} vs {sel['val_accuracy']:.4f}), but the selection rule "
            "fixed in advance was validation loss, which measures how good the probabilities are "
            "and not only whether they land on the right side of 0.5.")


def render() -> None:
    st.header("Model Performance")
    try:
        perf = svc.get_performance()
    except PerformanceDataError as e:
        st.error(f"The evaluation results could not be loaded: {e}")
        st.caption("Set SUPABASE_URL and SUPABASE_ANON_KEY in the secrets, or run the app from "
                   "the repository so the files in models/ are available.")
        return

    sel, test = perf["selected"], perf["test"]
    st.caption(f"Source: {perf['source_detail']}.")
    for w in perf.get("warnings") or []:
        st.caption(f"Note: {w}")

    st.write(
        f"Selected model: **{sel['name']}**"
        + (f" (run {sel['run_id']})" if sel.get("run_id") else "")
        + f", hidden layers {sel['hidden_sizes']}, {str(sel['activation']).upper()}, "
        f"dropout {sel['dropout']}, best epoch {sel['best_epoch']}. "
        "It was chosen on the **validation** set; the **test** set was used once, afterwards, "
        "so the numbers below are an honest estimate of performance on new data."
    )

    # -- headline metrics ------------------------------------------------------
    n = test.get("n")
    st.subheader("Test-set results" + (f" ({n:,} rows)" if n else ""))
    cols = st.columns(5)
    cols[0].metric("Accuracy", svc.fmt(test.get("accuracy")))
    cols[1].metric("ROC-AUC", svc.fmt(test.get("roc_auc")))
    cols[2].metric("Precision (>50K)", svc.fmt(test.get("precision")))
    cols[3].metric("Recall (>50K)", svc.fmt(test.get("recall")))
    cols[4].metric("F1 (>50K)", svc.fmt(test.get("f1")))
    st.caption(
        "Threshold 0.5 on the calibrated probability. Accuracy: share of all rows classified "
        "correctly. ROC-AUC: how well the probabilities rank >50K people above ≤50K people "
        "(1.0 is perfect, 0.5 is random). Precision: of the people predicted >50K, the share "
        "who really are. Recall: of the people who really are >50K, the share the model finds."
    )

    render_split_comparison(perf)
    render_learning_curves(perf)

    # -- confusion matrix ------------------------------------------------------
    cm = perf.get("confusion_matrix")
    if cm:
        st.subheader("Confusion matrix")
        left, right = st.columns([3, 2])
        left.altair_chart(confusion_chart(cm), use_container_width=True)
        (tn, fp), (fn, tp) = cm
        right.markdown(
            f"- **TN = {tn:,}**: ≤50K, predicted ≤50K\n"
            f"- **FP = {fp:,}**: ≤50K, wrongly predicted >50K\n"
            f"- **FN = {fn:,}**: >50K, wrongly predicted ≤50K\n"
            f"- **TP = {tp:,}**: >50K, predicted >50K"
        )
        right.write(
            f"**>50K is the harder class.** Of the {fn + tp:,} people who really earn >50K, the "
            f"model misses {fn:,} (false negatives), which is why recall for >50K is lower than "
            "precision. People earning >50K are the minority in the data, so the model has "
            "fewer examples of them and leans toward predicting ≤50K."
        )

    # -- per-class -------------------------------------------------------------
    per_class = perf.get("per_class") or {}
    if per_class:
        st.subheader("Per-class metrics")
        rows = [{"class": name, "precision": m.get("precision"), "recall": m.get("recall"),
                 "F1": m.get("f1"), "support (rows)": m.get("support")}
                for name, m in per_class.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ("precision", "recall", "F1")})

    # -- calibration -------------------------------------------------------------
    cal = perf.get("calibration") or {}
    cal_test = cal.get("test") or {}
    st.subheader("Calibration")
    t = cal.get("temperature")
    st.write(
        f"Method: **{cal.get('method') or 'n/a'}** scaling"
        + (f", T ≈ {t:.4f}" if t else "")
        + f", fitted on the **{cal.get('fitted_on_split') or 'val'}** split only. The calibrated "
        "probability is sigmoid(logit / T). Because T is positive, it never changes which side "
        "of 0.5 a prediction is on, so accuracy and the confusion matrix are unchanged; only the "
        "confidence of the probabilities moves."
    )
    cal_rows = [
        {"metric": "ECE (expected calibration error, lower is better)",
         "before": cal_test.get("ece_before"), "after": cal_test.get("ece_after")},
        {"metric": "Brier score (lower is better)",
         "before": cal_test.get("brier_before"), "after": cal_test.get("brier_after")},
    ]
    st.dataframe(pd.DataFrame(cal_rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("before", "after")})
    caption = "Measured on the test set, which played no part in fitting T."
    if t and abs(t - 1.0) < 0.05:
        caption += (" Before and after are almost identical: the network was already well "
                    "calibrated, so T stayed close to 1.")
    st.caption(caption)
    rel = reliability_frame(cal_test)
    if not rel.empty:
        st.markdown("**Reliability diagram (test set)**")
        st.altair_chart(reliability_chart(rel), use_container_width=True)
        st.caption("Each point is a bin of predictions. A perfectly calibrated model lies on the "
                   "dashed diagonal: among people given a 70% probability, 70% really earn >50K.")

    # -- experiments ---------------------------------------------------------------
    experiments = [e for e in (perf.get("experiments") or []) if e.get("val_loss") is not None]
    if experiments:
        st.subheader("Experiment comparison (validation set)")
        st.write(
            "Three configurations were trained under matched controls: the same train/validation/"
            "test split, random seed, learning rate, batch size, epoch budget, weight decay and "
            "early stopping. Each comparison changes one thing: `gelu` differs from `baseline` "
            "only in the activation, and `deep` only in the hidden layer sizes."
        )
        st.dataframe(
            experiments_frame(experiments), hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.4f")
                           for c in ("val loss", "val accuracy", "val precision", "val recall",
                                     "val F1", "val ROC-AUC")},
        )
        st.caption(f"Selection rule: {perf.get('selection_criterion')}. The test set played no "
                   "part in choosing the model.")
        note = accuracy_note(experiments)
        if note:
            st.info(note)
        st.altair_chart(experiments_chart(experiments), use_container_width=True)
        losses = [e["val_loss"] for e in experiments]
        spread = max(losses) - min(losses)
        st.caption(f"The y-axis is zoomed in: the best and worst validation losses differ by only "
                   f"{spread:.4f}, so the configurations perform almost the same.")

    # -- importance ----------------------------------------------------------------
    imp = perf.get("importance") or []
    if imp:
        st.subheader("Permutation importance (test set)")
        df = pd.DataFrame(imp)
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("importance_mean:Q", title="drop in ROC-AUC when the feature is shuffled"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("type:N", title=None),
                tooltip=["feature", "type", alt.Tooltip("importance_mean:Q", format=".4f"),
                         alt.Tooltip("importance_std:Q", format=".4f")],
            ).properties(height=320),
            use_container_width=True,
        )
        st.caption("Each of the 10 original features was shuffled (10 repeats) to break its link "
                   "with the target, and the drop in ROC-AUC was measured. A bigger drop means the "
                   "model relies on that feature more. This shows what the model uses, not what "
                   "causes income.")


def render_split_comparison(perf: dict) -> None:
    splits = perf.get("splits") or {}
    sel = perf["selected"]
    st.subheader(f"Train / validation / test comparison ({sel['name']})")
    if not any(splits.get(s) for s in ("train", "val", "test")):
        st.info("The stored results for this run do not include per-split metrics, so the "
                "comparison cannot be shown.")
        return
    st.dataframe(split_comparison_frame(splits), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in ("train", "validation", "test")})
    missing = [SPLIT_LABELS[s] for s in ("train", "val", "test") if not splits.get(s)]
    note = (f"All three columns describe the same saved checkpoint (best epoch "
            f"{sel['best_epoch']}), scored in evaluation mode (dropout off) at threshold 0.5 on "
            "the uncalibrated probability. Train and validation values were recorded during "
            "training; the test column was computed once, after the model was selected. "
            "Temperature scaling does not change the predicted labels or the ranking, so "
            "accuracy, precision, recall, F1 and ROC-AUC are the same after calibration.")
    cal_loss = (splits.get("test") or {}).get("loss_calibrated")
    if cal_loss is not None:
        note += f" Test BCE loss after calibration: {cal_loss:.4f}."
    if missing:
        note += f" Not stored for: {', '.join(missing)}."
    st.caption(note)
    train, val, test = (splits.get(s) or {} for s in ("train", "val", "test"))
    if train.get("accuracy") is not None and test.get("accuracy") is not None:
        gap = train["accuracy"] - test["accuracy"]
        st.write(
            f"Train accuracy is {gap:+.4f} relative to test"
            + (f" and validation is {val['accuracy'] - test['accuracy']:+.4f}"
               if val.get("accuracy") is not None else "")
            + ". A small gap like this means the network is not badly overfitting: it does "
            "about as well on people it never trained on as on the training rows."
            if abs(gap) < 0.02 else
            f"Train accuracy is {gap:+.4f} relative to test, a noticeable gap that suggests "
            "some overfitting to the training rows."
        )


def render_learning_curves(perf: dict) -> None:
    hist = perf.get("history") or {}
    sel = perf["selected"]
    st.subheader("Learning curves (train vs validation, by epoch)")
    if not hist.get("available"):
        st.info(f"Per-epoch history is unavailable for this run: {hist.get('detail') or 'not saved'}. "
                "No curves are drawn rather than inventing them.")
        return
    epochs = hist["epochs"]
    best = sel.get("best_epoch")
    left, right = st.columns(2)
    left.altair_chart(curve_chart(curves_frame(epochs, "loss"), "loss", "BCE loss", best),
                      use_container_width=True)
    right.altair_chart(curve_chart(curves_frame(epochs, "accuracy"), "accuracy", "accuracy", best),
                       use_container_width=True)
    st.caption(
        f"Source: {hist['detail']}. Both curves are measured after each epoch in evaluation "
        "mode (dropout off) on the full train and validation splits. The dashed line marks "
        f"epoch {best}, the lowest validation loss, whose weights were kept; training ran "
        f"{len(epochs)} epochs in total. The test set is not shown here because it was never "
        "looked at during training."
    )
