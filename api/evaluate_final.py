"""Final evaluation of the selected run: calibrate on val, then score test once.

Usage (from the repo root, with a .env holding the service-role key):

    python -m api.evaluate_final --run gelu

Steps:
  1. refuse unless <run> is the validation winner in
     models/experiments/controlled_comparison.json and its test split has not
     been evaluated yet (no evaluation.json, empty test_metrics in run.json)
  2. load the saved model.pt and preprocessor.joblib unchanged (nothing is refit)
  3. read the VAL rows from Supabase and check the loaded model reproduces the
     val loss recorded in run.json
  4. fit temperature scaling on val logits only; save calibrator.json and
     calibration.json. From here on the calibrator is frozen.
  5. only then read the TEST rows, and score them once at the fixed 0.5
     threshold: loss, accuracy, ROC-AUC, confusion matrix, per-class metrics,
     Brier/ECE before and after calibration, reliability bins, per-row
     calibrated probabilities
  6. permutation importance of the 10 original features on the same test rows
  7. write evaluation.json, test_predictions.csv, permutation_importance.json and
     the complete runs-table row (is_best = true) to run.json

There are no threshold, calibration or binning options: nothing about the
evaluation can be adjusted after seeing test numbers. Nothing is written to
Supabase; api.persist_runs does that later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from api.calibration import (
    CALIBRATOR_FILENAME,
    DECISION_THRESHOLD,
    DEFAULT_N_BINS,
    FIT_SPLIT,
    METHOD,
    TemperatureScaler,
    binary_nll,
    calibration_summary,
)
from api.model import MLP, load_checkpoint
from api.persist_runs import DEFAULT_COMPARISON, complete_run_row, validation_winner
from api.preprocessing import load_preprocessor, split_xy, transform_features
from api.train import MODELS_DIR, _repo_relative, load_adult_income
from shared.features import FEATURE_COLS, NUMERIC_COLS, TARGET_CLASSES

TEST_SPLIT = "test"
CALIBRATION_FILENAME = "calibration.json"
EVALUATION_FILENAME = "evaluation.json"
IMPORTANCE_FILENAME = "permutation_importance.json"
PREDICTIONS_FILENAME = "test_predictions.csv"
IMPORTANCE_REPEATS = 10
IMPORTANCE_SEED = 42
VAL_LOSS_TOLERANCE = 1e-4
CONFUSION_LAYOUT = ("[[TN, FP], [FN, TP]]: rows = true class, columns = predicted class; "
                    f"0 = '{TARGET_CLASSES[0]}', 1 = '{TARGET_CLASSES[1]}'")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require_split(df: pd.DataFrame, split: str) -> None:
    """Raise unless every row of ``df`` belongs to ``split``."""
    if "split" not in df.columns:
        raise ValueError("rows need the 'split' column")
    found = set(df["split"].unique())
    if found != {split}:
        raise ValueError(f"expected only {split!r} rows, got splits {sorted(map(str, found))}")


@torch.no_grad()
def pipeline_logits(model: MLP, preprocessor: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Raw rows -> fitted preprocessor (transform only) -> MLP logits, eval mode."""
    model.eval()
    X = transform_features(preprocessor, df)
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def predict_labels(prob) -> np.ndarray:
    """0/1 labels at the fixed DECISION_THRESHOLD (0.5)."""
    return (np.asarray(prob) >= DECISION_THRESHOLD).astype(int)


def confusion_matrix_2x2(y_true, y_pred) -> List[List[int]]:
    """[[TN, FP], [FN, TP]] as plain ints."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]


def per_class_metrics(y_true, y_pred) -> Dict[str, dict]:
    """Precision / recall / F1 / support for each class, keyed by class name."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        TARGET_CLASSES[i]: {"label": i, "precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
        for i in (0, 1)
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


# ---------------------------------------------------------------------------
# 1. calibration (validation only)
# ---------------------------------------------------------------------------
def fit_calibration(model: MLP, preprocessor: Pipeline, val_df: pd.DataFrame,
                    n_bins: int = DEFAULT_N_BINS) -> Tuple[TemperatureScaler, dict]:
    """Fit temperature scaling on VALIDATION rows only; return (calibrator, val report).

    The after-calibration numbers here are in-sample (val also fit T); the
    out-of-sample check is the test section of evaluation.json.
    """
    _require_split(val_df, FIT_SPLIT)
    _, y = split_xy(val_df)
    logits = pipeline_logits(model, preprocessor, val_df)
    calibrator = TemperatureScaler().fit(logits, y)
    report = {
        "method": METHOD,
        "fitted_on_split": FIT_SPLIT,
        "temperature": calibrator.temperature,
        "nll_before": calibrator.nll_before,
        "nll_after": calibrator.nll_after,
        "split": FIT_SPLIT,
        **calibration_summary(y, expit(logits), calibrator.predict_proba(logits), n_bins),
        "note": "validation 'after' numbers are in-sample; see evaluation.json for test",
    }
    return calibrator, report


# ---------------------------------------------------------------------------
# 2. test evaluation (frozen pipeline, once)
# ---------------------------------------------------------------------------
def evaluate_test(model: MLP, preprocessor: Pipeline, calibrator: TemperatureScaler,
                  test_df: pd.DataFrame,
                  n_bins: int = DEFAULT_N_BINS) -> Tuple[dict, pd.DataFrame]:
    """Score the frozen pipeline on TEST rows. Fits nothing.

    Returns (evaluation dict for evaluation.json, per-row predictions frame).
    Labels use the fixed 0.5 threshold on calibrated probabilities; temperature
    scaling keeps the sign of every logit, so they equal the uncalibrated labels.
    """
    if not isinstance(calibrator, TemperatureScaler) or not calibrator.is_fitted:
        raise RuntimeError("fit the calibrator on validation before touching test")
    _require_split(test_df, TEST_SPLIT)
    _, y = split_xy(test_df)
    logits = pipeline_logits(model, preprocessor, test_df)
    prob_raw = expit(logits)
    prob_cal = calibrator.predict_proba(logits)
    y_pred = predict_labels(prob_cal)

    metrics = {
        "loss": binary_nll(logits, y),  # same BCE as val_loss (uncalibrated logits)
        "loss_calibrated": binary_nll(calibrator.calibrate_logits(logits), y),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),  # class 1 (>50K)
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob_cal)),
        "threshold": DECISION_THRESHOLD,
        "n": int(len(y)),
    }
    evaluation = {
        "split": TEST_SPLIT,
        "n": int(len(y)),
        "threshold": DECISION_THRESHOLD,
        "positive_class": TARGET_CLASSES[1],
        "calibration_method": METHOD,
        "temperature": calibrator.temperature,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_2x2(y, y_pred),
        "confusion_matrix_layout": CONFUSION_LAYOUT,
        "per_class": per_class_metrics(y, y_pred),
        "calibration": calibration_summary(y, prob_raw, prob_cal, n_bins),
    }
    predictions = pd.DataFrame({
        "adult_income_id": test_df["id"].to_numpy() if "id" in test_df.columns else None,
        "y_true": y,
        "logit": logits,
        "prob_uncalibrated": prob_raw,
        "prob_calibrated": prob_cal,
        "predicted_label": y_pred,
    })
    return evaluation, predictions


# ---------------------------------------------------------------------------
# 3. permutation importance (frozen pipeline, original features)
# ---------------------------------------------------------------------------
def permutation_importance(model: MLP, preprocessor: Pipeline, calibrator: TemperatureScaler,
                           df: pd.DataFrame, n_repeats: int = IMPORTANCE_REPEATS,
                           seed: int = IMPORTANCE_SEED) -> dict:
    """Drop in ROC-AUC when one ORIGINAL feature column is shuffled.

    Each raw column is permuted before preprocessing, so all one-hot columns of
    a categorical feature move together and importance is reported for the 10
    model features, not the encoded columns. The full serving pipeline
    (preprocessor -> MLP -> calibrator) is scored; nothing is refit.
    """
    if not calibrator.is_fitted:
        raise RuntimeError("fit the calibrator before computing permutation importance")
    splits = sorted(map(str, df["split"].unique()))
    if len(splits) != 1 or splits[0] == "train":
        raise ValueError(f"permutation importance needs one held-out split, got {splits}")
    _, y = split_xy(df)
    X = df[FEATURE_COLS].reset_index(drop=True)

    def scores(frame: pd.DataFrame) -> Dict[str, float]:
        logits = pipeline_logits(model, preprocessor, frame)
        prob = calibrator.predict_proba(logits)
        return {
            "roc_auc": float(roc_auc_score(y, prob)),
            "log_loss": binary_nll(calibrator.calibrate_logits(logits), y),
            "accuracy": float(accuracy_score(y, predict_labels(prob))),
        }

    base = scores(X)
    rng = np.random.default_rng(seed)
    features = []
    for col in FEATURE_COLS:
        auc_drop, loss_rise, acc_drop = [], [], []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = X[col].to_numpy()[rng.permutation(len(X))]
            s = scores(shuffled)
            auc_drop.append(base["roc_auc"] - s["roc_auc"])
            loss_rise.append(s["log_loss"] - base["log_loss"])
            acc_drop.append(base["accuracy"] - s["accuracy"])
        features.append({
            "feature": col,
            "type": "numeric" if col in NUMERIC_COLS else "categorical",
            "importance_mean": float(np.mean(auc_drop)),
            "importance_std": float(np.std(auc_drop)),
            "importances": [float(v) for v in auc_drop],
            "log_loss_increase_mean": float(np.mean(loss_rise)),
            "accuracy_drop_mean": float(np.mean(acc_drop)),
        })
    features.sort(key=lambda f: (-f["importance_mean"], f["feature"]))
    for rank, f in enumerate(features, start=1):
        f["rank"] = rank
    return {
        "method": "permutation",
        "split": splits[0],
        "n": int(len(y)),
        "n_repeats": n_repeats,
        "seed": seed,
        "scoring": "roc_auc",
        "importance_definition": "mean decrease in ROC-AUC when the raw feature column "
                                 "is shuffled (higher = more important)",
        "level": "original features (before one-hot encoding)",
        "baseline": base,
        "features": features,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_final_evaluation(
    model: MLP,
    preprocessor: Pipeline,
    val_df: pd.DataFrame,
    load_test_df: Callable[[], pd.DataFrame],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> dict:
    """Calibrate on val, save the calibrator, and only then load and score test.

    ``load_test_df`` is called exactly once, after calibrator.json is written.
    Writes calibrator.json, calibration.json, evaluation.json,
    test_predictions.csv and permutation_importance.json into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrator, calibration = fit_calibration(model, preprocessor, val_df)
    calibrator.save(out_dir / CALIBRATOR_FILENAME)
    _write_json(out_dir / CALIBRATION_FILENAME, calibration)
    log(f"[calibration] temperature {calibrator.temperature:.4f} fit on {calibration['n']} val rows; "
        f"val ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}, "
        f"Brier {calibration['brier_before']:.4f} -> {calibration['brier_after']:.4f}")

    # The calibrator is saved and frozen; only now is test data requested.
    test_df = load_test_df()
    log(f"[test] scoring {len(test_df)} test rows once")
    evaluation, predictions = evaluate_test(model, preprocessor, calibrator, test_df)
    importance = permutation_importance(model, preprocessor, calibrator, test_df)

    predictions_path = out_dir / PREDICTIONS_FILENAME
    predictions.to_csv(predictions_path, index=False)
    evaluation["predictions_path"] = _repo_relative(predictions_path)
    _write_json(out_dir / EVALUATION_FILENAME, evaluation)
    _write_json(out_dir / IMPORTANCE_FILENAME, importance)
    return {"calibrator": calibrator, "calibration": calibration,
            "evaluation": evaluation, "permutation_importance": importance}


def build_final_run_row(base_row: dict, calibration: dict, evaluation: dict,
                        importance: dict) -> dict:
    """The selected run's complete runs-table row (is_best = true).

    Headline Brier/ECE columns are the TEST numbers; validation calibration
    numbers and both splits' reliability bins go in ``calibration_bins``.
    """
    m, cal = evaluation["metrics"], evaluation["calibration"]

    def split_block(c: dict) -> dict:
        return {k: c[k] for k in ("n", "brier_before", "brier_after", "ece_before",
                                  "ece_after", "bins_before", "bins_after")}

    row = {
        **base_row,
        "test_accuracy": m["accuracy"],
        "test_precision": m["precision"],
        "test_recall": m["recall"],
        "test_f1": m["f1"],
        "test_roc_auc": m["roc_auc"],
        "test_metrics": m,
        "calibration_method": METHOD,
        "ece_before": cal["ece_before"],
        "ece_after": cal["ece_after"],
        "brier_before": cal["brier_before"],
        "brier_after": cal["brier_after"],
        "calibration_bins": {
            "binning": cal["binning"],
            "n_bins": cal["n_bins"],
            "temperature": evaluation["temperature"],
            "fitted_on_split": calibration["fitted_on_split"],
            "val": split_block(calibration),
            "test": split_block(cal),
        },
        "confusion_matrix": evaluation["confusion_matrix"],
        "per_class_metrics": evaluation["per_class"],
        "permutation_importance": importance,
        "is_best": True,
    }
    return complete_run_row(row)


# ---------------------------------------------------------------------------
# pre-flight checks
# ---------------------------------------------------------------------------
def check_not_yet_evaluated(run_dir: Path, base_row: dict) -> None:
    if (run_dir / EVALUATION_FILENAME).exists() or base_row.get("test_metrics"):
        raise ValueError(f"{_repo_relative(run_dir)} has already been evaluated on test. "
                         "The test set is scored once; pass --overwrite only to redo an "
                         "interrupted run with the same code.")


def check_artifacts_match(base_row: dict, ckpt: dict, preprocessor: Pipeline) -> None:
    """model.pt, preprocessor.joblib and run.json must all come from the same run."""
    problems = []
    if ckpt.get("config") != base_row["config"]:
        problems.append("model.pt config differs from run.json config")
    if ckpt.get("best_epoch") != base_row["best_epoch"]:
        problems.append("model.pt best_epoch differs from run.json")
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    if ckpt.get("feature_names") != names:
        problems.append("preprocessor feature names differ from the checkpoint's")
    if problems:
        raise ValueError("artifacts do not belong together:\n  - " + "\n  - ".join(problems))


def check_reproduces_val_loss(model: MLP, preprocessor: Pipeline, val_df: pd.DataFrame,
                              expected: float, tol: float = VAL_LOSS_TOLERANCE) -> float:
    _, y = split_xy(val_df)
    loss = binary_nll(pipeline_logits(model, preprocessor, val_df), y)
    if abs(loss - expected) > tol:
        raise ValueError(f"loaded model gives val loss {loss:.6f}, run.json says {expected:.6f}; "
                         "the artifacts or the val rows have changed")
    return loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Calibrate the selected run on validation, then evaluate it on test once.")
    ap.add_argument("--run", required=True, help="run name, i.e. models/<run>/ (the validation winner)")
    ap.add_argument("--models-dir", default=str(MODELS_DIR), help="default: models/")
    ap.add_argument("--comparison", default=str(DEFAULT_COMPARISON),
                    help="validation comparison from api.run_experiments")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo an interrupted final evaluation (same code, same numbers)")
    # Deliberately no --threshold, calibration-method or binning options.
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.models_dir) / args.run
    run_json = run_dir / "run.json"
    try:
        winner = validation_winner(Path(args.comparison))
        if winner != args.run:
            raise ValueError(f"{args.run!r} is not the validation winner ({winner!r}); "
                             "only the selected run is evaluated on test")
        if not run_json.exists():
            raise ValueError(f"{run_json} not found")
        base_row = json.loads(run_json.read_text())
        if not args.overwrite:
            check_not_yet_evaluated(run_dir, base_row)
        model, ckpt = load_checkpoint(run_dir / "model.pt")
        preprocessor = load_preprocessor(run_dir / "preprocessor.joblib")
        check_artifacts_match(base_row, ckpt, preprocessor)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"[load] {_repo_relative(run_dir)}: model.pt, preprocessor.joblib (not refit)")

    val_df = load_adult_income(split=FIT_SPLIT)
    print(f"[data] {len(val_df)} val rows")
    try:
        val_loss = check_reproduces_val_loss(model, preprocessor, val_df, base_row["val_loss"])
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"[check] val loss reproduced: {val_loss:.6f}")

    def load_test_df() -> pd.DataFrame:
        return load_adult_income(split=TEST_SPLIT)

    out = run_final_evaluation(model, preprocessor, val_df, load_test_df, run_dir)
    ev = out["evaluation"]
    m, cal = ev["metrics"], ev["calibration"]
    print(f"[test] loss {m['loss']:.4f}  acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  "
          f"precision {m['precision']:.4f}  recall {m['recall']:.4f}  f1 {m['f1']:.4f}")
    print(f"[test] confusion {CONFUSION_LAYOUT.split(':')[0]} = {ev['confusion_matrix']}")
    print(f"[test] ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}  "
          f"Brier {cal['brier_before']:.4f} -> {cal['brier_after']:.4f}")
    top = ", ".join(f"{f['feature']} ({f['importance_mean']:.4f})"
                    for f in out["permutation_importance"]["features"][:3])
    print(f"[importance] top AUC drops: {top}")

    row = build_final_run_row(base_row, out["calibration"], ev, out["permutation_importance"])
    _write_json(run_json, row)
    print(f"[saved] {_repo_relative(run_dir)}/: {CALIBRATOR_FILENAME}, {CALIBRATION_FILENAME}, "
          f"{EVALUATION_FILENAME}, {PREDICTIONS_FILENAME}, {IMPORTANCE_FILENAME}, run.json")
    print("[runs] not written to Supabase (use python -m api.persist_runs)")


if __name__ == "__main__":
    sys.exit(main())
