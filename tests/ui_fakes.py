"""Offline fakes for the Streamlit UI tests (no network, no API, no Supabase).

The response bodies follow the FastAPI contract in api/main.py and the example
values in the backend handoff. FakeApi replaces requests.request and routes by
method + path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "ui"
if str(UI_DIR) not in sys.path:  # ui/ modules are imported by bare name, as in the app
    sys.path.insert(0, str(UI_DIR))

API_URL = "http://fake-api.test"

CATEGORIES = {
    "workclass": ["Federal-gov", "Local-gov", "Private", "Self-emp-inc", "Unknown"],
    "marital_status": ["Divorced", "Married-civ-spouse", "Never-married"],
    "occupation": ["Adm-clerical", "Exec-managerial", "Sales", "Unknown"],
    "relationship": ["Husband", "Not-in-family", "Own-child", "Wife"],
    "native_country": ["Canada", "Mexico", "United-States", "Unknown"],
}
NUMERIC = {  # name: (min, max, training median)
    "age": (17, 90, 37),
    "education_num": (1, 16, 10),
    "capital_gain": (0, 99999, 0),
    "capital_loss": (0, 4356, 0),
    "hours_per_week": (1, 99, 40),
}

SCHEMA = {
    "features": (
        [{"name": n, "kind": "numeric", "type": "integer", "required": True,
          "minimum": lo, "maximum": hi, "default": med, "categories": None}
         for n, (lo, hi, med) in NUMERIC.items()]
        + [{"name": n, "kind": "categorical", "type": "string", "required": True,
            "minimum": None, "maximum": None, "default": None, "categories": cats}
           for n, cats in CATEGORIES.items()]
    ),
    "numeric_features": list(NUMERIC),
    "categorical_features": list(CATEGORIES),
    "categories": CATEGORIES,
    "target_name": "income",
    "target_classes": ["<=50K", ">50K"],
    "threshold": 0.5,
    "run_name": "gelu",
}

HEALTH = {"status": "ok", "model_loader": True, "supabase": True, "run_loaded": True,
          "run_id": 3, "detail": None}

VERSION = {
    "project": "Income Insight", "api_version": "2.0.0", "run_name": "gelu", "run_id": 3,
    "model": {"architecture": "mlp", "hidden_sizes": [64, 32], "activation": "gelu",
              "dropout": 0.1, "best_epoch": 18, "n_inputs": 10, "n_encoded_features": 84,
              "calibration_method": "temperature", "temperature": 1.0238, "threshold": 0.5,
              "config": {}, "test_accuracy": 0.854, "test_roc_auc": 0.906},
    "git_sha": "abc1234", "torch_version": "2.4.1", "sklearn_version": "1.5.2",
    "supabase_project_ref": "example",
}

AUDIT = {
    "run_id": 3, "run_name": "gelu", "attribute": "sex", "split": "test",
    "source": "v_fairness_audit", "note": None,
    "groups": [
        {"group_value": "Female", "n": 2429, "tp": 155, "fp": 58, "tn": 2106, "fn": 110,
         "fpr": 0.0268, "fnr": 0.4151},
        {"group_value": "Male", "n": 4898, "tp": 922, "fp": 336, "tn": 3074, "fn": 566,
         "fpr": 0.0985, "fnr": 0.3804},
    ],
}

PREDICT = {"run_id": 3, "run_name": "gelu", "label": 1, "income": ">50K", "proba": 0.83,
           "threshold": 0.5, "calibration_method": "temperature",
           "request_hash": "f" * 64, "logged": True}


def batch_response(n_rows: int) -> dict:
    preds = [{"row": i, "label": i % 2, "income": ">50K" if i % 2 else "<=50K",
              "proba": 0.9 if i % 2 else 0.1, "request_hash": f"{i:064d}"}
             for i in range(1, n_rows + 1)]
    return {"run_id": 3, "run_name": "gelu", "threshold": 0.5, "calibration_method": "temperature",
            "n_rows": n_rows, "n_predicted_positive": sum(p["label"] for p in preds),
            "logged": n_rows, "ignored_columns": [], "predictions": preds}


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if body is not None else "<html>oops</html>"

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


class FakeApi:
    """Callable replacement for requests.request; records every call."""

    def __init__(self, routes: Optional[Dict[Tuple[str, str], Callable]] = None):
        self.calls = []
        self.routes = {
            ("GET", "/healthz"): lambda **kw: FakeResponse(200, HEALTH),
            ("GET", "/version"): lambda **kw: FakeResponse(200, VERSION),
            ("GET", "/schema"): lambda **kw: FakeResponse(200, SCHEMA),
            ("GET", "/audit"): lambda **kw: FakeResponse(200, AUDIT),
            ("POST", "/predict"): lambda **kw: FakeResponse(200, PREDICT),
        }
        self.routes.update(routes or {})

    def __call__(self, method, url, **kwargs):
        path = url.replace(API_URL, "", 1)
        self.calls.append({"method": method, "path": path, **kwargs})
        handler = self.routes.get((method, path))
        if handler is None:
            return FakeResponse(404, {"detail": "Not Found"})
        return handler(**kwargs)


# ---------------------------------------------------------------------------
# stored evaluation results (the files api.evaluate_final writes, trimmed)
# ---------------------------------------------------------------------------
BINS = [{"bin": b, "lower": b / 10, "upper": (b + 1) / 10, "count": 100,
         "mean_predicted": b / 10 + 0.05, "fraction_positive": b / 10 + 0.05} for b in range(10)]
CALIBRATION_TEST = {"n": 7327, "n_bins": 10, "binning": "uniform",
                    "brier_before": 0.1014, "brier_after": 0.1014,
                    "ece_before": 0.0100, "ece_after": 0.0099,
                    "bins_before": BINS, "bins_after": BINS}
PER_CLASS = {
    "<=50K": {"label": 0, "precision": 0.8846, "recall": 0.9293, "f1": 0.9064, "support": 5574},
    ">50K": {"label": 1, "precision": 0.7322, "recall": 0.6144, "f1": 0.6681, "support": 1753},
}
TEST_METRICS = {"loss": 0.31, "accuracy": 0.8540, "precision": 0.7322, "recall": 0.6144,
                "f1": 0.6681, "roc_auc": 0.9060, "threshold": 0.5, "n": 7327}
CONFUSION = [[5180, 394], [676, 1077]]
IMPORTANCE = {"method": "permutation", "split": "test", "features": [
    {"feature": "marital_status", "type": "categorical", "importance_mean": 0.0549,
     "importance_std": 0.002, "rank": 1},
    {"feature": "capital_gain", "type": "numeric", "importance_mean": 0.0370,
     "importance_std": 0.002, "rank": 2},
    {"feature": "age", "type": "numeric", "importance_mean": 0.0335,
     "importance_std": 0.002, "rank": 3},
]}
EXPERIMENTS = [  # validation metrics; values other than val_loss are illustrative
    {"name": "baseline", "hidden_sizes": [64, 32], "activation": "relu", "dropout": 0.1,
     "best_epoch": 18, "val_loss": 0.3163, "val_accuracy": 0.8567, "val_precision": 0.74,
     "val_recall": 0.62, "val_f1": 0.67, "val_roc_auc": 0.9074},
    {"name": "gelu", "hidden_sizes": [64, 32], "activation": "gelu", "dropout": 0.1,
     "best_epoch": 18, "val_loss": 0.3156, "val_accuracy": 0.8550, "val_precision": 0.74,
     "val_recall": 0.61, "val_f1": 0.67, "val_roc_auc": 0.9080},
    {"name": "deep", "hidden_sizes": [128, 64, 32], "activation": "relu", "dropout": 0.1,
     "best_epoch": 12, "val_loss": 0.3158, "val_accuracy": 0.8540, "val_precision": 0.73,
     "val_recall": 0.62, "val_f1": 0.67, "val_roc_auc": 0.9078},
]


def write_model_files(models_dir: Path) -> Path:
    """Write the result files perf_data reads, in the layout api.evaluate_final uses."""
    (models_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (models_dir / "gelu").mkdir(parents=True, exist_ok=True)
    ranked = sorted(EXPERIMENTS, key=lambda e: e["val_loss"])
    (models_dir / "experiments" / "controlled_comparison.json").write_text(json.dumps({
        "selection_criterion": "lowest validation loss", "winner": "gelu",
        "test_set_evaluated": False,
        "runs": [{"rank": i, **e} for i, e in enumerate(ranked, start=1)],
    }))
    (models_dir / "gelu" / "evaluation.json").write_text(json.dumps({
        "split": "test", "n": 7327, "threshold": 0.5, "calibration_method": "temperature",
        "temperature": 1.0238, "metrics": TEST_METRICS, "confusion_matrix": CONFUSION,
        "per_class": PER_CLASS, "calibration": CALIBRATION_TEST,
    }))
    (models_dir / "gelu" / "permutation_importance.json").write_text(json.dumps(IMPORTANCE))
    return models_dir


def runs_rows() -> list:
    """The same results as Supabase `runs` rows (the anon-key read)."""
    rows = []
    for i, e in enumerate(EXPERIMENTS, start=1):
        best = e["name"] == "gelu"
        val = {"loss": e["val_loss"], "accuracy": e["val_accuracy"], "precision": e["val_precision"],
               "recall": e["val_recall"], "f1": e["val_f1"], "roc_auc": e["val_roc_auc"]}
        rows.append({
            "id": {"deep": 1, "baseline": 2, "gelu": 3}[e["name"]], "name": e["name"], "is_best": best,
            "hidden_sizes": e["hidden_sizes"], "activation": e["activation"],
            "dropout": e["dropout"], "best_epoch": e["best_epoch"],
            "val_loss": e["val_loss"], "val_accuracy": e["val_accuracy"],
            "val_roc_auc": e["val_roc_auc"], "val_metrics": val,
            "test_metrics": TEST_METRICS if best else {},
            "test_accuracy": TEST_METRICS["accuracy"] if best else None,
            "calibration_method": "temperature" if best else None,
            "calibration_bins": ({"temperature": 1.0238, "fitted_on_split": "val",
                                  "test": CALIBRATION_TEST, "val": None} if best else None),
            "confusion_matrix": CONFUSION if best else None,
            "per_class_metrics": PER_CLASS if best else None,
            "permutation_importance": IMPORTANCE if best else None,
        })
    return rows
