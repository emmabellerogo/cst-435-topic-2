"""Read-only access to the finished evaluation results for the Model Performance
and Model Card tabs.

The FastAPI contract does not expose the confusion matrix, per-class metrics,
calibration bins, permutation importance or the three-way experiment
comparison, so this module reads them from where Emma's pipeline stored them:

  1. Supabase ``runs`` table, with the public ANON key (read-only; RLS allows
     anon SELECT on runs). This is the single source of truth.
  2. Fallback: the committed JSON files under models/ (written by
     api.run_experiments and api.evaluate_final), used when the anon key is not
     configured or the read fails.

Nothing is recomputed here: every number is copied from those sources and
reshaped into one dictionary (a "bundle") that the tabs display.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR_ENV = "INCOME_INSIGHT_MODELS_DIR"  # lets tests point at fixture files

RUN_COLUMNS = (
    "id,name,is_best,architecture,hidden_sizes,activation,dropout,best_epoch,"
    "val_loss,val_accuracy,val_roc_auc,val_metrics,"
    "test_accuracy,test_precision,test_recall,test_f1,test_roc_auc,test_metrics,"
    "calibration_method,ece_before,ece_after,brier_before,brier_after,calibration_bins,"
    "confusion_matrix,per_class_metrics,permutation_importance,created_at"
)
SELECTION_CRITERION = (
    "lowest validation loss (BCE at each run's best epoch); "
    "ties broken by higher validation ROC-AUC"
)
_CAL_KEYS = ("brier_before", "brier_after", "ece_before", "ece_after")


class PerformanceDataError(RuntimeError):
    """Neither Supabase nor the committed files could provide the results."""


def default_models_dir() -> Path:
    return Path(os.environ.get(MODELS_DIR_ENV) or (REPO_ROOT / "models"))


def _experiment(row: dict, val: dict, selected: bool, run_id: Optional[int] = None) -> dict:
    def pick(key: str):
        v = val.get(key)
        return row.get(f"val_{key}") if v is None else v

    return {
        "name": row.get("name"),
        "run_id": run_id,
        "hidden_sizes": list(row.get("hidden_sizes") or []),
        "activation": row.get("activation"),
        "dropout": row.get("dropout"),
        "best_epoch": row.get("best_epoch"),
        "val_loss": pick("loss"),
        "val_accuracy": pick("accuracy"),
        "val_precision": pick("precision"),
        "val_recall": pick("recall"),
        "val_f1": pick("f1"),
        "val_roc_auc": pick("roc_auc"),
        "selected": selected,
    }


def _sort_experiments(experiments: List[dict]) -> List[dict]:
    def key(e):
        loss = e.get("val_loss")
        return (loss is None, loss if loss is not None else 0.0, e.get("name") or "")
    return sorted(experiments, key=key)


# ---------------------------------------------------------------------------
# source 1: Supabase runs rows (anon key)
# ---------------------------------------------------------------------------
def from_runs_rows(rows: List[dict]) -> Dict[str, Any]:
    if not rows:
        raise PerformanceDataError("the runs table returned no rows")
    best = [r for r in rows if r.get("is_best") is True]
    if len(best) != 1:
        raise PerformanceDataError(f"expected exactly one is_best run, found {len(best)}")
    sel = best[0]

    test = dict(sel.get("test_metrics") or {})
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if test.get(k) is None:
            test[k] = sel.get(f"test_{k}")
    if test.get("accuracy") is None:
        raise PerformanceDataError(f"run {sel.get('name')!r} has no test metrics yet")

    bins = sel.get("calibration_bins") or {}
    cal_test = bins.get("test") or {k: sel.get(k) for k in _CAL_KEYS}
    calibration = {
        "method": sel.get("calibration_method"),
        "temperature": bins.get("temperature"),
        "fitted_on_split": bins.get("fitted_on_split"),
        "test": cal_test,
        "val": bins.get("val"),
    }
    importance = (sel.get("permutation_importance") or {}).get("features") or []

    experiments = [
        _experiment(r, r.get("val_metrics") or {}, r is sel, run_id=r.get("id")) for r in rows
    ]
    return {
        "source": "supabase",
        "source_detail": "Supabase `runs` table (read-only, anon key)",
        "selected": {
            "name": sel.get("name"),
            "run_id": sel.get("id"),
            "hidden_sizes": list(sel.get("hidden_sizes") or []),
            "activation": sel.get("activation"),
            "dropout": sel.get("dropout"),
            "best_epoch": sel.get("best_epoch"),
        },
        "test": test,
        "confusion_matrix": sel.get("confusion_matrix"),
        "per_class": sel.get("per_class_metrics") or {},
        "calibration": calibration,
        "importance": importance,
        "experiments": _sort_experiments(experiments),
        "selection_criterion": SELECTION_CRITERION,
        "warnings": [],
    }


def fetch_runs_rows(supabase_url: str, anon_key: str) -> List[dict]:
    from supabase import create_client  # imported lazily so tests need no network

    client = create_client(supabase_url, anon_key)
    return client.table("runs").select(RUN_COLUMNS).order("id").execute().data


# ---------------------------------------------------------------------------
# source 2: committed files under models/
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PerformanceDataError(f"{path} not found")
    return json.loads(path.read_text())


def from_model_files(models_dir: Path) -> Dict[str, Any]:
    models_dir = Path(models_dir)
    comparison = _read_json(models_dir / "experiments" / "controlled_comparison.json")
    winner = comparison.get("winner")
    if not winner:
        raise PerformanceDataError("controlled_comparison.json names no winner")
    run_dir = models_dir / winner
    evaluation = _read_json(run_dir / "evaluation.json")
    importance = _read_json(run_dir / "permutation_importance.json")
    val_cal = None
    if (run_dir / "calibration.json").exists():
        val_cal = _read_json(run_dir / "calibration.json")

    experiments = [
        _experiment(r, {}, r.get("name") == winner) for r in comparison.get("runs", [])
    ]
    sel = next((e for e in experiments if e["selected"]), {})
    test_cal = evaluation.get("calibration") or {}
    return {
        "source": "files",
        "source_detail": f"committed result files in models/ ({winner}/evaluation.json etc.)",
        "selected": {
            "name": winner,
            "run_id": None,
            "hidden_sizes": sel.get("hidden_sizes", []),
            "activation": sel.get("activation"),
            "dropout": sel.get("dropout"),
            "best_epoch": sel.get("best_epoch"),
        },
        "test": dict(evaluation.get("metrics") or {}),
        "confusion_matrix": evaluation.get("confusion_matrix"),
        "per_class": evaluation.get("per_class") or {},
        "calibration": {
            "method": evaluation.get("calibration_method"),
            "temperature": evaluation.get("temperature"),
            "fitted_on_split": (val_cal or {}).get("fitted_on_split", "val"),
            "test": test_cal,
            "val": val_cal,
        },
        "importance": importance.get("features") or [],
        "experiments": _sort_experiments(experiments),
        "selection_criterion": comparison.get("selection_criterion") or SELECTION_CRITERION,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def load_performance(supabase_url: Optional[str], anon_key: Optional[str],
                     models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Supabase first (when configured), then the committed files."""
    problems = []
    if supabase_url and anon_key:
        try:
            return from_runs_rows(fetch_runs_rows(supabase_url, anon_key))
        except Exception as e:  # fall back, but say why
            problems.append(f"Supabase runs read failed: {type(e).__name__}: {e}")
    else:
        problems.append("SUPABASE_URL / SUPABASE_ANON_KEY are not configured")
    try:
        bundle = from_model_files(Path(models_dir) if models_dir else default_models_dir())
    except Exception as e:
        problems.append(f"model files: {e}")
        raise PerformanceDataError("; ".join(problems))
    bundle["warnings"] = problems
    return bundle
