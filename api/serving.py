"""Serving core for the selected GELU run: frozen artifacts in, calibrated predictions out.

Used by api/main.py. Kept free of FastAPI and of training code (api.train needs
PyYAML, which the Render image does not install), so it is easy to test alone.

    raw 10-feature rows
      -> validate_records()        types, ranges, known categories
      -> preprocessor.joblib       transform only (fit on train rows, never refit)
      -> model.pt                  GELU MLP logits, eval mode
      -> calibrator.json           sigmoid(logit / T), T fit on validation
      -> label = proba >= 0.5      fixed threshold

``load_artifacts`` loads the three files once and refuses to serve unless they
are the selected GELU run's and agree with each other and with run.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from api.calibration import CALIBRATOR_FILENAME, DECISION_THRESHOLD, METHOD, TemperatureScaler
from api.model import MLP, load_checkpoint
from api.preprocessing import load_preprocessor, transform_features
from shared.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MISSING_CATEGORY,
    NUMERIC_BOUNDS,
    NUMERIC_COLS,
    TARGET_CLASSES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVED_RUN_NAME = "gelu"
SERVED_ACTIVATION = "gelu"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / SERVED_RUN_NAME

# Raw values that mean "not known" (the UCI '?' marker and blank cells). They map
# to the explicit 'Unknown' category, exactly as the preprocessor treated NULLs
# during training, but only for columns where 'Unknown' was seen in training.
_MISSING_TOKENS = {"", "?"}
MAX_REPORTED_ERRORS = 50

PathLike = Union[str, Path]


class ArtifactError(RuntimeError):
    """The model files are missing, inconsistent, or not the selected GELU run."""


class RunMismatchError(RuntimeError):
    """The Supabase runs row does not describe the loaded artifacts."""


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
@dataclass
class Artifacts:
    model: MLP
    preprocessor: Pipeline
    calibrator: TemperatureScaler
    run: dict                         # models/gelu/run.json (the runs-table row)
    checkpoint: dict                  # model.pt metadata (no weights needed here)
    model_dir: Path
    categories: Dict[str, List[str]] = field(default_factory=dict)   # from the fitted encoder
    numeric_medians: Dict[str, float] = field(default_factory=dict)  # from the fitted imputer

    @property
    def run_name(self) -> str:
        return self.run["name"]


def _fitted_categories(pre: Pipeline) -> Dict[str, List[str]]:
    encoder = pre.named_steps["columns"].named_transformers_["cat"].named_steps["onehot"]
    return {col: [str(c) for c in cats] for col, cats in zip(CATEGORICAL_COLS, encoder.categories_)}


def _fitted_medians(pre: Pipeline) -> Dict[str, float]:
    imputer = pre.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    return {col: float(v) for col, v in zip(NUMERIC_COLS, imputer.statistics_)}


def load_artifacts(model_dir: PathLike = DEFAULT_MODEL_DIR) -> Artifacts:
    """Load model.pt, preprocessor.joblib and calibrator.json once, and check them.

    Raises ArtifactError unless all three belong to the selected GELU run.
    """
    model_dir = Path(model_dir)
    paths = {name: model_dir / name for name in
             ("model.pt", "preprocessor.joblib", CALIBRATOR_FILENAME, "run.json")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ArtifactError(f"missing model artifacts: {missing}")

    run = json.loads(paths["run.json"].read_text())
    model, ckpt = load_checkpoint(paths["model.pt"])
    preprocessor = load_preprocessor(paths["preprocessor.joblib"])
    calibrator = TemperatureScaler.load(paths[CALIBRATOR_FILENAME])

    problems = []
    if run.get("name") != SERVED_RUN_NAME:
        problems.append(f"run.json is for {run.get('name')!r}, expected {SERVED_RUN_NAME!r}")
    if run.get("is_best") is not True:
        problems.append("run.json is not marked is_best (final evaluation missing)")
    if ckpt.get("activation") != SERVED_ACTIVATION or model.activation != SERVED_ACTIVATION:
        problems.append(f"model.pt activation is {ckpt.get('activation')!r}, expected 'gelu'")
    if ckpt.get("config") != run.get("config"):
        problems.append("model.pt config differs from run.json config")
    if ckpt.get("hidden_sizes") != run.get("hidden_sizes"):
        problems.append("model.pt hidden_sizes differ from run.json")
    if ckpt.get("best_epoch") != run.get("best_epoch"):
        problems.append("model.pt best_epoch differs from run.json")
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    if ckpt.get("feature_names") != names or model.in_dim != len(names):
        problems.append("preprocessor output columns do not match the checkpoint")
    if run.get("calibration_method") != METHOD:
        problems.append(f"run.json calibration_method is {run.get('calibration_method')!r}")
    expected_t = (run.get("calibration_bins") or {}).get("temperature")
    if expected_t is None or not math.isclose(calibrator.temperature, expected_t, rel_tol=1e-12):
        problems.append(f"calibrator temperature {calibrator.temperature} != run.json {expected_t}")
    if problems:
        raise ArtifactError("model artifacts are not the selected GELU run:\n  - "
                            + "\n  - ".join(problems))

    return Artifacts(
        model=model, preprocessor=preprocessor, calibrator=calibrator, run=run,
        checkpoint={k: v for k, v in ckpt.items() if k != "state_dict"},
        model_dir=model_dir,
        categories=_fitted_categories(preprocessor),
        numeric_medians=_fitted_medians(preprocessor),
    )


def verify_run_row(row: Optional[dict], art: Artifacts) -> dict:
    """Check the Supabase runs row describes the loaded artifacts; return it."""
    if not row:
        raise RunMismatchError("no runs row found for the served model")
    problems = []
    if row.get("name") != art.run_name:
        problems.append(f"runs.name is {row.get('name')!r}, artifacts are {art.run_name!r}")
    if row.get("is_best") is not True:
        problems.append(f"runs row {row.get('id')} is not is_best")
    if row.get("config") != art.run["config"]:
        problems.append("runs.config differs from run.json")
    if row.get("checkpoint_path") != art.run["checkpoint_path"]:
        problems.append(f"runs.checkpoint_path is {row.get('checkpoint_path')!r}, "
                        f"expected {art.run['checkpoint_path']!r}")
    if row.get("calibration_method") != METHOD:
        problems.append(f"runs.calibration_method is {row.get('calibration_method')!r}")
    if problems:
        raise RunMismatchError(f"Supabase run {row.get('id')} does not match the loaded model:\n  - "
                               + "\n  - ".join(problems))
    return row


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    records: List[dict]               # clean rows: ints for numerics, str for categoricals
    errors: List[dict]                # [{row, column, value, error}], 1-based rows
    n_errors: int = 0


def _display(v) -> object:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def validate_records(rows: Sequence[Mapping], categories: Mapping[str, Sequence[str]]) -> ValidationResult:
    """Check every row has valid values for the 10 model features.

    Numerics must be whole numbers inside NUMERIC_BOUNDS; categoricals must be
    one of the categories the fitted encoder learned. '' and '?' become
    'Unknown' where that category exists. Other keys (e.g. sex, race) are not
    read here; callers decide whether extra keys are an error.
    """
    clean: List[dict] = []
    errors: List[dict] = []
    allowed = {c: set(v) for c, v in categories.items()}
    for i, row in enumerate(rows, start=1):
        out = {}
        for col in NUMERIC_COLS:
            raw = row.get(col)
            lo, hi = NUMERIC_BOUNDS[col]
            msg = None
            try:
                if isinstance(raw, bool):
                    raise ValueError
                num = float(raw.strip()) if isinstance(raw, str) else float(raw)
                if not math.isfinite(num):
                    raise ValueError
            except (TypeError, ValueError):
                msg = "missing or not a number"
            else:
                if num != int(num):
                    msg = "must be a whole number"
                elif not lo <= num <= hi:
                    msg = f"must be between {lo} and {hi}"
                else:
                    out[col] = int(num)
            if msg:
                errors.append({"row": i, "column": col, "value": _display(raw), "error": msg})
        for col in CATEGORICAL_COLS:
            raw = row.get(col)
            value = raw.strip() if isinstance(raw, str) else raw
            if (value is None or value in _MISSING_TOKENS
                    or (isinstance(value, float) and math.isnan(value))):
                if MISSING_CATEGORY in allowed[col]:
                    out[col] = MISSING_CATEGORY
                else:
                    errors.append({"row": i, "column": col, "value": _display(raw),
                                   "error": "missing"})
            elif not isinstance(value, str) or value not in allowed[col]:
                errors.append({"row": i, "column": col, "value": _display(raw),
                               "error": f"unknown category; see /schema for the "
                                        f"{len(allowed[col])} allowed values"})
            else:
                out[col] = value
        clean.append(out)
    return ValidationResult(records=clean, errors=errors[:MAX_REPORTED_ERRORS], n_errors=len(errors))


class CSVInputError(ValueError):
    """The uploaded file is not a usable CSV of the 10 model features."""

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.detail = {"message": message, **extra}


def parse_csv(content: bytes) -> Tuple[List[dict], List[str]]:
    """Read an uploaded CSV into raw row dicts; return (rows, ignored extra columns).

    Every cell is read as text so validate_records() can report bad values per
    row instead of pandas silently turning them into NaN.
    """
    if not content or not content.strip():
        raise CSVInputError("the uploaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        raise CSVInputError("the uploaded file is not UTF-8 text")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise CSVInputError(f"could not parse the CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise CSVInputError("duplicate column names",
                            duplicate_columns=sorted(set(df.columns[df.columns.duplicated()])))
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise CSVInputError("the CSV is missing required columns",
                            missing_columns=missing, required_columns=FEATURE_COLS)
    if df.empty:
        raise CSVInputError("the CSV has a header but no data rows")
    ignored = [c for c in df.columns if c not in FEATURE_COLS]
    return df[FEATURE_COLS].to_dict(orient="records"), ignored


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(art: Artifacts, records: Sequence[Mapping]) -> np.ndarray:
    """Calibrated P(>50K) for validated rows, through the frozen pipeline."""
    frame = pd.DataFrame(list(records), columns=FEATURE_COLS)
    X = transform_features(art.preprocessor, frame)
    art.model.eval()
    logits = art.model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
    return art.calibrator.predict_proba(logits)


def predict_labels(proba) -> np.ndarray:
    """0/1 at the fixed DECISION_THRESHOLD (0.5); never tuned."""
    return (np.asarray(proba) >= DECISION_THRESHOLD).astype(int)


def income_label(label: int) -> str:
    return TARGET_CLASSES[int(label)]


# ---------------------------------------------------------------------------
# prediction logging
# ---------------------------------------------------------------------------
def canonical_features(record: Mapping) -> str:
    """Canonical JSON of the 10 validated features: fixed keys, sorted, no spaces."""
    return json.dumps({c: record[c] for c in FEATURE_COLS}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def request_hash(record: Mapping) -> str:
    """sha256 of canonical_features(); the same inputs always give the same hash."""
    return hashlib.sha256(canonical_features(record).encode("utf-8")).hexdigest()


def prediction_row(record: Mapping, label: int, proba: float, run_id: int) -> dict:
    """One row for the Supabase ``predictions`` table.

    Holds no raw features (only their hash) and no protected attributes.
    adult_income_id stays NULL: these are arbitrary user inputs with no true label.
    """
    return {
        "request_hash": request_hash(record),
        "predicted_label": int(label),
        "predicted_proba": float(proba),
        "served_by_run_id": int(run_id),
        "adult_income_id": None,
    }
