"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Serves ONE frozen model: the selected GELU run (models/gelu/). Nothing is
trained here. At startup (lifespan) the service:
  1. loads model.pt, preprocessor.joblib and calibrator.json once, and refuses
     to serve unless they are the GELU run's and agree with run.json
  2. reads the matching runs row from Supabase (the is_best row, or the row
     pinned by SERVED_RUN_ID) and checks it describes those artifacts

Endpoints:
    GET  /healthz         liveness + whether artifacts / Supabase / run are ready
    GET  /version         project, served run, model and library versions
    GET  /schema          the 10 input fields for the UI form (never sex/race)
    POST /predict         one row -> calibrated P(>50K), label at 0.5, logged
    POST /predict_batch   CSV upload -> one prediction per row, all logged
    GET  /audit           FPR/FNR by sex from the v_fairness_audit SQL view

There is NO UI code here and NO model logic in the UI. Run locally with:

    uvicorn api.main:app --reload --env-file .env
"""
from __future__ import annotations

import logging
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional, TypeVar

import sklearn
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import db, serving
from api.calibration import DECISION_THRESHOLD, METHOD
from shared.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_BOUNDS,
    NUMERIC_COLS,
    TARGET_CLASSES,
    TARGET_NAME,
)
from shared.schemas import (
    AuditGroup,
    AuditResponse,
    BatchPrediction,
    FieldSpec,
    Health,
    ModelInfo,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    SchemaResponse,
    Version,
)

PROJECT = "Income Insight"
API_VERSION = "2.0.0"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
AUDIT_ATTRIBUTE = "sex"  # the protected attribute v_fairness_audit groups by
AUDIT_SPLIT = "test"     # the view only counts labeled held-out rows

log = logging.getLogger("uvicorn.error")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# serving state (loaded once at startup)
# ---------------------------------------------------------------------------
class ServingState:
    """The loaded artifacts and the matching Supabase runs row."""

    def __init__(self, model_dir: Path, pinned_run_id: Optional[int] = None):
        self.model_dir = Path(model_dir)
        self.pinned_run_id = pinned_run_id
        self.artifacts: Optional[serving.Artifacts] = None
        self.artifacts_error: Optional[str] = None
        self.run: Optional[dict] = None
        self.run_error: Optional[str] = None

    def load_artifacts(self) -> None:
        try:
            self.artifacts = serving.load_artifacts(self.model_dir)
            self.artifacts_error = None
        except Exception as e:  # reported by /healthz; every model endpoint returns 503
            self.artifacts, self.artifacts_error = None, f"{type(e).__name__}: {e}"

    def read_run(self) -> dict:
        """Read the served run from Supabase and check it matches the artifacts.

        Raises on a Supabase error or a mismatch. Exactly one runs read.
        """
        if self.artifacts is None:
            raise serving.ArtifactError("model artifacts are not loaded")
        if self.pinned_run_id is not None:
            row = db.fetch_run(self.pinned_run_id)
        elif self.run is not None:
            row = db.fetch_run(self.run["id"])  # still the same run, still is_best?
        else:
            rows = db.fetch_best_runs()
            if len(rows) != 1:
                raise serving.RunMismatchError(f"expected exactly one is_best run, found {len(rows)}")
            row = rows[0]
        try:
            self.run = serving.verify_run_row(row, self.artifacts)
        except serving.RunMismatchError:
            self.run = None
            raise
        self.run_error = None
        return self.run

    def try_read_run(self) -> None:
        try:
            self.read_run()
        except Exception as e:
            self.run_error = f"{type(e).__name__}: {e}"


def _pinned_run_id() -> Optional[int]:
    raw = os.environ.get("SERVED_RUN_ID", "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise RuntimeError(f"SERVED_RUN_ID must be an integer, got {raw!r}")
    return int(raw)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ServingState(Path(os.environ.get("MODEL_DIR", serving.DEFAULT_MODEL_DIR)),
                         _pinned_run_id())
    state.load_artifacts()
    if state.artifacts is None:
        log.error("model artifacts failed to load: %s", state.artifacts_error)
    else:
        state.try_read_run()
        if state.run is None:
            log.error("served run not confirmed in Supabase: %s", state.run_error)
        else:
            log.info("serving run %s (%s) from %s", state.run["id"], state.run["name"],
                     state.model_dir)
    app.state.serving = state
    yield


app = FastAPI(
    title="Income-Insight API",
    description="Serves the selected GELU MLP (calibrated) for the three-cloud stack.",
    version=API_VERSION,
    lifespan=lifespan,
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def get_state(request: Request) -> ServingState:
    return request.app.state.serving


def _supabase_error(what: str, e: Exception) -> HTTPException:
    log.error("Supabase request failed while %s: %s", what, e)
    return HTTPException(status_code=503, detail={
        "message": f"Supabase request failed while {what}",
        "error": f"{type(e).__name__}: {e}",
    })


def _db_call(fn: Callable[[], T], what: str) -> T:
    try:
        return fn()
    except Exception as e:
        raise _supabase_error(what, e)


def require_artifacts(state: ServingState) -> serving.Artifacts:
    if state.artifacts is None:
        raise HTTPException(status_code=503, detail={
            "message": "model artifacts are not loaded", "error": state.artifacts_error})
    return state.artifacts


def require_run(state: ServingState) -> dict:
    """The served runs row; read from Supabase now if startup could not."""
    require_artifacts(state)
    if state.run is None:
        _read_run_or_503(state, "reading the served run")
    return state.run


def _read_run_or_503(state: ServingState, what: str) -> dict:
    try:
        return state.read_run()
    except serving.RunMismatchError as e:
        state.run_error = str(e)
        raise HTTPException(status_code=503, detail={
            "message": "the Supabase runs row does not match the loaded model",
            "error": str(e)})
    except Exception as e:
        state.run_error = f"{type(e).__name__}: {e}"
        raise _supabase_error(what, e)


def _invalid_values(checked: serving.ValidationResult) -> HTTPException:
    return HTTPException(status_code=422, detail={
        "message": f"{checked.n_errors} invalid feature value(s); nothing was scored or logged",
        "n_errors": checked.n_errors,
        "errors": checked.errors,
    })


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> Optional[str]:
    """The public project id from SUPABASE_URL (never the key)."""
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz(state: ServingState = Depends(get_state)):
    """200 with status 'ok' when artifacts, Supabase and the served run are all ready.

    503 only when the model artifacts failed to load (the service cannot predict).
    """
    model_ok = state.artifacts is not None
    supabase_ok = db.ping()
    if model_ok and supabase_ok and state.run is None:
        state.try_read_run()  # recover if Supabase was down at startup
    run_ok = state.run is not None
    ok = model_ok and supabase_ok and run_ok
    body = Health(
        status="ok" if ok else "degraded",
        model_loader=model_ok,
        supabase=supabase_ok,
        run_loaded=run_ok,
        run_id=state.run["id"] if run_ok else None,
        detail=None if ok else (state.artifacts_error or state.run_error
                                or ("Supabase unreachable" if not supabase_ok else None)),
    )
    if not model_ok:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.get("/version", response_model=Version, tags=["ops"])
def version(state: ServingState = Depends(get_state)) -> Version:
    art = state.artifacts
    model = None
    if art is not None:
        model = ModelInfo(
            architecture=art.run["architecture"],
            hidden_sizes=art.run["hidden_sizes"],
            activation=art.run["activation"],
            dropout=art.run["dropout"],
            best_epoch=art.run["best_epoch"],
            n_inputs=len(FEATURE_COLS),
            n_encoded_features=art.model.in_dim,
            calibration_method=METHOD,
            temperature=art.calibrator.temperature,
            threshold=DECISION_THRESHOLD,
            config=art.run["config"],
            test_accuracy=art.run.get("test_accuracy"),
            test_roc_auc=art.run.get("test_roc_auc"),
        )
    return Version(
        project=PROJECT,
        api_version=API_VERSION,
        run_name=art.run_name if art else None,
        run_id=state.run["id"] if state.run else None,
        model=model,
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        supabase_project_ref=_project_ref(),
    )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
@app.get("/schema", response_model=SchemaResponse, tags=["meta"])
def schema(state: ServingState = Depends(get_state)) -> SchemaResponse:
    """The 10 model inputs. Categories come from the fitted preprocessor's encoder;
    numeric defaults are its training medians."""
    art = require_artifacts(state)
    features = []
    for col in NUMERIC_COLS:
        lo, hi = NUMERIC_BOUNDS[col]
        default = min(max(int(round(art.numeric_medians[col])), lo), hi)
        features.append(FieldSpec(name=col, kind="numeric", type="integer",
                                  minimum=lo, maximum=hi, default=default))
    for col in CATEGORICAL_COLS:
        features.append(FieldSpec(name=col, kind="categorical", type="string",
                                  categories=art.categories[col]))
    return SchemaResponse(
        features=features,
        numeric_features=NUMERIC_COLS,
        categorical_features=CATEGORICAL_COLS,
        categories=art.categories,
        target_name=TARGET_NAME,
        target_classes=TARGET_CLASSES,
        threshold=DECISION_THRESHOLD,
        run_name=art.run_name,
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest, state: ServingState = Depends(get_state)) -> PredictResponse:
    """Score one row with the frozen GELU pipeline and log it to ``predictions``."""
    art = require_artifacts(state)
    run = require_run(state)
    if req.run_id is not None and req.run_id != run["id"]:
        raise HTTPException(status_code=409, detail={
            "message": f"this service serves run {run['id']} ({run['name']}) only",
            "requested_run_id": req.run_id})

    checked = serving.validate_records([req.features.model_dump()], art.categories)
    if checked.errors:
        raise _invalid_values(checked)
    record = checked.records[0]
    proba = float(serving.predict_proba(art, [record])[0])
    label = int(serving.predict_labels([proba])[0])

    row = serving.prediction_row(record, label, proba, run["id"])
    _db_call(lambda: db.insert_predictions([row]), "logging the prediction")
    return PredictResponse(
        run_id=run["id"], run_name=run["name"], label=label, income=serving.income_label(label),
        proba=proba, threshold=DECISION_THRESHOLD, calibration_method=METHOD,
        request_hash=row["request_hash"], logged=True,
    )


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
def predict_batch(
    file: UploadFile = File(..., description="CSV with a header row containing the 10 feature columns."),
    state: ServingState = Depends(get_state),
) -> PredictBatchResponse:
    """Score every row of an uploaded CSV and log each prediction to ``predictions``.

    Starts with a Supabase read of the served runs row, so a batch is never
    scored or logged against a run that is no longer the active is_best run.
    All rows are validated before any is scored; one bad value rejects the file.
    """
    require_artifacts(state)
    run = _read_run_or_503(state, "confirming the active run before batch scoring")
    art = state.artifacts

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={
            "message": f"the file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"})
    try:
        raw_rows, ignored = serving.parse_csv(content)
    except serving.CSVInputError as e:
        raise HTTPException(status_code=422, detail=e.detail)
    if len(raw_rows) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail={
            "message": f"at most {MAX_BATCH_ROWS} rows per request, got {len(raw_rows)}"})

    checked = serving.validate_records(raw_rows, art.categories)
    if checked.errors:
        raise _invalid_values(checked)
    proba = serving.predict_proba(art, checked.records)
    labels = serving.predict_labels(proba)

    rows = [serving.prediction_row(rec, lab, p, run["id"])
            for rec, lab, p in zip(checked.records, labels, proba)]
    _db_call(lambda: db.insert_predictions(rows), "logging the batch predictions")
    return PredictBatchResponse(
        run_id=run["id"], run_name=run["name"], threshold=DECISION_THRESHOLD,
        calibration_method=METHOD, n_rows=len(rows),
        n_predicted_positive=int(labels.sum()), logged=len(rows), ignored_columns=ignored,
        predictions=[
            BatchPrediction(row=i, label=int(lab), income=serving.income_label(lab),
                            proba=float(p), request_hash=r["request_hash"])
            for i, (lab, p, r) in enumerate(zip(labels, proba, rows), start=1)
        ],
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@app.get("/audit", response_model=AuditResponse, tags=["meta"])
def audit(state: ServingState = Depends(get_state)) -> AuditResponse:
    """FPR / FNR by sex for the served run, exactly as v_fairness_audit computes them.

    The rates come from SQL; this endpoint only passes the view's rows through.
    """
    run = require_run(state)
    rows = _db_call(lambda: db.fetch_fairness_audit(run["id"]), "reading v_fairness_audit")
    groups = [AuditGroup(**{k: r.get(k) for k in AuditGroup.model_fields}) for r in rows]
    note = None if groups else (
        "v_fairness_audit has no rows for this run yet: it only counts predictions "
        "linked to labeled held-out (test) adult_income rows.")
    return AuditResponse(
        run_id=run["id"], run_name=run["name"], attribute=AUDIT_ATTRIBUTE,
        split=AUDIT_SPLIT, source="v_fairness_audit", groups=groups, note=note,
    )
