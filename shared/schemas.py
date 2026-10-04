"""Pydantic request/response models shared by the API and (optionally) the UI.

Keeping every wire-format type in one module is the contract between the three
clouds. The Streamlit UI never imports model or SQL code -- it only imports (or
mirrors) these schemas so that the payloads it sends match what FastAPI expects.

The API serves one frozen model (the selected GELU run); there are no training
or dataset endpoints. Inputs are exactly the 10 model features in
shared.features -- never sex or race.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.features import NUMERIC_BOUNDS


def _bounded(col: str, example: int):
    lo, hi = NUMERIC_BOUNDS[col]
    return Field(..., ge=lo, le=hi, examples=[example])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictFeatures(BaseModel):
    """Exactly the 10 raw model features. Any other key (e.g. sex, race) is a 422."""

    model_config = ConfigDict(extra="forbid")

    age: int = _bounded("age", 45)
    education_num: int = _bounded("education_num", 13)
    capital_gain: int = _bounded("capital_gain", 0)
    capital_loss: int = _bounded("capital_loss", 0)
    hours_per_week: int = _bounded("hours_per_week", 45)
    workclass: str = Field(..., examples=["Private"])
    marital_status: str = Field(..., examples=["Married-civ-spouse"])
    occupation: str = Field(..., examples=["Exec-managerial"])
    relationship: str = Field(..., examples=["Husband"])
    native_country: str = Field(..., examples=["United-States"])


class PredictRequest(BaseModel):
    """Request body for POST /predict -- one row."""

    model_config = ConfigDict(extra="forbid")

    features: PredictFeatures
    run_id: Optional[int] = Field(
        None, description="Optional. If given, must equal the served run's id.")


class PredictResponse(BaseModel):
    run_id: int
    run_name: str
    label: int = Field(..., description="0 = <=50K, 1 = >50K.")
    income: str = Field(..., description="Human-readable class label.")
    proba: float = Field(..., description="Calibrated P(income > 50K).")
    threshold: float
    calibration_method: str
    request_hash: str
    logged: bool


class BatchPrediction(BaseModel):
    row: int = Field(..., description="1-based data row of the uploaded CSV.")
    label: int
    income: str
    proba: float
    request_hash: str


class PredictBatchResponse(BaseModel):
    run_id: int
    run_name: str
    threshold: float
    calibration_method: str
    n_rows: int
    n_predicted_positive: int
    logged: int
    ignored_columns: List[str]
    predictions: List[BatchPrediction]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class FieldSpec(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    type: Literal["integer", "string"]
    required: bool = True
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    default: Optional[Any] = None
    categories: Optional[List[str]] = None


class SchemaResponse(BaseModel):
    """Feature contract, so the UI can build its input form dynamically."""

    features: List[FieldSpec]
    numeric_features: List[str]
    categorical_features: List[str]
    categories: Dict[str, List[str]]
    target_name: str
    target_classes: List[str]
    threshold: float
    run_name: str


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
class AuditGroup(BaseModel):
    """One row of v_fairness_audit, passed through unchanged."""

    group_value: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    fpr: Optional[float] = Field(None, description="FP / (FP + TN), computed in SQL.")
    fnr: Optional[float] = Field(None, description="FN / (FN + TP), computed in SQL.")


class AuditResponse(BaseModel):
    run_id: int
    run_name: str
    attribute: str
    split: str
    source: str
    groups: List[AuditGroup]
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    status: str
    model_loader: bool = Field(..., description="Model, preprocessor and calibrator loaded.")
    supabase: bool
    run_loaded: bool
    run_id: Optional[int] = None
    detail: Optional[str] = None


class ModelInfo(BaseModel):
    architecture: str
    hidden_sizes: List[int]
    activation: str
    dropout: float
    best_epoch: int
    n_inputs: int
    n_encoded_features: int
    calibration_method: str
    temperature: float
    threshold: float
    config: Dict[str, Any]
    test_accuracy: Optional[float] = None
    test_roc_auc: Optional[float] = None


class Version(BaseModel):
    project: str
    api_version: str
    run_name: Optional[str]
    run_id: Optional[int]
    model: Optional[ModelInfo]
    git_sha: str
    torch_version: str
    sklearn_version: str
    supabase_project_ref: Optional[str]
